import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import chain
import os

from src.data.text    import VOCAB_SIZE
from src.data.dataset import TTSDataset, collate_fn
from src.pipeline.encoder   import Encoder
from src.pipeline.attention import Attention
from src.pipeline.decoder   import Decoder
from src.pipeline.hifigan   import Generador, Discriminador


def entrenar_modelo_acustico(encoder, attention, decoder,
                              dataloader, optimizador,
                              device, epoch):

    encoder.train()
    attention.train()
    decoder.train()

    perdida_total = 0.0
    loss_fn       = nn.MSELoss()

    for batch in dataloader:

        indices, mel_real, long_texto, long_mel = batch
        indices  = indices.to(device)
        mel_real = mel_real.to(device)

        enc_salida = encoder(indices)

        batch_size = indices.shape[0]
        h = torch.zeros(batch_size, decoder.hidden_dim).to(device)
        c  = torch.zeros(batch_size, decoder.hidden_dim).to(device)
        frame_anterior = torch.zeros(batch_size, decoder.n_mels).to(device)

        frames = []

        for step in range(mel_real.shape[2]):

            vector_contexto, _ = attention(enc_salida, h)

            frame_nuevo, _, (h, c) = decoder.forward_step(
                frame_anterior, vector_contexto, (h, c)
            )

            frame_anterior = mel_real[:, :, step]
            frames.append(frame_nuevo)

        mel_generado = torch.stack(frames, dim=1)
        mel_real_t   = mel_real.transpose(1, 2)

        perdida = loss_fn(mel_generado, mel_real_t)

        optimizador.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(
            list(encoder.parameters()) +
            list(attention.parameters()) +
            list(decoder.parameters()),
            max_norm=1.0
        )
        optimizador.step()

        perdida_total += perdida.item()

    perdida_media = perdida_total / len(dataloader)
    print(f'Época {epoch} — pérdida acústica: {perdida_media:.4f}')
    return perdida_media


def entrenar_hifigan(generador, discriminador,
                     dataloader,
                     opt_gen, opt_disc,
                     device, epoch):

    generador.train()
    discriminador.train()

    loss_fn       = nn.MSELoss()
    perdida_gen   = 0.0
    perdida_disc  = 0.0

    for batch in dataloader:

        _, mel_real, _, _ = batch
        mel_real = mel_real.to(device)

        # audio real para el discriminador
        # en producción vendría del dataset directamente
        # aquí lo generamos desde el mel real como aproximación
        audio_real = mel_real.mean(dim=1, keepdim=True)
        audio_real = audio_real.expand(-1, 1, mel_real.shape[2] * 256)

        # ── entrenar discriminador ────────────────────────────
        audio_falso  = generador(mel_real).detach()
        score_real   = discriminador(audio_real)
        score_falso  = discriminador(audio_falso)

        loss_d = (
            loss_fn(score_real,  torch.ones_like(score_real)) +
            loss_fn(score_falso, torch.zeros_like(score_falso))
        )

        opt_disc.zero_grad()
        loss_d.backward()
        opt_disc.step()

        # ── entrenar generador ────────────────────────────────
        audio_falso  = generador(mel_real)
        score_falso  = discriminador(audio_falso)

        loss_g = loss_fn(score_falso, torch.ones_like(score_falso))

        opt_gen.zero_grad()
        loss_g.backward()
        opt_gen.step()

        perdida_disc += loss_d.item()
        perdida_gen  += loss_g.item()

    print(f'Época {epoch} — pérdida discriminador: {perdida_disc/len(dataloader):.4f} '
          f'pérdida generador: {perdida_gen/len(dataloader):.4f}')
    return perdida_gen / len(dataloader)


def guardar_checkpoint(ruta, epoch, encoder, attention,
                       decoder, generador, discriminador,
                       opt_acustico, opt_gen, opt_disc):

    os.makedirs(os.path.dirname(ruta), exist_ok=True)

    torch.save({
        'epoch':          epoch,
        'encoder':        encoder.state_dict(),
        'attention':      attention.state_dict(),
        'decoder':        decoder.state_dict(),
        'generador':      generador.state_dict(),
        'discriminador':  discriminador.state_dict(),
        'opt_acustico':   opt_acustico.state_dict(),
        'opt_gen':        opt_gen.state_dict(),
        'opt_disc':       opt_disc.state_dict(),
    }, ruta)

    print(f'Checkpoint guardado: {ruta}')


def cargar_checkpoint(ruta, encoder, attention,
                      decoder, generador, discriminador,
                      opt_acustico, opt_gen, opt_disc):

    if not os.path.exists(ruta):
        print(f'No existe checkpoint en {ruta} — empezando desde cero')
        return 0

    checkpoint = torch.load(ruta)

    encoder.load_state_dict(checkpoint['encoder'])
    attention.load_state_dict(checkpoint['attention'])
    decoder.load_state_dict(checkpoint['decoder'])
    generador.load_state_dict(checkpoint['generador'])
    discriminador.load_state_dict(checkpoint['discriminador'])
    opt_acustico.load_state_dict(checkpoint['opt_acustico'])
    opt_gen.load_state_dict(checkpoint['opt_gen'])
    opt_disc.load_state_dict(checkpoint['opt_disc'])

    epoch = checkpoint['epoch']
    print(f'Checkpoint cargado — continuando desde época {epoch}')
    return epoch


def main():

    from itertools import chain

    # ── hiperparámetros ───────────────────────────────────────
    HIDDEN_DIM    = 512
    EMBEDDING_DIM = 512
    N_MELS        = 80
    BATCH_SIZE    = 16
    EPOCHS        = 100
    LR_ACUSTICO   = 0.001
    LR_HIFIGAN    = 0.0002
    CHECKPOINT    = 'data/checkpoints/ultimo.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Entrenando en: {device}')

    # ── crear modelos ─────────────────────────────────────────
    encoder       = Encoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    attention     = Attention(HIDDEN_DIM).to(device)
    decoder       = Decoder(HIDDEN_DIM, N_MELS).to(device)
    generador     = Generador(N_MELS).to(device)
    discriminador = Discriminador().to(device)

    # ── crear optimizadores ───────────────────────────────────
    opt_acustico = torch.optim.Adam(
        chain(encoder.parameters(),
              attention.parameters(),
              decoder.parameters()),
        lr=LR_ACUSTICO
    )
    
    # ── cargar checkpoint si existe ───────────────────────────
    epoch_inicio = cargar_checkpoint(
        CHECKPOINT,
        encoder, attention, decoder,
        generador, discriminador,
        opt_acustico
    )

    # ── dataset y dataloader ──────────────────────────────────
    dataset    = TTSDataset('data/css10_es')
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    print(f'Dataset: {len(dataset)} muestras')
    print(f'Batches por época: {len(dataloader)}')

    # ── bucle de épocas ───────────────────────────────────────
    for epoch in range(epoch_inicio + 1, EPOCHS + 1):
        print(f'\n{"="*50}')
        print(f'Época {epoch}/{EPOCHS}')
        print(f'{"="*50}')

        entrenar_modelo_acustico(
            encoder, attention, decoder,
            dataloader, opt_acustico,
            device, epoch
        )

        if epoch % 10 == 0:
            guardar_checkpoint(
                CHECKPOINT,
                epoch, encoder, attention, decoder,
                generador, discriminador,
                opt_acustico, opt_gen, opt_disc
            )


if __name__ == '__main__':
    main()