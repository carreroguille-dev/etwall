import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import chain
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.text    import VOCAB_SIZE
from src.data.dataset import TTSDataset, collate_fn
from src.pipeline.encoder   import Encoder
from src.pipeline.attention import Attention
from src.pipeline.decoder   import Decoder



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

        batch_size     = indices.shape[0]
        h              = torch.zeros(batch_size, decoder.hidden_dim).to(device)
        c              = torch.zeros(batch_size, decoder.hidden_dim).to(device)
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


def guardar_checkpoint(ruta, epoch, encoder, attention,
                       decoder, opt_acustico):

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    torch.save({
        'epoch':        epoch,
        'encoder':      encoder.state_dict(),
        'attention':    attention.state_dict(),
        'decoder':      decoder.state_dict(),
        'opt_acustico': opt_acustico.state_dict(),
        'scheduler':    scheduler.state_dict(),
    }, ruta)
    print(f'Checkpoint guardado: {ruta}')


def cargar_checkpoint(ruta, encoder, attention,
                      decoder, opt_acustico):

    if not os.path.exists(ruta):
        print(f'No existe checkpoint — empezando desde cero')
        return 0
    
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    checkpoint = torch.load(ruta, weights_only=True)
    encoder.load_state_dict(checkpoint['encoder'])
    attention.load_state_dict(checkpoint['attention'])
    decoder.load_state_dict(checkpoint['decoder'])
    opt_acustico.load_state_dict(checkpoint['opt_acustico'])

    epoch = checkpoint['epoch']
    print(f'Checkpoint cargado — continuando desde época {epoch}')
    return epoch


def main():

    # ── hiperparámetros ───────────────────────────────────────
    HIDDEN_DIM    = 512
    EMBEDDING_DIM = 512
    N_MELS        = 80
    BATCH_SIZE    = 32
    EPOCHS        = 200
    LR            = 0.001
    CHECKPOINT    = '/content/drive/MyDrive/etwall_checkpoints/ultimo.pt'
    DATA_DIR      = 'data/css10_es'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Entrenando en: {device}')

    # ── crear modelos ─────────────────────────────────────────
    encoder   = Encoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    attention = Attention(HIDDEN_DIM).to(device)
    decoder   = Decoder(HIDDEN_DIM, N_MELS).to(device)

    # ── optimizador ───────────────────────────────────────────
    opt_acustico = torch.optim.Adam(
        chain(encoder.parameters(),
              attention.parameters(),
              decoder.parameters()),
        lr=LR
    )

    # ── scheduler ────────────────────────────────────────────────
    scheduler = ReduceLROnPlateau(
        opt_acustico,
        mode='min',      
        factor=0.5,      
        patience=5,      
        verbose=True     
    )

    # ── cargar checkpoint si existe ───────────────────────────
    epoch_inicio = cargar_checkpoint(
        CHECKPOINT,
        encoder, attention, decoder,
        opt_acustico
    )

    # ── dataset y dataloader ──────────────────────────────────
    dataset    = TTSDataset(DATA_DIR)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    print(f'Dataset:          {len(dataset)} muestras')
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

        scheduler.step(perdida)

        if epoch % 5 == 0:
            guardar_checkpoint(
                CHECKPOINT,
                epoch, encoder, attention, decoder,
                opt_acustico
            )


if __name__ == '__main__':
    main()