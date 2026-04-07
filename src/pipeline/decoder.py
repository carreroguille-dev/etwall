import torch
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, hidden_dim, n_mels):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_mels     = n_mels

        self.prenet = nn.Sequential(
            nn.Linear(n_mels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTMCell(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim
        )

        self.capa_parada = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.capa_mel = nn.Linear(hidden_dim, n_mels)

    def forward_step(self, frame_anterior, vector_contexto, estado_lstm):
        x = self.prenet(frame_anterior)

        x = torch.cat((x, vector_contexto), dim=1)

        h, c = self.lstm(x, estado_lstm)

        prob_parada = self.capa_parada(h)

        frame_nuevo = self.capa_mel(h)

        return frame_nuevo, prob_parada, (h, c)

    def forward(self, encoder_salida, attention, max_frames=500):
        batch_size = encoder_salida.size(0)

        h = torch.zeros(batch_size, self.hidden_dim)
        c = torch.zeros(batch_size, self.hidden_dim)

        frame_anterior = torch.zeros(batch_size, self.n_mels)
        frames = []

        for _ in range(max_frames):
            vector_contexto, pesos_contexto = attention(encoder_salida, h)
            
            frame_nuevo, prob_parada, (h, c) = self.forward_step(frame_anterior, vector_contexto, (h, c))

            frames.append(frame_nuevo)
            frame_anterior = frame_nuevo

            if prob_parada.mean() > 0.5:
                break

        return torch.stack(frames, dim=1)

if __name__ == '__main__':
    from src.data.text import VOCAB_SIZE, texto_a_indices
    from src.pipeline.encoder import Encoder
    from src.pipeline.attention import Attention

    HIDDEN_DIM = 128
    N_MELS     = 80

    encoder   = Encoder(VOCAB_SIZE, 128, HIDDEN_DIM)
    attention = Attention(HIDDEN_DIM)
    decoder   = Decoder(HIDDEN_DIM, N_MELS)

    frase   = "hola"
    indices = torch.tensor(texto_a_indices(frase)).unsqueeze(0)

    with torch.no_grad():
        enc_salida = encoder(indices)
        mel        = decoder(enc_salida, attention)

    print(f"Entrada:  '{frase}'")
    print(f"Encoder:  {enc_salida.shape}")
    print(f"Mel spec: {mel.shape}")
    print(f"  → {mel.shape[1]} frames generados")
    print(f"  → {mel.shape[2]} frecuencias por frame")