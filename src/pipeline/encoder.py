import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_conv_layers=3):
        """
        vocab_size    → tamaño del vocabulario

        embedding_dim → tamaño de cada vector de embedding
                        típico: 512

        hidden_dim    → tamaño del estado interno de la LSTM
                        típico: 512

        num_conv_layers → número de capas de convolución
                        típico: 3
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        capas_conv = []

        for _ in range(num_conv_layers):
            capas_conv += [
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5),
            ]

        self.convolutions = nn.Sequential(*capas_conv)


        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.proyeccion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, indices):
        """
        indices → (batch, L)  tensor de índices de caracteres

        devuelve:
            contexto → (batch, L, hidden_dim)
        """

        x = self.embedding(indices)

        x = x.transpose(1, 2)
        x = self.convolutions(x)
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)

        x = self.proyeccion(x)

        x = torch.tanh(x)

        return x


if __name__ == '__main__':
    from src.data.text import VOCAB_SIZE, texto_a_indices

    EMBEDDING_DIM = 512
    HIDDEN_DIM    = 512

    encoder = Encoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

    total_params = sum(p.numel() for p in encoder.parameters()
                       if p.requires_grad)
    print(f"Parámetros del encoder: {total_params:,}")

    frase   = "Hola, ¿en qué te puedo ayudar?"
    indices = texto_a_indices(frase)
    tensor  = torch.tensor(indices).unsqueeze(0)

    print(f"\nFrase:          '{frase}'")
    print(f"Longitud texto: {len(indices)} caracteres")
    print(f"Forma entrada:  {tensor.shape}")

    with torch.no_grad():
        salida = encoder(tensor)

    print(f"Forma salida:   {salida.shape}")
    print(f"  → {salida.shape[1]} vectores  (uno por carácter)")
    print(f"  → {salida.shape[2]} números   (cada vector)")
    print(f"\nValores de salida (primeros 5 del primer carácter):")
    print(f"  {salida[0, 0, :5].tolist()}")
    print(f"  todos entre -1 y 1: {salida.min().item():.3f} / {salida.max().item():.3f}")