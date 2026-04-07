import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, canales, kernel_size=3, dilaciones=[1, 3, 5]):
        super().__init__()

        self.convs = nn.ModuleList()
        for d in dilaciones:
            self.convs.append(
                nn.Conv1d(
                    canales, canales,
                    kernel_size=kernel_size,
                    dilation=d,
                    padding=d * (kernel_size - 1) // 2
                )
            )

    def forward(self, x):
        for conv in self.convs:
            residual = x
            x = F.relu(x)
            x = conv(x)
            x = x + residual
        return x


class Generador(nn.Module):

    def __init__(self, n_mels=80):
        super().__init__()

        self.conv_inicial = nn.Conv1d(n_mels, 512, kernel_size=7, padding=3)

        self.upsample_blocks = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4),
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4),
            nn.ConvTranspose1d(128, 64,  kernel_size=4,  stride=2, padding=1),
            nn.ConvTranspose1d(64,  32,  kernel_size=4,  stride=2, padding=1),
        ])

        self.res_blocks = nn.ModuleList([
            ResBlock(256),
            ResBlock(128),
            ResBlock(64),
            ResBlock(32),
        ])

        self.conv_final = nn.Sequential(
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, mel):
        x = self.conv_inicial(mel)

        for upsample, resblock in zip(self.upsample_blocks, self.res_blocks):
            x = F.relu(x)
            x = upsample(x)
            x = resblock(x)

        return self.conv_final(x)


class Discriminador(nn.Module):

    def __init__(self):
        super().__init__()

        self.capas = nn.Sequential(
            nn.Conv1d(1,   16,  kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.1),
            nn.Conv1d(16,  64,  kernel_size=41, stride=4, padding=20),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64,  256, kernel_size=41, stride=4, padding=20),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 512, kernel_size=41, stride=4, padding=20),
            nn.LeakyReLU(0.1),
            nn.Conv1d(512, 1,   kernel_size=3,  stride=1, padding=1),
        )

    def forward(self, audio):
        return self.capas(audio)


if __name__ == '__main__':

    generador    = Generador(n_mels=80)
    discriminador = Discriminador()

    total_gen  = sum(p.numel() for p in generador.parameters())
    total_disc = sum(p.numel() for p in discriminador.parameters())
    print(f'Parámetros generador:     {total_gen:,}')
    print(f'Parámetros discriminador: {total_disc:,}')

    mel   = torch.randn(1, 80, 100)
    audio = generador(mel)
    print(f'\nMel entrada:    {mel.shape}')
    print(f'Audio generado: {audio.shape}')

    score = discriminador(audio)
    print(f'Score:          {score.shape}')
    print(f'  → puntuación por posición temporal')
    print(f'  → media = {score.mean().item():.4f}')