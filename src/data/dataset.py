import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.text  import texto_a_indices, PAD_IDX
from src.data.audio import cargar_wav, normalizar_volumen, wav_a_mel


DURACION_MIN = 1.0   
DURACION_MAX = 10.0  


class TTSDataset(Dataset):
    """
    Dataset que carga pares (texto, mel_spectrogram) del CSS10.

    Cada elemento devuelto es una tupla:
        (indices_texto, mel_spectrogram)

        indices_texto   → tensor LongTensor  (L,)
                          L = longitud del texto en caracteres

        mel_spectrogram → tensor FloatTensor (N_MELS, T)
                          N_MELS = 80 frecuencias
                          T      = frames de tiempo
    """

    def __init__(self, directorio_datos, verbose=True):
        """
        directorio_datos → ruta a la carpeta css10_es/
                           debe contener transcript.txt
                           y las subcarpetas de audio

        verbose          → si True muestra estadísticas al cargar
        """
        self.directorio = directorio_datos
        self.muestras   = []   

        ruta_transcript = os.path.join(directorio_datos, 'transcript.txt')

        total       = 0
        descartados = 0

        with open(ruta_transcript, 'r', encoding='utf-8') as f:
            for linea in f:
                linea = linea.strip()

                if not linea:
                    continue

                total += 1

                partes = linea.split('|')
                if len(partes) != 4:
                    descartados += 1
                    continue

                ruta_relativa = partes[0]   
                texto         = partes[2]   
                duracion      = float(partes[3])

                if duracion < DURACION_MIN or duracion > DURACION_MAX:
                    descartados += 1
                    continue

                if not texto.strip():
                    descartados += 1
                    continue

                ruta_wav = os.path.join(directorio_datos, ruta_relativa)

                if not os.path.exists(ruta_wav):
                    descartados += 1
                    continue

                self.muestras.append((ruta_wav, texto))

        if verbose:
            print(f"Dataset cargado:")
            print(f"  total líneas:     {total}")
            print(f"  descartados:      {descartados}")
            print(f"  muestras válidas: {len(self.muestras)}")
            print(f"  duración mín:     {DURACION_MIN}s")
            print(f"  duración máx:     {DURACION_MAX}s")

    def __len__(self):
        """Cuántas muestras tiene el dataset."""
        return len(self.muestras)

    def __getitem__(self, idx):
        """
        Carga y devuelve la muestra en la posición idx.

        PyTorch llama a esta función automáticamente
        cuando itera el DataLoader — no la llamas tú directamente.
        """
        ruta_wav, texto = self.muestras[idx]

        audio = cargar_wav(ruta_wav)
        audio = normalizar_volumen(audio)
        mel   = wav_a_mel(audio)

        indices = texto_a_indices(texto)
        indices = torch.tensor(indices, dtype=torch.long)

        return indices, mel


def collate_fn(batch):
    """
    Agrupa un batch de muestras en tensores.

    El problema es que cada muestra tiene distinta longitud:
        muestra 1: texto de 50 chars, mel de 300 frames
        muestra 2: texto de 30 chars, mel de 180 frames
        muestra 3: texto de 70 chars, mel de 420 frames

    Para meterlas en un tensor necesitamos que todas tengan
    el mismo tamaño — rellenamos con PAD hasta la más larga.

    Devuelve:
        indices_pad  → (batch, L_max)    texto con padding
        mel_pad      → (batch, N_MELS, T_max)  mel con padding
        long_texto   → lista con longitudes reales de cada texto
        long_mel     → lista con longitudes reales de cada mel
    """
    indices_lista, mel_lista = zip(*batch)

    long_texto = [len(x)         for x in indices_lista]
    long_mel   = [x.shape[1]     for x in mel_lista]

    max_texto  = max(long_texto)
    max_mel    = max(long_mel)
    n_mels     = mel_lista[0].shape[0]   

    indices_pad = torch.full(
        (len(batch), max_texto),
        fill_value=PAD_IDX,
        dtype=torch.long
    )
    mel_pad = torch.zeros(len(batch), n_mels, max_mel)

    for i, (indices, mel) in enumerate(zip(indices_lista, mel_lista)):
        indices_pad[i, :len(indices)]   = indices
        mel_pad    [i, :, :mel.shape[1]] = mel

    return indices_pad, mel_pad, long_texto, long_mel


def crear_dataloader(directorio_datos, batch_size=16,
                     shuffle=True, num_workers=0):
    """
    Crea y devuelve un DataLoader listo para entrenar.

    batch_size   → cuántas muestras por batch

    shuffle      → True durante entrenamiento
                   False durante validación

    num_workers  → procesos paralelos para cargar datos
                   0 = carga en el proceso principal (más simple)
    """
    dataset = TTSDataset(directorio_datos)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True   
    )

    return dataloader


if __name__ == '__main__':
    import sys

    directorio = sys.argv[1] if len(sys.argv) > 1 \
                 else 'data/css10_es'

    print(f"Probando dataset en: {directorio}\n")

    dataset = TTSDataset(directorio)
    print()

    indices, mel = dataset[0]
    print(f"Primera muestra:")
    print(f"  texto índices: {indices.shape}  → {indices.tolist()}")
    print(f"  mel shape:     {mel.shape}  (frecuencias × frames)")

    indices2, mel2 = dataset[1]
    print(f"\nSegunda muestra:")
    print(f"  texto índices: {indices2.shape}")
    print(f"  mel shape:     {mel2.shape}")

    print(f"\nProbando DataLoader con batch_size=4...")
    dataloader = crear_dataloader(directorio, batch_size=4, shuffle=False)

    indices_batch, mel_batch, long_texto, long_mel = next(iter(dataloader))

    print(f"\nBatch:")
    print(f"  indices shape:  {indices_batch.shape}")
    print(f"  mel shape:      {mel_batch.shape}")
    print(f"  long textos:    {long_texto}")
    print(f"  long mels:      {long_mel}")