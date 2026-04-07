import numpy as np
import librosa
import torch

SAMPLE_RATE = 22050   

N_FFT       = 1024    

HOP_LENGTH  = 256     

N_MELS      = 80      

F_MIN       = 0       
F_MAX       = 8000    

MIN_DB      = -100
REF_DB      = 20

def cargar_wav(ruta):
    """
    Carga un archivo WAV desde disco.

    Siempre usamos SAMPLE_RATE definido arriba.
    Si el archivo tiene otro sample rate librosa lo convierte.

    Devuelve:
        audio → array numpy de forma (N,)
                N = número de muestras
                valores entre -1.0 y 1.0
    """
    audio, _ = librosa.load(ruta, sr=SAMPLE_RATE)
    return audio


def normalizar_volumen(audio):
    """
    Normaliza el volumen del audio para que el valor
    máximo absoluto sea 1.0.

    ¿Por qué?
    Los audios del dataset pueden tener volúmenes muy distintos.
    Un audio grabado cerca del micrófono tiene valores grandes.
    Uno grabado lejos tiene valores pequeños.
    El modelo no debería aprender esas diferencias de volumen
    — solo le interesa el contenido fonético.

    Ejemplo:
        audio = [0.1, 0.3, -0.2, 0.5, -0.4]
        máximo absoluto = 0.5
        normalizado = [0.2, 0.6, -0.4, 1.0, -0.8]
    """
    valor_maximo = np.max(np.abs(audio))

    if valor_maximo == 0:
        return audio

    return audio / valor_maximo


def wav_a_mel(audio):
    """
    Convierte un array de audio en un Mel Spectrogram.

    El proceso internamente es:
        audio → STFT → magnitudes → filtros Mel → escala log

    Devuelve:
        mel → tensor PyTorch de forma (N_MELS, T)
              N_MELS = 80 frecuencias
              T      = número de frames temporales
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX
    )

    mel_db = librosa.power_to_db(mel, ref=REF_DB, top_db=None)


    mel_normalizado = np.clip((mel_db - MIN_DB) / (-MIN_DB), 0, 1)
    mel_normalizado = mel_normalizado * 2 - 1

    return torch.tensor(mel_normalizado, dtype=torch.float32)


def mel_a_wav(mel_tensor):
    """
    Convierte un Mel Spectrogram de vuelta a audio.
    Usa Griffin-Lim — reconstrucción matemática sin ML.

    Útil para escuchar lo que el modelo está generando
    durante el entrenamiento, sin necesitar HiFi-GAN.

    mel_tensor → tensor PyTorch de forma (N_MELS, T)
                 o (T, N_MELS) — lo detecta automáticamente

    Devuelve:
        audio → array numpy de forma (N,)
    """

    mel = mel_tensor.numpy() if isinstance(mel_tensor, torch.Tensor) \
          else mel_tensor

    if mel.shape[0] != N_MELS:
        mel = mel.T

    mel = (mel + 1) / 2                    
    mel_db = mel * (-MIN_DB) + MIN_DB      

    mel_potencia = librosa.db_to_power(mel_db, ref=REF_DB)

    stft_magnitud = librosa.feature.inverse.mel_to_stft(
        mel_potencia,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        fmin=F_MIN,
        fmax=F_MAX
    )

    audio = librosa.griffinlim(
        stft_magnitud,
        n_iter=60,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )

    return audio


def duracion_frames(ruta):
    """
    Calcula cuántos frames tendrá el Mel Spectrogram
    de un archivo WAV sin cargarlo completamente.

    Útil para filtrar audios demasiado largos o cortos
    antes de cargar el dataset completo.
    """
    duracion_segundos = librosa.get_duration(path=ruta)
    muestras          = duracion_segundos * SAMPLE_RATE
    frames            = int(muestras) // HOP_LENGTH
    return frames


if __name__ == '__main__':
    import sys
    import soundfile as sf
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print("Uso: uv run python src/data/audio.py ruta/al/audio.wav")
        sys.exit(1)

    ruta = sys.argv[1]
    print(f"Cargando: {ruta}")

    audio   = cargar_wav(ruta)
    audio   = normalizar_volumen(audio)
    mel     = wav_a_mel(audio)

    print(f"\nAudio:")
    print(f"  muestras:  {len(audio)}")
    print(f"  duración:  {len(audio)/SAMPLE_RATE:.2f} segundos")
    print(f"  min/max:   {audio.min():.3f} / {audio.max():.3f}")

    print(f"\nMel Spectrogram:")
    print(f"  forma:     {mel.shape}  (frecuencias × frames)")
    print(f"  min/max:   {mel.min():.3f} / {mel.max():.3f}")

    audio_reconstruido = mel_a_wav(mel)
    sf.write("reconstruido.wav", audio_reconstruido, SAMPLE_RATE)
    print(f"\nAudio reconstruido guardado en reconstruido.wav")
    print(f"Escúchalo y compara con el original")

    plt.figure(figsize=(12, 4))
    plt.imshow(
        mel.numpy(),
        aspect='auto',
        origin='lower',
        interpolation='none'
    )
    plt.colorbar(label='Amplitud normalizada')
    plt.xlabel('Frames de tiempo')
    plt.ylabel('Bandas de frecuencia Mel')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig('mel_spectrogram.png')
    print(f"Imagen guardada en mel_spectrogram.png")