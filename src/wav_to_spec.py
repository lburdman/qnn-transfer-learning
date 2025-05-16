import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torchaudio

def process_audio_to_melspec(
    input_path, output_path=None, label=None,
    sample_rate=16000, duration=3, n_mels=128,
    save=True
):
    """
    Convierte un archivo .wav a un espectrograma mel.
    Si save=True, guarda la imagen como .png en la carpeta output_path/label/.

    Parámetros:
        input_path (str): ruta al archivo .wav
        output_path (str): carpeta base donde guardar (si save=True)
        label (str): subcarpeta (por clase) donde guardar
        sample_rate (int): frecuencia de muestreo
        duration (int): duración máxima en segundos
        n_mels (int): número de bandas mel
        save (bool): si True, guarda como imagen; si False, solo muestra
    """
    signal, sr = librosa.load(input_path, sr=sample_rate)
    desired_length = sr * duration

    # Padding o recorte
    if len(signal) < desired_length:
        signal = np.pad(signal, (0, desired_length - len(signal)), mode='constant')
    else:
        signal = signal[:desired_length]

    # Espectrograma
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Visualización
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.axis('off')
    plt.tight_layout()

    if save and output_path and label:
        os.makedirs(os.path.join(output_path, label), exist_ok=True)
        base_name = os.path.basename(input_path).replace(".wav", ".png")
        out_file = os.path.join(output_path, label, base_name)
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
        plt.close()
    elif not save:
        plt.show()
    else:
        raise ValueError("Si save=True, se debe especificar output_path y label")

# ------------------------------------------------------
# 🧪 TEST: solo visualizar un archivo sin guardar nada
# ------------------------------------------------------

if __name__ == "__main__":
    _SAMPLE_DIR = r"C:\Users\Lucas\Documents\Facultad\Tesis\CremaD\data_filtered"
    file_path = os.path.join(_SAMPLE_DIR, "1003_IOM_SAD_XX.wav")
    waveform, sr = torchaudio.load(file_path)

    # test_file = "C:\Users\Lucas\Documents\Facultad\Tesis\CremaD\data_filtered\1003_IOM_SAD_XX.wav" 

    print("🔍 Mostrando espectrograma de prueba (sin guardar)")
    process_audio_to_melspec(
        input_path=file_path,
        save=False
    )
