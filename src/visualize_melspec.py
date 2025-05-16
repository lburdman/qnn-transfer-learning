import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# Función auxiliar para convertir waveform → MelTensor
# =====================================================
def audio_to_melspec_tensor(waveform, sample_rate=16000, n_mels=128, duration=3, hop_length=512):
    fixed_length = sample_rate * duration
    if waveform.shape[1] < fixed_length:
        padding = fixed_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :fixed_length]

    y = waveform.squeeze().numpy()
    import librosa
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_tensor = torch.tensor(mel_spec_db).unsqueeze(0)  # (1, n_mels, time)
    return mel_tensor

# =====================================================
# Visualiza una grilla de espectrogramas desde archivos
# =====================================================
def plot_melspectrograms_from_files(file_dict):
    plt.figure(figsize=(15, 5))
    for idx, (label, file_path) in enumerate(file_dict.items()):
        waveform, sr = torchaudio.load(file_path)
        mel_tensor = audio_to_melspec_tensor(waveform, sample_rate=sr)
        mel_np = mel_tensor.squeeze(0).numpy()

        plt.subplot(1, len(file_dict), idx + 1)
        plt.imshow(mel_np, origin='lower', aspect='auto', cmap='magma')
        plt.title(label)
        plt.xlabel("Time")
        plt.ylabel("Mel frequency bins")
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

    plt.show()

# =====================================================
# Ejemplo de uso para 3 emociones (test manual)
# =====================================================
if __name__ == "__main__":
    BASE_DIR = "C:/Users/Lucas/Documents/Facultad/Tesis"
    _SAMPLE_DIR = os.path.join(BASE_DIR,"./CremaD/data_filtered")

    files = {
        "Anger": os.path.join(_SAMPLE_DIR, "1003_IOM_ANG_XX.wav"),
        "Happy": os.path.join(_SAMPLE_DIR, "1003_IOM_HAP_XX.wav"),
        "Sad":   os.path.join(_SAMPLE_DIR, "1003_IOM_SAD_XX.wav")
    }

    plot_melspectrograms_from_files(files)
