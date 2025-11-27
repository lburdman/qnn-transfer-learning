"""
Visualization helpers for waveforms, mel-spectrograms, and related features.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Sequence

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


# Used in: src.wav_to_spec.py (internal helper reuse), data_analysis.ipynb (mfcc previews)
def audio_to_melspec_tensor(waveform: torch.Tensor, sample_rate: int = 16000, n_mels: int = 128,
                            duration: int = 3, hop_length: int = 512) -> torch.Tensor:
    """
    Convert a waveform tensor to a mel-spectrogram tensor with a fixed duration.

    Args:
        waveform: Input waveform tensor with shape (channels, time).
        sample_rate: Sampling rate of the waveform.
        n_mels: Number of mel filter banks.
        duration: Duration in seconds for padding/truncation.
        hop_length: Hop length for STFT.

    Returns:
        Tensor with shape (1, n_mels, time_frames).
    """
    fixed_length = sample_rate * duration
    if waveform.shape[1] < fixed_length:
        padding = fixed_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :fixed_length]

    audio_np = waveform.squeeze().numpy()
    mel_spec = librosa.feature.melspectrogram(y=audio_np, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_tensor = torch.tensor(mel_spec_db).unsqueeze(0)
    return mel_tensor


# Used in: audio_preprocessing.ipynb (quick visualization), data_analysis.ipynb (examples)
def show_example_spectrograms(spec_root: str | Path, n_classes: int = 6) -> None:
    """
    Display one spectrogram image per class from a root directory.

    Args:
        spec_root: Root directory containing class subfolders with PNG images.
        n_classes: Maximum number of classes to display.
    """
    spec_root = Path(spec_root)
    class_dirs = sorted([directory for directory in spec_root.iterdir() if directory.is_dir()])[:n_classes]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.ravel()
    for ax, class_dir in zip(axes, class_dirs):
        files = list(class_dir.glob("*.png"))
        if not files:
            ax.axis("off")
            continue
        img_path = random.choice(files)
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(class_dir.name)
        ax.axis("off")

    for ax in axes[len(class_dirs):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# Used in: audio_preprocessing.ipynb (MFCC preview)
def show_example_mfcc(mfcc: np.ndarray, title: str = "MFCC example") -> None:
    """
    Display an MFCC array with time axis.

    Args:
        mfcc: MFCC array.
        title: Plot title.
    """
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mfcc, x_axis="time")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Used in: data_analysis.ipynb (waveform/mel visualization)
def plot_waveform(audio_path: str | Path, ax=None, sr: int = 22050):
    """
    Plot waveform amplitude over time.

    Args:
        audio_path: Path to an audio file.
        ax: Optional matplotlib axis.
        sr: Sampling rate for loading.

    Returns:
        Matplotlib axis with the waveform.
    """
    waveform, sr_loaded = librosa.load(audio_path, sr=sr)
    times = np.linspace(0, len(waveform) / sr_loaded, num=len(waveform))
    ax = ax or plt.gca()
    ax.plot(times, waveform, color="steelblue", linewidth=0.8)
    ax.set_title(f"Waveform: {Path(audio_path).name}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    return ax


# Used in: data_analysis.ipynb (waveform/mel visualization)
def plot_melspectrogram(audio_path: str | Path, sr: int | None = None, ax=None,
                        n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128):
    """
    Plot a mel-spectrogram for an audio file.

    Args:
        audio_path: Path to the audio file.
        sr: Optional sampling rate override.
        ax: Optional matplotlib axis.
        n_fft: FFT window length.
        hop_length: Hop length for STFT.
        n_mels: Number of mel filter banks.

    Returns:
        Matplotlib axis with the mel-spectrogram.
    """
    waveform, sr_loaded = librosa.load(audio_path, sr=sr or 22050)
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr_loaded,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    ax = ax or plt.gca()
    img = librosa.display.specshow(
        log_mel,
        sr=sr_loaded,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )
    ax.set_title(f"Mel-spectrogram: {Path(audio_path).name}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mel frequency bins")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    return ax


# Used in: data_analysis.ipynb (example visualization)
def plot_example_waveform_and_mel(row: Dict[str, str]) -> None:
    """
    Plot waveform and mel-spectrogram side by side for a metadata row.

    Args:
        row: Metadata row containing `file_path` and `label`.
    """
    audio_path = row["file_path"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    plot_waveform(audio_path, ax=axes[0])
    plot_melspectrogram(audio_path, ax=axes[1])
    fig.suptitle(f"{row['label']} - {Path(audio_path).name}")
    plt.tight_layout()
    plt.show()


# Used in: data_analysis.ipynb (MFCC visualization)
def plot_example_waveform_and_mfcc(row: Dict[str, str], audio_path: str | None = None) -> None:
    """
    Plot waveform and MFCC features for a metadata row.

    Args:
        row: Metadata row containing optional `mfcc_path` and `file_path`.
        audio_path: Optional explicit audio path override.
    """
    mfcc_path = row.get("mfcc_path")
    mfcc_arr = None
    if mfcc_path and Path(mfcc_path).exists():
        data = np.load(mfcc_path, allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile):
            key = "mfcc" if "mfcc" in data.files else data.files[0]
            mfcc_arr = data[key]
        else:
            mfcc_arr = data

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    candidate_audio = audio_path or row.get("file_path")
    if candidate_audio and Path(candidate_audio).exists():
        plot_waveform(candidate_audio, ax=axes[0])
    else:
        axes[0].text(0.5, 0.5, "Audio not found", ha="center", va="center")
        axes[0].set_title("Waveform unavailable")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

    if mfcc_arr is not None:
        mfcc_2d = np.atleast_2d(mfcc_arr)
        if mfcc_2d.ndim > 2:
            mfcc_2d = mfcc_2d.squeeze()
        img = librosa.display.specshow(mfcc_2d, x_axis="time", ax=axes[1])
        axes[1].set_title(f"MFCC: {Path(mfcc_path).name}")
        axes[1].set_ylabel("MFCC coefficients")
        axes[1].set_xlabel("Frames")
        plt.colorbar(img, ax=axes[1], format="%+2.0f")
    else:
        axes[1].text(0.5, 0.5, "MFCC not found", ha="center", va="center")
        axes[1].set_title("MFCC unavailable")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    fig.suptitle(f"{row.get('label', 'Unknown')} - {Path(candidate_audio).name if candidate_audio else row.get('file_stem', 'N/A')}")
    plt.tight_layout()
    plt.show()


# Used in: src.visualize_melspec.py (manual test), visualize_melspec.ipynb derivatives
def plot_melspectrograms_from_files(file_dict: Dict[str, str]) -> None:
    """
    Plot a grid of mel-spectrograms given a mapping from label to audio path.

    Args:
        file_dict: Mapping of label to audio file path.
    """
    plt.figure(figsize=(15, 5))
    for idx, (label, file_path) in enumerate(file_dict.items()):
        waveform, sr = torchaudio.load(file_path)
        mel_tensor = audio_to_melspec_tensor(waveform, sample_rate=sr)
        mel_np = mel_tensor.squeeze(0).numpy()

        plt.subplot(1, len(file_dict), idx + 1)
        plt.imshow(mel_np, origin="lower", aspect="auto", cmap="magma")
        plt.title(label)
        plt.xlabel("Time")
        plt.ylabel("Mel frequency bins")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()

    plt.show()


# Example usage for manual testing
if __name__ == "__main__":
    base_dir = "C:/Users/Lucas/Documents/Facultad/Tesis"
    sample_dir = os.path.join(base_dir, "./CremaD/data_filtered")
    files = {
        "Anger": os.path.join(sample_dir, "1003_IOM_ANG_XX.wav"),
        "Happy": os.path.join(sample_dir, "1003_IOM_HAP_XX.wav"),
        "Sad": os.path.join(sample_dir, "1003_IOM_SAD_XX.wav"),
    }
    plot_melspectrograms_from_files(files)
