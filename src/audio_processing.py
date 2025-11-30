"""
Audio feature helpers for MEL spectrograms and MFCCs.

These functions mirror the preprocessing used in the notebooks and the
utilities defined in :mod:`src.wav_to_spec`, but operate in-memory for
dataset-style pipelines.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torchaudio


def pad_or_trim_waveform(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    """
    Pad with zeros or trim the waveform to a fixed number of samples.

    Args:
        waveform: Tensor with shape (1, num_samples) or (num_channels, num_samples).
        target_num_samples: Desired length in samples.

    Returns:
        Waveform tensor with shape (1, target_num_samples).
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    num_samples = waveform.shape[1]
    if num_samples < target_num_samples:
        pad_amount = target_num_samples - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    else:
        waveform = waveform[:, :target_num_samples]
    return waveform


def compute_log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
) -> torch.Tensor:
    """
    Compute a log-MEL spectrogram for a mono waveform.

    Args:
        waveform: Tensor shaped (1, num_samples).
        sample_rate: Sampling rate of the waveform.
        n_fft: FFT window length.
        hop_length: Hop length between frames.
        n_mels: Number of MEL filter banks.

    Returns:
        Tensor with shape (n_mels, time_frames).
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_spec = mel_transform(waveform)
    log_mel = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
    return log_mel.squeeze(0)


def compute_mfcc(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
) -> torch.Tensor:
    """
    Compute MFCC coefficients for a mono waveform.

    Args:
        waveform: Tensor shaped (1, num_samples).
        sample_rate: Sampling rate of the waveform.
        n_mfcc: Number of MFCC coefficients.
        n_fft: FFT window length.
        hop_length: Hop length between frames.
        n_mels: Number of MEL filter banks used inside the MFCC transform.

    Returns:
        Tensor with shape (n_mfcc, time_frames).
    """
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
        },
    )
    mfcc = mfcc_transform(waveform)
    return mfcc.squeeze(0)


def normalize_feature(feature: torch.Tensor) -> torch.Tensor:
    """
    Apply per-feature normalization (zero mean, unit variance).

    Args:
        feature: Spectrogram-like tensor.

    Returns:
        Normalized tensor.
    """
    mean = feature.mean()
    std = feature.std().clamp_min(1e-6)
    return (feature - mean) / std


def load_audio(
    audio_path: str,
    target_sr: int,
    target_num_samples: int | None = None,
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file, resample if needed, and ensure mono format.

    Args:
        audio_path: Path to the WAV file.
        target_sr: Desired sampling rate.
        target_num_samples: Optional fixed length; pads or trims when provided.

    Returns:
        Tuple of (waveform tensor with shape (1, num_samples), sampling rate).
    """
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    if target_num_samples is not None:
        waveform = pad_or_trim_waveform(waveform, target_num_samples)
    return waveform, sr
