"""
Audio preprocessing utilities: waveform-to-spectrogram conversion, embeddings, and MFCC extraction.
"""

from __future__ import annotations

import glob
import os
from typing import Sequence

import librosa
import librosa.display
import numpy as np
import torch
import torchaudio
from PIL import Image
from panns_inference import AudioTagging
from tqdm import tqdm

from src.utils import create_dir, load_image


# DEPRECATED: currently not used anywhere in the project, kept for backwards compatibility.
# Used in: src.wav_to_spec.py (module manual test)
def process_audio_to_melspec(
    input_path: str,
    output_path: str | None = None,
    label: str | None = None,
    sample_rate: int = 16000,
    duration: int = 3,
    n_mels: int = 128,
    save: bool = True,
) -> None:
    """
    Convert a WAV file to a mel-spectrogram image.

    Args:
        input_path: Path to the input WAV file.
        output_path: Base directory for saving images.
        label: Subdirectory label for saving.
        sample_rate: Target sample rate.
        duration: Duration in seconds to retain.
        n_mels: Number of mel filter banks.
        save: Save the spectrogram when True; otherwise display it.
    """
    signal, sr = librosa.load(input_path, sr=sample_rate)
    desired_length = sr * duration
    if len(signal) < desired_length:
        signal = np.pad(signal, (0, desired_length - len(signal)), mode="constant")
    else:
        signal = signal[:desired_length]

    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt = librosa.display  # type: ignore
    import matplotlib.pyplot as mpl_plt

    mpl_plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", cmap="magma")
    mpl_plt.axis("off")
    mpl_plt.tight_layout()

    if save and output_path and label:
        create_dir(os.path.join(output_path, label))
        base_name = os.path.basename(input_path).replace(".wav", ".png")
        out_file = os.path.join(output_path, label, base_name)
        mpl_plt.savefig(out_file, bbox_inches="tight", pad_inches=0)
        mpl_plt.close()
    elif not save:
        mpl_plt.show()
    else:
        raise ValueError("If save=True, both output_path and label must be provided.")


# Used in: audio_preprocessing.ipynb (spectrogram generation)
def save_melspectrogram(audio_path: str, save_path: str, target_sr: int, n_fft: int, hop_length: int,
                        n_mels: int) -> None:
    """
    Generate and save a 224x224 log-mel spectrogram PNG from an audio file.

    Args:
        audio_path: Path to the input WAV file.
        save_path: Path to save the generated PNG file.
        target_sr: Target sampling rate.
        n_fft: FFT window length.
        hop_length: Hop length for STFT.
        n_mels: Number of mel filter banks.
    """
    try:
        waveform, sr = librosa.load(audio_path, sr=target_sr)
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_norm = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)
        mel_img = (log_mel_norm * 255).astype(np.uint8)
        img = Image.fromarray(mel_img)
        img = img.resize((224, 224))
        img.save(save_path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error generating spectrogram for {audio_path}: {exc}")


# Used in: audio_preprocessing.ipynb (spectrogram generation)
def process_spectrograms(split_df, split_name: str, spec_dir: str, target_sr: int, n_fft: int, hop_length: int,
                         n_mels: int) -> None:
    """
    Generate mel-spectrogram PNG files for each audio in a split DataFrame.

    Args:
        split_df: DataFrame with columns file_path, file_name, and label.
        split_name: Name of the split ("train", "test", or "val").
        spec_dir: Root directory where spectrograms are stored.
        target_sr: Target sampling rate.
        n_fft: FFT window length.
        hop_length: Hop length for STFT.
        n_mels: Number of mel filter banks.
    """
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Generating spectrograms for {split_name}"):
        file_path = row["file_path"]
        file_name_png = os.path.splitext(row["file_name"])[0] + ".png"
        label = row["label"]
        save_path = os.path.join(spec_dir, split_name, label, file_name_png)
        create_dir(os.path.dirname(save_path))
        if not os.path.exists(save_path):
            save_melspectrogram(file_path, save_path, target_sr=target_sr, n_fft=n_fft,
                                hop_length=hop_length, n_mels=n_mels)


# Used in: audio_preprocessing.ipynb (embedding extraction)
def build_embedding_transform():
    """
    Create the torchvision transform used for image-based embedding extraction.

    Returns:
        Torchvision transform pipeline.
    """
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# Used in: audio_preprocessing.ipynb (embedding extraction)
def extract_embeddings(model, model_name: str, split_name: str, classes: Sequence[str], spec_dir: str, emb_dir: str,
                       device, embedding_transform) -> None:
    """
    Extract and save embeddings from spectrogram images for a pretrained model and split.

    Args:
        model: Pretrained backbone with classification head removed.
        model_name: Name of the backbone (e.g., "ResNet18").
        split_name: Dataset split name.
        classes: Iterable of class labels.
        spec_dir: Root directory containing spectrogram PNGs.
        emb_dir: Root directory where embeddings will be saved.
        device: Torch device for inference.
        embedding_transform: Transform pipeline applied to each spectrogram image.
    """
    model.to(device)
    model.eval()
    base_dir = os.path.join(emb_dir, model_name, split_name)
    for label in classes:
        create_dir(os.path.join(base_dir, label))

    split_root = os.path.join(spec_dir, split_name)
    for label in classes:
        class_dir = os.path.join(split_root, label)
        image_files = glob.glob(os.path.join(class_dir, "*.png"))
        for img_path in tqdm(image_files, desc=f"Embedding {model_name} - {split_name} - {label}", leave=False):
            try:
                image = load_image(img_path)
                tensor = embedding_transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    feature = model(tensor)
                embedding = feature.squeeze().cpu().numpy().flatten()
                file_name = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(base_dir, label, f"{file_name}.npz")
                np.savez(save_path, embedding=embedding, label=label, file_name=file_name)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Error extracting embedding for {img_path}: {exc}")


# Used in: audio_preprocessing.ipynb (PANNs embedding extraction)
def extract_panns_embedding(audio_path: str, at_model: AudioTagging, target_sr: int = 32000) -> np.ndarray:
    """
    Load an audio file, resample, and compute a 2048-dim PANNs embedding.

    Args:
        audio_path: Path to the input WAV file.
        at_model: PANNs AudioTagging model.
        target_sr: Target sampling rate for the model.

    Returns:
        Embedding array with shape (2048,).
    """
    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    audio = audio.astype(np.float32)[None, :]
    with torch.no_grad():
        _, embedding = at_model.inference(audio)
    return np.asarray(embedding[0], dtype=np.float32)


# Used in: audio_preprocessing.ipynb (PANNs embedding extraction)
def process_panns_embeddings(split_df, split_name: str, pann_dir: str, classes: Sequence[str],
                             audio_tagging: AudioTagging, target_sr: int = 32000) -> None:
    """
    Extract and save PANNs embeddings for each audio file in a split.

    Args:
        split_df: DataFrame containing file_path, file_name, and label.
        split_name: Name of the split.
        pann_dir: Root directory where PANN embeddings are saved.
        classes: Iterable of class labels.
        audio_tagging: Initialized AudioTagging model.
        target_sr: Target sampling rate for the model.
    """
    base_dir = os.path.join(pann_dir, split_name)
    for label in classes:
        create_dir(os.path.join(base_dir, label))

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"PANNs embeddings for {split_name}"):
        file_path = row["file_path"]
        file_name = os.path.splitext(row["file_name"])[0]
        label = row["label"]
        save_path = os.path.join(base_dir, label, f"{file_name}.npz")
        if os.path.exists(save_path):
            continue
        try:
            emb = extract_panns_embedding(file_path, audio_tagging, target_sr=target_sr)
            np.savez(save_path, embedding=emb, label=label, file_name=os.path.basename(file_path))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error extracting PANNs embedding for {file_path}: {exc}")


# Used in: audio_preprocessing.ipynb (MFCC generation)
def process_mfccs(split_df, split_name: str, mfcc_dir: str, target_sr: int, n_mfcc: int, n_fft: int,
                  hop_length: int) -> None:
    """
    Generate and save MFCC feature arrays for each audio file in a split.

    Args:
        split_df: DataFrame containing file_path, file_name, and label.
        split_name: Name of the split ("train", "test", or "val").
        mfcc_dir: Root directory where MFCC files are saved.
        target_sr: Target sampling rate.
        n_mfcc: Number of MFCC coefficients.
        n_fft: FFT window length.
        hop_length: Hop length for STFT.
    """
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Generating MFCCs for {split_name}"):
        file_path = row["file_path"]
        file_name = os.path.splitext(row["file_name"])[0]
        label = row["label"]
        save_path = os.path.join(mfcc_dir, split_name, label, f"{file_name}.npz")
        create_dir(os.path.dirname(save_path))
        if os.path.exists(save_path):
            continue
        try:
            waveform, sr = librosa.load(file_path, sr=target_sr)
            mfcc = librosa.feature.mfcc(
                y=waveform,
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            mfcc = mfcc.astype(np.float32)
            np.savez(save_path, mfcc=mfcc, label=label, file_name=file_name)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error generating MFCC for {file_path}: {exc}")
