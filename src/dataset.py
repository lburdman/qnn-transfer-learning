"""
Dataset utilities for loading image, embedding, and MFCC features.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.audio_processing import compute_log_mel_spectrogram, compute_mfcc, load_audio, normalize_feature


# Used in: crema-d.ipynb, crema-d-updated.ipynb, crema-d-enhanced.ipynb (SpecAugment option)
class SpecAugment:
    """
    Apply SpecAugment-style time and frequency masking to spectrogram tensors.

    Args:
        time_mask_param: Maximum mask width over time axis.
        freq_mask_param: Maximum mask width over frequency axis.
        num_masks: Number of masks per spectrogram.
        fill_value: Value used to fill masked regions; defaults to ImageNet-normalized black.
    """

    def __init__(self, time_mask_param: int = 30, freq_mask_param: int = 13, num_masks: int = 1,
                 fill_value: torch.Tensor | None = None) -> None:
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks
        self.fill_value = fill_value if fill_value is not None else torch.tensor([[-2.118], [-2.036], [-1.804]])

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Mask a spectrogram tensor with shape [C, H, W] along time and frequency axes.
        """
        channels, height, width = spec.shape
        for _ in range(self.num_masks):
            time_mask = random.randint(0, self.time_mask_param)
            time_start = random.randint(0, max(1, width - time_mask))
            spec[:, :, time_start:time_start + time_mask] = (
                self.fill_value if channels == 1 else self.fill_value[:, None]
            )

            freq_mask = random.randint(0, self.freq_mask_param)
            freq_start = random.randint(0, max(1, height - freq_mask))
            spec[:, freq_start:freq_start + freq_mask, :] = (
                self.fill_value if channels == 1 else self.fill_value[:, None]
            )
        return spec


# Used in: crema-d.ipynb, crema-d-updated.ipynb, crema-d-enhanced.ipynb (ImageFolder training)
def get_data_transforms(spec_augment: bool = False) -> Dict[str, transforms.Compose]:
    """
    Build torchvision transforms for train and validation splits.

    Args:
        spec_augment: Whether to apply SpecAugment to spectrogram images.

    Returns:
        Dictionary with train/val transforms.
    """
    train_transform: List[transforms.Compose | transforms.Normalize | transforms.Resize] = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if spec_augment:
        train_transform.append(SpecAugment(time_mask_param=30, freq_mask_param=15))

    return {
        "train": transforms.Compose(train_transform),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }


# Used in: crema-d.ipynb, crema-d-updated.ipynb, crema-d-enhanced.ipynb, emotion_classification.py
def get_dataloaders(data_dir: str, batch_size: int = 4, shuffle: bool = True, num_workers: int = 0,
                    spec_augment: bool = False) -> Tuple[Dict[str, DataLoader], Dict[str, int], List[str]]:
    """
    Create ImageFolder-based dataloaders for train/val splits.

    Args:
        data_dir: Root directory containing `train` and `val` folders.
        batch_size: Batch size for all dataloaders.
        shuffle: Whether to shuffle the training set.
        num_workers: Number of worker processes.
        spec_augment: Apply SpecAugment to training data.

    Returns:
        Tuple of (dataloaders dict, dataset sizes dict, class names list).
    """
    data_transforms = get_data_transforms(spec_augment=spec_augment)
    image_datasets = {
        split: datasets.ImageFolder(os.path.join(data_dir, split), data_transforms[split])
        for split in ["train", "val"]
    }

    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        for split in ["train", "val"]
    }
    dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val"]}
    class_names = image_datasets["train"].classes
    return dataloaders, dataset_sizes, class_names


# Used in: crema-d-updated.ipynb, crema-d-enhanced.ipynb (grayscale experiments)
def create_dataloaders(data_dir: str, batch_size: int = 8, shuffle: bool = True, num_workers: int = 4,
                       grayscale: bool = False) -> Tuple[Dict[str, DataLoader], Dict[str, int], List[str]]:
    """
    Create ImageFolder dataloaders for RGB or grayscale inputs.

    Args:
        data_dir: Root directory with `train` and `val` subfolders.
        batch_size: Samples per batch.
        shuffle: Shuffle the training set.
        num_workers: Number of worker processes.
        grayscale: Convert images to one channel when True.

    Returns:
        Tuple of (dataloaders dict, dataset sizes dict, class names list).
    """
    mean, std = ([0.5], [0.5]) if grayscale else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform_list: List[transforms.Compose | transforms.Normalize | transforms.Resize] = [transforms.Resize((224, 224))]
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    transform_list += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    data_transform = transforms.Compose(transform_list)

    image_datasets = {
        split: datasets.ImageFolder(os.path.join(data_dir, split), transform=data_transform)
        for split in ["train", "val"]
    }
    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=(shuffle if split == "train" else False),
            num_workers=num_workers,
        )
        for split in ["train", "val"]
    }
    dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val"]}
    class_names = image_datasets["train"].classes
    return dataloaders, dataset_sizes, class_names


# Used in: crema-d-updated.ipynb, crema-d-enhanced.ipynb (class distribution reporting)
def count_images_per_class_from_dataset(dataset: datasets.ImageFolder, class_names: Sequence[str]) -> Dict[str, int]:
    """
    Count images per class using the targets stored in an ImageFolder dataset.

    Args:
        dataset: ImageFolder dataset instance.
        class_names: Ordered list of class labels.

    Returns:
        Mapping of class name to count.
    """
    counts = Counter(dataset.targets)
    return {class_names[idx]: counts[idx] for idx in range(len(class_names))}


# Used in: crema_d_hybrid_qnn.ipynb (hybrid data pipeline)
class AudioFeatureDataset(Dataset):
    """
    Generic dataset supporting spectrogram images, embeddings, and MFCC features.

    Args:
        filepaths: List of feature file paths.
        labels: Integer labels aligned with `filepaths`.
        base_model: Identifier for the backbone model family.
        grayscale: Whether to load inputs as grayscale.
        force_three_channels: Expand grayscale inputs to three channels when True.
    """

    def __init__(self, filepaths: Sequence[str], labels: Sequence[int], base_model: str, grayscale: bool = False,
                 force_three_channels: bool = False) -> None:
        super().__init__()
        self.filepaths = list(filepaths)
        self.labels = list(labels)
        self.base_model = base_model
        self.grayscale = grayscale
        self.force_three_channels = force_three_channels

        if base_model in ["resnet18", "vgg16", "custom_cnn"]:
            if grayscale and not force_three_channels:
                mean, std = [0.5], [0.5]
            else:
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            transform_steps: List[transforms.Compose | transforms.Normalize | transforms.Resize] = [
                transforms.Resize((224, 224))
            ]
            if grayscale:
                num_channels = 3 if force_three_channels else 1
                transform_steps.append(transforms.Grayscale(num_output_channels=num_channels))
            transform_steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
            self.transform = transforms.Compose(transform_steps)
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.filepaths[idx]
        label = self.labels[idx]

        if self.base_model in ["resnet18", "vgg16", "custom_cnn"]:
            img = Image.open(path)
            img = img.convert("L") if self.grayscale and not self.force_three_channels else img.convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

        if self.base_model in ["emb_resnet18", "emb_vgg16", "emb_panns_cnn14"]:
            data = np.load(path)
            key = "embedding" if "embedding" in data else list(data.keys())[0]
            embedding = np.array(data[key], dtype=np.float32)
            return torch.tensor(embedding, dtype=torch.float32), label

        if self.base_model in ["mfcc", "cnn_mfcc"]:
            data = np.load(path)
            key = "mfcc" if "mfcc" in data else list(data.keys())[0]
            mfcc = np.array(data[key], dtype=np.float32)
            # Pad/trim MFCC along time axis to ensure consistent shapes
            target_len = getattr(self, "pad_mfcc_len", None)
            if target_len is not None:
                current_len = mfcc.shape[1]
                if current_len < target_len:
                    pad_width = ((0, 0), (0, target_len - current_len))
                    mfcc = np.pad(mfcc, pad_width, mode="constant")
                elif current_len > target_len:
                    mfcc = mfcc[:, :target_len]
            if self.base_model == "cnn_mfcc":
                # Return as 1 x H x W tensor for CNNs
                return torch.tensor(mfcc[None, ...], dtype=torch.float32), label
            features = mfcc.flatten()
            return torch.tensor(features, dtype=torch.float32), label

        raise ValueError(f"Unsupported base_model: {self.base_model}")


class CREMADWaveformDataset(Dataset):
    """
    On-the-fly CREMA-D dataset that produces MEL spectrograms or MFCC tensors.

    Args:
        dataframe: Metadata with columns `file_path` (or `file_name`) and `label`.
        label_to_idx: Mapping from string label to integer index.
        representation: Either "mel" or "mfcc".
        sample_rate: Target sampling rate for resampling.
        duration: Duration in seconds to keep per clip.
        n_fft: FFT window length for spectral transforms.
        hop_length: Hop length between frames.
        n_mels: Number of MEL bins (used for MEL and MFCC).
        n_mfcc: Number of MFCC coefficients when representation="mfcc".
        audio_root: Optional base directory for audio paths when `file_path` is missing.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_to_idx: Dict[str, int],
        representation: str = "mel",
        sample_rate: int = 16000,
        duration: float = 3.0,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 40,
        audio_root: str | None = None,
    ) -> None:
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.representation = representation.lower()
        self.sample_rate = sample_rate
        self.target_num_samples = int(sample_rate * duration)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.audio_root = audio_root
        self.classes = list(label_to_idx.keys())

        if self.representation not in {"mel", "mfcc"}:
            raise ValueError(f"Unsupported representation: {representation}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, idx: int) -> str:
        row = self.df.iloc[idx]
        if "file_path" in row and isinstance(row["file_path"], (str, os.PathLike)):
            return str(row["file_path"])
        if "file_name" not in row or self.audio_root is None:
            raise FileNotFoundError("Audio path could not be resolved for dataset row.")
        return os.path.join(self.audio_root, str(row["file_name"]))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path = self._resolve_path(idx)
        label_name = str(self.df.iloc[idx]["label"])
        label = self.label_to_idx[label_name]

        waveform, _ = load_audio(audio_path, target_sr=self.sample_rate, target_num_samples=self.target_num_samples)

        if self.representation == "mel":
            feature = compute_log_mel_spectrogram(
                waveform,
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
        else:
            feature = compute_mfcc(
                waveform,
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )

        feature = normalize_feature(feature).unsqueeze(0)
        return feature, label


# Used in: crema_d_hybrid_qnn.ipynb (hybrid data pipeline)
def create_dataloaders_all(config: Dict[str, object], shuffle: bool = True, num_workers: int = 4
                           ) -> Tuple[Dict[str, DataLoader], Dict[str, int], List[str], Dict[str, Dict[str, int]]]:
    """
    Create dataloaders for spectrograms, embeddings, or MFCC features based on configuration.

    Args:
        config: Experiment configuration dictionary.
        shuffle: Shuffle training data.
        num_workers: Number of worker processes.

    Returns:
        dataloaders: Mapping of split name to DataLoader.
        dataset_sizes: Mapping of split name to dataset size.
        class_names: Ordered list of class labels.
        counts_per_class: Mapping of split to class-count dictionaries.
    """
    base_model = config["base_model"]
    # Normalize aliases for clarity in notebooks (keep cnn_mfcc separate to preserve 2D MFCC handling)
    alias_map = {
        "cnn_specs": "resnet18",
    }
    base_model = alias_map.get(base_model, base_model)
    selected_classes = config["selected_classes"]
    batch_size = config["batch_size"]
    grayscale = config["grayscale"]
    use_pretrained = config["use_pretrained"]
    force_three_channels = bool(grayscale and use_pretrained)

    dataloaders: Dict[str, DataLoader] = {}
    dataset_sizes: Dict[str, int] = {}
    counts_per_class: Dict[str, Dict[str, int]] = {}

    if base_model in ["resnet18", "vgg16", "custom_cnn"]:
        root = str(config["specs_dir"])
        phases = [phase for phase in ["train", "val", "test"] if os.path.isdir(os.path.join(root, phase))]
        class_names: List[str] | None = None
        for phase in phases:
            phase_dir = os.path.join(root, phase)
            ds = datasets.ImageFolder(phase_dir)
            if class_names is None:
                class_names = [cls for cls in ds.classes if selected_classes is None or cls in selected_classes]
            class_to_new_idx = {cls: idx for idx, cls in enumerate(class_names)}
            filtered_samples = [
                (path, lbl) for (path, lbl) in ds.samples
                if selected_classes is None or ds.classes[lbl] in selected_classes
            ]
            files = [path for path, _ in filtered_samples]
            labels = [class_to_new_idx[ds.classes[lbl]] for _, lbl in filtered_samples]
            dataset = AudioFeatureDataset(
                files,
                labels,
                base_model,
                grayscale=grayscale,
                force_three_channels=force_three_channels,
            )
            dataloaders[phase] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(shuffle if phase == "train" else False),
                num_workers=num_workers,
            )
            dataset_sizes[phase] = len(dataset)
            counter = Counter(labels)
            counts_per_class[phase] = {cls: counter.get(class_to_new_idx[cls], 0) for cls in class_names}

    elif base_model in ["emb_resnet18", "emb_vgg16", "emb_panns_cnn14"]:
        root = str(config["embedding_dir"])
        feature_type_map = {
            "emb_resnet18": "ResNet18",
            "emb_vgg16": "VGG16",
            "emb_panns_cnn14": "PANNs_Cnn14",
        }
        feature_type = feature_type_map[base_model]
        phases = [phase for phase in ["train", "val", "test"] if os.path.isdir(os.path.join(root, feature_type, phase))]
        class_names = None
        for phase in phases:
            ds_path = os.path.join(root, feature_type, phase)
            files: List[str] = []
            labels: List[str] = []
            for cls_name in sorted(os.listdir(ds_path)):
                class_dir = os.path.join(ds_path, cls_name)
                if not os.path.isdir(class_dir):
                    continue
                if selected_classes and cls_name not in selected_classes:
                    continue
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(".npz"):
                        files.append(os.path.join(class_dir, fname))
                        labels.append(cls_name)
            if class_names is None:
                class_names = sorted(list(set(labels)))
            label_indices = [class_names.index(lbl) for lbl in labels]
            dataset = AudioFeatureDataset(files, label_indices, base_model)
            dataloaders[phase] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(shuffle if phase == "train" else False),
                num_workers=num_workers,
            )
            dataset_sizes[phase] = len(dataset)
            counter = Counter(labels)
            counts_per_class[phase] = {cls: counter.get(cls, 0) for cls in class_names}

    elif base_model in ["mfcc", "cnn_mfcc"]:
        root = str(config["mfcc_dir"])
        pad_frames = int(config.get("mfcc_pad_frames", 500))
        phases = [phase for phase in ["train", "val", "test"] if os.path.isdir(os.path.join(root, phase))]
        class_names = None
        for phase in phases:
            ds_path = os.path.join(root, phase)
            files: List[str] = []
            labels: List[str] = []
            for cls_name in sorted(os.listdir(ds_path)):
                class_dir = os.path.join(ds_path, cls_name)
                if not os.path.isdir(class_dir):
                    continue
                if selected_classes and cls_name not in selected_classes:
                    continue
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(".npz"):
                        files.append(os.path.join(class_dir, fname))
                        labels.append(cls_name)
            if class_names is None:
                class_names = sorted(list(set(labels)))
            label_indices = [class_names.index(lbl) for lbl in labels]
            dataset = AudioFeatureDataset(files, label_indices, base_model)
            setattr(dataset, "pad_mfcc_len", pad_frames)
            dataloaders[phase] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(shuffle if phase == "train" else False),
                num_workers=num_workers,
            )
            dataset_sizes[phase] = len(dataset)
            counter = Counter(labels)
            counts_per_class[phase] = {cls: counter.get(cls, 0) for cls in class_names}
    else:
        raise ValueError(f"Unsupported base_model: {base_model}")

    assert class_names is not None, "Class names could not be inferred from the datasets."
    return dataloaders, dataset_sizes, class_names, counts_per_class


# Used in: data_analysis.ipynb (metadata loading)
def load_metadata(selected_classes: Sequence[str] | None = None, splits_dir: Path | None = None,
                  audio_dir: Path | None = None) -> pd.DataFrame:
    """
    Load metadata for waveform and spectrogram visualization from CSV split files.

    Args:
        selected_classes: Optional subset of labels to keep.
        splits_dir: Directory containing train/val/test CSV files.
        audio_dir: Directory containing the WAV files referenced in the CSVs.

    Returns:
        DataFrame with split, label, file_path, and file_stem columns.
    """
    splits_dir = splits_dir or Path("splits")
    audio_dir = audio_dir or Path("AudioWAV")
    frames = []
    for split in ["train", "val", "test"]:
        csv_path = splits_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split file: {csv_path}")
        df_split = pd.read_csv(csv_path)
        df_split["split"] = split
        df_split["file_stem"] = df_split["file_name"].apply(lambda name: Path(name).stem)
        df_split["file_path"] = audio_dir / df_split["file_name"]
        frames.append(df_split)

    meta = pd.concat(frames, ignore_index=True)
    if selected_classes:
        meta = meta[meta["label"].isin(selected_classes)].reset_index(drop=True)
    return meta


# Used in: data_analysis.ipynb (feature loading helpers)
def _load_feature_arrays(base_dir: Path, splits: Iterable[str], selected_classes: Sequence[str] | None,
                         preferred_keys: Sequence[str], flatten: bool = True,
                         path_col_name: str = "feature_path") -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load feature arrays stored under base_dir/split/label/*.npz or *.npy.

    Args:
        base_dir: Root directory containing split subfolders.
        splits: Iterable of split names to include.
        selected_classes: Optional subset of labels to keep.
        preferred_keys: Ordered keys to look for inside NPZ files.
        flatten: Flatten arrays when they have more than one dimension.
        path_col_name: Column name for storing file paths in the returned metadata.

    Returns:
        Tuple of (feature matrix X, labels y, metadata DataFrame).
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {base_dir}")

    if isinstance(splits, str):
        splits = (splits,)

    features: List[np.ndarray] = []
    labels: List[str] = []
    records: List[dict] = []
    load_errors: List[Tuple[str, str]] = []

    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            print(f"[load_feature_arrays] Split directory not found, skipping: {split_dir}")
            continue

        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue

            label = label_dir.name
            if selected_classes and label not in selected_classes:
                continue

            files = list(label_dir.glob("*.npz")) + list(label_dir.glob("*.npy"))
            if not files:
                continue

            for fpath in files:
                try:
                    data = np.load(fpath, allow_pickle=False)
                    if isinstance(data, np.lib.npyio.NpzFile):
                        key = None
                        for candidate in preferred_keys:
                            if candidate in data.files:
                                key = candidate
                                break
                        if key is None:
                            key = data.files[0] if data.files else None
                        if key is None:
                            raise ValueError("Empty NPZ file")
                        arr = data[key]
                    else:
                        arr = data

                    arr = np.asarray(arr)
                    if flatten and arr.ndim > 1:
                        arr = arr.reshape(-1)

                    features.append(arr)
                    labels.append(label)
                    records.append({
                        path_col_name: str(fpath),
                        "split": split,
                        "label": label,
                        "file_stem": fpath.stem,
                    })
                except Exception as exc:  # pylint: disable=broad-except
                    load_errors.append((str(fpath), repr(exc)))
                    continue

    if not features:
        raise RuntimeError(f"No features loaded from {base_dir} with current filters.")

    if load_errors:
        print(f"[load_feature_arrays] {len(load_errors)} files failed to load from {base_dir}.")
        print("  Example errors:", load_errors[:3])

    X = np.stack(features)
    y = np.array(labels)
    meta_loaded = pd.DataFrame(records).reset_index(drop=True)
    return X, y, meta_loaded


# Used in: data_analysis.ipynb (embedding loading)
def load_embeddings(embedding_type: str, embedding_paths: Dict[str, Path], splits: Iterable[str],
                    selected_classes: Sequence[str] | None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load embeddings from the directory structure: base_dir/split/label/<file>.

    Args:
        embedding_type: Key identifying the embedding directory.
        embedding_paths: Mapping from embedding key to base directory.
        splits: Iterable of split names to load.
        selected_classes: Optional subset of labels to keep.

    Returns:
        Tuple of (X embeddings, labels y, metadata DataFrame).
    """
    base_key = embedding_type.lower()
    if base_key not in embedding_paths:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    base_dir = embedding_paths[base_key]
    return _load_feature_arrays(
        base_dir=base_dir,
        splits=splits,
        selected_classes=selected_classes,
        preferred_keys=("embedding",),
        flatten=True,
        path_col_name="embedding_path",
    )


# Used in: data_analysis.ipynb (MFCC loading)
def load_mfcc_features(splits: Iterable[str], mfcc_dir: Path, selected_classes: Sequence[str] | None
                       ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load MFCC feature arrays stored under mfcc_dir/train|val|test/<label>/.

    Args:
        splits: Iterable of split names to load.
        mfcc_dir: Root MFCC directory.
        selected_classes: Optional subset of labels to keep.

    Returns:
        Tuple of (X MFCC features, labels y, metadata DataFrame).
    """
    return _load_feature_arrays(
        base_dir=mfcc_dir,
        splits=splits,
        selected_classes=selected_classes,
        preferred_keys=("mfcc", "feature", "features", "arr_0"),
        flatten=True,
        path_col_name="mfcc_path",
    )


def build_cremad_audio_dataloaders(
    audio_dir: str,
    splits_dir: str,
    representation: str,
    batch_size: int = 8,
    num_workers: int = 4,
    sample_rate: int = 16000,
    duration: float = 3.0,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    n_mfcc: int = 40,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], List[str]]:
    """
    Build CREMA-D dataloaders that compute MEL or MFCC features on the fly.

    Args:
        audio_dir: Directory containing WAV files referenced by the split CSVs.
        splits_dir: Directory containing train/val[/test] CSV split files.
        representation: Either \"mel\" or \"mfcc\".
        batch_size: Batch size for all splits.
        num_workers: Number of dataloader workers.
        sample_rate: Target sampling rate.
        duration: Clip duration (seconds).
        n_fft: FFT window length.
        hop_length: Hop length for STFT.
        n_mels: Number of MEL bins.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        Tuple of (dataloaders dict, dataset sizes dict, class names list).
    """
    meta = load_metadata(splits_dir=Path(splits_dir), audio_dir=Path(audio_dir))
    class_names = sorted(meta["label"].unique().tolist())
    label_to_idx = {lbl: idx for idx, lbl in enumerate(class_names)}

    dataloaders: Dict[str, DataLoader] = {}
    dataset_sizes: Dict[str, int] = {}
    for split in ["train", "val", "test"]:
        split_df = meta[meta["split"] == split]
        if split_df.empty:
            continue
        dataset = CREMADWaveformDataset(
            split_df,
            label_to_idx=label_to_idx,
            representation=representation,
            sample_rate=sample_rate,
            duration=duration,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            audio_root=audio_dir,
        )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        dataset_sizes[split] = len(dataset)
    return dataloaders, dataset_sizes, class_names


def build_cremad_mel_and_mfcc_dataloaders(
    audio_dir: str,
    splits_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    sample_rate: int = 16000,
    duration: float = 3.0,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    n_mfcc: int = 40,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader], Dict[str, int], Dict[str, int], List[str]]:
    """
    Convenience wrapper to build both MEL and MFCC dataloaders.

    Returns:
        dataloaders_mel, dataloaders_mfcc, dataset_sizes_mel, dataset_sizes_mfcc, class_names
    """
    dataloaders_mel, dataset_sizes_mel, class_names = build_cremad_audio_dataloaders(
        audio_dir=audio_dir,
        splits_dir=splits_dir,
        representation="mel",
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=sample_rate,
        duration=duration,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
    )
    dataloaders_mfcc, dataset_sizes_mfcc, _ = build_cremad_audio_dataloaders(
        audio_dir=audio_dir,
        splits_dir=splits_dir,
        representation="mfcc",
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=sample_rate,
        duration=duration,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
    )
    return dataloaders_mel, dataloaders_mfcc, dataset_sizes_mel, dataset_sizes_mfcc, class_names
