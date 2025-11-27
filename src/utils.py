"""
General-purpose utilities for configuration, I/O helpers, and visualization.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader


# Used in: src.training.py (visualize_model), ants_bees.ipynb (data preview), emotion_classification.py
def imshow(tensor, title: str | None = None) -> None:
    """
    Display an image tensor after undoing normalization.

    Args:
        tensor: Image tensor with shape (C, H, W).
        title: Optional title for the plot.
    """
    np_img = tensor.numpy()
    num_channels = np_img.shape[0]

    if num_channels == 1:
        mean = np.array([0.5])
        std = np.array([0.5])
        np_img = (np_img * std[:, None, None] + mean[:, None, None]).squeeze(0)
        np_img = np.clip(np_img, 0, 1)
        plt.imshow(np_img, cmap="gray", vmin=0, vmax=1)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_img = np_img.transpose((1, 2, 0))
        np_img = std * np_img + mean
        np_img = np.clip(np_img, 0, 1)
        plt.imshow(np_img)

    if title is not None:
        plt.title(title)


# Used in: crema-d-enhanced.ipynb (dataset preview), crema-d-updated.ipynb (dataset preview)
def show_exact_images_from_dataloader(dataloader: DataLoader, phase: str = "train", n_images: int = 4,
                                      prefix: str = "", grayscale: bool = False) -> None:
    """
    Visualize the first `n_images` samples from a dataloader with class labels.

    Args:
        dataloader: DataLoader for the desired phase.
        phase: Phase label to show in the title.
        n_images: Number of samples to show.
        prefix: Optional prefix for the figure title.
        grayscale: Whether the images are single-channel.
    """
    class_names = dataloader.dataset.classes
    imgs, labels = [], []
    for inputs, labs in dataloader:
        for img, lab in zip(inputs, labs):
            imgs.append(img.detach().cpu().numpy())
            labels.append(int(lab))
            if len(imgs) >= n_images:
                break
        if len(imgs) >= n_images:
            break

    proc_imgs = []
    for img in imgs:
        channels = img.shape[0]
        if channels == 1:
            mean = np.array([0.5])
            std = np.array([0.5])
            img = (img * std[:, None, None] + mean[:, None, None]).squeeze(0)
        else:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            for channel_idx in range(channels):
                img[channel_idx] = img[channel_idx] * std[channel_idx] + mean[channel_idx]
            img = np.transpose(img, (1, 2, 0))
        proc_imgs.append(img)

    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 3, 3))
    axes = [axes] if n_images == 1 else axes

    for ax, img, lab in zip(axes, proc_imgs, labels):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", aspect="auto", vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(img, 0, 1), aspect="auto")
        ax.set_title(class_names[lab], fontsize=10)
        ax.axis("off")

    fig.suptitle(f"{prefix} - conjunto {phase}", fontsize=14)
    plt.tight_layout()
    plt.show()


# Used in: crema_d_hybrid_qnn.ipynb (experiment setup)
def configure_run(base_model: str, quantum: bool, classical_model: str = "512_nq_2", n_qubits: int = 4,
                  q_depth: int = 3, selected_classes: Sequence[str] | None = None, batch_size: int = 8,
                  num_epochs: int = 20, learning_rate: float = 1e-3, data_root: str = "/content/drive/MyDrive/CREMAD",
                  specs_dir: str | None = None, embedding_dir: str | None = None, mfcc_dir: str | None = None,
                  use_pretrained: bool = True, freeze_backbone: bool = False, use_generic_weights: bool = False,
                  grayscale: bool = False, rng_seed: int = 42, **kwargs) -> dict:
    """
    Build a configuration dictionary for an experiment run and ensure directories exist.

    Args:
        base_model: Backbone identifier.
        quantum: Whether to use the quantum head.
        classical_model: Classifier head selection for classical runs.
        n_qubits: Number of qubits for the quantum layer.
        q_depth: Quantum circuit depth.
        selected_classes: Optional subset of labels to keep.
        batch_size: Batch size for training.
        num_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        data_root: Root directory for dataset artifacts.
        specs_dir: Spectrogram directory override.
        embedding_dir: Embedding directory override.
        mfcc_dir: MFCC directory override.
        use_pretrained: Whether to load pretrained weights for classical backbones.
        freeze_backbone: Whether to freeze backbone weights.
        use_generic_weights: Initialize linear layers with a generic normal distribution.
        grayscale: Whether to train on grayscale inputs.
        rng_seed: Random seed to store in the configuration.
        **kwargs: Ignored extras preserved for compatibility.

    Returns:
        Configuration dictionary with run directories included.
    """
    specs_dir = specs_dir or os.path.join(data_root, "Spectrograms")
    embedding_dir = embedding_dir or os.path.join(data_root, "Embeddings")
    mfcc_dir = mfcc_dir or os.path.join(data_root, "MFCCs")

    if quantum:
        use_pretrained = False
        freeze_backbone = True
        use_generic_weights = False

    save_root = os.path.join(data_root, "Models")
    model_category = f"{base_model}_{'quantum' if quantum else 'classic'}"
    model_base_dir = os.path.join(save_root, base_model, model_category)
    run_id = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    run_dir = os.path.join(model_base_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "run_id": run_id,
        "base_model": base_model,
        "quantum": quantum,
        "classical_model": classical_model,
        "n_qubits": n_qubits,
        "q_depth": q_depth,
        "selected_classes": selected_classes,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "data_root": data_root,
        "specs_dir": specs_dir,
        "embedding_dir": embedding_dir,
        "mfcc_dir": mfcc_dir,
        "use_pretrained": use_pretrained,
        "freeze_backbone": freeze_backbone,
        "use_generic_weights": use_generic_weights,
        "grayscale": grayscale,
        "rng_seed": rng_seed,
        "save_root": save_root,
        "model_dir": run_dir,
    }
    return config


# Used in: audio_preprocessing.ipynb (directory setup), data_analysis.ipynb (file operations)
def create_dir(path: str | Path) -> None:
    """
    Create a directory if it does not already exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


# Used in: audio_preprocessing.ipynb (embedding extraction)
def load_image(image_path: str | Path) -> Image.Image:
    """
    Load an image from disk and convert it to RGB.

    Args:
        image_path: Path to the image file.

    Returns:
        Loaded PIL image.
    """
    return Image.open(image_path).convert("RGB")


# DEPRECATED: replaced by the version used in crema_d_hybrid_qnn.ipynb.
# The entire function below is kept commented out for historical reference.
# def plot_tensorboard_metric(run_name: str, metric: str, phase: str, runs_dir: str = "runs") -> None:
#     \"\"\"
#     Plot a TensorBoard scalar metric for a given run and phase.
#
#     Args:
#         run_name: Run directory inside `runs/`.
#         metric: Metric name ("Loss" or "Accuracy").
#         phase: "train" or "val".
#         runs_dir: Base directory containing TensorBoard event files.
#     \"\"\"
#     from tensorboard.backend.event_processing import event_accumulator
#     import os
#     import matplotlib.pyplot as plt
#
#     log_dir = os.path.join(runs_dir, run_name)
#     event_file = next(
#         (os.path.join(log_dir, file) for file in os.listdir(log_dir) if file.startswith("events.out.tfevents")),
#         None,
#     )
#     if event_file is None:
#         raise FileNotFoundError(f"No event file found in {log_dir}")
#
#     accumulator = event_accumulator.EventAccumulator(event_file)
#     accumulator.Reload()
#     tag = f"{phase}/{metric}"
#     if tag not in accumulator.Tags().get("scalars", []):
#         raise ValueError(f"Metric '{tag}' is not available in this run.")
#
#     events = accumulator.Scalars(tag)
#     steps = [event.step for event in events]
#     values = [event.value for event in events]
#
#     plt.figure(figsize=(7, 4))
#     plt.plot(steps, values, label=f"{phase} {metric}")
#     plt.xlabel("Epoch")
#     plt.ylabel(metric)
#     plt.title(f"{metric} - {phase} ({run_name})")
#     plt.ylim(0, 1)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
