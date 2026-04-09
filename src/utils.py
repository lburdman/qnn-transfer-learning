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
import json
import torch
from PIL import Image
from torch.utils.data import DataLoader


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


def configure_run(base_model: str, quantum: bool, classical_model: str = "512_nq_2", n_qubits: int = 4,
                  q_depth: int = 3, selected_classes: Sequence[str] | None = None, batch_size: int = 8,
                  num_epochs: int = 20, learning_rate: float = 1e-3, data_root: str = "/content/drive/MyDrive/CREMAD",
                  specs_dir: str | None = None, embedding_dir: str | None = None, mfcc_dir: str | None = None,
                  grayscale: bool = False, rng_seed: int = 42, **kwargs) -> dict:
    """
    Build a configuration dictionary for an experiment run and ensure directories exist.
    Saves the full config to config.json in the run directory.
    """
    specs_dir = specs_dir or os.path.join(data_root, "Spectrograms")
    embedding_dir = embedding_dir or os.path.join(data_root, "Embeddings")
    mfcc_dir = mfcc_dir or os.path.join(data_root, "MFCC")

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
        "rng_seed": rng_seed,
        "save_root": save_root,
        "model_dir": run_dir,
    }

    # Persist configuration for reproducibility
    try:
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=4)
    except Exception:
        pass
    return config

def create_dir(path: str | Path) -> None:
    """
    Create a directory if it does not already exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def load_image(image_path: str | Path) -> Image.Image:
    """
    Load an image from disk and convert it to RGB.

    Args:
        image_path: Path to the image file.

    Returns:
        Loaded PIL image.
    """
    return Image.open(image_path).convert("RGB")


# Model summary helpers
def count_params_by_kind(model) -> tuple[int, int, int, int]:
    """
    Count total, trainable, classical, and quantum parameters.
    Quantum params are those belonging to PennyLane TorchLayers when available.
    """
    try:
        import pennylane as qml  # type: ignore
        torchlayer_cls = qml.qnn.TorchLayer  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        torchlayer_cls = tuple()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum = 0
    counted = set()
    for module in model.modules():
        if torchlayer_cls and isinstance(module, torchlayer_cls):
            for param in module.parameters():
                if id(param) not in counted:
                    quantum += param.numel()
                    counted.add(id(param))
    # Fallback name-based scan
    for name, param in model.named_parameters():
        if ("torchlayer" in name.lower() or "qlayer" in name.lower()) and id(param) not in counted:
            quantum += param.numel()
            counted.add(id(param))
    classical = total - quantum
    return total, trainable, classical, quantum


def print_model_summary(model) -> None:
    """
    Print a brief model summary and parameter counts.
    """
    try:
        from torchinfo import summary as torchinfo_summary  # type: ignore
        torchinfo_summary(model, verbose=0)
    except Exception:
        print("Torchinfo not available; showing minimal structure:")
        for name, module in model.named_children():
            print(f"- {name}: {module.__class__.__name__}")
    total, trainable, classical, quantum = count_params_by_kind(model)
    print(model)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Classical params: {classical:,}")
    print(f"Quantum params: {quantum:,}")
