"""
Plotting utilities for training and evaluation metrics.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


# Used in: crema_d_hybrid_qnn.ipynb (metrics visualization), plots_models.ipynb (metrics visualization)
def load_training_data(json_path: str) -> Dict:
    """
    Load training metrics from a JSON file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Dictionary with stored metrics.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metrics file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# Used in: crema_d_hybrid_qnn.ipynb (metrics visualization), plots_models.ipynb (metrics visualization)
def get_model_name_from_path(json_path: str) -> str:
    """
    Infer a model name from a metrics file path.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Parsed model name without timestamps.
    """
    filename = Path(json_path).stem
    if "_" in filename:
        parts = filename.split("_")
        for idx in range(len(parts) - 1, -1, -1):
            if len(parts[idx]) == 4 and parts[idx].isdigit():
                if idx > 0 and len(parts[idx - 1]) == 4 and parts[idx - 1].isdigit():
                    return "_".join(parts[:idx - 1])
            elif len(parts[idx]) == 6 and parts[idx].isdigit():
                if idx > 0 and len(parts[idx - 1]) == 8 and parts[idx - 1].isdigit():
                    return "_".join(parts[:idx - 1])
    return filename


# Used in: plots_models.ipynb (single model plots)
def plot_training_metrics(json_path: str, metrics: str = "both", figsize: Sequence[int] = (12, 5),
                          save_path: str | None = None) -> None:
    """
    Plot training and validation curves for a single model.

    Args:
        json_path: Path to the JSON metrics file.
        metrics: "loss", "accuracy", or "both".
        figsize: Figure size for the plots.
        save_path: Optional path to save the figure.
    """
    data = load_training_data(json_path)
    model_name = get_model_name_from_path(json_path)
    epochs = range(1, len(data["train_losses"]) + 1)

    if metrics == "both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
        axes = [ax]

    if metrics in ["loss", "both"]:
        ax = axes[0]
        ax.plot(epochs, data["train_losses"], "b-", label="Training Loss", linewidth=2)
        ax.plot(epochs, data["val_losses"], "r-", label="Validation Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model_name} - Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if metrics in ["accuracy", "both"]:
        ax = axes[1] if metrics == "both" else axes[0]
        ax.plot(epochs, data["train_accs"], "b-", label="Training Accuracy", linewidth=2)
        ax.plot(epochs, data["val_accs"], "r-", label="Validation Accuracy", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{model_name} - Training & Validation Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.show()
    print(f"\nModel statistics for {model_name}:")
    print(f"  Best Validation Loss: {min(data['val_losses']):.4f}")
    print(f"  Best Validation Accuracy: {max(data['val_accs']):.4f}")
    print(f"  Epochs: {len(epochs)}")


# Used in: plots_models.ipynb (comparison plots)
def plot_multiple_models(json_paths: Iterable[str], metrics: str = "both", figsize: Sequence[int] = (12, 5),
                         save_path: str | None = None) -> None:
    """
    Plot metrics for multiple models for side-by-side comparison.

    Args:
        json_paths: Iterable of JSON metrics file paths.
        metrics: "loss", "accuracy", or "both".
        figsize: Figure size for the plots.
        save_path: Optional path to save the figure.
    """
    json_paths = list(json_paths)
    if not json_paths:
        raise ValueError("At least one JSON path must be provided.")

    models_data: List[Dict] = []
    model_names: List[str] = []
    for json_path in json_paths:
        data = load_training_data(json_path)
        models_data.append(data)
        model_names.append(get_model_name_from_path(json_path))

    epochs = range(1, len(models_data[0]["train_losses"]) + 1)

    if metrics == "both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
        axes = [ax]

    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))

    if metrics in ["loss", "both"]:
        ax = axes[0]
        for data, name, color in zip(models_data, model_names, colors):
            ax.plot(epochs, data["train_losses"], "--", color=color, label=f"{name} (Train)", linewidth=2, alpha=0.7)
            ax.plot(epochs, data["val_losses"], "-", color=color, label=f"{name} (Val)", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if metrics in ["accuracy", "both"]:
        ax = axes[1] if metrics == "both" else axes[0]
        for data, name, color in zip(models_data, model_names, colors):
            ax.plot(epochs, data["train_accs"], "--", color=color, label=f"{name} (Train)", linewidth=2, alpha=0.7)
            ax.plot(epochs, data["val_accs"], "-", color=color, label=f"{name} (Val)", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training & Validation Accuracy Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.show()

    print("\nComparison summary:")
    print("=" * 60)
    for data, name in zip(models_data, model_names):
        print(f"\n{name}:")
        print(f"  Best Validation Loss: {min(data['val_losses']):.4f}")
        print(f"  Best Validation Accuracy: {max(data['val_accs']):.4f}")
        print(f"  Epochs: {len(epochs)}")


# Used in: data_analysis.ipynb (dimensionality reduction plots)
def plot_2d_projection(X_2d: np.ndarray, labels: np.ndarray, title: str, label_order: Sequence[str] | None = None,
                       alpha: float = 0.7) -> None:
    """
    Plot a 2D projection with points colored by label.

    Args:
        X_2d: 2D coordinates with shape (n_samples, 2).
        labels: Array of labels.
        title: Plot title.
        label_order: Optional ordering of labels.
        alpha: Point opacity.
    """
    labels = np.asarray(labels)
    label_order = label_order or sorted(np.unique(labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl in label_order:
        mask = labels == lbl
        if mask.any():
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=16, alpha=alpha, label=lbl)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(title="Label")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.show()


# Used in: data_analysis.ipynb (baseline classifier evaluation)
def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], title: str):
    """
    Plot a confusion matrix with labels on both axes.

    Args:
        cm: Confusion matrix array.
        labels: Label names.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.matshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center", color="black", fontsize=9)
    plt.tight_layout()
    return fig


# DEPRECATED: replaced by the version used in crema_d_hybrid_qnn.ipynb.
# The entire function below is kept commented out for historical reference.
# def plot_overlapped_metrics(json_path, figsize=(10, 6), save_path=None):
#     \"\"\"
#     Plot loss and accuracy on dual axes for train/validation metrics.
#
#     Args:
#         json_path: Path to the JSON metrics file.
#         figsize: Figure size for the plot.
#         save_path: Optional path to save the figure.
#     \"\"\"
#     data = load_training_data(json_path)
#     model_name = get_model_name_from_path(json_path)
#     epochs = range(1, len(data['train_losses']) + 1)
#     fig, ax1 = plt.subplots(figsize=figsize)
#     color1 = 'tab:red'
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss', color=color1)
#     line1 = ax1.plot(epochs, data['train_losses'], '--', color=color1,
#                      label='Training Loss', linewidth=2, alpha=0.7)
#     line2 = ax1.plot(epochs, data['val_losses'], '-', color=color1,
#                      label='Validation Loss', linewidth=2)
#     ax1.tick_params(axis='y', labelcolor=color1)
#     ax1.grid(True, alpha=0.3)
#     ax2 = ax1.twinx()
#     color2 = 'tab:blue'
#     ax2.set_ylabel('Accuracy', color=color2)
#     line3 = ax2.plot(epochs, data['train_accs'], '--', color=color2,
#                      label='Training Accuracy', linewidth=2, alpha=0.7)
#     line4 = ax2.plot(epochs, data['val_accs'], '-', color=color2,
#                      label='Validation Accuracy', linewidth=2)
#     ax2.tick_params(axis='y', labelcolor=color2)
#     ax2.set_ylim(0, 1)
#     lines = line1 + line2 + line3 + line4
#     labels = [line.get_label() for line in lines]
#     ax1.legend(lines, labels, loc='center right')
#     plt.title(f'{model_name} - Loss & Accuracy Overlapped')
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Figure saved to: {save_path}")
#     plt.show()
#     print(f"\\nModel statistics for {model_name}:")
#     print(f"  Best Validation Loss: {min(data['val_losses']):.4f}")
#     print(f"  Best Validation Accuracy: {max(data['val_accs']):.4f}")
#     print(f"  Epochs: {len(epochs)}")


# Used in: crema_d_hybrid_qnn.ipynb (canonical metrics plot), plots_models.ipynb (overlapped view)
def plot_overlapped_metrics(json_path: str, figsize: Sequence[int] = (10, 6),
                            save_path: str | None = None) -> None:
    """
    Plot training and testing loss/accuracy on shared epochs with dual axes.

    Args:
        json_path: Path to the JSON metrics file.
        figsize: Figure size for the plot.
        save_path: Optional path to save the figure.
    """
    data = load_training_data(json_path)
    model_name = get_model_name_from_path(json_path)
    epochs = range(1, len(data["train_losses"]) + 1)

    fig, ax1 = plt.subplots(figsize=figsize)
    color1 = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color1)
    line1 = ax1.plot(epochs, data["train_losses"], "--", color=color1,
                     label="Training Loss", linewidth=2, alpha=0.7)
    line2 = ax1.plot(epochs, data.get("test_losses", data.get("val_losses", [])), "-",
                     color=color1, label="Test Loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color2)
    line3 = ax2.plot(epochs, data["train_accs"], "--", color=color2,
                     label="Training Accuracy", linewidth=2, alpha=0.7)
    line4 = ax2.plot(epochs, data.get("test_accs", data.get("val_accs", [])), "-",
                     color=color2, label="Testing Accuracy", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1)

    lines = line1 + line2 + line3 + line4
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right")

    plt.title(f"{model_name} - Loss & Accuracy Overlapped")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.show()

    print(f"\nModel statistics for {model_name}:")
    if "test_losses" in data:
        print(f"  Best Test Loss: {min(data['test_losses']):.4f}")
    elif "val_losses" in data:
        print(f"  Best Validation Loss: {min(data['val_losses']):.4f}")
    if "test_accs" in data:
        print(f"  Best Test Accuracy: {max(data['test_accs']):.4f}")
    elif "val_accs" in data:
        print(f"  Best Validation Accuracy: {max(data['val_accs']):.4f}")
    print(f"  Epochs trained: {len(epochs)}")


# Used in: plots_models.ipynb (automatic directory visualization)
def plot_models_from_directory(directory: str = "runs_updated", pattern: str = "*.json", metrics: str = "both",
                               figsize: Sequence[int] = (12, 5), save_path: str | None = None) -> None:
    """
    Plot metrics for all JSON files matching a pattern in a directory.

    Args:
        directory: Directory to search for metric files.
        pattern: Glob pattern for file matching.
        metrics: "loss", "accuracy", or "both".
        figsize: Figure size for the plots.
        save_path: Optional path to save the figure.
    """
    json_files = list(Path(directory).glob(pattern))
    if not json_files:
        print(f"No {pattern} files found in {directory}")
        return

    json_paths = [str(path) for path in json_files]
    print(f"Found {len(json_paths)} model(s) in {directory}:")
    for path in json_paths:
        print(f"  - {get_model_name_from_path(path)}")

    if len(json_paths) == 1:
        plot_training_metrics(json_paths[0], metrics=metrics, figsize=figsize, save_path=save_path)
    else:
        plot_multiple_models(json_paths, metrics=metrics, figsize=figsize, save_path=save_path)


# Used in: plots_models.ipynb (latest run helper)
def find_latest_metrics_file(directory: str = "runs_updated") -> str | None:
    """
    Locate the most recently modified metrics JSON in a directory.

    Args:
        directory: Directory to search.

    Returns:
        Path to the latest metrics file or None when none exist.
    """
    json_files = glob.glob(os.path.join(directory, "*.json"))
    if json_files:
        json_files.sort(key=os.path.getmtime, reverse=True)
        return json_files[0]
    return None


# Used in: plots_models.ipynb (latest run helper)
def plot_current_experiment(directory: str = "runs_updated", figsize: Sequence[int] = (12, 6),
                            save_path: str | None = None) -> None:
    """
    Plot metrics for the most recent experiment in a directory.

    Args:
        directory: Directory to search for metric files.
        figsize: Figure size for the plot.
        save_path: Optional path to save the figure.
    """
    latest_file = find_latest_metrics_file(directory)
    if latest_file:
        print(f"Latest metrics file: {latest_file}")
        print("\nVisualizing current experiment metrics:")
        plot_overlapped_metrics(latest_file, figsize=figsize, save_path=save_path)
    else:
        print(f"No JSON files found in {directory}/. Ensure training saved the metrics correctly.")


plt.style.use("default")
