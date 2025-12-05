"""
Plotting utilities for training and evaluation metrics.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText


def safe_load_json(path: Path) -> dict | None:
    """
    Safely load a JSON file, returning None on failure.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not read {path} ({exc}). Skipping.")
        return None


def discover_runs(base_dir: Path) -> List[Dict[str, str]]:
    """
    Recursively find run folders containing metrics.json and either config.json or hyperparameters.json.
    """
    runs = []
    for root, _dirs, files in os.walk(base_dir):
        files_set = set(files)
        has_metrics = "metrics.json" in files_set
        has_config = "config.json" in files_set
        has_hparams = "hyperparameters.json" in files_set
        if has_metrics and (has_config or has_hparams):
            run_path = Path(root)
            rel_parts = run_path.relative_to(base_dir).parts
            model_family = rel_parts[0] if len(rel_parts) >= 1 else "unknown"
            variant = rel_parts[1] if len(rel_parts) >= 2 else "unknown"
            runs.append(
                {
                    "model_family": model_family,
                    "variant": variant,
                    "run_name": run_path.name,
                    "full_path": str(run_path),
                }
            )
    return runs


def load_metrics(run_path: Path) -> Dict[str, Any]:
    return safe_load_json(run_path / "metrics.json") or {}


def load_config(run_path: Path) -> Dict[str, Any]:
    config_path = run_path / "config.json"
    config = safe_load_json(config_path)
    if config is not None:
        return config

    hparams_path = run_path / "hyperparameters.json"
    hparams = safe_load_json(hparams_path)
    if hparams is not None:
        return hparams

    return {}


def _collect_scalar_keys(hparams_list: Iterable[dict], max_keys: int = 12) -> List[str]:
    freq: Dict[str, int] = {}
    for hparams in hparams_list:
        if not isinstance(hparams, dict):
            continue
        for key, val in hparams.items():
            if isinstance(val, (int, float, str, bool)) and len(str(val)) < 100:
                freq[key] = freq.get(key, 0) + 1
    sorted_keys = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in sorted_keys[:max_keys]]


def build_summary_dataframe(base_dir: Path) -> pd.DataFrame:
    runs = discover_runs(base_dir)
    if not runs:
        print("No runs found. Check BASE_DIR or ensure metrics.json plus config.json or hyperparameters.json are present.")
        return pd.DataFrame()

    config_cache = {run["full_path"]: load_config(Path(run["full_path"])) for run in runs}
    scalar_keys = _collect_scalar_keys(list(config_cache.values()))

    rows = []
    for run in runs:
        row = run.copy()
        cfg = config_cache.get(run["full_path"], {})
        for key in scalar_keys:
            row[key] = cfg.get(key)
        rows.append(row)

    df = pd.DataFrame(rows)
    sort_cols = [col for col in ["model_family", "variant", "run_name"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def select_runs(summary_df: pd.DataFrame, indices=None, run_names=None):
    indices = indices or []
    run_names = run_names or []
    mask = pd.Series(False, index=summary_df.index)
    if indices:
        mask |= summary_df.index.isin(indices)
    if run_names:
        mask |= summary_df["run_name"].isin(run_names)
    return summary_df[mask].to_dict(orient="records")


def plot_training_for_run(run: dict) -> None:
    run_path = Path(run["full_path"])
    metrics = load_metrics(run_path)
    if not metrics:
        print(f"No metrics found for {run_path}")
        return
    metrics_path = run_path / "metrics.json"
    save_path = run_path / "training_overview.png"
    try:
        plot_overlapped_metrics(str(metrics_path), save_path=str(save_path))
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to plot {metrics_path}: {exc}")


def plot_comparison_across_runs(
    selected_runs,
    task_name: str | None = None,
    comparison_dir: Path | None = None,
) -> None:
    """
    Plot comparative accuracy and loss curves for multiple runs.
    """
    if not selected_runs:
        print("No runs selected for comparison.")
        return

    comparison_dir = Path(comparison_dir or "comparisons")
    comparison_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

    colors = plt.cm.tab10.colors
    global_min_acc = 1.0
    best_results_text = "Best Test Accuracy:\n"
    plotted_any = False

    for i, run in enumerate(selected_runs):
        run_path = Path(run["full_path"])
        metrics = load_metrics(run_path)
        data = metrics.get("history", metrics)

        train_acc = data.get("train_acc") or data.get("train_accs") or data.get("acc_train")
        train_loss = data.get("train_loss") or data.get("train_losses") or data.get("loss_train")

        test_acc = data.get("test_acc") or data.get("val_acc") or data.get("test_accs") or data.get("val_accs")
        test_loss = data.get("test_loss") or data.get("val_loss") or data.get("test_losses") or data.get("val_losses")

        if not test_acc:
            print(f"Skipping {run['run_name']}: incomplete data.")
            continue

        plotted_any = True
        epochs = range(1, len(test_acc) + 1)
        train_epochs = range(1, len(train_acc) + 1) if train_acc else []

        color = colors[i % len(colors)]
        label_base = f"{run['variant']}"

        if train_acc:
            ax_acc.plot(train_epochs, train_acc, color=color, linestyle="--", alpha=0.65, linewidth=2)
        ax_acc.plot(
            epochs,
            test_acc,
            color=color,
            linestyle="-",
            marker="o",
            markersize=5,
            linewidth=2.4,
            label=label_base,
        )

        current_best = max(test_acc)
        global_min_acc = min(global_min_acc, min(test_acc))
        best_results_text += f"- {label_base}: {current_best:.4f}\n"

        if train_loss:
            loss_train_epochs = range(1, len(train_loss) + 1)
            ax_loss.plot(loss_train_epochs, train_loss, color=color, linestyle="--", alpha=0.65, linewidth=2)
        if test_loss:
            loss_test_epochs = range(1, len(test_loss) + 1)
            ax_loss.plot(
                loss_test_epochs,
                test_loss,
                color=color,
                linestyle="-",
                marker="o",
                markersize=5,
                linewidth=2.4,
                label=label_base,
            )

    if not plotted_any:
        print("No metrics available to plot.")
        plt.close(fig)
        return

    title_suffix = f" on {task_name}" if task_name else ""

    ax_acc.set_title(f"Accuracy Evolution{title_suffix}")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")

    bottom_limit = max(0, global_min_acc - 0.05)
    ax_acc.set_ylim(bottom_limit, 1.02)

    ax_acc.grid(True, alpha=0.25, linestyle="--")
    ax_acc.legend(fontsize=10, frameon=True, facecolor="white", framealpha=0.85, loc="upper left")

    at = AnchoredText(best_results_text.strip(), prop=dict(size=10), frameon=True, loc="lower right")
    at.patch.set_boxstyle("round,pad=0.5,rounding_size=0.2")
    at.patch.set_facecolor("#e8f5e9")
    at.patch.set_edgecolor("#a5d6a7")
    at.patch.set_alpha(0.95)
    ax_acc.add_artist(at)

    ax_loss.set_title(f"Loss Evolution{title_suffix}")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.25, linestyle="--")

    custom_lines = [
        Line2D([0], [0], color="gray", lw=2.4, linestyle="-"),
        Line2D([0], [0], color="gray", lw=2.0, linestyle="--"),
    ]
    ax_loss.legend(custom_lines, ["Test (Solid)", "Train (Dashed)"], loc="upper right", fontsize=10, frameon=True)

    file_suffix = f"_{task_name.replace(' ', '_')}" if task_name else ""
    save_path = comparison_dir / f"comparison_{selected_runs[-1]['variant']}{file_suffix}.png"

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved Thesis Plot: {save_path}")
    plt.show()


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
    epochs = range(1, len(data.get("train_losses", [])) + 1)

    if metrics == "both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
        axes = [ax]

    # Handle optional keys gracefully
    train_losses = data.get("train_losses") or data.get("loss_train")
    val_losses = data.get("val_losses") or data.get("validation_losses") or data.get("loss_val")
    train_accs = data.get("train_accs") or data.get("acc_train")
    val_accs = data.get("val_accs") or data.get("validation_accs") or data.get("acc_val")

    if metrics in ["loss", "both"] and train_losses and val_losses:
        ax = axes[0]
        ax.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
        ax.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model_name} - Training & Validation Loss")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3, linestyle="--")

    if metrics in ["accuracy", "both"] and train_accs and val_accs:
        ax = axes[1] if metrics == "both" else axes[0]
        ax.plot(epochs, train_accs, "b-", label="Training Accuracy", linewidth=2)
        ax.plot(epochs, val_accs, "r-", label="Validation Accuracy", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{model_name} - Training & Validation Accuracy")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.show()
    print(f"\nModel statistics for {model_name}:")
    if val_losses:
        print(f"  Best Validation Loss: {min(val_losses):.4f}")
    if val_accs:
        print(f"  Best Validation Accuracy: {max(val_accs):.4f}")
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

    epochs = range(1, len(models_data[0].get("train_losses", [])) + 1)

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
            train_losses = data.get("train_losses")
            val_losses = data.get("val_losses") or data.get("test_losses")
            if not train_losses or not val_losses:
                continue
            ax.plot(epochs, train_losses, "--", color=color, label=f"{name} (Train)", linewidth=2, alpha=0.7)
            ax.plot(epochs, val_losses, "-", color=color, label=f"{name} (Val/Test)", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss Comparison")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3, linestyle="--")

    if metrics in ["accuracy", "both"]:
        ax = axes[1] if metrics == "both" else axes[0]
        for data, name, color in zip(models_data, model_names, colors):
            train_accs = data.get("train_accs")
            val_accs = data.get("val_accs") or data.get("test_accs")
            if not train_accs or not val_accs:
                continue
            ax.plot(epochs, train_accs, "--", color=color, label=f"{name} (Train)", linewidth=2, alpha=0.7)
            ax.plot(epochs, val_accs, "-", color=color, label=f"{name} (Val/Test)", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training & Validation Accuracy Comparison")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3, linestyle="--")
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
        if data.get("val_losses"):
            print(f"  Best Validation Loss: {min(data['val_losses']):.4f}")
        if data.get("test_losses"):
            print(f"  Best Test Loss: {min(data['test_losses']):.4f}")
        if data.get("val_accs"):
            print(f"  Best Validation Accuracy: {max(data['val_accs']):.4f}")
        if data.get("test_accs"):
            print(f"  Best Test Accuracy: {max(data['test_accs']):.4f}")
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
    epochs = range(1, len(data.get("train_losses", [])) + 1)

    # Resolve primary and fallback metric names to stay robust across logs
    train_losses = data.get("train_losses") or data.get("loss_train")
    val_losses = data.get("val_losses") or data.get("validation_losses")
    test_losses = data.get("test_losses")
    train_accs = data.get("train_accs") or data.get("acc_train")
    val_accs = data.get("val_accs") or data.get("validation_accs")
    test_accs = data.get("test_accs")

    fig, ax1 = plt.subplots(figsize=figsize)
    color1 = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color1)

    lines = []
    if train_losses:
        lines += ax1.plot(epochs, train_losses, "--", color=color1, label="Training Loss", linewidth=2, alpha=0.7)
    target_losses = test_losses or val_losses
    if target_losses:
        lines += ax1.plot(epochs, target_losses, "-", color=color1, label="Val/Test Loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.35, linestyle="--")

    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color2)
    if train_accs:
        lines += ax2.plot(epochs, train_accs, "--", color=color2, label="Training Accuracy", linewidth=2, alpha=0.7)
    target_accs = test_accs or val_accs
    if target_accs:
        lines += ax2.plot(epochs, target_accs, "-", color=color2, label="Val/Test Accuracy", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1)

    if lines:
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper right", frameon=False)

    plt.title(f"{model_name} - Loss & Accuracy Overlapped")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.show()

    print(f"\nModel statistics for {model_name}:")
    if target_losses:
        print(f"  Best Validation/Test Loss: {min(target_losses):.4f}")
    if target_accs:
        print(f"  Best Validation/Test Accuracy: {max(target_accs):.4f}")
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
