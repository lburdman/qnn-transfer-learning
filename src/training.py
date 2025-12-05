"""
Training and evaluation routines for classical and hybrid models.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

from tqdm.auto import tqdm
from src.model_builder import (
    AudioBackboneCNN,
    AudioEmbeddingHead,
    BackboneWithClassifier,
    FrozenBackboneWithHead,
    build_quantum_head,
    load_backbone_checkpoint,
    save_backbone_checkpoint,
)


@dataclass
class FineTuneConfig:
    """
    Configuration for the CREMA-D backbone pretraining and head fine-tuning pipeline.
    """

    representation: str = "mel"
    n_qubits: int = 10
    q_depth: int = 2
    pretrain_epochs: int = 4
    finetune_epochs: int = 12
    learning_rate_pretrain: float = 1e-3
    learning_rate_finetune_classical: float = 5e-4
    learning_rate_finetune_quantum: float = 1e-3
    batch_size: int = 8
    num_workers: int = 2
    sample_rate: int = 16000
    duration: float = 3.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 40
    backbone_dir: str = os.path.join("CREMAD", "Models", "backbone")
    embedding_dropout: float = 0.1
    model_tag: str = "cremad"
    device_override: str | None = None


def resolve_device(device_override: str | None = None):
    """
    Resolve the torch device, preferring GPU when available.
    """
    if device_override:
        return torch.device(device_override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_with_history(
    model: nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    dataset_sizes: Dict[str, int],
    device,
    num_epochs: int,
    learning_rate: float,
    model_dir: str,
    phases: Tuple[str, ...] | None = None,
    optimizer: optim.Optimizer | None = None,
    early_stop_acc: float | None = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Generic training loop that logs loss/accuracy for the requested phases.
    """
    phases = phases or tuple(phase for phase in ("train", "test") if phase in dataloaders)
    if not phases:
        raise ValueError("At least one dataloader phase is required.")

    criterion = nn.CrossEntropyLoss()
    opt = optimizer or optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history: Dict[str, List[float]] = {f"{phase}_loss": [] for phase in phases}
    history.update({f"{phase}_acc": [] for phase in phases})
    history.update({f"{phase}_f1": [] for phase in phases})
    eval_phase = None
    if early_stop_acc is not None:
        if "test" in phases:
            eval_phase = "test"
        elif "val" in phases:
            eval_phase = "val"

    os.makedirs(model_dir, exist_ok=True)
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for phase in phases:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            all_preds = []
            all_labels = []
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} {epoch + 1}/{num_epochs}", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                opt.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        opt.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())
            history[f"{phase}_f1"].append(epoch_f1)
            print(f"{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}  F1: {epoch_f1:.4f}")

            if phase != "train" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()

        if early_stop_acc is not None and eval_phase is not None:
            current_acc = history.get(f"{eval_phase}_acc", [])
            if current_acc and current_acc[-1] >= early_stop_acc:
                print(f"Early stopping at epoch {epoch + 1} with {eval_phase} accuracy {current_acc[-1]:.4f}")
                break
        print()

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    history_path = os.path.join(model_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=4)

    metrics = {
        "history": history,
        "best_acc": best_acc.item() if hasattr(best_acc, "item") else float(best_acc),
        "best_train_acc": max(history.get("train_acc", [0])) if history.get("train_acc") else 0,
        "best_test_acc": max(history.get("test_acc", [0])) if history.get("test_acc") else 0,
        "best_train_f1": max(history.get("train_f1", [0])) if history.get("train_f1") else 0,
        "best_test_f1": max(history.get("test_f1", [0])) if history.get("test_f1") else 0,
    }
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=4)
    print(f"History saved to {history_path}")
    print(f"Metrics saved to {metrics_path}")
    return model, history


def pretrain_backbone_and_embedding(
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    dataset_sizes: Dict[str, int],
    class_names: List[str],
    config: FineTuneConfig,
    representation_tag: str,
) -> Tuple[nn.Module, Dict[str, List[float]], str]:
    """
    Stage 1: train the embedding projector + linear classifier on precomputed embeddings.
    """
    device = resolve_device(config.device_override)
    sample_inputs, _ = next(iter(dataloaders["train"]))
    input_dim = sample_inputs.shape[1]
    embedding = AudioEmbeddingHead(input_dim, config.n_qubits, dropout=config.embedding_dropout)
    classifier = nn.Linear(config.n_qubits, len(class_names))
    model = nn.Sequential(embedding, classifier)

    model_dir = os.path.join(config.backbone_dir, representation_tag)
    model, history = train_with_history(
        model,
        dataloaders,
        dataset_sizes,
        device=device,
        num_epochs=config.pretrain_epochs,
        learning_rate=config.learning_rate_pretrain,
        model_dir=model_dir,
        phases=tuple(phase for phase in ("train", "test") if phase in dataloaders),
        early_stop_acc=0.9,
    )

    checkpoint_path = os.path.join(config.backbone_dir, f"{representation_tag}_backbone.pt")
    metadata = {
        "class_names": class_names,
        "representation": representation_tag,
        "config": asdict(config),
        "backbone_arch": "identity",
        "input_dim": input_dim,
    }
    save_backbone_checkpoint(nn.Identity(), embedding, checkpoint_path, metadata=metadata)
    print(f"Backbone checkpoint saved to {checkpoint_path}")
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({"history": history}, handle, indent=4)
    return model, history, checkpoint_path


def finetune_head_only(
    checkpoint_path: str,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    dataset_sizes: Dict[str, int],
    class_names: List[str],
    config: FineTuneConfig,
    head_type: str,
    representation_tag: str,
) -> Tuple[nn.Module, Dict[str, List[float]], float]:
    """
    Stage 2: freeze backbone+embedding and train only the selected head (classical or quantum).
    """
    device = resolve_device(config.device_override)
    backbone, embedding, metadata = load_backbone_checkpoint(
        checkpoint_path,
        input_channels=1,
        n_qubits=config.n_qubits,
        map_location=device,
    )
    backbone = backbone.to(device)
    embedding = embedding.to(device)

    if head_type == "classical":
        layers: List[nn.Module] = []
        in_dim = config.n_qubits
        for _ in range(max(config.q_depth - 1, 0)):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_dim, len(class_names)))
        head = nn.Sequential(*layers)
        lr = config.learning_rate_finetune_classical
    elif head_type == "quantum":
        head = build_quantum_head(config.n_qubits, len(class_names), config.q_depth)
        lr = config.learning_rate_finetune_quantum
    else:
        raise ValueError(f"Unsupported head type: {head_type}")

    freeze_module_params(embedding)
    model = FrozenBackboneWithHead(backbone, embedding, head)
    model_dir = os.path.join(config.backbone_dir, representation_tag, f"{head_type}_finetune")
    os.makedirs(model_dir, exist_ok=True)

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model, history = train_with_history(
        model,
        dataloaders,
        dataset_sizes,
        device=device,
        num_epochs=config.finetune_epochs,
        learning_rate=lr,
        model_dir=model_dir,
        phases=tuple(phase for phase in ("train", "test") if phase in dataloaders),
        optimizer=opt,
        early_stop_acc=None,
    )

    if "test_acc" in history and history["test_acc"]:
        final_acc = history["test_acc"][-1]
    elif "val_acc" in history and history["val_acc"]:
        final_acc = history["val_acc"][-1]
    else:
        final_acc = 0.0
    summary = {
        "head_type": head_type,
        "representation": representation_tag,
        "n_qubits": config.n_qubits,
        "q_depth": config.q_depth,
        "final_acc": final_acc,
        "class_names": class_names,
        "metadata": metadata,
        "history": history,
    }
    with open(os.path.join(model_dir, "finetune_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)
    return model, history, final_acc


def summarize_experiments(results: List[Dict[str, object]]) -> None:
    """
    Print a concise summary table of experiment outcomes.
    """
    if not results:
        print("No experiment results to summarize.")
        return
    header = f"{'repr':<6} | {'head':<9} | {'n_qubits':<8} | {'q_depth':<7} | {'val_acc':<8}"
    print(header)
    print("-" * len(header))
    for res in results:
        print(
            f"{res.get('representation',''):>6} | {res.get('head_type',''):>9} | "
            f"{res.get('n_qubits',''):>8} | {res.get('q_depth',''):>7} | {res.get('final_acc',0):>8.4f}"
        )


def quantum_probability_helper(
    model: nn.Module,
    inputs: torch.Tensor,
    class_names: List[str],
    device,
    max_samples: int = 4,
) -> torch.Tensor:
    """
    Compute and optionally plot class probabilities from a quantum head.
    """
    model.eval()
    with torch.no_grad():
        logits = model(inputs.to(device))
        probs = F.softmax(logits, dim=1).cpu()

    num_samples = min(max_samples, probs.shape[0])
    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
    axes = [axes] if num_samples == 1 else axes
    for idx in range(num_samples):
        ax = axes[idx]
        ax.bar(range(len(class_names)), probs[idx].numpy())
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"Sample {idx}")
    plt.tight_layout()
    return probs


def freeze_module_params(module: nn.Module) -> None:
    """
    Set requires_grad=False for all parameters in a module.
    """
    for param in module.parameters():
        param.requires_grad = False


# Used in: crema_d_hybrid_qnn.ipynb (canonical training loop)
def train_model(model: nn.Module, dataloaders: Dict[str, torch.utils.data.DataLoader],
                dataset_sizes: Dict[str, int], device, num_epochs: int, learning_rate: float,
                model_dir: str) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train a model using AdamW and StepLR, saving the best checkpoint to disk.

    Args:
        model: Model to train.
        dataloaders: Mapping of phase to DataLoader.
        dataset_sizes: Mapping of phase to dataset size.
        device: Torch device.
        num_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        model_dir: Directory to save the model checkpoint.

    Returns:
        Tuple containing the trained model and a history dictionary.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "train_f1": [], "test_loss": [], "test_acc": [], "test_f1": []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        for phase in ["train", "test"]:
            if phase not in dataloaders:
                continue
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            dataloader = dataloaders[phase]
            all_preds, all_labels = [], []
            for inputs, labels in tqdm(dataloader, desc=f"{phase} {epoch + 1}/{num_epochs}", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())
            history[f"{phase}_f1"].append(epoch_f1)
            print(f"{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}  F1: {epoch_f1:.4f}")
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        scheduler.step()
        print()

    print(f"Training complete. Best test accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    metrics = {
        "history": history,
        "best_train_acc": max(history.get("train_acc", [0])) if history.get("train_acc") else 0,
        "best_test_acc": max(history.get("test_acc", [0])) if history.get("test_acc") else 0,
        "best_train_f1": max(history.get("train_f1", [0])) if history.get("train_f1") else 0,
        "best_test_f1": max(history.get("test_f1", [0])) if history.get("test_f1") else 0,
    }
    with open(os.path.join(model_dir, "history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=4)
    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=4)
    print(f"Model saved to {model_path}")
    return model, history

# Used in: crema_d_hybrid_qnn.ipynb (evaluation and confusion matrix)
def evaluate_model(model: nn.Module, dataloader, class_names, device, model_dir: str,
                   split_name: str = "test") -> Dict[str, float]:
    """
    Evaluate a model on a dataloader and save metrics and confusion matrix.

    Args:
        model: Trained model.
        dataloader: DataLoader to evaluate.
        class_names: Ordered list of class labels.
        device: Torch device.
        model_dir: Directory to save metrics and plots.
        split_name: Split label for filenames.

    Returns:
        Dictionary with accuracy, precision, recall, and F1.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating ({split_name})"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    metrics = {
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }

    metrics_path = os.path.join(model_dir, f"{split_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=4)
    print(f"{split_name.upper()} metrics saved to {metrics_path}")
    print(f"{split_name.upper()} Accuracy: {acc:.4f} | F1: {f1:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicciones",
        ylabel="Etiqueta real",
        title=f"Matriz de confusión ({split_name})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    cm_path = os.path.join(model_dir, f"confusion_matrix_{split_name}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    return metrics
