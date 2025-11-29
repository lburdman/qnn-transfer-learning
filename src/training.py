"""
Training and evaluation routines for classical and hybrid models.
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn, optim
from torch.optim import lr_scheduler

from src.utils import imshow
from tqdm.auto import tqdm


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
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

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
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())
            print(f"{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        scheduler.step()
        print()

    print(f"Training complete. Best test accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
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


# Used in: crema-d-enhanced.ipynb (legacy training), crema-d-updated.ipynb (legacy training)
def train_model1(model: nn.Module, dataloaders: Dict[str, torch.utils.data.DataLoader],
                dataset_sizes: Dict[str, int], device, criterion, optimizer, scheduler, num_epochs: int,
                save_metrics: bool = True, metrics_dir: str = "runs_updated") -> nn.Module:
    """
    Train a Torch model with metric logging to JSON.

    Args:
        model: Model to train.
        dataloaders: Mapping of phase to DataLoader.
        dataset_sizes: Mapping of phase to dataset size.
        device: Torch device.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: LR scheduler.
        num_epochs: Number of epochs to train.
        save_metrics: Save metrics to JSON when True.
        metrics_dir: Output directory for metrics files.

    Returns:
        Model with the best validation weights loaded.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_train = 0.0
    best_loss_train = float("inf")
    best_loss = float("inf")

    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    print("Training started:")
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_corrects = 0
            n_batches = len(dataloaders[phase])

            for it, (inputs, labels) in enumerate(dataloaders[phase]):
                since_batch = time.time()
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
                running_corrects += torch.sum(preds == labels.data).item()
                print(
                    f"Phase: {phase}    Epoch: {epoch + 1}/{num_epochs}    Iter: {it + 1}/{n_batches + 1} "
                    f"Batch time: {time.time() - since_batch:.4f}s",
                    end="\r",
                    flush=True,
                )

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(
                f"Phase: {phase}    Epoch: {epoch + 1}/{num_epochs}    Loss: {epoch_loss:.4f}    Acc: {epoch_acc:.4f}        "
            )

            if phase == "train":
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
            if phase == "train" and epoch_acc > best_acc_train:
                best_acc_train = epoch_acc
            if phase == "train" and epoch_loss < best_loss_train:
                best_loss_train = epoch_loss

        scheduler.step()

    time_elapsed = time.time() - since
    print(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val loss: {best_loss:.4f} | Best val accuracy: {best_acc:.4f}")

    if save_metrics:
        timestamp = time.strftime("%d%m_%H%M")
        metrics_data = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "timestamp": timestamp,
            "best_val_loss": best_loss,
            "best_val_acc": best_acc,
            "best_train_loss": best_loss_train,
            "best_train_acc": best_acc_train,
            "num_epochs": num_epochs,
        }
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_file = os.path.join(metrics_dir, f"training_metrics_{timestamp}.json")
        with open(metrics_file, "w", encoding="utf-8") as handle:
            json.dump(metrics_data, handle, indent=4)
        print(f"Metrics saved to: {metrics_file}")

    model.load_state_dict(best_model_wts)
    return model

# Used in: crema-d-enhanced.ipynb (prediction visualization)
def visualize_model(model: nn.Module, dataloader_val: torch.utils.data.DataLoader,
                    class_names: list, device, num_images: int = 6,
                    fig_name: str = "Predictions") -> None:
    """
    Display validation images with predicted labels.

    Args:
        model: Trained model.
        dataloader_val: Validation dataloader.
        class_names: List of class names.
        device: Torch device.
        num_images: Number of images to display.
        fig_name: Figure title.
    """
    images_so_far = 0
    fig = plt.figure(fig_name)
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader_val:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size(0)):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"[{class_names[preds[j]]}]")
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    plt.tight_layout()
                    return


# Used in: crema-d-enhanced.ipynb (model persistence), ants_bees.ipynb (model persistence)
def save_model(model: nn.Module, quantum: bool, name: str, models_dir: str = "models") -> None:
    """
    Save a trained model to disk.

    Args:
        model: Model to save.
        quantum: Whether the model is quantum-hybrid.
        name: Base filename without extension.
        models_dir: Directory where the model will be stored.
    """
    os.makedirs(models_dir, exist_ok=True)
    suffix = "q" if quantum else "c"
    path = os.path.join(models_dir, f"{suffix}_{name}.pt")
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")


# Used in: crema-d-enhanced.ipynb (model persistence), ants_bees.ipynb (model persistence)
def load_model(model: nn.Module, quantum: bool, name: str, models_dir: str = "models") -> nn.Module:
    """
    Load model weights from disk.

    Args:
        model: Instantiated model to load weights into.
        quantum: Whether the stored model is quantum-hybrid.
        name: Base filename without extension.
        models_dir: Directory containing the model file.

    Returns:
        Model with loaded state dict.
    """
    suffix = "quantum" if quantum else "classical"
    path = os.path.join(models_dir, f"{name}_{suffix}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from: {path}")
    return model


