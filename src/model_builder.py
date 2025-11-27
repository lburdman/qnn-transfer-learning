"""
Model construction utilities for classical and quantum-hybrid architectures.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pennylane as qml
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, VGG16_Weights


# Used in: crema_d_hybrid_qnn.ipynb (custom CNN backbone)
def create_custom_cnn(input_channels: int = 3) -> nn.Module:
    """
    Build a lightweight CNN backbone returning a flattened feature vector.

    Args:
        input_channels: Number of input channels (1 or 3).

    Returns:
        CNN backbone with attribute `output_dim`.
    """

    class CustomCNNBackbone(nn.Module):
        def __init__(self, input_channels: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.output_dim = 512

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            return torch.flatten(x, 1)

    return CustomCNNBackbone(input_channels)


# Used in: crema_d_hybrid_qnn.ipynb (quantum layer construction)
def build_quantum_layer(n_qubits: int, q_depth: int):
    """
    Build a PennyLane TorchLayer with AngleEmbedding and BasicEntanglerLayers.

    Args:
        n_qubits: Number of qubits.
        q_depth: Depth of the entangler layers.

    Returns:
        TorchLayer wrapping the defined quantum circuit.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (q_depth, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


# Used in: crema_d_hybrid_qnn.ipynb (model assembly)
def build_model(config: Dict[str, object], class_names: Sequence[str], dataloaders: Dict[str, DataLoader], device):
    """
    Assemble a classical or quantum-hybrid model based on configuration.

    Args:
        config: Experiment configuration dictionary.
        class_names: Ordered list of class labels.
        dataloaders: Mapping of phase to DataLoader (used to infer embedding shapes).
        device: Torch device.

    Returns:
        Configured model moved to the specified device.
    """
    base_model = config["base_model"]
    quantum = config["quantum"]
    classical_model = config["classical_model"]
    n_qubits = config["n_qubits"]
    q_depth = config["q_depth"]
    use_pretrained = config["use_pretrained"]
    freeze_backbone = config["freeze_backbone"]
    use_generic_weights = config["use_generic_weights"]
    grayscale = config["grayscale"]
    n_classes = len(class_names)

    model = None
    if base_model == "custom_cnn":
        force_three_channels = bool(grayscale and use_pretrained)
        input_channels = 1 if (grayscale and not force_three_channels) else 3
        backbone = create_custom_cnn(input_channels=input_channels)
        feature_dim = getattr(backbone, "output_dim", 512)
        if quantum:
            qlayer = build_quantum_layer(n_qubits, q_depth)
            head = nn.Sequential(
                nn.Linear(feature_dim, n_qubits),
                nn.ReLU(),
                qlayer,
                nn.Linear(n_qubits, n_classes),
            )
        else:
            if classical_model == "512_2":
                head = nn.Linear(feature_dim, n_classes)
            elif classical_model == "512_nq_2":
                head = nn.Sequential(
                    nn.Linear(feature_dim, n_qubits),
                    nn.ReLU(),
                    nn.Linear(n_qubits, n_qubits),
                    nn.ReLU(),
                    nn.Linear(n_qubits, n_classes),
                )
            elif classical_model == "551_512_2":
                head = nn.Sequential(
                    nn.Linear(feature_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, n_classes),
                )
            else:
                raise ValueError(f"Unsupported classifier for custom_cnn: {classical_model}")
        model = nn.Sequential(backbone, head)

    elif base_model == "resnet18":
        model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT if use_pretrained else None)
        if grayscale:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        if quantum:
            qlayer = build_quantum_layer(n_qubits, q_depth)
            model.fc = nn.Sequential(
                nn.Linear(in_features, n_qubits),
                nn.ReLU(),
                qlayer,
                nn.Linear(n_qubits, n_classes),
            )
        else:
            if classical_model == "512_2":
                model.fc = nn.Linear(in_features, n_classes)
            elif classical_model == "512_nq_2":
                model.fc = nn.Sequential(
                    nn.Linear(in_features, n_qubits),
                    nn.ReLU(),
                    nn.Linear(n_qubits, n_qubits),
                    nn.ReLU(),
                    nn.Linear(n_qubits, n_classes),
                )
            elif classical_model == "551_512_2":
                model.fc = nn.Sequential(
                    nn.Linear(in_features, 512),
                    nn.ReLU(),
                    nn.Linear(512, n_classes),
                )

    elif base_model == "vgg16":
        model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT if use_pretrained else None)
        if grayscale:
            first_layer = model.features[0]
            if isinstance(first_layer, nn.Conv2d) and first_layer.in_channels == 3:
                model.features[0] = nn.Conv2d(
                    1,
                    first_layer.out_channels,
                    kernel_size=first_layer.kernel_size,
                    stride=first_layer.stride,
                    padding=first_layer.padding,
                    bias=False,
                )
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[6].in_features
        if quantum:
            qlayer = build_quantum_layer(n_qubits, q_depth)
            model.classifier[6] = nn.Sequential(
                nn.Linear(in_features, n_qubits),
                nn.ReLU(),
                qlayer,
                nn.Linear(n_qubits, n_classes),
            )
        else:
            if classical_model == "512_2":
                model.classifier[6] = nn.Linear(in_features, n_classes)
            elif classical_model == "512_nq_2":
                model.classifier[6] = nn.Sequential(
                    nn.Linear(in_features, 512),
                    nn.ReLU(),
                    nn.Linear(512, n_qubits),
                    nn.ReLU(),
                    nn.Linear(n_qubits, n_classes),
                )
            elif classical_model == "551_512_2":
                model.classifier[6] = nn.Sequential(
                    nn.Linear(in_features, 512),
                    nn.ReLU(),
                    nn.Linear(512, n_classes),
                )

    elif base_model in ["emb_resnet18", "emb_vgg16", "emb_panns_cnn14", "mfcc"]:
        sample_inputs, _ = next(iter(dataloaders["train"]))
        input_dim = sample_inputs.shape[1] if sample_inputs.ndim > 1 else sample_inputs.numel()
        if quantum:
            qlayer = build_quantum_layer(n_qubits, q_depth)
            model = nn.Sequential(
                nn.Linear(input_dim, n_qubits),
                nn.ReLU(),
                qlayer,
                nn.Linear(n_qubits, n_classes),
            )
        else:
            if classical_model == "512_2":
                model = nn.Linear(input_dim, n_classes)
            elif classical_model == "512_nq_2":
                model = nn.Sequential(
                    nn.Linear(input_dim, n_qubits),
                    nn.ReLU(),
                    nn.Linear(n_qubits, n_qubits),
                    nn.ReLU(),
                    nn.Linear(n_qubits, n_classes),
                )
            elif classical_model == "551_512_2":
                model = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, n_classes),
                )
    else:
        raise ValueError(f"Unsupported base_model: {base_model}")

    if use_generic_weights and not quantum and model is not None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    if model is None:
        raise RuntimeError("Model construction failed; check configuration.")
    return model.to(device)
