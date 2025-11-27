"""
Quantum circuit components and hybrid model definitions.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn


# Used in: src.quantum_circuit.build_qnode (Hadamard initialization)
def H_layer(nqubits: int) -> None:
    """
    Apply a Hadamard gate to all qubits.

    Args:
        nqubits: Number of qubits.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


# Used in: src.quantum_circuit.build_qnode (data embedding)
def RY_layer(weights) -> None:
    """
    Apply a rotation around the Y axis for each qubit.

    Args:
        weights: Iterable of rotation angles.
    """
    for idx, element in enumerate(weights):
        qml.RY(element, wires=idx)


# Used in: src.quantum_circuit.build_qnode (entanglement)
def entangling_layer(nqubits: int) -> None:
    """
    Apply alternating CNOT gates to entangle neighboring qubits.

    Args:
        nqubits: Number of qubits.
    """
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


# Used in: ants_bees.ipynb (hybrid head), crema-d*.ipynb (hybrid head)
def build_qnode(n_qubits: int, q_depth: int, max_layers: int, dev) -> qml.QNode:
    """
    Construct a PennyLane QNode using simple RY entangling layers.

    Args:
        n_qubits: Number of qubits.
        q_depth: Quantum depth for trainable layers.
        max_layers: Maximum number of parameter layers.
        dev: PennyLane device.

    Returns:
        Configured QNode compatible with Torch.
    """

    @qml.qnode(dev, interface="torch")
    def q_net(q_in, q_weights_flat):
        q_weights = q_weights_flat.reshape(max_layers, n_qubits)
        H_layer(n_qubits)
        RY_layer(q_in)
        for depth_idx in range(q_depth):
            entangling_layer(n_qubits)
            RY_layer(q_weights[depth_idx + 1])
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

    return q_net


# Used in: ants_bees.ipynb (hybrid head), crema-d*.ipynb (hybrid head)
class Quantumnet(nn.Module):
    """
    Hybrid quantum-classical head that wraps a PennyLane QNode.
    """

    def __init__(self, n_qubits: int, q_depth: int, max_layers: int, q_delta: float, dev, n_classes: int = 2,
                 base_model: str = "resnet18") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.max_layers = max_layers
        self.q_net = build_qnode(n_qubits, q_depth, max_layers, dev)

        if base_model == "resnet18":
            self.pre_net = nn.Linear(512, n_qubits)
        elif base_model == "vgg16":
            self.pre_net = nn.Sequential(
                nn.Linear(4096, 512),
                nn.ReLU(),
                nn.Linear(512, n_qubits),
            )
        else:
            self.pre_net = nn.Linear(4096, n_qubits)

        self.q_params = nn.Parameter(q_delta * torch.randn(max_layers * n_qubits))
        self.post_net = nn.Linear(n_qubits, n_classes)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        q_out = torch.zeros((0, self.n_qubits), device=input_features.device)
        for elem in q_in:
            q_out_elem = torch.stack(self.q_net(elem, self.q_params)).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        return self.post_net(q_out)


# Used in: src.quantum_circuit.build_qnode2 (feature map)
def zz_feature_map(x, nqubits: int, reps: int = 2) -> None:
    """
    Apply a ZZFeatureMap-style embedding with repeated layers.

    Args:
        x: Input angles.
        nqubits: Number of qubits.
        reps: Number of repetitions.
    """
    for _ in range(reps):
        for i in range(nqubits):
            qml.Hadamard(wires=i)
            qml.PhaseShift(2 * x[i], wires=i)
        for i in range(nqubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.PhaseShift(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)


# Used in: src.quantum_circuit.build_qnode2 (variational block)
def real_amplitudes_block(weights, nqubits: int) -> None:
    """
    Apply a RealAmplitudes-inspired rotation and entanglement block.

    Args:
        weights: Rotation angles for each qubit.
        nqubits: Number of qubits.
    """
    for i in range(nqubits):
        qml.RY(weights[i], wires=i)
    for i in range(nqubits - 1):
        qml.CNOT(wires=[i, i + 1])


# Used in: crema-d-enhanced.ipynb (enhanced hybrid head), ants_bees.ipynb (hybrid head)
def build_qnode2(n_qubits: int, q_depth: int, max_layers: int, dev) -> qml.QNode:
    """
    Construct a PennyLane QNode with ZZFeatureMap and RealAmplitudes-style layers.

    Args:
        n_qubits: Number of qubits.
        q_depth: Number of variational blocks.
        max_layers: Maximum number of parameter layers.
        dev: PennyLane device.

    Returns:
        Configured QNode compatible with Torch.
    """

    @qml.qnode(dev, interface="torch")
    def q_net(q_in, q_weights_flat):
        q_weights = q_weights_flat.reshape(max_layers, n_qubits)
        zz_feature_map(q_in, n_qubits, reps=2)
        for layer_idx in range(q_depth):
            real_amplitudes_block(q_weights[layer_idx], n_qubits)
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

    return q_net


# Used in: crema-d-enhanced.ipynb (enhanced hybrid head)
class DressedQuantumCircuit(nn.Module):
    """
    Hybrid quantum-classical head using the enhanced QNode with feature maps.
    """

    def __init__(self, n_qubits: int, q_depth: int, max_layers: int, q_delta: float, dev, n_classes: int = 2,
                 base_model: str = "resnet18") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.max_layers = max_layers
        self.q_net = build_qnode2(n_qubits, q_depth, max_layers, dev)

        if base_model == "resnet18":
            self.pre_net = nn.Linear(512, n_qubits)
        elif base_model == "vgg16":
            self.pre_net = nn.Sequential(
                nn.Linear(4096, 512),
                nn.ReLU(),
                nn.Linear(512, n_qubits),
            )
        else:
            self.pre_net = nn.Linear(4096, n_qubits)

        self.q_params = nn.Parameter(q_delta * torch.randn(max_layers * n_qubits))
        self.post_net = nn.Linear(n_qubits, n_classes)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        q_out = torch.zeros((0, self.n_qubits), device=input_features.device)
        for elem in q_in:
            q_out_elem = torch.stack(self.q_net(elem, self.q_params)).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        return self.post_net(q_out)


if __name__ == "__main__":
    n_qubits = 4
    q_depth = 3
    max_layers = 10
    q_delta = 0.01
    n_classes = 3
    batch_size = 2

    device = qml.device("default.qubit", wires=n_qubits)
    model = Quantumnet(n_qubits, q_depth, max_layers, q_delta, device, n_classes)
    model.eval()
    sample_input = torch.randn(batch_size, 512)
    with torch.no_grad():
        out = model(sample_input)
    print("Test completed successfully")
    print("Output shape:", out.shape)
    print("Sample output:\n", out)
