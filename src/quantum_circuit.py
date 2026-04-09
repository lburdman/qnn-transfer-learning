"""
Quantum circuit components and hybrid model definitions.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path


def H_layer(nqubits: int) -> None:
    """
    Apply a Hadamard gate to all qubits.

    Args:
        nqubits: Number of qubits.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(weights) -> None:
    """
    Apply a rotation around the Y axis for each qubit.

    Args:
        weights: Iterable of rotation angles.
    """
    for idx, element in enumerate(weights):
        qml.RY(element, wires=idx)


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



def analyze_trained_quantum_head(
    model: nn.Module,
    device,
    sample_input: Optional[np.ndarray] = None,
    print_density: bool = False,
    save_dir: Optional[str] = None,
):
    """
    Print the quantum circuit diagram used in the head and report a simple purity diagnostic
    using the trained weights. Uses the last TorchLayer when multiple are present.
    """
    try:
        torchlayer_cls = qml.qnn.TorchLayer
    except Exception:
        print("PennyLane not available; cannot analyze quantum head.")
        return

    qlayers = [m for m in model.modules() if isinstance(m, torchlayer_cls)]
    if not qlayers:
        print("No PennyLane TorchLayer found in model.")
        return
    qlayer = qlayers[-1]

    # Extract trained weights
    trained_weights = None
    for name, param in qlayer.named_parameters():
        if name == "weights":
            trained_weights = param.detach().to(device)
            break
    if trained_weights is None:
        print("No trainable weights found in quantum layer.")
        return

    n_qubits = trained_weights.shape[1]
    q_depth = trained_weights.shape[0]
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return qml.state()

    inputs = sample_input if sample_input is not None else np.zeros((n_qubits,), dtype=np.float32)
    weights_np = trained_weights.cpu().numpy()
    print("Quantum head circuit (decomposed) with trained parameters:")
    try:
        fig, _ = qml.draw_mpl(qnode, expansion_strategy="device")(inputs, weights_np)
        plt.show()
    except Exception:  # pragma: no cover
        fig = None
        diagram = qml.draw(qnode)(inputs, weights_np)
        print(diagram)

    if print_density:
        state = qnode(inputs, weights_np)
        rho = np.outer(state, state.conj())
        purity = np.trace(rho @ rho).real
        print(f"Purity of provided-input state: {purity:.4f}")

    if save_dir:
        save_dir = Path(save_dir)
        if fig is not None:
            try:
                fig.savefig(save_dir / "quantum_circuit_decomposed.png", dpi=300, bbox_inches="tight")  # type: ignore[arg-type]
            except Exception:
                pass


class BaseHybridHead(nn.Module):
    """
    Base class for hybrid quantum-classical heads using a parameterized QNode.
    """
    def __init__(self, n_qubits: int, q_depth: int, max_layers: int, q_delta: float, dev,
                 qnode_func, n_classes: int = 2, base_model: str = "resnet18") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.max_layers = max_layers
        self.q_net = qnode_func(n_qubits, q_depth, max_layers, dev)

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
        
        q_outs = [torch.stack(self.q_net(elem, self.q_params)).float() for elem in q_in]
        q_out = torch.stack(q_outs)
        
        return self.post_net(q_out)


class Quantumnet(BaseHybridHead):
    """
    Hybrid quantum-classical head that wraps a PennyLane QNode.
    """
    def __init__(self, n_qubits: int, q_depth: int, max_layers: int, q_delta: float, dev, n_classes: int = 2,
                 base_model: str = "resnet18") -> None:
        super().__init__(n_qubits, q_depth, max_layers, q_delta, dev, build_qnode, n_classes, base_model)



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


class DressedQuantumCircuit(BaseHybridHead):
    """
    Hybrid quantum-classical head using the enhanced QNode with feature maps.
    """
    def __init__(self, n_qubits: int, q_depth: int, max_layers: int, q_delta: float, dev, n_classes: int = 2,
                 base_model: str = "resnet18") -> None:
        super().__init__(n_qubits, q_depth, max_layers, q_delta, dev, build_qnode2, n_classes, base_model)


def draw_qnode_circuit_example(
    n_qubits: int = 2,
    q_depth: int = 2,
    max_layers: int | None = None,
    seed: int | None = None,
    style: str = "pennylane",
):
    """
    Draw the qnode built by build_qnode both at the template level and decomposed into basic gates.
    This mirrors the notebook helper but uses the circuit defined in this module.
    """

    # max_layers should at least cover the accessed slices inside build_qnode
    effective_max_layers = max_layers if max_layers is not None else q_depth + 1

    rng = np.random.default_rng(seed)
    q_in = torch.tensor(rng.uniform(-np.pi, np.pi, size=(n_qubits,)), dtype=torch.float32)
    q_weights_flat = torch.tensor(
        rng.uniform(-1.0, 1.0, size=(effective_max_layers * n_qubits,)), dtype=torch.float32
    )

    dev = qml.device("default.qubit", wires=n_qubits)
    qnode = build_qnode(n_qubits, q_depth, effective_max_layers, dev)

    # Run once so the tape is constructed
    _ = qnode(q_in, q_weights_flat)

    print("ASCII circuit diagram (templates):")
    print(qml.draw(qnode)(q_in, q_weights_flat))

    try:
        drawer = qml.draw_mpl(qnode, expansion_strategy="device", style=style)
        fig, ax = drawer(q_in, q_weights_flat)
        plt.show()
    except Exception as exc:
        print(f"Matplotlib draw (templates) failed: {exc}")

    # Decompose to a basic gate set for a lower-level view
    decomp_qnode = qml.transforms.decompose(
        qnode,
        gate_set={qml.RX, qml.RY, qml.RZ, qml.CNOT, qml.Hadamard, qml.PhaseShift},
    )

    _ = decomp_qnode(q_in, q_weights_flat)

    print("\nASCII decomposed circuit (basic gates):")
    print(qml.draw(decomp_qnode)(q_in, q_weights_flat))

    try:
        decomp_drawer = qml.draw_mpl(decomp_qnode, expansion_strategy="device", style=style)
        fig2, ax2 = decomp_drawer(q_in, q_weights_flat)
        plt.show()
    except Exception as exc:
        print(f"Matplotlib draw (decomposed) failed: {exc}")



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
