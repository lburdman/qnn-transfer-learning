import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

# -------------------------------
# Capas del circuito cuántico
# -------------------------------

def H_layer(nqubits):
    """Aplica Hadamard a todos los qubits."""
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)

def RY_layer(w):
    """Rota cada qubit alrededor de Y usando un vector de ángulos w."""
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def entangling_layer(nqubits):
    """Crea entrelazamiento alternado en anillos pares e impares."""
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

# -------------------------------
# Constructor de QNode
# -------------------------------

def build_qnode(n_qubits, q_depth, max_layers, dev):
    @qml.qnode(dev, interface='torch')
    def q_net(q_in, q_weights_flat):
        q_weights = q_weights_flat.reshape(max_layers, n_qubits)
        H_layer(n_qubits)
        RY_layer(q_in)
        for k in range(q_depth):
            entangling_layer(n_qubits)
            RY_layer(q_weights[k + 1])
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
    
    return q_net

# -------------------------------
# Clase del modelo híbrido
# -------------------------------

class Quantumnet(nn.Module):
    def __init__(self, n_qubits, q_depth, max_layers, q_delta, dev, n_classes=2, base_model='resnet18'):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.max_layers = max_layers
        self.q_net = build_qnode(n_qubits, q_depth, max_layers, dev)

        if base_model == 'resnet18':
            self.reduction = nn.Linear(512, n_qubits)
        elif base_model == 'vgg16':
            self.reduction = nn.Sequential(
                nn.Linear(4096, 512),
                nn.ReLU(),
                nn.Linear(512, n_qubits)
            )
        # self.pre_net = nn.Linear(4096, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(max_layers * n_qubits))
        self.post_net = nn.Linear(n_qubits, n_classes)

    def forward(self, input_features):
        """Proyección a n_qubits, ejecución del QNode por elemento y clasificación clásica final.
        Nota: PennyLane no vectoriza este QNode; por eso se itera por elemento.
        """
        # pre_out = self.pre_net(input_features)
        pre_out = self.reduction(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        batch_size = q_in.shape[0]
        # Preasignación para evitar concatenaciones O(n^2)
        q_out = torch.empty((batch_size, self.n_qubits), device=input_features.device, dtype=torch.float32)

        for i, elem in enumerate(q_in):
            q_out_elem = torch.stack(self.q_net(elem, self.q_params)).float()
            q_out[i] = q_out_elem

        return self.post_net(q_out)


if __name__ == "__main__":
    import torch
    import pennylane as qml

    # Hiperparámetros de test
    n_qubits = 4
    q_depth = 3
    max_layers = 10
    q_delta = 0.01
    n_classes = 3
    batch_size = 2

    # Crear dispositivo cuántico
    dev = qml.device("default.qubit", wires=n_qubits)

    # Instanciar el modelo
    model = Quantumnet(n_qubits, q_depth, max_layers, q_delta, dev, n_classes)
    model.eval()

    # Generar input dummy (como salida de ResNet18)
    x_dummy = torch.randn(batch_size, 512)

    # Hacer forward
    with torch.no_grad():
        out = model(x_dummy)

    print("✅ Test completado correctamente")
    print("🔢 Output shape:", out.shape)  # debería ser (batch_size, n_classes)
    print("📊 Output ejemplo:\n", out)
