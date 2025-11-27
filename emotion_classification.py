import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import pennylane as qml
import matplotlib.pyplot as plt
import time
import copy
import sys
sys.path.append('src')
from src.dataset import get_dataloaders
from src.utils import imshow
from src.quantum_circuit import Quantumnet
from torch.utils.tensorboard import SummaryWriter

# Función de entrenamiento
def train_model(model, dataloaders, dataset_sizes, device,
                criterion, optimizer, scheduler, num_epochs,
                writer=None):
    """
    Entrena un modelo PyTorch e integra logs en TensorBoard si `writer` está definido.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')

    print('🚀 Training started:')

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            n_batches = dataset_sizes[phase] // dataloaders[phase].batch_size

            for it, (inputs, labels) in enumerate(dataloaders[phase]):
                since_batch = time.time()
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

                print('Phase: {} Epoch: {}/{} Iter: {}/{} Batch time: {:.4f}s'.format(
                    phase, epoch + 1, num_epochs, it + 1, n_batches + 1, time.time() - since_batch),
                    end='\r', flush=True)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('Phase: {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch + 1, num_epochs, epoch_loss, epoch_acc))

            # Log en TensorBoard
            if writer:
                writer.add_scalar(f'{phase}/Loss', epoch_loss, epoch)
                writer.add_scalar(f'{phase}/Accuracy', epoch_acc, epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

    time_elapsed = time.time() - since
    print('✅ Entrenamiento finalizado en {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('🎯 Mejor loss (val): {:.4f} | Mejor accuracy (val): {:.4f}'.format(best_loss, best_acc))

    model.load_state_dict(best_model_wts)
    return model

# Configuración
base_model = 'resnet18'
n_qubits = 5
quantum = True
step = 0.0004
batch_size = 4
num_epochs = 20
q_depth = 6
gamma_lr_scheduler = 0.1
q_delta = 0.01
rng_seed = 0
data_dir = 'CremaD/mel_spec_reduced'

# Dispositivos
dev = qml.device('default.qubit', wires=n_qubits)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ Usando dispositivo: {device}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# Carga de datos
dataloaders, dataset_sizes, class_names = get_dataloaders(
    data_dir=data_dir, batch_size=batch_size, shuffle=True, spec_augment=True
)
n_classes = len(class_names)

# Visualizar un batch
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

# Definición del modelo
model_hybrid = torchvision.models.resnet18(pretrained=True)
for param in model_hybrid.parameters():
    param.requires_grad = False

if quantum:
    model_hybrid.fc = Quantumnet(
        n_qubits=n_qubits,
        q_depth=q_depth,
        max_layers=15,
        q_delta=q_delta,
        dev=dev,
        n_classes=n_classes
    )
else:
    model_hybrid.fc = nn.Linear(512, n_classes)

model_hybrid = model_hybrid.to(device)
torch.manual_seed(rng_seed)

# Pérdida, optimizador y scheduler
criterion = nn.CrossEntropyLoss()
optimizer_hybrid = optim.Adam(model_hybrid.fc.parameters(), lr=step)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_hybrid, step_size=10, gamma=gamma_lr_scheduler)

# TensorBoard Writer (opcional)
writer = SummaryWriter(log_dir='runs/emotion_classification')

# Entrenar el modelo
print("🚀 Starting training...")
model_hybrid = train_model(
    model=model_hybrid,
    dataloaders=dataloaders,
    dataset_sizes=dataset_sizes,
    device=device,
    criterion=criterion,
    optimizer=optimizer_hybrid,
    scheduler=exp_lr_scheduler,
    num_epochs=num_epochs,
    writer=writer
)

# Cerrar TensorBoard writer
writer.close()

# Visualización del circuito cuántico
if quantum:
    @qml.qnode(dev)
    def sample_circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    dummy_inputs = torch.randn(n_qubits)
    dummy_weights = torch.randn(q_depth, n_qubits, 3)
    
    print("\nQuantum Circuit Diagram:")
    print(qml.draw(sample_circuit)(dummy_inputs, dummy_weights))
    
    fig, ax = qml.draw_mpl(sample_circuit)(dummy_inputs, dummy_weights)
    plt.title("Quantum Circuit Visualization")
    plt.show()