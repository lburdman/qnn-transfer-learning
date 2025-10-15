import time
import copy
import torch
import os
import json
import matplotlib.pyplot as plt
from utils import imshow  # función para mostrar imágenes normalizadas

# -----------------------------------------
# 🏋️ Función de entrenamiento principal (comentada a petición del usuario)
# -----------------------------------------
"""
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

            # 👉 Log en TensorBoard
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
"""


def train_model1(model, dataloaders, dataset_sizes, device,
                criterion, optimizer, scheduler, num_epochs, save_metrics=True, metrics_dir="runs_updated"):
    """
    Entrena un modelo PyTorch usando entrenamiento supervisado.
    
    Parámetros:
    - model: instancia de nn.Module
    - dataloaders: dict con 'train' y 'val'
    - dataset_sizes: dict con tamaños de cada dataset
    - device: 'cuda' o 'cpu'
    - criterion: función de pérdida (loss)
    - optimizer: optimizador de PyTorch
    - scheduler: scheduler de learning rate
    - num_epochs: cantidad total de épocas
    - save_metrics: si guardar métricas en JSON
    - metrics_dir: directorio donde guardar métricas

    Devuelve:
    - model: modelo con los mejores pesos validados
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_train = 0.0
    best_loss_train = float("inf")
    best_loss = float("inf")

    # Listas para almacenar métricas
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    print('🚀 Training started:')

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            n_batches = len(dataloaders[phase])

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

                print('Phase: {}    Epoch: {}/{}    Iter: {}/{} Batch time: {:.4f}s'.format(
                    phase, epoch + 1, num_epochs, it + 1, n_batches + 1, time.time() - since_batch),
                    end='\r', flush=True)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('Phase: {}    Epoch: {}/{}    Loss: {:.4f}    Acc: {:.4f}        '.format(
                phase, epoch + 1, num_epochs, epoch_loss, epoch_acc))

            # Guardar métricas
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

            # Actualizar mejores pesos según validación
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
            if phase == 'train' and epoch_acc > best_acc_train:
                best_acc_train = epoch_acc
            if phase == 'train' and epoch_loss < best_loss_train:
                best_loss_train = epoch_loss

        # Avanzar el scheduler solo 1 vez por epoch
        scheduler.step()

    time_elapsed = time.time() - since
    print('Entrenamiento completado en {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Mejor loss (val): {:.4f} | Mejor accuracy (val): {:.4f}'.format(
        best_loss, best_acc))

    # Guardar métricas en JSON si se solicita
    if save_metrics:
        timestamp = time.strftime('%d%m_%H%M')  # Formato: día-mes_hora-minuto
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
            "num_epochs": num_epochs
        }
        
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_file = os.path.join(metrics_dir, f"training_metrics_{timestamp}.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        print(f"📊 Métricas guardadas en: {metrics_file}")

    model.load_state_dict(best_model_wts)
    return model

# -----------------------------------------
# 🖼️ Función para visualizar predicciones
# -----------------------------------------
def visualize_model(model, dataloader_val, class_names, device,
                    num_images=6, fig_name='Predictions'):
    """
    Visualiza imágenes de validación con sus predicciones.
    
    Parámetros:
    - model: modelo entrenado
    - dataloader_val: dataloader de validación
    - class_names: lista de nombres de clase
    - device: 'cuda' o 'cpu'
    - num_images: cantidad de imágenes a mostrar
    - fig_name: nombre del gráfico
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
                ax.axis('off')
                ax.set_title('[{}]'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    plt.tight_layout()
                    return

def save_model(model, quantum: bool, name: str, models_dir="models"):
    """
    Guarda el modelo entrenado en formato .pt

    Parámetros:
    - model: modelo a guardar
    - quantum (bool): si es un modelo híbrido cuántico
    - name (str): nombre base del archivo, sin extensión
    - models_dir (str): carpeta donde guardar los modelos
    """
    os.makedirs(models_dir, exist_ok=True)
    suffix = "q" if quantum else "c"
    path = os.path.join(models_dir, f"{suffix}_{name}.pt")
    torch.save(model.state_dict(), path)
    print(f"💾 Modelo guardado en: {path}")


def load_model(model, quantum: bool, name: str, models_dir="models"):
    """
    Carga los pesos de un modelo guardado.

    Parámetros:
    - model: instancia del modelo (ya creada)
    - quantum (bool): si es cuántico o clásico
    - name (str): nombre base del archivo, sin extensión
    - models_dir (str): carpeta donde buscar los modelos

    Devuelve:
    - model con pesos cargados (o error si no existe el archivo)
    """
    suffix = "quantum" if quantum else "classical"
    path = os.path.join(models_dir, f"{name}_{suffix}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"❌ No se encontró el modelo en: {path}")

    model.load_state_dict(torch.load(path))
    print(f"✅ Modelo cargado desde: {path}")
    return model
