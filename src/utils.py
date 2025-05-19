import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def imshow(inp, title=None):
    """
    Muestra una imagen a partir de un tensor normalizado (como el que devuelve un DataLoader).
    
    Parámetros:
    - inp (Tensor): tensor de imagen con shape (C, H, W)
    - title (str): texto opcional para mostrar como título

    Detalles:
    Esta función:
    1. Transforma el tensor de PyTorch a un arreglo NumPy de shape (H, W, C)
    2. Aplica la inversión de la normalización de ImageNet
    3. Muestra la imagen con matplotlib
    """
    inp = inp.numpy().transpose((1, 2, 0))  # (C, H, W) → (H, W, C)

    # Inversión de la normalización de ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  
    inp = np.clip(inp, 0, 1)

    # Mostrar imagen
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def plot_tensorboard_metric(run_name: str, metric: str, phase: str, runs_dir="runs"):
    """
    Grafica una métrica específica (Loss o Accuracy) para una fase (train o val).

    Parámetros:
    - run_name (str): subcarpeta dentro de 'runs/' (ej: 'q_ants_bees_resnet18_20240516_2135')
    - metric (str): nombre de la métrica ('Loss' o 'Accuracy')
    - phase (str): 'train' o 'val'
    - runs_dir (str): carpeta base donde se guardan los logs (default: 'runs/')
    """
    from tensorboard.backend.event_processing import event_accumulator
    import os
    import matplotlib.pyplot as plt

    log_dir = os.path.join(runs_dir, run_name)

    event_file = next(
        (os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")),
        None
    )
    if event_file is None:
        raise FileNotFoundError(f"No se encontró archivo de eventos en {log_dir}")

    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    tag = f"{phase}/{metric}"
    if tag not in ea.Tags().get("scalars", []):
        raise ValueError(f"La métrica '{tag}' no existe en este run.")

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    # Graficar
    plt.figure(figsize=(7, 4))
    plt.plot(steps, values, label=f"{phase} {metric}")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} - {phase} ({run_name})")
    # if metric.lower() == "accuracy":
    plt.ylim(0, 1)  # 🔒 Limita el eje y para accuracy
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
