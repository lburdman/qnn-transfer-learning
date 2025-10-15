import numpy as np
import matplotlib.pyplot as plt
# from tensorboard.backend.event_processing import event_accumulator

def imshow(inp, title=None):
    """
    Muestra una imagen a partir de un tensor normalizado (como el que devuelve un DataLoader).
    
    Parámetros:
    - inp (Tensor): tensor de imagen con shape (C, H, W)
    - title (str): texto opcional para mostrar como título

    Nota: Detecta dinámicamente si la imagen es 1 canal (grayscale) o 3 canales (RGB)
    y aplica la inversión de normalización correspondiente.
    """
    np_img = inp.numpy()  # (C, H, W)
    num_channels = np_img.shape[0]

    if num_channels == 1:
        mean = np.array([0.5])
        std = np.array([0.5])
        np_img = (np_img * std[:, None, None] + mean[:, None, None]).squeeze(0)  # (H, W)
        np_img = np.clip(np_img, 0, 1)
        plt.imshow(np_img, cmap='gray', vmin=0, vmax=1)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_img = np_img.transpose((1, 2, 0))  # (H, W, C)
        np_img = std * np_img + mean
        np_img = np.clip(np_img, 0, 1)
        plt.imshow(np_img)

    if title is not None:
        plt.title(title)

def show_exact_images_from_dataloader(dataloader,
                                     phase='train',
                                     n_images=4,
                                     prefix='',
                                     grayscale=False):
    """
    Display the first n_images from a DataLoader, unnormalizing them
    so you see the exact dataset images, and show their class above each.

    Args:
        dataloader (DataLoader): loader for 'train' or 'val'.
        phase (str): 'train' or 'val', used in the suptitle.
        n_images (int): how many images to display.
        prefix (str): title prefix.
        grayscale (bool): True if images are single-channel.
    """
    # Las estadísticas de normalización pueden variar por canal.
    # Se decidirá por imagen en función del número de canales.

    # If using ImageFolder, get class names from the dataset
    class_names = dataloader.dataset.classes

    # Collect first n_images and their labels
    imgs, labels = [], []
    for inputs, labs in dataloader:
        for img, lab in zip(inputs, labs):
            imgs.append(img.detach().cpu().numpy())
            labels.append(int(lab))
            if len(imgs) >= n_images:
                break
        if len(imgs) >= n_images:
            break

    # Unnormalize and reorder channels
    proc_imgs = []
    for img in imgs:
        # img shape: [C, H, W]
        channels = img.shape[0]
        if channels == 1:
            mean = np.array([0.5])
            std  = np.array([0.5])
            img = (img * std[:, None, None] + mean[:, None, None]).squeeze(0)  # (H, W)
        else:
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            # aplicar por canal y reordenar a HWC
            for c in range(channels):
                img[c] = img[c] * std[c] + mean[c]
            img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        proc_imgs.append(img)

    # Plot
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 3, 3))
    if n_images == 1:
        axes = [axes]

    for ax, img, lab in zip(axes, proc_imgs, labels):
        if img.ndim == 2:  # grayscale
            # matplotlib's default colormap for 2D arrays is 'viridis'
            # so we must specify cmap='gray' to render true grayscale
            ax.imshow(img, cmap='gray', aspect='auto', vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(img, 0, 1), aspect='auto')

        # Show class name above each image
        ax.set_title(class_names[lab], fontsize=10)
        ax.axis('off')

    # Global title
    fig.suptitle(f"{prefix} - conjunto {phase}", fontsize=14)
    plt.tight_layout()
    plt.show()

# def plot_tensorboard_metric(run_name: str, metric: str, phase: str, runs_dir="runs"):
#     """
#     Grafica una métrica específica (Loss o Accuracy) para una fase (train o val).

#     Parámetros:
#     - run_name (str): subcarpeta dentro de 'runs/' (ej: 'q_ants_bees_resnet18_20240516_2135')
#     - metric (str): nombre de la métrica ('Loss' o 'Accuracy')
#     - phase (str): 'train' o 'val'
#     - runs_dir (str): carpeta base donde se guardan los logs (default: 'runs/')
#     """
#     from tensorboard.backend.event_processing import event_accumulator
#     import os
#     import matplotlib.pyplot as plt

#     log_dir = os.path.join(runs_dir, run_name)

#     event_file = next(
#         (os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")),
#         None
#     )
#     if event_file is None:
#         raise FileNotFoundError(f"No se encontró archivo de eventos en {log_dir}")

#     ea = event_accumulator.EventAccumulator(event_file)
#     ea.Reload()

#     tag = f"{phase}/{metric}"
#     if tag not in ea.Tags().get("scalars", []):
#         raise ValueError(f"La métrica '{tag}' no existe en este run.")

#     events = ea.Scalars(tag)
#     steps = [e.step for e in events]
#     values = [e.value for e in events]

#     # Graficar
#     plt.figure(figsize=(7, 4))
#     plt.plot(steps, values, label=f"{phase} {metric}")
#     plt.xlabel("Epoch")
#     plt.ylabel(metric)
#     plt.title(f"{metric} - {phase} ({run_name})")
#     # if metric.lower() == "accuracy":
#     plt.ylim(0, 1)  # 🔒 Limita el eje y para accuracy
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
