import numpy as np
import matplotlib.pyplot as plt

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
