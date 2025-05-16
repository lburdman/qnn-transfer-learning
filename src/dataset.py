import os
from torchvision import datasets, transforms
import torch

# ---------------------------------------------------
# 🔧 Función que define las transformaciones de imagen
# ---------------------------------------------------
def get_data_transforms():
    """
    Devuelve un diccionario con las transformaciones a aplicar a las imágenes
    en los conjuntos de entrenamiento y validación.

    Estas transformaciones:
    - Redimensionan la imagen a 256 px en su lado menor
    - Recortan el centro a 224x224 px (lo que espera ResNet)
    - La convierten a tensor
    - La normalizan con la media y desvío estándar de ImageNet
    """
    return {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

# ---------------------------------------------------
# 📦 Función que prepara DataLoaders para entrenamiento
# ---------------------------------------------------
def get_dataloaders(data_dir, batch_size=4, shuffle=True, num_workers=0):
    """
    Crea los DataLoaders a partir de un directorio con estructura estilo ImageFolder.
    
    Parámetros:
    - data_dir (str): carpeta raíz que contiene 'train/' y 'val/' con subcarpetas por clase
    - batch_size (int): tamaño del batch a usar
    - shuffle (bool): si se mezclan los datos por época
    - num_workers (int): procesos paralelos para cargar datos (0 en Windows)

    Retorna:
    - dataloaders: diccionario con 'train' y 'val'
    - dataset_sizes: cantidad de imágenes por conjunto
    - class_names: lista de nombres de clases detectadas en 'train'
    """
    data_transforms = get_data_transforms()
    
    # Cargar los datasets desde carpetas organizadas por clase
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    # Crear DataLoaders para cada conjunto
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        for x in ['train', 'val']
    }

    # Obtener tamaños de datasets y nombres de clases
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names
