import os
from torchvision import datasets, transforms
import torch
import random

# SpecAugment
class SpecAugment:
    def __init__(self, time_mask_param=30, freq_mask_param=13, num_masks=1, fill_value=None):
        """
        time_mask_param: ancho máximo del masking en tiempo
        freq_mask_param: ancho máximo del masking en frecuencia
        num_masks: cantidad de máscaras por espectrograma
        fill_value: valor con el que se rellenan las zonas (por defecto negro post-normalización)
        """
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks

        # Valor negro por defecto tras normalización ImageNet para RGB
        if fill_value is None:
            self.fill_value = torch.tensor([-2.118, -2.036, -1.804])  # R, G, B como vector 1D
        else:
            self.fill_value = fill_value  # tensor o escalar; se adapta en __call__

    def __call__(self, spec):
        """
        Aplica masking a un tensor [C, H, W] (como imagen RGB o espectrograma 1 canal).
        """
        c, h, w = spec.shape

        # Preparar tensor de relleno con broadcasting seguro [C, 1, 1]
        fill_value = self.fill_value
        if not isinstance(fill_value, torch.Tensor):
            fill_value = torch.as_tensor(fill_value)
        fill_value = fill_value.to(device=spec.device, dtype=spec.dtype)
        if fill_value.dim() == 0 or fill_value.numel() == 1:
            fill_tensor = fill_value.view(1, 1, 1).expand(c, 1, 1)
        elif fill_value.numel() == c:
            fill_tensor = fill_value.view(c, 1, 1)
        else:
            # Si el tamaño no coincide, usar media como fallback seguro
            fill_tensor = fill_value.mean().view(1, 1, 1).expand(c, 1, 1)

        for _ in range(self.num_masks):
            # 🎯 Masking en tiempo (columnas negras)
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(1, w - t))
            if t > 0:
                spec[:, :, t0:t0 + t] = fill_tensor

            # 🎯 Masking en frecuencia (filas negras)
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(1, h - f))
            if f > 0:
                spec[:, f0:f0 + f, :] = fill_tensor

        return spec



# ---------------------------------------------------
# 🔧 Función que define las transformaciones de imagen
# ---------------------------------------------------
def get_data_transforms(spec_augment: bool = False):
    """
    Devuelve transformaciones para los conjuntos de entrenamiento y validación.
    Si spec_augment=True, aplica SpecAugment (en tiempo y frecuencia) a los tensores.
    """
    train_transform = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
    ]

    if spec_augment:
        train_transform.append(SpecAugment(time_mask_param=30, freq_mask_param=15))

    return {
        'train': transforms.Compose(train_transform),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }


# ---------------------------------------------------
# 📦 Función que prepara DataLoaders para entrenamiento
# ---------------------------------------------------
def get_dataloaders(data_dir, batch_size=4, shuffle=True, num_workers=0, spec_augment=False):
    """
    Crea los DataLoaders desde un directorio con estructura tipo ImageFolder.

    Agrega SpecAugment visual si spec_augment=True.
    """
    data_transforms = get_data_transforms(spec_augment=spec_augment)
    
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    # TODO: Exponer 'prefetch_factor' y 'persistent_workers' como parámetros cuando num_workers > 0
    #       para tunear I/O según hardware y tamaño de batch.
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0)
        )
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names
