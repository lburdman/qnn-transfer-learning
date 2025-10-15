"""
📊 Plotting Functions for Training Metrics

This module provides functions to visualize training metrics (loss and accuracy) 
from saved model runs. Designed for publication-ready plots.

Features:
- Plot single model training curves
- Compare multiple models in the same figure
- Support for Loss, Accuracy, or both metrics
- Overlapped plots with dual y-axes
- Clean, professional styling
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import glob


def load_training_data(json_path):
    """
    Carga los datos de entrenamiento desde un archivo JSON.
    
    Args:
        json_path (str): Ruta al archivo JSON con métricas de entrenamiento
    
    Returns:
        dict: Diccionario con las métricas de entrenamiento
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encontró el archivo: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data


def get_model_name_from_path(json_path):
    """
    Extrae el nombre del modelo desde la ruta del archivo.
    
    Args:
        json_path (str): Ruta al archivo JSON
    
    Returns:
        str: Nombre del modelo
    """
    filename = Path(json_path).stem
    # Remover timestamp si existe
    if '_' in filename:
        parts = filename.split('_')
        # Buscar el último segmento que sea un timestamp (formato YYYYMMDD_HHMMSS o DDMM_HHMM)
        for i in range(len(parts)-1, -1, -1):
            if len(parts[i]) == 4 and parts[i].isdigit():  # HHMM
                if i > 0 and len(parts[i-1]) == 4 and parts[i-1].isdigit():  # DDMM
                    return '_'.join(parts[:i-1])
            elif len(parts[i]) == 6 and parts[i].isdigit():  # HHMMSS
                if i > 0 and len(parts[i-1]) == 8 and parts[i-1].isdigit():  # YYYYMMDD
                    return '_'.join(parts[:i-1])
    return filename


def plot_training_metrics(json_path, metrics='both', figsize=(12, 5), save_path=None):
    """
    Plotea las métricas de entrenamiento de un modelo individual.
    
    Args:
        json_path (str): Ruta al archivo JSON con métricas
        metrics (str): 'loss', 'accuracy', o 'both'
        figsize (tuple): Tamaño de la figura
        save_path (str, optional): Ruta para guardar la figura
    """
    # Cargar datos
    data = load_training_data(json_path)
    model_name = get_model_name_from_path(json_path)
    
    # Preparar datos
    epochs = range(1, len(data['train_losses']) + 1)
    
    # Determinar número de subplots
    if metrics == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        axes = [ax]
    
    # Plot Loss
    if metrics in ['loss', 'both']:
        ax = axes[0] if metrics == 'both' else axes[0]
        ax.plot(epochs, data['train_losses'], 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, data['val_losses'], 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{model_name} - Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot Accuracy
    if metrics in ['accuracy', 'both']:
        ax = axes[1] if metrics == 'both' else axes[0]
        ax.plot(epochs, data['train_accs'], 'b-', label='Training Accuracy', linewidth=2)
        ax.plot(epochs, data['val_accs'], 'r-', label='Validation Accuracy', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model_name} - Training & Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Guardar si se especifica
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()
    
    # Mostrar estadísticas
    print(f"\nEstadisticas del modelo {model_name}:")
    print(f"  • Mejor Loss (val): {min(data['val_losses']):.4f}")
    print(f"  • Mejor Accuracy (val): {max(data['val_accs']):.4f}")
    print(f"  • Épocas entrenadas: {len(epochs)}")


def plot_multiple_models(json_paths, metrics='both', figsize=(12, 5), save_path=None):
    """
    Plotea las métricas de entrenamiento de múltiples modelos para comparación.
    
    Args:
        json_paths (list): Lista de rutas a archivos JSON
        metrics (str): 'loss', 'accuracy', o 'both'
        figsize (tuple): Tamaño de la figura
        save_path (str, optional): Ruta para guardar la figura
    """
    if not json_paths:
        raise ValueError("Debe proporcionar al menos una ruta de archivo")
    
    # Cargar datos de todos los modelos
    models_data = []
    model_names = []
    
    for json_path in json_paths:
        data = load_training_data(json_path)
        models_data.append(data)
        model_names.append(get_model_name_from_path(json_path))
    
    # Usar el número de épocas del primer modelo como referencia
    epochs = range(1, len(models_data[0]['train_losses']) + 1)
    
    # Determinar número de subplots
    if metrics == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        axes = [ax]
    
    # Colores para diferentes modelos
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))
    
    # Plot Loss
    if metrics in ['loss', 'both']:
        ax = axes[0] if metrics == 'both' else axes[0]
        
        for i, (data, name, color) in enumerate(zip(models_data, model_names, colors)):
            ax.plot(epochs, data['train_losses'], '--', color=color, 
                   label=f'{name} (Train)', linewidth=2, alpha=0.7)
            ax.plot(epochs, data['val_losses'], '-', color=color, 
                   label=f'{name} (Val)', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot Accuracy
    if metrics in ['accuracy', 'both']:
        ax = axes[1] if metrics == 'both' else axes[0]
        
        for i, (data, name, color) in enumerate(zip(models_data, model_names, colors)):
            ax.plot(epochs, data['train_accs'], '--', color=color, 
                   label=f'{name} (Train)', linewidth=2, alpha=0.7)
            ax.plot(epochs, data['val_accs'], '-', color=color, 
                   label=f'{name} (Val)', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training & Validation Accuracy Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Guardar si se especifica
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()
    
    # Mostrar estadísticas comparativas
    print(f"\nEstadisticas comparativas:")
    print("=" * 60)
    for data, name in zip(models_data, model_names):
        print(f"\n{name}:")
        print(f"  • Mejor Loss (val): {min(data['val_losses']):.4f}")
        print(f"  • Mejor Accuracy (val): {max(data['val_accs']):.4f}")
        print(f"  • Épocas: {len(epochs)}")


def plot_overlapped_metrics(json_path, figsize=(10, 6), save_path=None):
    """
    Plotea Loss y Accuracy superpuestos en el mismo gráfico.
    
    Args:
        json_path (str): Ruta al archivo JSON con métricas
        figsize (tuple): Tamaño de la figura
        save_path (str, optional): Ruta para guardar la figura
    """
    # Cargar datos
    data = load_training_data(json_path)
    model_name = get_model_name_from_path(json_path)
    
    # Preparar datos
    epochs = range(1, len(data['train_losses']) + 1)
    
    # Crear figura con dos ejes Y
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot Loss (eje izquierdo)
    color1 = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color1)
    line1 = ax1.plot(epochs, data['train_losses'], '--', color=color1, 
                     label='Training Loss', linewidth=2, alpha=0.7)
    line2 = ax1.plot(epochs, data['val_losses'], '-', color=color1, 
                     label='Validation Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Crear segundo eje Y para Accuracy
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color2)
    line3 = ax2.plot(epochs, data['train_accs'], '--', color=color2, 
                     label='Training Accuracy', linewidth=2, alpha=0.7)
    line4 = ax2.plot(epochs, data['val_accs'], '-', color=color2, 
                     label='Validation Accuracy', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1)
    
    # Combinar leyendas
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title(f'{model_name} - Loss & Accuracy Overlapped')
    plt.tight_layout()
    
    # Guardar si se especifica
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()
    
    # Mostrar estadísticas
    print(f"\nEstadisticas del modelo {model_name}:")
    print(f"  • Mejor Loss (val): {min(data['val_losses']):.4f}")
    print(f"  • Mejor Accuracy (val): {max(data['val_accs']):.4f}")
    print(f"  • Épocas entrenadas: {len(epochs)}")


def plot_models_from_directory(directory='runs_updated', pattern='*.json', 
                              metrics='both', figsize=(12, 5), save_path=None):
    """
    Plotea automáticamente todos los modelos encontrados en un directorio.
    
    Args:
        directory (str): Directorio donde buscar archivos JSON
        pattern (str): Patrón de búsqueda para archivos
        metrics (str): 'loss', 'accuracy', o 'both'
        figsize (tuple): Tamaño de la figura
        save_path (str, optional): Ruta para guardar la figura
    """
    # Buscar archivos JSON
    json_files = list(Path(directory).glob(pattern))
    
    if not json_files:
        print(f"No se encontraron archivos {pattern} en {directory}")
        return
    
    # Convertir a strings
    json_paths = [str(f) for f in json_files]
    
    print(f"Encontrados {len(json_paths)} modelos en {directory}:")
    for path in json_paths:
        print(f"  • {get_model_name_from_path(path)}")
    
    # Plotear todos los modelos
    if len(json_paths) == 1:
        plot_training_metrics(json_paths[0], metrics=metrics, figsize=figsize, save_path=save_path)
    else:
        plot_multiple_models(json_paths, metrics=metrics, figsize=figsize, save_path=save_path)


def find_latest_metrics_file(directory='runs_updated'):
    """
    Encuentra el archivo JSON de métricas más reciente en un directorio.
    
    Args:
        directory (str): Directorio donde buscar archivos JSON
    
    Returns:
        str or None: Ruta al archivo más reciente, o None si no se encuentra
    """
    json_files = glob.glob(os.path.join(directory, '*.json'))
    if json_files:
        # Ordenar por fecha de modificación (más reciente primero)
        json_files.sort(key=os.path.getmtime, reverse=True)
        return json_files[0]
    return None


def plot_current_experiment(directory='runs_updated', figsize=(12, 6), save_path=None):
    """
    Plotea las métricas del experimento más reciente.
    
    Args:
        directory (str): Directorio donde buscar archivos JSON
        figsize (tuple): Tamaño de la figura
        save_path (str, optional): Ruta para guardar la figura
    """
    latest_file = find_latest_metrics_file(directory)
    
    if latest_file:
        print(f"Archivo de metricas encontrado: {latest_file}")
        print("\nVisualizando metricas del experimento actual:")
        plot_overlapped_metrics(latest_file, figsize=figsize, save_path=save_path)
    else:
        print(f"No se encontraron archivos JSON en {directory}/")
        print("Asegurate de que el entrenamiento haya guardado las metricas correctamente.")


# Configurar estilo por defecto (sin fondo gris)
plt.style.use('default')
