"""
Reusable utilities for inspecting trained hybrid models and extracting 
quantum-related trainable parameters and metadata.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def find_quantum_layer(model: nn.Module) -> nn.Module | None:
    """
    Find and return the quantum layer/block within a PyTorch model.
    Checks for custom instances (e.g. `BaseHybridHead`) or PennyLane's `TorchLayer`.

    Args:
        model (nn.Module): The full hybrid model.

    Returns:
        nn.Module | None: The quantum layer module, or None if not found.
    """
    try:
        from pennylane.qnn import TorchLayer
    except ImportError:
        TorchLayer = type("PlaceholderTorchLayer", (), {})

    try:
        from src.quantum_circuit import BaseHybridHead
    except ImportError:
        BaseHybridHead = type("PlaceholderBaseHybridHead", (), {})

    # Direct match
    if isinstance(model, (BaseHybridHead, TorchLayer)):
        return model

    # Search through children
    for module in model.modules():
        if isinstance(module, (BaseHybridHead, TorchLayer)):
            return module

    return None


def extract_quantum_weights(model_or_state_dict: nn.Module | dict) -> dict[str, torch.Tensor]:
    """
    Extract only the quantum trainable parameters from a model or state dictionary.

    Args:
        model_or_state_dict: An nn.Module or its state_dict (dict) containing parameters.

    Returns:
        dict[str, torch.Tensor]: Dictionary mapping parameter names to their 
                                detached quantum weight tensors.

    Raises:
        ValueError: If no quantum layer or parameters can be identified.
    """
    if isinstance(model_or_state_dict, nn.Module):
        qlayer = find_quantum_layer(model_or_state_dict)
        if qlayer is None:
            raise ValueError("Quantum layer not found in the provided model.")

        try:
            from src.quantum_circuit import BaseHybridHead
            is_base_hybrid = isinstance(qlayer, BaseHybridHead)
        except ImportError:
            is_base_hybrid = False

        if is_base_hybrid:
            return {"q_params": qlayer.q_params.detach()}

        weights = {}
        for name, param in qlayer.named_parameters():
            if param.requires_grad:
                weights[name] = param.detach()
        return weights

    elif isinstance(model_or_state_dict, dict):
        q_weights = {}
        for k, v in model_or_state_dict.items():
            # In our repository, 'q_params' corresponds to Custom classical-quantum heads
            if "q_params" in k:
                q_weights[k] = v
            # Common pattern for PennyLane TorchLayer: parameters are often named 'weights'
            elif k.endswith(".weights") or k == "weights":
                # Ensure it's not a generic nn.Linear 'weight' parameter by accident
                q_weights[k] = v

        if not q_weights:
            raise ValueError("No quantum parameters found in the provided state_dict.")
        return q_weights

    else:
        raise TypeError("Input must be an nn.Module or a state_dict (dict).")


def summarize_quantum_weights(model_or_state_dict: nn.Module | dict) -> dict[str, dict]:
    """
    Generate a summary of metadata for the quantum weights.

    Args:
        model_or_state_dict: An nn.Module or its state_dict (dict).

    Returns:
        dict[str, dict]: A formatted dictionary of parameter metadata.
    """
    weights = extract_quantum_weights(model_or_state_dict)
    summary = {}

    for name, tensor in weights.items():
        summary[name] = {
            "name": name,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std().item()),
            "first_few_values": tensor.flatten()[:5].tolist()
        }
    return summary


def find_classical_to_quantum_mapper(model: nn.Module) -> nn.Module | None:
    """
    Identify the classical layer immediately preceding the quantum layer.
    This is usually the projection mapping classical features into N-qubits space.

    Args:
        model (nn.Module): The full hybrid model.

    Returns:
        nn.Module | None: The preceding classical nn.Linear layer, or None.
    """
    qlayer = find_quantum_layer(model)
    if qlayer is None:
        raise ValueError("Quantum layer not found in the model.")

    try:
        from src.quantum_circuit import BaseHybridHead
        if isinstance(qlayer, BaseHybridHead):
            return getattr(qlayer, "pre_net", None)
    except ImportError:
        pass

    try:
        from pennylane.qnn import TorchLayer
    except ImportError:
        TorchLayer = type("PlaceholderTorchLayer", (), {})

    # Traverse backwards in Sequential blocks that contain TorchLayer
    for module in model.modules():
        if isinstance(module, nn.Sequential):
            children = list(module.children())
            for i, child in enumerate(children):
                if isinstance(child, TorchLayer):
                    # Step backwards to find the nearest nn.Linear mapper
                    for j in range(i - 1, -1, -1):
                        if isinstance(children[j], nn.Linear):
                            return children[j]

    return None
