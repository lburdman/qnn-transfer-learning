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

    # Allow injection for mocking via globals
    if 'BaseHybridHead' in globals():
        BaseHybridHead = globals()['BaseHybridHead']
    else:
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

    if 'BaseHybridHead' in globals():
        BaseHybridHead = globals()['BaseHybridHead']
        if isinstance(qlayer, BaseHybridHead):
            return getattr(qlayer, "pre_net", None)
    else:
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

def infer_quantum_metadata(model: nn.Module) -> dict[str, int]:
    """
    Infer n_qubits and q_depth from the model's quantum layer weights.
    
    Args:
        model (nn.Module): The full hybrid model.
        
    Returns:
        dict: A dictionary containing 'n_qubits' and 'q_depth'.
        
    Raises:
        ValueError: If the metadata cannot be inferred.
    """
    summary = summarize_quantum_weights(model)
    
    # Check for BaseHybridHead's q_params
    if "q_params" in summary:
        shape = summary["q_params"]["shape"]
        if len(shape) == 1:
            # For BaseHybridHead: shape is (max_layers * n_qubits,)
            qlayer = find_quantum_layer(model)
            if hasattr(qlayer, "n_qubits") and hasattr(qlayer, "max_layers"):
                return {"n_qubits": qlayer.n_qubits, "q_depth": qlayer.q_depth, "max_layers": qlayer.max_layers}
            
    # Check for TorchLayer's weights
    for name, meta in summary.items():
        if name.endswith("weights") or name == "weights":
            shape = meta["shape"]
            if len(shape) == 2:
                # Common PennyLane shape: (q_depth, n_qubits)
                return {"n_qubits": shape[1], "q_depth": shape[0]}
                
    raise ValueError("Could not infer n_qubits and q_depth from the quantum weights.")

def get_default_quantum_dummy_input(n_qubits: int, default_val: float = 0.5) -> torch.Tensor:
    """
    Create a constant dummy input tensor for quantum circuit visualization.
    
    Args:
        n_qubits (int): Number of qubits (length of the input).
        default_val (float): The constant value to fill the tensor with.
        
    Returns:
        torch.Tensor: A 1D tensor of shape (n_qubits,).
    """
    return torch.full((n_qubits,), default_val, dtype=torch.float32)

def draw_quantum_circuit_from_model(model: nn.Module, dummy_input: torch.Tensor | None = None, style: str = "pennylane") -> None:
    """
    Reconstruct and draw the quantum circuit corresponding to the model's quantum layer.
    
    Args:
        model (nn.Module): The full hybrid model.
        dummy_input (torch.Tensor | None): Optional input tensor. If None, a default is generated.
        style (str): The drawing style (e.g., "pennylane", "black_white").
        
    Raises:
        ValueError: If PennyLane is not installed or the quantum layer cannot be found/inferred.
    """
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
        import matplotlib.pyplot as plt
    except ImportError:
        raise ValueError("PennyLane and Matplotlib are required to draw the quantum circuit.")
        
    qlayer = find_quantum_layer(model)
    if qlayer is None:
        raise ValueError("No quantum layer found in the model.")
        
    metadata = infer_quantum_metadata(model)
    n_qubits = metadata["n_qubits"]
    
    if dummy_input is None:
        dummy_input = get_default_quantum_dummy_input(n_qubits)
        
    # Attempt to extract weights
    weights_dict = extract_quantum_weights(model)
    if "q_params" in weights_dict:
        weights = weights_dict["q_params"].flatten().cpu().numpy()
        max_layers = metadata.get("max_layers", metadata["q_depth"])
        q_depth = metadata["q_depth"]
        
        # Use the BaseHybridHead builder
        try:
            from src.quantum_circuit import build_qnode, build_qnode2, DressedQuantumCircuit
            
            dev = qml.device("default.qubit", wires=n_qubits)
            
            # Check if it's the enhanced circuit
            if isinstance(qlayer, DressedQuantumCircuit):
                qnode = build_qnode2(n_qubits, q_depth, max_layers, dev)
            else:
                qnode = build_qnode(n_qubits, q_depth, max_layers, dev)
                
            print("ASCII circuit diagram (templates):")
            try:
                print(qml.draw(qnode, expansion_strategy="device")(dummy_input, weights))
            except Exception:
                try:
                    print(qml.draw(qnode, level="device")(dummy_input, weights))
                except Exception:
                    print(qml.draw(qnode)(dummy_input, weights))
            
            try:
                drawer = qml.draw_mpl(qnode, expansion_strategy="device", style=style)
                fig, ax = drawer(dummy_input, weights)
                plt.show()
            except Exception:
                try:
                    drawer = qml.draw_mpl(qnode, level="device", style=style)
                    fig, ax = drawer(dummy_input, weights)
                    plt.show()
                except Exception as e:
                    print(f"Matplotlib draw failed: {e}")
                
            return
            
        except ImportError:
            pass

    # Fallback for standard TorchLayer
    weights_tensor = None
    for name, w in weights_dict.items():
        if name.endswith("weights") or name == "weights":
            weights_tensor = w.detach().cpu().numpy()
            break
            
    if weights_tensor is not None:
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev)
        def qnode(inputs, w):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(w, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        print("ASCII circuit diagram (TorchLayer):")
        try:
            print(qml.draw(qnode, expansion_strategy="device")(dummy_input, weights_tensor))
        except Exception:
            try:
                print(qml.draw(qnode, level="device")(dummy_input, weights_tensor))
            except Exception:
                print(qml.draw(qnode)(dummy_input, weights_tensor))
        
        try:
            drawer = qml.draw_mpl(qnode, expansion_strategy="device", style=style)
            fig, ax = drawer(dummy_input, weights_tensor)
            plt.show()
        except Exception:
            try:
                drawer = qml.draw_mpl(qnode, level="device", style=style)
                fig, ax = drawer(dummy_input, weights_tensor)
                plt.show()
            except Exception as e:
                print(f"Matplotlib draw failed: {e}")
            
        return
        
    raise ValueError("Could not extract weights to draw the circuit.")
