import os
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from src.dataset import create_dataloaders_all
from src.model_builder import build_model


def _write_npz(path: Path, key: str, arr: np.ndarray, label: str, file_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: arr}, label=label, file_name=file_name)


def _minimal_config(base_model: str, root: Path) -> dict:
    return {
        "base_model": base_model,
        "selected_classes": None,
        "batch_size": 2,
        "grayscale": False,
        "use_pretrained": False,
        "use_generic_weights": False,
        "specs_dir": str(root / "Spectrograms"),
        "embedding_dir": str(root / "Embeddings"),
        "mfcc_dir": str(root / "MFCCs"),
    }


def test_embedding_dataloader_and_model_build():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        emb_root = root / "Embeddings" / "ResNet18" / "train" / "ANG"
        _write_npz(emb_root / "sample1.npz", "embedding", np.random.rand(512).astype(np.float32), "ANG", "sample1")
        config = _minimal_config("emb_resnet18", root)
        dls, sizes, class_names, _ = create_dataloaders_all(config, shuffle=False, num_workers=0)
        xb, yb = next(iter(dls["train"]))
        assert xb.shape[1] == 512
        assert yb.shape[0] == xb.shape[0]
        model = build_model({**config, "quantum": False, "classical_model": "512_2", "n_qubits": 4, "q_depth": 2}, class_names, dls, device=torch.device("cpu"))
        out = model(xb)
        assert out.shape[0] == xb.shape[0]
        assert out.shape[1] == len(class_names)


def test_mfcc_dataloader_alias():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        mfcc_root = root / "MFCCs" / "train" / "ANG"
        _write_npz(mfcc_root / "sample1.npz", "mfcc", np.random.rand(40, 10).astype(np.float32), "ANG", "sample1")
        config = _minimal_config("cnn_mfcc", root)
        dls, sizes, class_names, _ = create_dataloaders_all(config, shuffle=False, num_workers=0)
        xb, yb = next(iter(dls["train"]))
        assert xb.ndim == 2
        assert xb.shape[0] == yb.shape[0]


try:
    import pennylane as qml  # noqa: F401
    HAS_QML = True
except Exception:  # pragma: no cover
    HAS_QML = False


@pytest.mark.skipif(not HAS_QML, reason="PennyLane not installed for quantum head test")
def test_model_build_quantum_head_cpu():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        emb_root = root / "Embeddings" / "ResNet18" / "train" / "ANG"
        _write_npz(emb_root / "sample1.npz", "embedding", np.random.rand(8).astype(np.float32), "ANG", "sample1")
        config = _minimal_config("emb_resnet18", root)
        dls, sizes, class_names, _ = create_dataloaders_all(config, shuffle=False, num_workers=0)
        xb, _ = next(iter(dls["train"]))
        model = build_model({**config, "quantum": True, "classical_model": "512_nq_2", "n_qubits": xb.shape[1], "q_depth": 1}, class_names, dls, device=torch.device("cpu"))
        out = model(xb)
        assert out.shape[0] == xb.shape[0]
        assert out.shape[1] == len(class_names)
