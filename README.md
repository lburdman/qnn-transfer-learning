# Quantum Transfer Learning for Audio Emotion Recognition

Speech emotion classification on [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) using classical CNNs and **hybrid quantum–classical neural networks** built with [PennyLane](https://pennylane.ai/) and PyTorch. The project investigates whether a variational quantum circuit head—trained on top of a frozen classical backbone—can match or exceed a purely classical counterpart on a multi-class emotion recognition task.

---

## Motivation

Quantum machine learning (QML) proposes that variational quantum circuits (VQCs) may offer expressivity advantages over classical counterparts when operating in high-dimensional feature spaces. This project evaluates the practical viability of **quantum transfer learning** for real-world audio classification:

1. Pretrain a classical backbone (ResNet18, VGG16, or PANNs CNN14) to extract audio representations from mel-spectrograms or embeddings.
2. Freeze the backbone and fine-tune either a **classical head** or a **quantum head** (PennyLane `TorchLayer`) built from `AngleEmbedding` + `BasicEntanglerLayers`.
3. Compare accuracy, F1, and training dynamics between the two head types under identical conditions.

---

## Repository Structure

```
qnn-transfer-learning/
│
├── qnn_speech_recognition.ipynb     ← Main training & evaluation notebook (START HERE)
├── audio_preprocessing.ipynb         Generates splits, spectrograms, MFCCs, and embeddings
├── data_analysis.ipynb               EDA on CREMA-D splits, embeddings, and features
├── models_visualization.ipynb        Cross-run metrics comparison and training curves
├── quantum_analysis.ipynb            Standalone quantum circuit analysis (purity, entanglement)
├── hardware_verification.ipynb       Hybrid inference verification on IBM Quantum hardware
│
├── crema_d_hybrid_qnn.ipynb         Legacy hybrid training notebook (kept for reference)
├── ants_bees.ipynb                   Early transfer learning proof-of-concept (kept for reference)
│
└── src/
    ├── dataset.py          Dataloaders: spectrograms, precomputed embeddings, MFCCs
    ├── audio_processing.py In-memory audio feature computation (MEL / MFCC)
    ├── model_builder.py    Backbone constructors; classical and quantum head builders
    ├── training.py         Training loops, fine-tuning stages, evaluation utilities
    ├── quantum_circuit.py  Quantum circuit builders (QNode, Quantumnet, DressedQuantumCircuit)
    ├── quantum_weights.py  Weight extraction, artifact export, and circuit reconstruction
    ├── plot_functions.py   Metric plotting, run discovery, and comparison utilities
    ├── analysis_utils.py   Dimensionality reduction (PCA/t-SNE/MDS) and baseline classifiers
    ├── visualize_melspec.py Waveform and mel-spectrogram visualization helpers
    ├── wav_to_spec.py      Offline spectrogram, MFCC, and embedding generation utilities
    └── utils.py            Config builder, directory helpers, and model parameter counters
```

---

## Data Layout (Google Drive)

The pipeline expects data under `/content/drive/MyDrive/CREMAD/`:

```
CREMAD/
├── AudioWAV/          Raw WAV files (from the CREMA-D dataset)
├── splits/            train.csv / val.csv / test.csv with columns: file_name, label
├── Spectrograms/      Mel-spectrogram PNGs organized by split/label/
├── Embeddings/        Precomputed embeddings: ResNet18/, VGG16/, PANNs_Cnn14/
├── MFCCs/             MFCC .npz files organized by split/label/
└── Models/            Saved runs: <backbone>/<backbone>_<classic|quantum>/run_<timestamp>/
```

Generate spectrograms, MFCCs, and embeddings by running `audio_preprocessing.ipynb` once.

---

## How to Run

### Main workflow (Colab)

1. Open `qnn_speech_recognition.ipynb` in Google Colab.
2. Run the first cell to mount Drive and clone this repository.
3. Configure your experiment in the config cell:
   - `base_model`: e.g. `"emb_resnet18"`, `"emb_panns_cnn14"`, `"cnn_specs"`
   - `USE_QUANTUM`: `True` for hybrid head, `False` for classical head
   - `selected_classes`, `n_qubits`, `q_depth`, and hyperparameters
4. Run all cells. The two-stage pipeline will:
   - **Stage 1** – pretrain the embedding projector on the chosen representation.
   - **Stage 2** – fine-tune only the selected head (classical or quantum) on the frozen backbone.
5. Metrics, checkpoints, config, and quantum artifacts are saved under `CREMAD/Models/`.

### Hardware verification (IBM Quantum)

After training a quantum model, open `hardware_verification.ipynb`. It loads the exported quantum artifacts (weights, inputs, circuit metadata) and runs inference on a real IBM QPU using Qiskit Runtime Sampler. Requires an IBM Quantum account and credentials via Colab Secrets.

---

## Classical vs. Quantum Comparison

| Component | Classical head | Quantum head |
|---|---|---|
| Projector | `nn.Linear(in → n_qubits)` | `nn.Linear(in → n_qubits)` |
| Core | MLP stack (depth ≈ q_depth) | PennyLane `TorchLayer` (AngleEmbedding + BasicEntanglerLayers) |
| Output | `nn.Linear(n_qubits → n_classes)` | `nn.Linear(n_qubits → n_classes)` |
| Trainable params | O(n_qubits² × depth) | O(n_qubits × q_depth) |

Both heads receive the same frozen backbone embeddings, enabling a direct apples-to-apples comparison.

---

## Environment

This project is designed to run on **Google Colab** with the standard Colab GPU runtime. Key dependencies:

- Python ≥ 3.10
- PyTorch ≥ 2.0, TorchVision, TorchAudio
- PennyLane ≥ 0.38
- scikit-learn, librosa, pandas, matplotlib
- `panns_inference` (for PANNs CNN14 embeddings)
- `qiskit`, `qiskit-ibm-runtime` (optional, for hardware verification)

Install all dependencies locally using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

> **Note:** `numpy<2` is pinned because the current PyTorch binary wheels require the NumPy 1.x ABI. Upgrading NumPy to 2.x will cause a runtime incompatibility with torch until a matching torch wheel is built. Colab notebooks automatically use compatible versions.

---

## Tests

```bash
pytest tests/ -v
```

`tests/test_quantum_weights.py` — unit tests for quantum weight extraction, artifact export, and circuit reconstruction utilities.  
`tests/test_dataloaders.py` — sanity checks for dataloader construction and model forward passes.

---

## License

This project is licensed under the terms in [LICENSE](LICENSE).
