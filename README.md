# QNN Transfer Learning for CREMA-D

Speech emotion recognition on the CREMA-D dataset using classical CNNs, precomputed embeddings, and hybrid quantum/classical heads built with PennyLane + PyTorch. The project is optimized for Google Colab with data stored in Drive under `/content/drive/MyDrive/CREMAD`.

## Notebooks
- `qnn_speech_recognition.ipynb` **Main** training & evaluation notebook with two-stage fine-tuning (embedding/backbone pretraining + head-only finetuning), supporting classical and quantum heads.
- `audio_preprocessing.ipynb` Generates splits, spectrograms, MFCCs, and embeddings; defines the expected folder structure under `CREMAD`.
- `data_analysis.ipynb` Exploratory data analysis on CREMA-D splits and features.
- `models_visualization.ipynb` Visualizes learned representations and model behaviors.
- `quantum_analysis.ipynb` Quantum circuit inspection (density matrices, purity, etc.).
- `ants_bees.ipynb` Legacy example from the early pipeline (kept for reference).
- `crema_d_hybrid_qnn.ipynb` Deprecated; replaced by `qnn_speech_recognition.ipynb`.

## Directory Structure (under `/content/drive/MyDrive/CREMAD`)
- `AudioWAV/` Raw audio files.
- `splits/` CSV split files (train/val/test) referencing WAVs.
- `Spectrograms/` Mel-spectrogram PNGs organized by split/label.
- `Embeddings/` Precomputed embeddings (e.g., `ResNet18/`, `VGG16/`, `PANNs_Cnn14/`).
- `MFCCs/` MFCC feature `.npz` files organized by split/label.
- `Models/<base_model>/<base_model>_<classic|quantum>/run_<timestamp>/` Saved models, hyperparams, and metrics.

## Running on Colab
1. Open `qnn_speech_recognition.ipynb` in Colab.
2. Run the setup cell to mount Drive, clone the repo, and checkout the `fine-tunning` branch.
3. Ensure your data lives at `/content/drive/MyDrive/CREMAD` following the structure above (generated via `audio_preprocessing.ipynb`).
4. Configure the experiment (base model, classes, quantum vs. classical, hyperparameters) and run the two-stage pipeline.
5. Metrics and checkpoints are saved under `CREMAD/Models/...` as described above.

## Source Layout
- `src/dataset.py` � Dataloaders for spectrograms, embeddings, and MFCCs.
- `src/audio_processing.py` � In-memory audio feature computation (MEL/MFCC).
- `src/model_builder.py` � Backbone/embedding constructors, classical/quantum heads.
- `src/training.py` � Training loops, fine-tuning helpers, evaluation utilities.
- `src/quantum_circuit.py` � Quantum circuit builders and analysis helpers.
- `src/plot_functions.py` � Metric plotting utilities.
- `tests/` � Basic shape/loader/model sanity tests.

## Environment
- Python 3.x, PyTorch, TorchVision.
- PennyLane for quantum layers.
- torchaudio, librosa for audio features.

## License
See `LICENSE` for licensing details.
