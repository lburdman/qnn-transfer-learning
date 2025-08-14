#!/usr/bin/env python3
import os
import sys
import json
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

try:
	from PIL import Image
	exists_pil = True
except Exception:
	exists_pil = False
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
	("ants_bees.ipynb", {
		"NB_DATA_DIR": str(WORKSPACE / "data" / "hymenoptera_data"),
		"NB_OUTPUTS_DIR": str(WORKSPACE / "outputs" / "notebooks" / "ants_bees"),
	}),
	("crema-d.ipynb", {
		"NB_DATA_DIR": str(WORKSPACE / "data" / "CremaD" / "mel_spec_reduced"),
		"NB_OUTPUTS_DIR": str(WORKSPACE / "outputs" / "notebooks" / "crema-d"),
		"NB_SPEC_AUGMENT": "False",
	}),
]

PARAM_CELL_CODE = """
# NB_PARAMETERS: injected for reproducible execution
import os, random, numpy as np, torch

# Defaults (overridable via env)
n_qubits = int(os.getenv('NB_N_QUBITS', str(globals().get('n_qubits', 4))))
quantum = os.getenv('NB_QUANTUM', str(globals().get('quantum', True))).lower() in ('1','true','yes')
classical_model = os.getenv('NB_CLASSICAL_MODEL', str(globals().get('classical_model', '512_nq_2')))
step = float(os.getenv('NB_LR', str(globals().get('step', 4e-4))))
batch_size = int(os.getenv('NB_BATCH_SIZE', str(globals().get('batch_size', 4))))
num_epochs = int(os.getenv('NB_NUM_EPOCHS', str(globals().get('num_epochs', 1))))
q_depth = int(os.getenv('NB_Q_DEPTH', str(globals().get('q_depth', 6))))
gamma_lr_scheduler = float(os.getenv('NB_LR_GAMMA', str(globals().get('gamma_lr_scheduler', 0.1))))
max_layers = int(os.getenv('NB_MAX_LAYERS', str(globals().get('max_layers', 15))))
q_delta = float(os.getenv('NB_Q_DELTA', str(globals().get('q_delta', 0.01))))
rng_seed = int(os.getenv('NB_SEED', str(globals().get('rng_seed', 0))))

# Outputs dir
outputs_dir = os.getenv('NB_OUTPUTS_DIR', './outputs/notebooks')
os.makedirs(outputs_dir, exist_ok=True)

# Device selection (optional)
_device_choice = os.getenv('NB_DEVICE', 'auto').lower()
if _device_choice == 'cpu':
	device = torch.device('cpu')
elif _device_choice == 'cuda' and torch.cuda.is_available():
	device = torch.device('cuda:0')
else:
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Seeding
random.seed(rng_seed)
np.random.seed(rng_seed)
torch.manual_seed(rng_seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(rng_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start_time = time.time()
""".strip()

TAIL_CELL_CODE = """
# NB_TAIL: save minimal run metadata
import json, time, os
run_meta = {
	"device": str(globals().get('device', 'unknown')),
	"batch_size": int(globals().get('batch_size', -1)),
	"num_epochs": int(globals().get('num_epochs', -1)),
	"data_dir": str(globals().get('data_dir', '')),
	"start_time": float(globals().get('start_time', time.time())),
	"end_time": float(time.time()),
}
_outputs_dir = os.getenv('NB_OUTPUTS_DIR', './outputs/notebooks')
os.makedirs(_outputs_dir, exist_ok=True)
with open(os.path.join(_outputs_dir, 'run.json'), 'w') as f:
	json.dump(run_meta, f, indent=2)
""".strip()


def ensure_minimal_imagefolder(root: Path, classes: list[str], num_train: int = 4, num_val: int = 2, image_size: int = 224) -> None:
	"""Create a tiny ImageFolder structure with random images to allow smoke tests."""
	if not exists_pil:
		print("PIL not available; skipping dataset generation.")
		return
	for split, n in [("train", num_train), ("val", num_val)]:
		for cls in classes:
			cls_dir = root / split / cls
			cls_dir.mkdir(parents=True, exist_ok=True)
			# Create images only if directory empty
			if any(cls_dir.iterdir()):
				continue
			for i in range(n):
				arr = (np.random.rand(image_size, image_size, 3) * 255).astype(np.uint8)
				img = Image.fromarray(arr)
				img.save(cls_dir / f"img_{i}.png")


def patch_notebook_params(nb_path: Path) -> None:
	"""Idempotently insert a parameters cell at the top (after imports)."""
	nb = nbformat.read(nb_path, as_version=4)
	# If already patched, skip
	for cell in nb.cells:
		if cell.cell_type == 'code' and 'NB_PARAMETERS' in cell.source:
			break
	else:
		# Insert as the 2nd or 3rd cell to ensure imports exist
		insert_idx = 2 if len(nb.cells) >= 2 else 0
		nb.cells.insert(insert_idx, nbformat.v4.new_code_cell(PARAM_CELL_CODE))
		# Append metadata tail saver if not present
		nb.cells.append(nbformat.v4.new_code_cell(TAIL_CELL_CODE))
		nbformat.write(nb, nb_path)


def execute_notebook(nb_path: Path, cwd: Path, env: dict[str, str]) -> None:
	print(f"Executing {nb_path.name}...")
	nb = nbformat.read(nb_path, as_version=4)
	client = NotebookClient(nb, timeout=600, kernel_name="python3")
	# Merge env vars
	run_env = os.environ.copy()
	run_env.update(env)
	# Execute
	try:
		client.execute()
	except CellExecutionError as e:
		raise RuntimeError(f"Execution failed for {nb_path}: {e}") from e
	# Save executed notebook alongside outputs
	out_dir = Path(env.get('NB_OUTPUTS_DIR', str(cwd / 'outputs' / 'notebooks' / nb_path.stem)))
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / nb_path.name
	nbformat.write(nb, out_path)
	print(f"Saved executed notebook to {out_path}")


def main() -> int:
	# Ensure minimal datasets exist
	ants_bees_root = WORKSPACE / 'data' / 'hymenoptera_data'
	ensure_minimal_imagefolder(ants_bees_root, classes=["ants", "bees"])
	cremad_root = WORKSPACE / 'data' / 'CremaD' / 'mel_spec_reduced'
	ensure_minimal_imagefolder(cremad_root, classes=["ang", "hap"])  # example 2 classes

	# Patch and execute notebooks
	for nb_name, base_env in NOTEBOOKS:
		nb_path = WORKSPACE / nb_name
		if not nb_path.exists():
			print(f"Skipping {nb_name}: not found")
			continue
		patch_notebook_params(nb_path)
		# Common smoke parameters
		env = {
			"NB_BATCH_SIZE": "2",
			"NB_NUM_EPOCHS": "1",
			"NB_DEVICE": "cpu",
			"NB_SEED": "0",
		}
		env.update(base_env)
		execute_notebook(nb_path, WORKSPACE, env)

	print("All notebooks executed successfully.")
	return 0


if __name__ == "__main__":
	sys.exit(main())