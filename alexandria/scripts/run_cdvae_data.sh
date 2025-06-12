#!/bin/bash
set -e
mkdir -p ../models/cdvae/data/alexandria/
uv pip install jarvis-tools pymatgen numpy pandas tqdm
python scripts/generate_data_cdvae.py
python - <<'PYCODE'
import os
path = "../models/cdvae/data/alexandria"
files = ["train.csv", "test.csv", "val.csv"]
for file in files:
	file_path = os.path.join(path,file)
	if os.path.exists(file_path):
		os.remove(file_path)
	os.rename(file, file_path)
PYCODE

