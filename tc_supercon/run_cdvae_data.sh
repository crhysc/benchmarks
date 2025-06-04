#!/bin/bash
set -e
mkdir -p ../models/cdvae/data/supercon/
uv pip install jarvis-tools pymatgen numpy pandas tqdm
python generate_data_cdvae.py
wget "https://github.com/crhysc/utilities/blob/main/supercon.yaml"
python - <<'PYCODE'
import os
path = "../models/cdvae/data/supercon"
files = ["train.csv", "test.csv", "val.csv"]
for file in files:
	file_path = os.path.join(path,file)
	if os.path.exists(file_path):
		os.remove(file_path)
	os.rename(file, file_path)
yaml_path = "../models/cdvae/conf/data/supercon.yaml"
if not os.path.exists(yaml_path):
	os.rename("supercon.yaml", yaml_path)
else:
	os.remove("supercon.yaml")
PYCODE

