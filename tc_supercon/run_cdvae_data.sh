#!/bin/bash
set -e
mkdir -p ../models/cdvae/data/supercon/
uv pip install jarvis-tools pymatgen numpy pandas tqdm
python generate_data_cdvae.py
python - <<'PYCODE'
import os
import urllib.request
path = "../models/cdvae/data/supercon"
files = ["train.csv", "test.csv", "val.csv"]
for file in files:
	file_path = os.path.join(path,file)
	if os.path.exists(file_path)
		os.remove(file_path)
	os.rename(file, file_path)
yaml_path = "../models/cdvae/conf/data/supercon.yaml"
url = "https://github.com/crhysc/utilities/blob/main/supercon.yaml"
if not os.path.exists(yaml_path)
	urllib.request.urlretrieve(url, yaml_path)
PYCODE

