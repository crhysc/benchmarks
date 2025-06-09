#!/bin/bash

cd /lab/mml/kipp/677/jarvis/rhys/benchmarks/models/flowmm

mamba env create -f environment.yml -y
source /home/crc8/miniconda3/etc/profile.d/conda.sh
conda activate flowmm
pip install uv
uv pip install "jarvis-tools>=2024.5" "pymatgen>=2024.1" pandas numpy tqdm
uv pip install -e . \
	       -e remote/riemannian-fm \
	       -e remote/cdvae \
	       -e remote/DiffCSP-official
	       print("Done")
