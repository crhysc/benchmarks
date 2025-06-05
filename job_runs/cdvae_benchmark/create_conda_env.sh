#!/bin/bash

cd /lab/mml/kipp/677/jarvis/rhys/benchmarks/models/cdvae

mamba env create -f env.yml -y
mamba activate cdvae
mamba install -c conda-forge "torchmetrics<0.8" --yes
mamba install mkl=2024.0 --yes
pip install uv
uv pip install "monty==2022.9.9"
mamba install -c conda-forge "pymatgen>=2022.0.8,<2023" --yes
uv pip install pandas jarvis-tools
uv pip install --upgrade torch_geometric==1.7.0
uv pip install -e .
