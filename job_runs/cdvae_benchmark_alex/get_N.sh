#!/bin/bash

# see how many structures are in test.csv

export PROJECT_ROOT=/lab/mml/kipp/677/jarvis/rhys/benchmarks/models/cdvae
source /home/crc8/miniconda3/etc/profile.d/conda.sh
conda activate cdvae
cd ../../models/cdvae/scripts
pwd
python - <<'PY'
from jarvis.core.atoms import Atoms
from eval_utils import load_model
from pathlib import Path
model, test_loader, _ = load_model(
    Path("/lab/mml/kipp/677/jarvis/rhys/benchmarks/job_runs/cdvae_benchmark_alex/hydra_outputs/singlerun/2025-06-06/alexandria"),
    load_data=True
)
print("Test set size:", len(test_loader.dataset))
PY

