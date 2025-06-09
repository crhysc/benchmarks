#!/bin/bash
#run data preprocessor
python scripts/supercon_preprocess.py \
        --dataset dft_3d \
        --id-key jid \
        --target Tc_supercon \
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
        --seed 123 \
        --max-size 25

# move everything to the right spot
mkdir ../models/flowmm/data/supercon
mv ./train.csv ../models/flowmm/data/supercon/
mv ./val.csv ../models/flowmm/data/supercon/
mv ./test.csv ../models/flowmm/data/supercon/

# pull hydra config yaml
cd ../models/flowmm/scripts_model/conf/data/
wget https://raw.githubusercontent.com/crhysc/utilities/refs/heads/main/supercon.yaml

# remove yamls that might exist from past pipeline runs
rm /lab/mml/kipp/677/jarvis/rhys/benchmarks/models/flowmm/src/flowmm/model/stats_supercon*
python - <<'PYCODE'
import os
files = ["atom_density.yaml", "spd_pLTL_stats.yaml", "spd_std_coef.yaml", "lattice_params_stats.yaml"]
path = "/lab/mml/kipp/677/jarvis/rhys/benchmarks/models/flowmm/src/flowmm/rfm/manifolds"
for file in files:
	filepath = os.path.join(file,path)
	if os.path.exists(filepath):
		os.remove(filepath)
PYCODE

# create necessary yamls for training and inference
cd /lab/mml/kipp/677/jarvis/rhys/benchmarks/models/flowmm 
python -u -m flowmm.rfm.manifolds.spd
python -u -m flowmm.rfm.manifolds.lattice_params
python -u -m flowmm.model.standardize data=supercon







