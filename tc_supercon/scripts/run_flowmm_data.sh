#!/bin/bash
# run data preprocessor
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








