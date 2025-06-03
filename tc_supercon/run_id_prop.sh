#!/bin/bash
set -e
mkdir agpt_tc_supercon
pip install -r data_reqs.txt
python id_prop.py \
	--dataset dft_3d \
        --id-key jid \
	--output agpt_tc_supercon
        --target Tc_supercon \
        --seed 123 \
        --max-size 1000
