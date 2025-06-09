# data downloader and preprocessor
!wget -q https://raw.githubusercontent.com/crhysc/utilities/refs/heads/main/supercon_preprocess.py

#run data preprocessor
%cd /content/flowmm
!conda run -p /usr/local/envs/flowmm_env --live-stream \
    python supercon_preprocess.py \
        --dataset dft_3d \
        --id-key jid \
        --target Tc_supercon \
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
        --seed 123 \
        --max-size 25
print("Done")

# mv everything to the correct spot
%cd /content
%mkdir /content/flowmm/data/supercon
%mv /content/flowmm/train.csv /content/flowmm/data/supercon/
%mv /content/flowmm/val.csv /content/flowmm/data/supercon/
%mv /content/flowmm/test.csv /content/flowmm/data/supercon/
print("Done")

# pull hydra config yaml
%cd /content/flowmm/scripts_model/conf/data/
!wget https://raw.githubusercontent.com/crhysc/utilities/refs/heads/main/supercon.yaml
%cat supercon.yaml

# generate necessary yaml files for training
%rm /content/flowmm/src/flowmm/rfm/manifolds/atom_density.yaml
%rm /content/flowmm/src/flowmm/rfm/manifolds/spd_pLTL_stats.yaml
%rm /content/flowmm/src/flowmm/rfm/manifolds/spd_std_coef.yaml

%cd /content/flowmm
!bash create_env_file.sh && \
 echo "successfully ran create_env_file.sh" && \
 HYDRA_FULL_ERROR=1 \
 FLOWMM_DEBUG=1 \
 conda run -p /usr/local/envs/flowmm_env --live-stream \
    python -u -m flowmm.rfm.manifolds.spd

# generate lattice_params_stats.yaml
!rm /content/flowmm/src/flowmm/rfm/manifolds/lattice_params_stats.yaml
%cd /content/flowmm
!bash create_env_file.sh && \
 echo "successfully ran create_env_file.sh" && \
 HYDRA_FULL_ERROR=1 \
 conda run -p /usr/local/envs/flowmm_env --live-stream \
    python -u -m flowmm.rfm.manifolds.lattice_params

# create affine stats yaml
%rm /content/flowmm/src/flowmm/model/stats_supercon*
%cd /content/flowmm
!bash create_env_file.sh && \
 echo "successfully ran create_env_file.sh" && \
 HYDRA_FULL_ERROR=1 \
 conda run -p /usr/local/envs/flowmm_env --live-stream \
    python -u -m flowmm.model.standardize \
                 data=supercon







