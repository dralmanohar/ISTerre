#!/bin/bash
##!/bin/bash
#OAR -n 900_phi_dir_same_32_32_256_unet_u_all__deri_u_b_t_nlinu_ub_bu
#OAR -l /nodes=1/gpu=1, walltime=48:00:00
##OAR -l /nodes=1/gpu=1,walltime=02:00:00
#OAR -p gpumodel='A100'
##OAR -O result_non.out
#OAR -E error_non.out
#OAR --project pr-geodynamo-ai
##OAR --project pr-geodyn_nn
##OAR -t devel

#export PATH="/home/sharmam/anaconda3/bin":$PATH
source /applis/environments/cuda_env.sh 11.7
source /applis/environments/conda.sh
#conda activate PyTorch
conda activate torch

#python3 data_training_mlp_loop.py . 1>out_log_non 2>err_log_non

python3 data_training_3D_unet.py . 1>out_log_non 2>err_log_non


##python3 evaluate_cnn_loop.py . 1>out_log_non 2>err_log_non

