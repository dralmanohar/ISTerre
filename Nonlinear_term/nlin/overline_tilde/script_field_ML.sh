#!/bin/bash
#OAR -n iter_save_100
#OAR -l /nodes=1/core=16,walltime=200:30:00
##OAR -O result_non.out
#OAR -E error_non.out
#OAR --project iste-equ-geodynamo

####OAR -p network_address='ist-calcul1.ujf-grenoble.fr'

#source /soft/env.bash
#module load MATLAB/R2016a
export PATH="/home/sharmam/anaconda3/bin":$PATH
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PyTorch

python3 data_featuring_from_diff_folder.py . 1>out_log_non 2>err_log_non


