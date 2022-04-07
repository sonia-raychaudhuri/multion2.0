#!/bin/bash
#SBATCH --job-name=multion
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20 # You can give each process multiple threads/cpus
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=0-2:59     # DD-HH:MM:SS
#SBATCH --mail-user=sonia_raychaudhuri@sfu.ca
#SBATCH --mail-type=ALL

source /home/sraychau/miniconda3/etc/profile.d/conda.sh
# activate environment
conda activate multion
hostname
echo $CUDA_AVAILABLE_DEVICES

srun -u \
python -u habitat_baselines/run_map.py \
    --exp-config habitat_baselines/config/multinav/ppo_multinav_oracle_1ON_w_distr.yaml \
    --run-type train  \
    --agent-type oracle
