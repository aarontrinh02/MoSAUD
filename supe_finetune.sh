#!/bin/bash
#SBATCH --job-name=v3complete_tune
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-gpu=96G
#SBATCH --nodes=1
#SBATCH --time=10:00:00          # Request 5 hours runtime
#SBATCH --output=outputs/%x_%j.out      # Output file name (%x is job name, %j is job ID)
#SBATCH --error=outputs/%x_%j.err       # Error file name

# Your commands here
cd ~/scratch/SUPE

module load anaconda3
module load cuda

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export D4RL_SUPPRESS_IMPORT_ERROR=1

conda activate supe

python train_finetuning_supe.py --config.backup_entropy=False --config.num_min_qs=2 --offline_relabel_type=pred --use_rnd_offline=True --use_rnd_online=True --env_name=kitchen-complete-v0 --seed=1 --config.init_temperature=1.0