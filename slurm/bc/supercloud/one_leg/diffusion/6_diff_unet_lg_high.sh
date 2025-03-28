#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=6_diff_unet_lg_high
#SBATCH -c 20

python -m src.train.bc +experiment=state/diff_unet \
    randomness='[high]' \
    data.data_subset=50 \
    rollout.randomness=high \
    rollout.rollouts=false \
    wandb.mode=offline \
    wandb.project=ol-state-dr-high-1 \
    dryrun=false
