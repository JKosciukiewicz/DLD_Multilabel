#!/bin/bash
#SBATCH --job-name=DLD_bray_gigadb
#SBATCH --time=24:00:00
#SBATCH --account=plgwtln2-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu

source /net/pr2/projects/plgrid/plggwtln/jk/DLD/bin/activate
cd /net/people/plgrid/plgjkosciukiewi/DLD_Multilabel/

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export WANDB_MODE=online

python train_on_Bray.py \
    --data_file /net/pr2/projects/plgrid/plggwtln/jk/datasets/gigadb/gigadb.csv \
    --labels_file /net/pr2/projects/plgrid/plggwtln/jk/datasets/gigadb/gigadb_top_3_moas.csv \
    --use_wandb \
    --wandb_project "DLD_Bray" \
    --wandb_run_name "bray_top_3_${TIMESTAMP}"
