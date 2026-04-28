#!/bin/bash
#SBATCH --job-name=DLD_mnist_20
#SBATCH --time=24:00:00
#SBATCH --account=plgwtln2-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu

source /Users/jkosciukiewicz/Developer/Research/DLD/.venv/bin/activate
python train_on_MNIST_20.py --device cuda
