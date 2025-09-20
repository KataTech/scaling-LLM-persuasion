#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -t 4:00:00
#SBATCH -c 16
#SBATCH --gres=gpu:h200:1

# Set up environment
module load miniforge

# Run your application
conda activate gpt-oss
python3 classify.py