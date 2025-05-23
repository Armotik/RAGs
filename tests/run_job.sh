#!/bin/bash
#SBATCH --partition=gpu-a40
#SBATCH --time=4:00:00
#SBATCH --job-name=test_preprocessing
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate rag_env

python test_preprocessing.py