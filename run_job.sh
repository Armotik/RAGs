#!/bin/bash
#SBATCH --partition=gpu-a40
#SBATCH --time=16:00:00
#SBATCH --job-name=rag_improve
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate rag_env

python tests/test_preprocessing.py