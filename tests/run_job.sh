#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --time=1:00:00
#SBATCH --job-name=milvus_indexation
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1GB
#SBATCH --gres=gpu:1

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate rag_env

python test.py