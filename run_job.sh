#!/bin/bash
#SBATCH --partition=gpu-2080ti
#SBATCH --time=48:00:00
#SBATCH --job-name=rag_post-retrieval
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate rag_env

python post-retrieval.py