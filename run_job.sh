#!/bin/bash
#SBATCH --partition=gpu-2080ti
#SBATCH --time=01:00:00
#SBATCH --job-name=mon_script_python
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate stage_l3i

python mon_script.py