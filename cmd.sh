#!/bin/bash
#SBATCH --job-name=maskgan            # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --ntasks=36                   # Run on a single CPU
#SBATCH --mem=120gb                   # Job memory request
#SBATCH --gres=gpu:4                  # Job GPU request
#SBATCH --output=logs/%j.log          # Standard output and error log
#SBATCH --account=jerin

module load use.own
module load python/3.7.0
python3 -W ignore -m mgan.main --path datasets/aclImdb/ --spm_prefix datasets/aclImdb/train/imdb
