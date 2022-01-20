#!/bin/bash
#SBATCH -n 1 #number of parallel tasks ; default: same value as #SBATCH -N
#SBATCH -c 1 #number of CPUs per task ; default: 1
#SBATCH --mem=20G # mem
#SBATCH -p GPU-DEPINFO # partition to use ; default: LONG
#SBATCH -t 4-0
#SBATCH -D /home/guk06997/beir/examples/dataset
#SBATCH --gres=gpu:0

source /home/guk06997/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate nlp
export PYTHONPATH=/home/guk06997/beir

unset http_proxy
unset https_proxy

srun python3 index_corpus_anserini.py --dataset quora --port 8000

