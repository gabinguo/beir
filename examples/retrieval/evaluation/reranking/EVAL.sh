#!/bin/bash
#SBATCH -n 1 #number of parallel tasks ; default: same value as #SBATCH -N
#SBATCH -c 1 #number of CPUs per task ; default: 1
#SBATCH --mem=20G # mem
#SBATCH -p GPU-DEPINFO # partition to use ; default: LONG
#SBATCH --nodelist calcul-gpu-depinfo-3
#SBATCH -t 4-0
#SBATCH -D /home/guk06997/beir/examples/retrieval/evaluation/reranking
#SBATCH --gres=gpu:1

source /home/guk06997/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate nlp
unset http_proxy
unset https_proxy

export PYTHONPATH=/home/guk06997/beir

srun python3 evaluate_anserini_bm25_ce_reranking.py --dataset quora --port 8000;
