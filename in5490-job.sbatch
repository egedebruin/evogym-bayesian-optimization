#!/bin/bash

#SBATCH --job-name=in5490
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=1G

conda activate py310

source revolve2_ege/.venv/bin/activate

srun --ntasks=1 --cpus-per-task=50 python3 evolve_robots_ddpg.py

wait
