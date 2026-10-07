#!/bin/bash
#SBATCH --job-name=CAESAR_hurricane
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=7:45:00
#SBATCH --output=caesar_hurricane_%j.out
#SBATCH --error=caesar_hurricane_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=klaskyethan@gmail.com

# Submit from the CAESAR repository root.
exec bash examples/slurm/run_hurricane.sh
