#!/bin/bash
#SBATCH --job-name=CAESAR_re3200
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=7:45:00
#SBATCH --output=caesar_re3200_%j.out
#SBATCH --error=caesar_re3200_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=klaskyethan@gmail.com

# Submit from the CAESAR repository root.
exec bash examples/slurm/run_re3200.sh
