#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:1
#SBATCH --mem=20G

module load cuda11.6/toolkit
python drone_tiny_nerf_pytorch.py
