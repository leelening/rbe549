#!/bin/bash
#SBATCH --job-name=nerf
#SBATCH -p gpu
#SBATCH --time=72:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=64gb

module load cuda11.6/toolkit

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_PER_TASK  = $SLURM_GPUS_PER_TASK "

gpuList=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/,/ /g')
N=0
devList=""
for gpu in $gpuList
do
    devList="$devList $N"
    N=$(($N + 1))
done
devList=$(echo $devList | sed -e 's/ /,/g')
echo "devList = $devList"


srun python copy_of_nerf_from_nothing.py

date