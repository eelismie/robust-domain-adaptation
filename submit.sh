#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time 00:15:00
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem 32768
#SBATCH --partition=gpu
#SBATCH --account=master
#SBATCH --output jupyter-log-%J.out
module load gcc/8.4.0-cuda
module load python/3.7.7
source ~/environments/test4/bin/activate
CUDA_VISIBLE_DEVICES=0 python ~/debug.py 
