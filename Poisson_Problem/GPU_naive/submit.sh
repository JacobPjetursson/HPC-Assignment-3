#!/bin/bash

# Setup for batch stuff
#BSUB -J POISSON_GPU
#BSUB -o poisson_gpu_sim%J.out
#BSUB -q hpcintrogpu 
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 30
#BSUB -n 1 -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"

module load cuda/10.0
module load gcc/7.3.0

# Put environment variables in here and call poisson.gcc
mkdir -p sim_gpu_naive
rm -r sim_gpu_naive/jacobi_naive*.txt

poisson jacobi 16384 10000 >> sim_gpu_naive/jacobi_gpu_naive.txt

