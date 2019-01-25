#!/bin/bash

# Setup for batch stuff
#BSUB -J POISSON_CPU
#BSUB -o poisson_cpu_sim%J.out
#BSUB -q hpcintrogpu 
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 15
#BSUB -n 1 -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"

module load cuda/10.0
module load gcc/7.3.0

# Put environment variables in here and call poisson.gcc
mkdir -p sim_gpu_seq
rm -r sim_gpu_seq/jacobi_gpu_seq.txt

for i in {1..6}
do
	poisson.gcc jacobi $(($i*1000)) 1000 >> sim_gpu_seq/jacobi_gpu_seq.txt
done

