#!/bin/bash

export MATMULT_COMPARE=0
export MFLOPS_MAX_IT=5
#BSUB -J MATMULT_SIM
#BSUB -o matmult_sim_%J.out
#BSUB -q hpcintrogpu
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 30
#BSUB -R "rusage[mem=8GB]"

mkdir gpu_sim

nvprof --print-gpu-summary ./matmult_f.nvcc gpu1 256 256 256  >> gpu_sim/gpu1.txt 2>&1
nvprof --print-gpu-summary ./matmult_f.nvcc gpu1 512 512 512  >> gpu_sim/gpu1.txt 2>&1
nvprof --print-gpu-summary ./matmult_f.nvcc gpu1 768 768 768  >> gpu_sim/gpu1.txt 2>&1
nvprof --print-gpu-summary ./matmult_f.nvcc gpu1 1024 1024 1024  >> gpu_sim/gpu1.txt 2>&1
