#!/bin/bash

export MATMULT_COMPARE=0
#BSUB -J MATMULT_SIM
#BSUB -o matmult_sim_%J.out
#BSUB -q hpcintrogpu
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 30
#BSUB -R "rusage[mem=4GB]"

mkdir -p gpu_sim

rm -rf gpu_sim/gpu3*

for i in {2..20}
do
	nvprof --print-gpu-summary ./matmult_f.nvcc gpu3 $((512*$i)) $((512*$i)) $((512*$i)) >> gpu_sim/gpu3.txt 2>&1
done

