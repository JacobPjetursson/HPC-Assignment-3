#!/bin/bash

export MATMULT_COMPARE=0
export MFLOPS_MAX_IT=1
export OMP_NUM_THREADS=12
#BSUB -J MATMULT_SIM
#BSUB -o matmult_sim_%J.out
#BSUB -q hpcintrogpu
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -n 12 -R "span[hosts=1]"
#BSUB -W 30
#BSUB -R "rusage[mem=4GB]"


mkdir -p cpu_sim
rm -f cpu_sim/cpu.txt

for i in {2..20}
do
	numactl --cpunodebind=0 ./matmult_f.nvcc lib $((512*$i)) $((512*$i)) $((512*$i)) >> cpu_sim/cpu.txt 2>&1
done

