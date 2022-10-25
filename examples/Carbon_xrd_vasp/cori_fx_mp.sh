#!/bin/bash
#SBATCH -A m2571
#SBATCH -N 16
#SBATCH -C knl
#SBATCH -q regular
#SBATCH -t 17:59:59

module load vasp-tpc/6.2.1-knl
#module load vasp-tpc/5.4.4-knl

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_STACKSIZE=512m
export OMP_NUM_THREADS=4


source /global/homes/z/zisheng/.bashrc
conda activate /global/homes/z/zisheng/Software/anaconda3/envs/fantastx
export PYTHONPATH=$PYTHONPATH:/global/homes/z/zisheng/Software/fantastx
export PYTHONPATH=$PYTHONPATH:/global/homes/z/zisheng/Software/GSASII

time python -u /global/cfs/cdirs/m2571/zishengz/fx_xrd_prod/run_fx_mp.py

