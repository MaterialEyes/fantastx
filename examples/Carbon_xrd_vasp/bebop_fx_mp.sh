#!/bin/bash
##SBATCH --job-name=bulk_pdf_fantastx
#SBATCH --account=FANTASTX
#SBATCH --partition=bdwall
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=120gb
#SBATCH -t 0-2:00
#SBATCH --output=fx_mp.out
#SBATCH --error=fx_mp.err

export I_MPI_FABRICS=shm:tmi
ulimit -s unlimited

module purge
module add StdEnv 
module delete intel intel-mkl intel-mpi
module add gcc/9.2.0-pkmzczt
module add intel-parallel-studio/cluster.2020.2-y7ijupg
export PATH=/soft/vasp/6.2.1/bdw_vaspsol:$PATH

#module add anaconda3/2021.05
#eval "$(conda shell.bash hook)"
#conda activate miniconda3
#conda list >> module.log

source ~/.bashrc
conda activate ~/miniconda3/envs/fantastx
export PYTHONPATH=$PYTHONPATH:/home/zisheng.zhang/software/GSASII

time python -u /home/zisheng.zhang/fantastx_proj/fx_xrd/run_fx_mp.py

