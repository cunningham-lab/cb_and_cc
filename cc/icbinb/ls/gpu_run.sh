#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
#SBATCH -A stats # The account name for the job.
#SBATCH --job-name=myPythonJob # The job name.
##SBATCH -c 4 # The number of cpu cores to use.
##SBATCH --exclusive # run on single node
#SBATCH --time=10:00:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=16gb # The memory the job will use per cpu core.
#SBATCH --gres=gpu:1
##SBATCH -o log.log
source /moto/home/eg2912/.bashrc
module load cuda10.1/toolkit
module load cudnn/cuda_10.1_v7.6.4
python $1
