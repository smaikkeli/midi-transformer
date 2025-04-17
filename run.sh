#!/bin/bash

#SBATCH --job-name=midi-trans    # job name
#SBATCH --time=00:30:00          # time hours/min/sec
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --cpus-per-task=8        # cpu-cores per task
#SBATCH --mem=40G                # total memory
#SBATCH --gpus=1                 # number of gpus
#SBATCH --time=00:05:00          # time limit
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-user=miksuhok@gmail.com

module purge
module load mamba
source activate midi-transformer
python train.py