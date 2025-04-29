#!/bin/bash

#SBATCH --job-name=midi_trans    # job name
#SBATCH --time=24:00:00       # time hours/min/sec
#SBATCH --mem=24G             # total memory
#SBATCH --gpus=1              # number of gpus
#SBATCH --cpus-per-task=8
#SBATCH --output=log.out

module purge
module load mamba
source activate midi-transformer
python train.py --data_to_tmp

