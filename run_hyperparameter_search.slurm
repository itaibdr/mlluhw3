#!/bin/bash
#SBATCH --job-name=roberta_hyperparameter_search_ibd214
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ibd214/%j_%x.out
#SBATCH --error=/scratch/ibd214/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4
#SBATCH --mail-user=<ibd214@nyu.edu>
#SBATCH --mail-type=ALL

singularity exec --nv --bind /scratch/ibd214 --overlay /scratch/ibd214/overlay-25GB-500K.ext3:ro /scratch/ibd214/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
cd /scratch/ibd214
python3 run_hyperparameter_search.py /scratch/ibd214/BoolQ
"
