#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)
#SBATCH -o slurm.out
#SBATCH -e slurm.err

nvidia-smi
ip a
module purge
# module load anaconda3-2021.05-gcc-9.3.0-r6itwa7
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
#module load CUDA/10.2.89-GCC-8.3.0
#module load zlib/1.2.11
conda init bash 
# #conda create --name py38 python=3.8
source activate ZeroShotEdit
# #pip install ultralytics
# python real_image_edit.py
# python task_concept.py

uvicorn api_main:app --host 0.0.0.0 --port 8000
# python cc.py
conda deactivate
