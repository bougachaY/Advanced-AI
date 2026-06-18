#!/bin/bash
#SBATCH -J vlm_train
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p shared
#SBATCH --output=/users/formation/tpirtbgchs/job_results/out/job_%j.out
#SBATCH --error=/users/formation/tpirtbgchs/job_results/err/job_%j.err

cd /users/formation/tpirtbgchs/Advanced-AI/Project

apptainer exec \
--env PATH=$HOME/.local/bin:$PATH \
--env UV_PROJECT_ENVIRONMENT=/tmpdir/tpirtbgchs/envs/aai \
--env UV_NO_SYNC=true \
--env HF_HOME=/work/formation/tpirtbgchs/huggingface \
--env HF_HUB_OFFLINE=1 \
--env HF_DATASETS_OFFLINE=1 \
--env TRANSFORMERS_OFFLINE=1 \
--bind /tmpdir,/work \
--nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif \
uv run python train.py \
    --dataset_type cauldron \
    --max_steps 10000 
