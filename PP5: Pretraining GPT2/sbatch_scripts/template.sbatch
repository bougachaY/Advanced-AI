#!/bin/bash
#SBATCH -J mon_job
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p shared
#SBATCH --time=00:15:00 
#SBATCH --output=/users/formation/tpirtbgchs/job_results/out/job_%j.out
#SBATCH --error=/users/formation/tpirtbgchs/job_results/err/job_%j.err


apptainer exec \
--env PATH=$HOME/.local/bin:$PATH \
--env UV_PROJECT_ENVIRONMENT=/tmpdir/tpirtbgchs/envs/aai \
--bind /tmpdir,/work \
--nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif \
<<<<<<< HEAD
<<<<<<< HEAD
uv run python test_sbatch.py
=======
uv run --no-sync python mon_script.py
>>>>>>> upstream/main
=======
uv run --no-sync python mon_script.py
>>>>>>> upstream/main
