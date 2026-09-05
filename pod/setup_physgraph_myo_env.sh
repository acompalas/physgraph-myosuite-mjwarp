#!/usr/bin/env bash
# setup_physgraph_myo_env.sh
# Run this on the pod AFTER cloning this repo (physgraph-myosuite-mjwarp)
# to /data/, and AFTER setup_physgraph_env.sh has already run (this
# script reuses that env's existing /data/miniconda3 and /data/PhysGraph).
#
# VERIFIED WORKING END-TO-END on the pod, 2026-09-05 (RTX 3090,
# driver 595.91.07 / CUDA 13.2, base image nvidia/cuda:11.8.0-cudnn8-
# devel-ubuntu20.04).
#
# Usage: bash pod/setup_physgraph_myo_env.sh

set -e
export DEBIAN_FRONTEND=noninteractive

echo "== [1/6] System packages (mujoco rendering deps) =="
apt-get update && apt-get install -y \
    libgl1-mesa-glx libxrender1 libxrandr2 libglib2.0-0 \
    libosmesa6-dev libglfw3 patchelf

echo "== [2/6] Verify prerequisites =="
if [ ! -d /data/miniconda3 ]; then
  echo "ERROR: /data/miniconda3 not found. Run setup_physgraph_env.sh first"
  echo "(this script reuses that shared miniconda install)."
  exit 1
fi
if [ ! -d /data/PhysGraph ]; then
  echo "ERROR: /data/PhysGraph not found. Run setup_physgraph_env.sh first"
  echo "(needed for the DexHand base class + ManipData we inherit from)."
  exit 1
fi
source /data/miniconda3/bin/activate

echo "== [2.5/6] Accept conda ToS =="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "== [3/6] Create physgraph-myo env (Python 3.11) =="
if ! conda env list | grep -q physgraph-myo; then
  conda create -y -n physgraph-myo python=3.11
fi
conda activate physgraph-myo

echo "== [4/6] Torch (cu130) -- matches local dev version 2.13.0+cu130 =="
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130

echo "== [5/6] MuJoCo + retargeting deps =="
pip install mujoco==3.12.0
pip install pytorch_kinematics
pip install rl_games  # NOT pinned -- 1.6.1 requires Python <3.11, incompatible with this env's 3.11
pip install omegaconf

echo "== [6/6] mujoco-warp (GPU-parallel MuJoCo, google-deepmind/mujoco_warp) =="
# CONFIRMED WORKING on the pod: real PyPI package, requires an NVIDIA GPU
# (which we have), version tracks the installed mujoco version (both
# landed on 3.12.0 here) -- no compatibility issues found.
pip install mujoco-warp
# numpy last -- some upstream deps pull a newer version than we want
pip install numpy==2.4.6

echo "== Verifying imports =="
python -c "
import mujoco; print('mujoco', mujoco.__version__)
import mujoco_warp; print('mujoco_warp OK')
import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
import pytorch_kinematics; print('pytorch_kinematics OK')
import rl_games; print('rl_games OK')
import numpy; print('numpy', numpy.__version__)
print('ALL CORE IMPORTS OK')
"

echo "== Done. Activate with: source /data/miniconda3/bin/activate physgraph-myo =="
