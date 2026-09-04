#!/usr/bin/env bash
# setup_physgraph_myo_env.sh
# Run this on the pod AFTER cloning this repo (physgraph-myosuite-mjwarp)
# to /data/, and AFTER setup_physgraph_env.sh has already run (this
# script reuses that env's existing /data/miniconda3 and /data/PhysGraph).
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

echo "== [3/6] Create physgraph-myo env (Python 3.11) =="
if ! conda env list | grep -q physgraph-myo; then
  conda create -y -n physgraph-myo python=3.11
fi
conda activate physgraph-myo

echo "== [4/6] Torch (cu130) -- matches local dev version 2.13.0+cu130 =="
# NOTE: this specific version/index-url combo is copied from a verified-
# working LOCAL install, not independently re-verified on the pod yet --
# if this exact command fails, check https://pytorch.org for the current
# correct install command for CUDA 13.0 and adjust.
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130

echo "== [5/6] MuJoCo + retargeting deps (pinned to verified local versions) =="
pip install mujoco==3.12.0
pip install pytorch_kinematics
pip install rl_games==1.6.1
pip install omegaconf

echo "== [6/6] mujoco-warp -- UNTESTED, best-effort placeholder =="
echo "WARNING: mujoco-warp itself has never been installed/tested locally"
echo "(all local work so far has been single-env MuJoCo for retargeting)."
echo "This step is a best-effort guess -- verify against the actual"
echo "mujoco-warp repo's own install instructions and adjust as needed:"
# pip install mujoco-warp  # <-- placeholder, confirm real package name/method

# numpy last -- some upstream deps pull a newer version than we want
pip install numpy==2.4.6

echo "== Verifying imports =="
python -c "
import mujoco; print('mujoco', mujoco.__version__)
import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
import pytorch_kinematics; print('pytorch_kinematics OK')
import rl_games; print('rl_games OK')
print('ALL CORE IMPORTS OK (mujoco-warp not yet verified)')
"

echo "== Done. Activate with: source /data/miniconda3/bin/activate physgraph-myo =="
