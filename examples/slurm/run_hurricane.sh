#!/bin/bash
source ~/.bashrc || true
source ../data/test/set_env_ufl_caesar.sh
source ../caesar_venv/bin/activate
set -eo pipefail
# Keep the training source unchanged, using eager execution on cluster Torch.
export TORCH_COMPILE_DISABLE=1
python -u -m pyCAESAR.train_vae3d \
  --config examples/config_hurricane.yaml \
  --train_set hurricane --test_set hurricane \
  --save_path snapshots/hurricane-adios \
  --batch_size 32 --iterations 100 --model_dim 16 --sr_dim 16 \
  --pretrain pretrained/model_bs64_ep100k.pt
