#!/bin/bash
set -eo pipefail
experiment_dataset=$1
experiment_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$experiment_root"
source ~/.bashrc
source ../data/test/set_env_ufl_caesar.sh
source ../caesar_venv/bin/activate
set -u
which bpls
python -c 'import adios2, torch; print("ADIOS2:", adios2.__version__); print("Torch:", torch.__version__); assert torch.cuda.is_available(), "CUDA required"'
case "$experiment_dataset" in
  re3200) experiment_bp=/lustre/blue2/ranka/eklasky/data/513.513.513.0.000625.Re3200OG.bp ;;
  hurricane) experiment_bp=/lustre/blue2/ranka/eklasky/software_X_data/data/Hurricane_Isabel.bp ;;
  input300) experiment_bp=/lustre/blue2/ranka/eklasky/data/input_300stepsOG.bp ;;
  *) exit 2 ;;
esac
experiment_output="${EXPERIMENT_OUTPUT:-$experiment_root/snapshots/adios-experiment}"
mkdir -p "$experiment_output/$experiment_dataset"
bpls -la "$experiment_bp" > "$experiment_output/$experiment_dataset/bpls.txt"
# Each job performs three independent runs; hyperparameters stay identical.
# Default batch_size is 64. Override BATCH_SIZE for ALL jobs if GPU memory requires it.
for experiment_size in 256 512 128; do
  experiment_run="$experiment_output/$experiment_dataset/size-$experiment_size"
  if [[ -e "$experiment_run/COMPLETE" ]]; then
    echo "Skipping completed run: $experiment_run"
    continue
  fi
  mkdir -p "$experiment_run"
  python -u -m pyCAESAR.train_vae3d \
    --config examples/config_adios_experiment.yaml \
    --train_set "$experiment_dataset" --test_set "$experiment_dataset" \
    --save_path "$experiment_run" --spatial_size "$experiment_size" \
    --iterations 100 --model_dim 16 --sr_dim 16 \
    --batch_size "${BATCH_SIZE:-64}" --workers 4 --seed 0 \
    2>&1 | tee "$experiment_run/train.log"
  touch "$experiment_run/COMPLETE"
done
