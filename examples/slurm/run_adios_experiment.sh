#!/bin/bash
set -Eo pipefail
trap 'experiment_status=$?; echo "ERROR: line $LINENO: $BASH_COMMAND (exit $experiment_status)" >&2; exit "$experiment_status"' ERR
experiment_dataset=$1
echo "Starting CAESAR dataset=$experiment_dataset job=${SLURM_JOB_ID:-local} host=$(hostname)"
experiment_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$experiment_root"
# Shell startup files can return nonzero in a noninteractive batch shell.
# Do not apply errexit/the error trap to them; validate the tools afterward.
trap - ERR
set +e
echo "Loading ~/.bashrc"
source ~/.bashrc || echo "~/.bashrc returned nonzero; continuing batch environment setup" >&2
set +e
set +u
for experiment_env in ../data/test/set_env_ufl_caesar.sh ../caesar_venv/bin/activate; do
  if [[ ! -r "$experiment_env" ]]; then
    echo "Missing environment file: $experiment_root/$experiment_env" >&2
    exit 1
  fi
  echo "Loading $experiment_env"
  experiment_env_status=0
  source "$experiment_env" || experiment_env_status=$?
  set +e
  set +u
  if (( experiment_env_status != 0 )); then
    echo "Environment file returned $experiment_env_status; checking required tools next" >&2
  fi
done
trap 'experiment_status=$?; echo "ERROR: line $LINENO: $BASH_COMMAND (exit $experiment_status)" >&2; exit "$experiment_status"' ERR
set -Eeuo pipefail
echo "Checking ADIOS and Python/CUDA"
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
