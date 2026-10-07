# Three datasets, three latent sizes

Submit from the CAESAR repository root:

```bash
sbatch examples/slurm/train_re3200.slurm
sbatch examples/slurm/train_hurricane.slurm
sbatch examples/slurm/train_input300.slurm
```

Each job uses your `hpg-b200`, one GPU, six CPUs, 200 GB RAM, and 7:45:00
limit. It sources `~/.bashrc`, `../data/test/set_env_ufl_caesar.sh`, and
`../caesar_venv/bin/activate`. Check those environment paths on the cluster;
the `test/` environment script is not included in this checkout. Install
`adios2` in that environment beforehand if needed (`pip install adios2`).

The three `run_<dataset>.sh` launchers share `run_adios_experiment.sh`.
Each trains independent models at sizes 256, 512, then 128. There is no
checkpoint transfer between sizes. All use identical training defaults:
batch 64, model dimension 16, SR dimension 16, learning rate 0.001,
lr gamma 0.5, beta 1e-5 to 2e-5, beta switch 0.75, seed 0,
16 frames, and `--iterations 100` (the existing convention: **100,000
optimizer steps per model**, not 100 steps or 100 epochs).
Training uses eager autograd; `torch.compile` is disabled after a compiled
backward gradient-shape failure on cluster PyTorch 2.11. Hyperparameters,
architecture, and normalization remain the same.

| Input patch | Logical latent (excluding batch) | Hyperlatent per latent frame |
| --- | --- | --- |
| 16 × 256 × 256 | 64 × 4 × 16 × 16 | 64 × 4 × 4 |
| 16 × 512 × 512 | 64 × 4 × 32 × 32 | 64 × 8 × 8 |
| 16 × 128 × 128 | 64 × 4 × 8 × 8 | 64 × 2 × 2 |

The model internally flattens latent time into batch, so returned `q_latent`
is `[batch*4, 64, L, L]` rather than a five-dimensional tensor.

Batch 64 at 512 pixels may exceed GPU memory. If necessary use the same
smaller batch for all three jobs, e.g.
`sbatch --export=ALL,BATCH_SIZE=1 examples/slurm/train_re3200.slurm`.
The 7:45 limit may not cover three 100,000-step runs; runtime is unmeasured.
A rerun skips sizes with a `COMPLETE` marker; interrupted sizes restart.

## BP reading and results

`config_adios_experiment.yaml` contains your exact paths and the variable
names/layouts from your `bpls -lv` output:

| Dataset | Fields | Layout |
| --- | --- | --- |
| Re3200 | critq, pp, ux, uy, uz, vort | 10 ADIOS steps of `[section=513,height=513,width=513]` |
| Hurricane | all 19 float fields from bpls | `[time=100,height=500,width=500]` per field |
| input300 | data | `[section=60,time=300,height=640,width=640]` |

Hurricane's first spatial dimension is the model frame direction. Re3200 uses
actual ADIOS time steps, with each depth slice as a section. Its 10 frames are
padded by repeating the last frame to make 16. Hurricane 512-pixel crops are
padded from 500×500. Evaluation excludes padded frames/pixels. These conventions
are explicit in YAML and should stay identical across all three sizes.

The reader reads only one hyperslab patch at a time, including one ADIOS
step at a time for Re3200. It does not load the 55 GB input300 file into RAM.
Evaluation also accumulates metrics per block without allocating a full
reconstruction. The `bpls.txt` file records input metadata for each job.

Each run writes to `snapshots/adios-experiment/<dataset>/size-<size>/`:
`train.log`, `model_bs64_ep100k.pt` (best NRMSE),
`model_bs64_ep100k_final.pt` (latest), and `model_bs64_ep100k.json`.
The batch number in filenames follows the selected batch size.
JSON records arguments, NRMSE, bpp, and compression ratio.
These are neural-model metrics **before GAE**. By default training and evaluation
use the same dataset. Set separate `train_subset.frame_range` and
`test_subset.frame_range` for held-out measurements.

## Later C++ GAE comparison: required changes

Training does not require changing the compiler or C++. The existing C++
installation cannot yet reproduce these exact shapes: it fixes 8 frames,
256-pixel blocks, 16-pixel latents, and 4-pixel hyperlatents.
For each trained checkpoint, use a separate exported installation and make
these changes before running the comparison:

1. The exporter currently only accepts registered UFL checkpoints through
   `model_registry.read_selection`. Add a separate local-experiment checkpoint
   path to `compile_model.py`/the exporter context, and give each local
   installation an identity tied to its checkpoint hash. Do not relabel a
   trained checkpoint as an existing registered foundation model. Keep the
   compressor, hyper-decoder, decoder, and six entropy tables from the same
   checkpoint together; C++ metadata validation must also accept the local
   experiment identity.
2. In `CAESAR_compressor.py`, change the export example from
   `[8,1,8,256,256]` to `[8,1,16,P,P]`, where P is 128, 256, or 512.
   In `CAESAR_decompressor.py`, change `[10,2,64,16,16]` to
   `[10,4,64,L,L]`, where L=P/16. In `CAESAR_hyper_decompressor.py`,
   change `[8,64,4,4]` to `[8,64,Q,Q]`, where Q=P/64.
   Export each size separately: these dimensions are currently static.
3. In `CAESAR/models/caesar_compress.cpp`, permit `n_frame=16` and
   set `dataset_config.test_size={P,P}` before constructing the dataset.
   In `CAESAR/models/caesar_decompress.cpp`, permit 16 frames, replace
   hyperlatent shape/strides `64*4*4` with `64*Q*Q`, and latent
   shape/strides `64*16*16` with `64*L*L`. Replace hardcoded two-latent-frame
   batching/grouping (`*2`, `%2`, and related sample counts) with
   `n_frame/4` (=4 for these experiments). Audit all related reshape counts.
   Use matching installation shape metadata to choose P/L/Q, or build
   a separate experimental binary per size. Reduce the fixed inference batch
   of 128 if necessary for 512-pixel blocks. Update CLI frame-count checks too.
4. Run one identical held-out block/field with each installation, using the
   same relative error bound. Read BP hyperslabs into a float32 tensor for the
   library API, or write float32 raw data for the CLI. The CLI does not directly
   read BP. Its 5D path selects variable 0, so compare one field at a time.

After those changes, a CLI comparison would use:

```bash
export CAESAR_MODEL_DIR=/path/to/exported/<dataset>/size-512
build/CAESAR/caesar compress heldout.bin --shape 1,1,16,512,512 \
  --n-frame 16 --error-bound 0.001 --correction gae --output result.cae
build/CAESAR/caesar decompress result.cae --output restored.bin \
  --verify --original heldout.bin
```

That example will be rejected by the current unmodified C++ code. Record
measured error, total compressed bytes including GAE, compression ratio,
and encode/decode times at identical error bounds. GAE corrects reconstruction
toward the bound, so compare total bytes as well as final error. Training logs
alone do not measure GAE performance.
