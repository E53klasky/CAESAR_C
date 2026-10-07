# Three datasets, three latent sizes with fixed 256×256 inputs

Submit from the CAESAR repository root:

```bash
sbatch examples/slurm/train_re3200.slurm
sbatch examples/slurm/train_hurricane.slurm
sbatch examples/slurm/train_input300.slurm
```

Each job uses hpg-b200, one GPU, six CPUs, 200 GB RAM, and the original
7:45:00 limit. It sources `~/.bashrc`, `../data/test/set_env_ufl_caesar.sh`,
and `../caesar_venv/bin/activate`. Install ADIOS2 and the training dependencies
in that environment before submitting.

## What each job trains

Each dataset launcher trains three independent models, in order 16, 32, 8.
**Every run uses 256×256 training crops and 256×256 test blocks, batch 32,
and 16 frames. Only the model's latent spatial size changes.**

| Input (all runs) | Logical latent, excluding batch | Hyperlatent per latent frame |
| --- | --- | --- |
| 16×256×256 | 64×4×16×16 | 64×4×4 |
| 16×256×256 | 64×4×32×32 | 64×8×8 |
| 16×256×256 | 64×4×8×8 | 64×2×2 |

`--latent_size 8|16|32` changes the final encoder's spatial stride to 4|2|1,
respectively, and the first decoder's matching spatial upsampling. Temporal
stride stays 2 at each of the two 3D stages, giving four latent frames from
16 inputs. The stride-1 transpose convolution is cropped by one trailing
pixel to retain its matching spatial size. Kernel weights and parameter
shapes are preserved so all variants can strictly load the same pretrained
checkpoint. The network internally flattens latent time into batch:
`q_latent` is `[batch*4,64,L,L]`.

All runs load `pretrained/model_bs64_ep100k.pt` (CAESAR v2), then fine-tune
independently. There is no weight transfer between experiments. If missing:

```bash
python model_registry.py caesar_v2 --output pretrained
```

Use `PRETRAIN=/absolute/path/to/compatible.pt` in the submission environment
to select a different checkpoint for all jobs. Other settings are identical:
learning rate 0.001, LR gamma 0.5, beta 1e-5 to 2e-5 at 75%, model dimension
16, SR dimension 16, seed 0, mean_range instance normalization, no overlap.
`--iterations 100` retains the original convention: **100,000 optimizer steps
per model**, not 100 steps or 100 epochs. Eager autograd is used because
compiled backward failed on cluster PyTorch 2.11.

## BP reading and outputs

The YAML contains explicit paths and variables from your bpls output:

| Dataset | Fields | Layout |
| --- | --- | --- |
| Re3200 | critq, pp, ux, uy, uz, vort | 10 steps of [section=513,height=513,width=513] |
| Hurricane | all 19 float fields | [time=100,height=500,width=500] per field |
| input300 | data | [section=60,time=300,height=640,width=640] |

Hurricane's depth is treated as the frame direction. Re3200 uses ADIOS steps
as time and depth as section. Re3200's 10 frames are padded to 16 by repeating
the last frame. Spatial edges are also padded. Evaluation excludes padded
values. The lazy reader reads patches and evaluates blocks without loading or
reconstructing the full 55 GB file in memory.

Outputs go to `snapshots/adios-latent256-bs32/<dataset>/latent-<L>/`:
`train.log`, `model_bs32_ep100k.pt` (best NRMSE),
`model_bs32_ep100k_final.pt` (latest evaluated model), and JSON metrics.
Each dataset also gets `bpls.txt`. Outputs from the earlier variable-input-size
experiment remain in their previous directories. Existing jobs continue using
the old settings until restarted with the updated scripts.

The JSON metrics are before GAE. Training and evaluation use the same data by
default; configure separate train/test frame ranges for held-out results.
A rerun skips sizes marked COMPLETE. Interrupted sizes restart from pretrained
weights; optimizer/scheduler resume is not implemented.

## Runtime

The old 256-input, batch-64 run measured 0.678 seconds per optimizer step:
about 18.8 hours for 100,000 steps, excluding input waits and evaluation.
**Those timings do not measure the new batch-32 latent variants.** Benchmark
these runs to estimate runtime. The original 7:45 limit may be insufficient;
request a cluster-permitted longer limit or reduce the step count consistently.
The scripts retain your requested step count and original Slurm time limit.

## Later C++ GAE comparison

The training changes do not modify deployment/C++. Current C++ fixes eight
frames and the default 16×16 latent; it cannot reproduce the 16-frame variants
yet. Keep inputs and spatial blocking at 256×256 in the later comparison too.

Required changes before that comparison:

1. Add a local-experiment checkpoint export path to `compile_model.py` and
   `model_registry.export_context`. Current exports require registered UFL
   models. Give each experimental installation a checkpoint/variant identity
   and support it in C++ metadata validation; do not relabel it as a registered
   foundation model. Keep all three pt2 packages and six probability tables
   from the same weights and architecture together.
2. Apply the same final encoder spatial stride and first decoder upsampling
   used by the training variant in all three deployment model scripts. The
   decoder must include the stride-1 spatial crop for L=32. Shape changes in
   export examples alone are insufficient.
3. Export compressor example `[8,1,16,256,256]`, decoder
   `[10,4,64,L,L]`, and hyper-decoder `[8,64,Q,Q]` with Q=L/4.
   Export each L into a separate installation; these dimensions are static.
4. In `CAESAR/models/caesar_compress.cpp`, support n_frame=16 and keep
   `dataset_config.test_size={256,256}`. In `caesar_decompress.cpp`, support
   16 frames, use latent shapes/strides `64*L*L` and hyperlatent shapes/strides
   `64*Q*Q`, and change all two-latent-frame batching/grouping/counts to
   `n_frame/4` (four for these experiments). Update CLI frame checks too.
   Read variant dimensions from installation metadata, or use a separate
   experimental binary per variant. Benchmark inference memory requirements.
5. Compare identical held-out float32 fields, error bounds, and GAE settings.
   Read BP hyperslabs into tensors for the library API, or convert to raw
   float32 for the CLI (it does not directly read BP). Compare one variable
   at a time; the C++ 5D path selects variable index zero.

After implementing those changes, the CLI command would be:

```bash
export CAESAR_MODEL_DIR=/path/to/exported/<dataset>/latent-32
build/CAESAR/caesar compress heldout.bin --shape 1,1,16,256,256 \
  --n-frame 16 --error-bound 0.001 --correction gae --output result.cae
build/CAESAR/caesar decompress result.cae --output restored.bin \
  --verify --original heldout.bin
```

The current C++ rejects that frame count. Record measured error, total bytes
including GAE, compression ratio, and encode/decode times. GAE corrects toward
the requested bound, so compare total bytes as well as final error.
