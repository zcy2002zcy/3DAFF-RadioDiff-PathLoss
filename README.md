# 3DAFF-RadioDiff: Path-Loss Reconstruction

Clean research implementation of 3DAFF-RadioDiff for path-loss radio-map
reconstruction. This repository contains the two 10% measurement protocols used
for comparison: random sampling and uniform sampling. Both use adaptive LoS-depth
fusion (AFF) and a 200-step DDPM trained and evaluated with a near-Gaussian
terminal state.

The repository intentionally excludes datasets, checkpoints, logs, and generated
outputs so that it can be published directly on GitHub.

## Protocol

- Target: normalized path-loss slice `psi`, with range `[-1, 1]`.
- Geometry: depth map and LoS mask fused by sample-wise AFF weights.
- Other conditions: transmitter/base-station map `q`, receiver altitude `z`, and
  a sparse measurement map `s`.
- Sampling: either random 10% or uniform 10%.
- Diffusion: 200 linear steps, `beta_start=0.0005`, `beta_end=0.10`.
- Training objective: noise-prediction MSE with a random diffusion timestep.
- Evaluation: reverse DDPM sampling from independent `N(0, I)` noise.
- Metrics: NMSE, RMSE, SSIM, and PSNR (`data_range=2`).
- Measurement consistency: predictions at observed positions are replaced by the
  corresponding authorized sparse measurements before the main metrics.

The complete test target is not passed to the network. It is read to construct the
authorized 10% observation map and to compute metrics. The unobserved 90% remains
hidden from the model during evaluation.

## Repository Layout

```text
configs/       random and uniform experiment configurations
datasets/      NPZ loading and sparse-measurement construction
diffusion/     beta schedule, forward process, loss, and DDPM sampler
models/        AFF, residual U-Net, Transformer, and embeddings
scripts/       experiment check, training, and full-noise evaluation
trainers/      shared training/evaluation helpers
utils/         metrics, logging, I/O, seeds, and visualization
```

## Data

Each NPZ sample must contain:

```text
q, depth, los, psi, z, id
```

The default Windows paths are:

```text
D:\datas\processed_pl_aff_dataset\train
D:\datas\processed_pl_aff_dataset\val
D:\datas\processed_pl_aff_dataset\test
```

Change `train_dir`, `val_dir`, and `test_dir` in both YAML files when using a
different location.

## Environment

```powershell
conda activate rmdm5070
pip install -r requirements.txt
cd "C:\Users\Administrator\Desktop\3DAFF-RadioDiff-PathLoss"
```

The launchers first try `D:\Anaconda3\envs\rmdm5070\python.exe` and otherwise use
the active environment's `python` command.

## Random 10%

Check the complete pipeline before training:

```powershell
.\run_check.ps1 -Mode random -Gpu 0
```

Train a dedicated random-sampling checkpoint:

```powershell
.\run_train.ps1 -Mode random -Gpu 0
```

Evaluate 1,000 samples as a quick test, then all test samples:

```powershell
.\run_eval.ps1 -Mode random -Gpu 0 -MaxSamples 1000
.\run_eval.ps1 -Mode random -Gpu 0
```

## Uniform 10%

```powershell
.\run_check.ps1 -Mode uniform -Gpu 0
.\run_train.ps1 -Mode uniform -Gpu 0
.\run_eval.ps1 -Mode uniform -Gpu 0 -MaxSamples 1000
.\run_eval.ps1 -Mode uniform -Gpu 0
```

Random and uniform sampling use separate checkpoints and must be trained
independently. Change `-Gpu` to the desired physical GPU index.

## Outputs

Random results are written under:

```text
outputs_pl_aff_random10_fullnoise/
```

Uniform results are written under:

```text
outputs_pl_aff_uniform10_fullnoise_200/
```

Each full evaluation writes `metrics_summary.csv`, `metrics_per_sample.csv`, and
the first ten prediction archives. Use `-NoConsistency` only for an explicitly
reported ablation; do not mix those results with the default protocol.

## Reproducibility Notes

- The evaluation seed defaults to 42 and can be changed with `-Seed`.
- Test sampling is deterministic for both protocols.
- Random training masks are resampled, while uniform grid positions are fixed.
- The reported random and uniform results should use the same test set, metric
  implementation, diffusion schedule, and measurement-consistency policy.
