# Data-efficiency (two-stage training)

Two-stage AMOR training on the large nonlinear diffusion set
`nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3_big.npz` (generate with
`AMOR/data/generate_nonlinear_diffusion.py --preset medium --num_samples 20000 --suffix big`).

| Script | Role |
|---|---|
| `train_baseline.py` | One-stage AMOR on the full train split |
| `train_stage1.py` | Stage 1 in PCA coefficient space |
| `train_stage2.py` | Stage 2 fine-tune (`--version` variants; see `--help`) |
| `train_two_stage.py` | Combined driver |

Paths default through `AMOR/paths.py` (`AMOR_SCRATCH_DATA`). The scripts put the repository root on `sys.path` themselves, so they run from any working directory.

Optimizer defaults follow the main nonlinear trainers (AdamW, cosine schedule, relative \(L^1\)). Exact flags: each script’s `--help`.
