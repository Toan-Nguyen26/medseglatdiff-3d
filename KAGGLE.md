# Running on Kaggle

How to run inference on a Kaggle notebook when you have the checkpoints but
not the training data or the original split.

**The data is handled in code.** `run_smoke.sh` streams the cases it needs
straight out of the HuggingFace tars, so the only thing you upload is the
checkpoints.

---

## What this produces

`run_smoke.sh` is a **crash test**, not an evaluation. It pulls cases in the
order they were written into the archive, which is not sorted and not the
held-out split — so most of the cases it picks were in the training set, and
the Dice scores it reports are inflated.

Use it to confirm the pipeline runs. Do not put its numbers in a paper.

Producing valid numbers needs the original test split reconstructed
(`resplit_data.py` is deterministic — `sorted()` names plus `seed=42` — so it
can be rebuilt from the full case-name list, but that is separate work).

---

## 1. Upload the checkpoints as a Dataset

Checkpoints are gitignored, so they cannot come from the repo.

On kaggle.com: **Datasets → New Dataset**, drag in your `cache/` folder, name
it `medseg-checkpoints`.

About 436 MB:

```
cache/image_vae/final.pth              63 MB
cache/mask_vae/best.pth                15 MB
cache/latent_diffusion/step_150000.pth 358 MB
```

Do this once. Every session mounts it read-only at
`/kaggle/input/medseg-checkpoints/`.

---

## 2. Notebook settings

**Code → New Notebook**, then in the right-hand panel:

| Setting | Value |
|---|---|
| Accelerator | `GPU T4 x2` (or P100) |
| Internet | **On** |
| Add Input | `medseg-checkpoints` |

**Internet must be on** — both the `git clone` and the HuggingFace fetch need
it. Kaggle requires phone verification to enable it; do that first, or
nothing below works.

---

## 3. Cells

Clone the branch:

```python
!git clone -q -b ddp https://github.com/Toan-Nguyen26/medseglatdiff-3d.git /kaggle/working/repo
%cd /kaggle/working/repo
```

Install what Kaggle is missing. It already has torch, numpy, scipy,
matplotlib, tqdm, scikit-image and pyyaml:

```python
!pip install -q monai nibabel einops timm
```

Check the upload survived with its folder structure intact:

```python
!ls -R /kaggle/input/medseg-checkpoints/ | head -20
```

Run it:

```python
%%bash
export IMAGE_VAE_CKPT=/kaggle/input/medseg-checkpoints/image_vae/final.pth
export MASK_VAE_CKPT=/kaggle/input/medseg-checkpoints/mask_vae/best.pth
export DIFF_CKPT=/kaggle/input/medseg-checkpoints/latent_diffusion/step_150000.pth
export NUM_CASES=6
export DEVICE=cuda
bash run_smoke.sh
```

Setting the three paths explicitly rather than `CKPT_SRC` means it works
regardless of how Kaggle nested things during upload.

Once that passes, scale up to a real sweep:

```python
%%bash
export IMAGE_VAE_CKPT=/kaggle/input/medseg-checkpoints/image_vae/final.pth
export MASK_VAE_CKPT=/kaggle/input/medseg-checkpoints/mask_vae/best.pth
export DIFF_CKPT=/kaggle/input/medseg-checkpoints/latent_diffusion/step_150000.pth
export NUM_CASES=50 ALL_COMBOS=1 N_SAMPLES=5 INFER_STEPS=50 DEVICE=cuda
bash run_smoke.sh
```

Results land in `eval_output/smoke/`, downloadable from the notebook's Output
tab.

---

## 4. Disk — the constraint you will actually hit

`/kaggle/working` is 20 GB. Measured per case:

| | size |
|---|---|
| raw `.nii.gz` | 9.0 MB |
| full-res `.npy` (float32) | **142.8 MB** |
| 128³ crop (uint8) | 10.5 MB |

The full-res intermediate dominates. `run_smoke.sh` deletes it after
cropping, but only once both preprocessing stages finish, so it sets the peak:

| cases | peak disk | |
|---|---|---|
| 6 | 1.0 GB | fine |
| 20 | 3.2 GB | fine |
| 50 | 8.1 GB | fine |
| 100 | 16.2 GB | tight |
| 250 | 40.6 GB | **exceeds 20 GB** |

**~50 cases is the practical ceiling per session.**

Going higher needs a one-pass `.nii.gz → crop` preprocessing path that never
materialises the full-res array. That is the same 143 MB/case blowup that
took the H200's disk from 65 GB to 1 TB.

---

## 5. Time

Kaggle gives 30 GPU-hours/week, 12h maximum per session.

Measured on MPS: **0.45 s** per (case × sample × DDIM step). Scaling that by
an assumed T4 speedup:

| run | est. T4 time |
|---|---|
| 6 cases × 1 combo (smoke) | a few minutes |
| 20 cases × 15 combos | ~1–2 h |
| 50 cases × 15 combos | ~3–6 h |

Only the smoke row is trustworthy. The others scale a measured MPS rate by an
estimated ratio — run the small one first and extrapolate from the real
number.

The disk ceiling (~50 cases) and the session ceiling (~3–6 h at 50 cases)
land in the same place, so **50 is the number to aim at**.

---

## Knobs

| Variable | Default | |
|---|---|---|
| `NUM_CASES` | 6 | cases to fetch |
| `DATASET` | `brats2023` | or `brats2024` |
| `ALL_COMBOS` | 0 | `1` sweeps all 15 modality combinations |
| `N_SAMPLES` | 2 | DDIM chains per case |
| `INFER_STEPS` | 10 | DDIM steps per chain |
| `DEVICE` | auto | `cuda` / `mps` / `cpu` |
| `OUT_DIR` | `eval_output/smoke` | |

Re-running skips the fetch and preprocessing if `data/smoke/` already has
them, so iterating on the inference step is fast.

---

## Troubleshooting

**`git clone` hangs or fails** — Internet is off in notebook settings, or the
account is not phone-verified.

**`MISSING <path>`** — the checkpoint paths are wrong. Run the `ls -R` cell
and correct the three exports; Kaggle sometimes adds a nesting level.

**`No space left on device`** — too many `NUM_CASES`. See the disk table.

**`No cases found`** — the fetch got no complete cases. Re-run; if it
persists, the archive layout may have changed and `_case_name()` in
`scripts/fetch_sample_cases.py` needs updating.
