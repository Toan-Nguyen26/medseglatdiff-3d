"""
Convert raw BraTS 2023 GLI .nii.gz files to .npy for fast training I/O.

Processes both the segmentation mask AND all 4 MRI modalities in one pass,
so this never needs to be re-run as the pipeline grows.

Output layout:
    output_dir/
      seg/{case_name}_seg.npy     # (240, 240, 155) uint8, labels 0-3
      vol/{case_name}_vol.npy     # (240, 240, 155, 4) float32, z-score normalised
      train.txt / val.txt / test.txt

Modality channel order (index 0-3 in vol):
    0: FLAIR  (t2f)
    1: T1ce   (t1c)
    2: T1     (t1n)
    3: T2     (t2w)

Z-score normalisation: per-modality, per-case, computed on non-zero voxels
only (brain mask excludes background air). Standard practice for BraTS.

Usage:
    python3 scripts/preprocess_brats.py \\
        --data_root medseglatdiff-3d/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \\
        --output_dir medseglatdiff-3d/data/brats2023_processed
"""

import argparse
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import multiprocessing 
from multiprocessing import Pool, cpu_count
import tqdm

MODALITY_SUFFIXES = ["t2f", "t1c", "t1n", "t2w"]  # → channels 0,1,2,3 (FLAIR,T1ce,T1,T2)


def zscore_normalise(volume: np.ndarray) -> np.ndarray:
    """Z-score per modality using only non-zero (brain) voxels."""
    mask = volume > 0
    if mask.sum() == 0:
        return volume.astype(np.float32)
    mean = volume[mask].mean()
    std = volume[mask].std()
    if std < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)
    out = np.zeros_like(volume, dtype=np.float32)
    out[mask] = (volume[mask] - mean) / std
    return out


def process_case(args) -> bool:
    (case_dir, seg_out, vol_out) = args
    name = case_dir.name

    seg_path = case_dir / f"{name}-seg.nii.gz"
    mod_paths = [case_dir / f"{name}-{s}.nii.gz" for s in MODALITY_SUFFIXES]

    if not seg_path.exists():
        print(f"  SKIP {name}: missing seg file")
        return None
    for p in mod_paths:
        if not p.exists():
            print(f"  SKIP {name}: missing {p.name}")
            return None

    seg_arr = nib.load(seg_path).get_fdata().astype(np.uint8)
    np.save(seg_out / f"{name}_seg.npy", seg_arr)

    modalities = []
    for p in mod_paths:
        arr = nib.load(p).get_fdata().astype(np.float32)
        modalities.append(zscore_normalise(arr))
    vol_arr = np.stack(modalities, axis=-1)  # (H, W, D, 4)
    np.save(vol_out / f"{name}_vol.npy", vol_arr)
    return name


def make_splits(names: list[str], val_frac: float, test_frac: float, seed: int) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = names[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder containing BraTS-GLI-* subfolders")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write seg/, vol/, and split .txt files")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.output_dir)
    seg_out = out / "seg"
    vol_out = out / "vol"
    seg_out.mkdir(parents=True, exist_ok=True)
    vol_out.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    tasks = [(case_dir, seg_out, vol_out) for case_dir in case_dirs]
    print(f"Found {len(case_dirs)} cases in {data_root}")
    with multiprocessing.Pool(8) as pool:
        done = list(tqdm.tqdm(pool.imap(process_case, tasks), total=len(tasks), desc="Processing cases"))
    
    # for i, case_dir in enumerate(case_dirs):
    #     print(f"[{i+1}/{len(case_dirs)}] {case_dir.name}", end=" ... ", flush=True)
    #     ok = process_case(case_dir, seg_out, vol_out)
    #     if ok:
    #         done.append(case_dir.name)
    #         print("done")
    done = [x for x in done if x is not None]

    train, val, test = make_splits(done, args.val_frac, args.test_frac, args.seed)
    (out / "train.txt").write_text("\n".join(train))
    (out / "val.txt").write_text("\n".join(val))
    (out / "test.txt").write_text("\n".join(test))

    print(f"\nDone. {len(done)} cases processed.")
    print(f"  train: {len(train)}  val: {len(val)}  test: {len(test)}")
    print(f"  Output: {out}")


if __name__ == "__main__":
    main()
