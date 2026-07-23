"""
Extract fixed ROI crops from preprocessed BraTS data.

Takes the output of preprocess_brats.py (full 240×240×155 volumes) and
saves a fixed 96³ crop centered on the tumor bounding-box centroid for
every case. Fixed crops enable latent caching — the VAE only needs to run
once, not once per training step.

Input layout (output of preprocess_brats.py):
    data_root/
      vol/{name}_vol.npy    (H, W, D, 4) float32  channel-last
      seg/{name}_seg.npy    (H, W, D)    uint8     labels {0,1,2,4}
      train.txt / val.txt / test.txt

Output layout (same naming, new directory):
    output_dir/
      vol/{name}_vol.npy    (96, 96, 96, 4) float16  (cast to float32 on load)
      seg/{name}_seg.npy    (96, 96, 96)    uint8
      train.txt / val.txt / test.txt        (copied from data_root)
      roi_coords.json                       (crop metadata per case)

When tumor bounding box > 96 in any dim:
    The centroid-centred 96³ window is kept — it maximises coverage by
    definition. Coverage < threshold cases are flagged in roi_coords.json
    so you can inspect or exclude them later.

Usage:
    python3 scripts/preprocess_roi.py \\
        --data_root  data/brats_combined \\
        --output_dir data/brats_roi96 \\
        --crop_size  96 \\
        --coverage_warn 0.9
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bbox_centroid(seg: np.ndarray) -> tuple[int, int, int] | None:
    """
    Bounding-box centre of all non-zero voxels.
    Returns None when the volume has no tumour label.
    """
    coords = np.where(seg > 0)
    if len(coords[0]) == 0:
        return None
    ch = int((coords[0].min() + coords[0].max()) // 2)
    cw = int((coords[1].min() + coords[1].max()) // 2)
    cd = int((coords[2].min() + coords[2].max()) // 2)
    return ch, cw, cd


def crop_start(center: int, crop: int, size: int) -> int:
    """
    Start index for a window of `crop` voxels centered on `center`,
    clamped so the window stays within [0, size).
    """
    start = center - crop // 2
    return max(0, min(start, size - crop))


def pad_to_crop(arr: np.ndarray, crop: int, spatial_axes: tuple) -> np.ndarray:
    """
    Zero-pad `arr` along each axis in `spatial_axes` if smaller than `crop`.
    Used when (rarely) a BraTS volume is smaller than the requested crop.
    """
    pad = [(0, 0)] * arr.ndim
    for ax in spatial_axes:
        deficit = crop - arr.shape[ax]
        if deficit > 0:
            pad[ax] = (deficit // 2, deficit - deficit // 2)
    if any(p != (0, 0) for p in pad):
        arr = np.pad(arr, pad, mode="constant", constant_values=0)
    return arr


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

ET_LABELS = {3, 4}   # BraTS raw: 4 = ET; some remapped datasets use 3

def _et_label(seg: np.ndarray) -> int:
    return 4 if np.any(seg == 4) else 3


def region_masks(seg: np.ndarray) -> dict[str, np.ndarray]:
    et = _et_label(seg)
    return {
        "WT": seg > 0,
        "TC": (seg == 1) | (seg == et),
        "ET": seg == et,
    }


def compute_coverage(
    seg_full: np.ndarray,
    h0: int, w0: int, d0: int,
    crop: int,
) -> dict[str, float]:
    """
    Fraction of each tumour region that falls inside the crop window.
    Regions with zero voxels in the full volume report 1.0 (nothing lost).
    """
    seg_crop = seg_full[h0:h0+crop, w0:w0+crop, d0:d0+crop]
    full_masks = region_masks(seg_full)
    crop_masks = region_masks(seg_crop)

    cov: dict[str, float] = {}
    for r in ("WT", "TC", "ET"):
        total = int(full_masks[r].sum())
        cov[r] = 1.0 if total == 0 else float(crop_masks[r].sum()) / total
    return cov


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_root",     required=True,
                   help="Preprocessed data folder (output of preprocess_brats.py)")
    p.add_argument("--output_dir",    required=True,
                   help="Where to write ROI-cropped data")
    p.add_argument("--crop_size",     type=int, default=96)
    p.add_argument("--coverage_warn", type=float, default=0.9,
                   help="Warn when WT coverage < this value (default 0.9 = 90%%)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    src     = Path(args.data_root)
    dst     = Path(args.output_dir)
    crop    = args.crop_size

    vol_src = src / "vol"
    seg_src = src / "seg"
    vol_dst = dst / "vol"
    seg_dst = dst / "seg"
    vol_dst.mkdir(parents=True, exist_ok=True)
    seg_dst.mkdir(parents=True, exist_ok=True)

    # Discover all cases in the source directory
    cases = sorted(p.name.replace("_vol.npy", "") for p in vol_src.glob("*_vol.npy"))
    if not cases:
        raise FileNotFoundError(f"No *_vol.npy files found in {vol_src}")
    print(f"Found {len(cases)} cases in {vol_src}")
    print(f"Crop size  : {crop}³")
    print(f"Output     : {dst}\n")

    roi_coords: dict = {}
    n_partial   = 0
    n_no_tumour = 0
    coverage_log: list[tuple[float, str]] = []  # (WT coverage, name)

    def process_case(
                        args
                    ):
        (name,
        vol_src,
        seg_src,
        vol_dst,
        seg_dst,
        crop,
        coverage_warn,
        roi_coords) = args
        # ---------------- Load ----------------
        vol = np.asarray(np.load(vol_src / f"{name}_vol.npy", mmap_mode="r"))
        seg = np.asarray(np.load(seg_src / f"{name}_seg.npy", mmap_mode="r"))

        # ---------------- Pad -----------------
        vol = pad_to_crop(vol, crop, spatial_axes=(0, 1, 2))
        seg = pad_to_crop(seg, crop, spatial_axes=(0, 1, 2))

        H, W, D = seg.shape

        # ---------------- Centroid ----------------
        centroid = bbox_centroid(seg)
        no_tumour = False

        if centroid is None:
            centroid = (H // 2, W // 2, D // 2)
            no_tumour = True

        ch, cw, cd = centroid

        # ---------------- Crop ----------------
        h0 = crop_start(ch, crop, H)
        w0 = crop_start(cw, crop, W)
        d0 = crop_start(cd, crop, D)

        vol_crop = vol[h0:h0+crop, w0:w0+crop, d0:d0+crop, :]
        seg_crop = seg[h0:h0+crop, w0:w0+crop, d0:d0+crop]

        vol_crop = np.interp(
            np.clip(vol_crop, -5, 5),
            (-5, 5),
            (0, 255)
        )

        vol_crop = np.ascontiguousarray(vol_crop.astype(np.uint8))
        seg_crop = np.ascontiguousarray(seg_crop.astype(np.uint8))
        # ------------------------------------------------------------------
        # Coverage stats
        # ------------------------------------------------------------------
        cov      = compute_coverage(seg, h0, w0, d0, crop)
        partial  = cov["WT"] < args.coverage_warn
        if partial:
            n_partial += 1
            tqdm.write(
                f"  [partial] {name}  "
                f"WT={cov['WT']:.2%}  TC={cov['TC']:.2%}  ET={cov['ET']:.2%}"
            )

        roi_coords[name] = {
            "crop_start":   [int(h0), int(w0), int(d0)],
            "volume_shape": [int(H),  int(W),  int(D)],
            "centroid":     [int(ch), int(cw), int(cd)],
            "coverage":     {r: round(v, 4) for r, v in cov.items()},
            "partial":      bool(partial),
        }
        coverage_log.append((cov["WT"], name))
        # ------------------------------------------------------------------
        # Save (keep channel-last for BraTSDataset compatibility)
        # ------------------------------------------------------------------
        np.save(vol_dst / f"{name}_vol.npy", vol_crop)
        np.save(seg_dst / f"{name}_seg.npy", seg_crop)

        return (
            name,
            {
                "crop_start": [int(h0), int(w0), int(d0)],
                "volume_shape": [int(H), int(W), int(D)],
                "centroid": [int(ch), int(cw), int(cd)],
                "coverage": {r: round(v, 4) for r, v in cov.items()},
                "partial": partial,
            },
            cov["WT"],
            partial,
            no_tumour,
            cov,
        )
    
    tasks = [
            (
                name,
                vol_src,
                seg_src,
                vol_dst,
                seg_dst,
                crop,
                args.coverage_warn,
                roi_coords
            )
            for name in cases
            ]
    
    with Pool(8) as pool:
        results = list(
            tqdm(
                pool.starmap(process_case, tasks),
                total=len(tasks),
                desc="ROI crop",
                unit="case",
            )
        )

    # for name in tqdm(cases, desc="ROI crop", unit="case"):
    #     # ------------------------------------------------------------------
    #     # Load (mmap avoids full allocation until sliced)
    #     # ------------------------------------------------------------------
    #     vol = np.asarray(np.load(vol_src / f"{name}_vol.npy", mmap_mode="r"))
    #     seg = np.asarray(np.load(seg_src / f"{name}_seg.npy", mmap_mode="r"))
    #     # vol: (H, W, D, 4)  seg: (H, W, D)

    #     # ------------------------------------------------------------------
    #     # Pad if any spatial dim < crop (extremely rare for BraTS 240×240×155)
    #     # ------------------------------------------------------------------
    #     vol = pad_to_crop(vol, crop, spatial_axes=(0, 1, 2))
    #     seg = pad_to_crop(seg, crop, spatial_axes=(0, 1, 2))

    #     H, W, D = seg.shape

    #     # ------------------------------------------------------------------
    #     # Find tumour centroid
    #     # ------------------------------------------------------------------
    #     centroid = bbox_centroid(seg)
    #     if centroid is None:
    #         tqdm.write(f"  [warn] no tumour in {name} — using volume centre")
    #         centroid = (H // 2, W // 2, D // 2)
    #         n_no_tumour += 1

    #     ch, cw, cd = centroid

    #     # ------------------------------------------------------------------
    #     # Compute crop window
    #     # ------------------------------------------------------------------
    #     h0 = crop_start(ch, crop, H)
    #     w0 = crop_start(cw, crop, W)
    #     d0 = crop_start(cd, crop, D)

    #     # ------------------------------------------------------------------
    #     # Slice
    #     # ------------------------------------------------------------------
    #     vol_crop = vol[h0:h0+crop, w0:w0+crop, d0:d0+crop, :]   # (96,96,96,4)
    #     seg_crop = seg[h0:h0+crop, w0:w0+crop, d0:d0+crop]       # (96,96,96)
        
    #     vol_crop = np.interp(vol_crop, (vol_crop.min(), vol_crop.max()), (0, 1)) * 255
        
    #     vol_crop = np.ascontiguousarray(vol_crop.astype(int), dtype=np.uint8)
    #     seg_crop = np.ascontiguousarray(seg_crop, dtype=np.uint8)

    #     # ------------------------------------------------------------------
    #     # Coverage stats
    #     # ------------------------------------------------------------------
    #     cov      = compute_coverage(seg, h0, w0, d0, crop)
    #     partial  = cov["WT"] < args.coverage_warn
    #     if partial:
    #         n_partial += 1
    #         tqdm.write(
    #             f"  [partial] {name}  "
    #             f"WT={cov['WT']:.2%}  TC={cov['TC']:.2%}  ET={cov['ET']:.2%}"
    #         )

    #     roi_coords[name] = {
    #         "crop_start":   [int(h0), int(w0), int(d0)],
    #         "volume_shape": [int(H),  int(W),  int(D)],
    #         "centroid":     [int(ch), int(cw), int(cd)],
    #         "coverage":     {r: round(v, 4) for r, v in cov.items()},
    #         "partial":      bool(partial),
    #     }
    #     coverage_log.append((cov["WT"], name))
    #     # ------------------------------------------------------------------
    #     # Save (keep channel-last for BraTSDataset compatibility)
    #     # ------------------------------------------------------------------
    #     np.save(vol_dst / f"{name}_vol.npy", vol_crop)
    #     np.save(seg_dst / f"{name}_seg.npy", seg_crop)

    # ----------------------------------------------------------------------
    # Copy split files (train.txt / val.txt / test.txt)
    # ----------------------------------------------------------------------
    for split in ("train.txt", "val.txt", "test.txt"):
        src_split = src / split
        if src_split.exists():
            shutil.copy(src_split, dst / split)
        else:
            tqdm.write(f"  [warn] {split} not found in {src}, skipping copy")

    # ----------------------------------------------------------------------
    # Save roi_coords.json
    # ----------------------------------------------------------------------
    coords_path = dst / "roi_coords.json"
    with open(coords_path, "w") as f:
        json.dump(roi_coords, f, indent=2)

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    n_total = len(cases)
    n_full  = n_total - n_partial - n_no_tumour

    coverage_log.sort()   # ascending WT coverage → worst cases first

    print(f"\n{'='*60}")
    print(f"ROI Crop Summary  (crop={crop}³,  warn_threshold={args.coverage_warn:.0%})")
    print(f"{'='*60}")
    print(f"  Total cases      : {n_total}")
    print(f"  Full coverage    : {n_full}  ({100*n_full/n_total:.1f}%)")
    print(f"  Partial coverage : {n_partial}  ({100*n_partial/n_total:.1f}%)")
    if n_no_tumour:
        print(f"  No tumour (warn) : {n_no_tumour}")

    if coverage_log:
        print(f"\n  Worst WT coverage (bottom 5):")
        for wt_cov, name in coverage_log[:5]:
            e = roi_coords[name]
            print(f"    {name}  "
                  f"WT={wt_cov:.2%}  "
                  f"TC={e['coverage']['TC']:.2%}  "
                  f"ET={e['coverage']['ET']:.2%}")

    print(f"\n  Output → {dst}/")
    print(f"    vol/            {n_total} files")
    print(f"    seg/            {n_total} files")
    print(f"    roi_coords.json")
    for split in ("train.txt", "val.txt", "test.txt"):
        if (dst / split).exists():
            print(f"    {split}")


if __name__ == "__main__":
    main()
