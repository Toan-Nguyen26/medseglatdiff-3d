"""
Create train/val/test split files for BraTS processed data.

Discovers case names from data_root/vol/*.npy and writes split txt files
to a SEPARATE output directory — the processed data folder is never touched.

Usage:
    # Full 70/20/10 split
    python3 scripts/resplit_data.py \\
        --data_root data/brats2023_processed \\
        --output_dir splits/brats2023_full

    # 5% smoke-test split
    python3 scripts/resplit_data.py \\
        --data_root data/brats2023_processed \\
        --output_dir splits/brats2023_smoke \\
        --subset_frac 0.05

Then point training scripts at the split dir:
    python3 -m training.train_image_vae \\
        --data_root data/brats2023_processed \\
        --splits_dir splits/brats2023_smoke
"""

import argparse
import random
from pathlib import Path


def discover_cases(data_root: Path) -> list[str]:
    vol_dir = data_root / "vol"
    if not vol_dir.exists():
        raise FileNotFoundError(f"Expected vol/ directory under {data_root}")
    names = sorted(p.name.replace("_vol.npy", "") for p in vol_dir.glob("*_vol.npy"))
    if not names:
        raise ValueError(f"No *_vol.npy files found in {vol_dir}")
    return names


def resplit(
    data_root: Path,
    output_dir: Path,
    seed: int,
    train_frac: float,
    val_frac: float,
    subset_frac: float,
) -> None:
    all_names = discover_cases(data_root)
    print(f"Found {len(all_names)} cases in {data_root / 'vol'}")

    random.seed(seed)
    random.shuffle(all_names)

    if subset_frac < 1.0:
        n_keep = max(3, round(len(all_names) * subset_frac))
        all_names = all_names[:n_keep]
        print(f"Keeping {n_keep} cases ({subset_frac:.0%} subset)")

    n = len(all_names)
    n_train = round(n * train_frac)
    n_val   = round(n * val_frac)

    train_names = all_names[:n_train]
    val_names   = all_names[n_train : n_train + n_val]
    test_names  = all_names[n_train + n_val :]

    output_dir.mkdir(parents=True, exist_ok=True)
    for fname, names in [("train.txt", train_names), ("val.txt", val_names), ("test.txt", test_names)]:
        (output_dir / fname).write_text("\n".join(names) + "\n")
        print(f"  {fname}: {len(names)} cases")

    print(f"\n{n} cases → {len(train_names)}/{len(val_names)}/{len(test_names)} "
          f"({len(train_names)/n:.0%}/{len(val_names)/n:.0%}/{len(test_names)/n:.0%})")
    print(f"Split files written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root",  required=True,
                        help="Processed data folder (contains vol/ and seg/). Never modified.")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write train.txt / val.txt / test.txt")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--train_frac",  type=float, default=0.70)
    parser.add_argument("--val_frac",    type=float, default=0.20)
    parser.add_argument("--subset_frac", type=float, default=1.0,
                        help="Keep only this fraction of all cases before splitting. "
                             "0.05 = 5%% smoke test.")
    args = parser.parse_args()

    resplit(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        subset_frac=args.subset_frac,
    )


if __name__ == "__main__":
    main()
