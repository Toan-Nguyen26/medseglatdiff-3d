"""
Quick utility: count subfolders in a directory (e.g. BraTS case folders like
BraTS-GLI-00001-000) to sanity-check how many cases you actually downloaded.

Usage:
    python3 scripts/count_cases.py /path/to/BraTS-GLI/training_data
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=str, help="Path to the folder to inspect")
    parser.add_argument("--pattern", type=str, default="*", help="Glob pattern to filter folder names, e.g. 'BraTS-GLI-*'")
    args = parser.parse_args()

    root = Path(args.folder)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    subfolders = sorted(p for p in root.glob(args.pattern) if p.is_dir())

    print(f"Folder: {root}")
    print(f"Subfolders found: {len(subfolders)}")
    if subfolders:
        print(f"First: {subfolders[0].name}")
        print(f"Last:  {subfolders[-1].name}")


if __name__ == "__main__":
    main()
