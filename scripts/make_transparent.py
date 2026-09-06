"""
Make the background of an existing figure PNG transparent, in place of a re-run.

Regenerating a combo grid means re-running inference, because the sample
stacks are not persisted. This works on the saved PNG instead.

The background is found by flood fill from the image border, not by matching
colour globally: only near-white pixels reachable from the edge without
crossing a panel are cleared. Bright voxels inside a panel are enclosed by
darker ones, so the fill never reaches them and the panels are left untouched.

Usage:
    python3 scripts/make_transparent.py figure.png
    python3 scripts/make_transparent.py figure.png --out clear.png --tol 12

Writes <name>_transparent.png next to the input unless --out is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("image", help="PNG to process.")
    p.add_argument("--out", default=None,
                   help="Output path. Default: <name>_transparent.png")
    p.add_argument("--tol", type=int, default=10,
                   help="How far from pure white still counts as background, "
                        "per channel (0-255). Default 10.")
    p.add_argument("--keep_text", action="store_true",
                   help="Do not clear the halo around dark text. Use if the "
                        "title or labels come out with ragged edges.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.image)
    if not src.exists():
        raise SystemExit(f"Not found: {src}")

    img = Image.open(src).convert("RGBA")
    a = np.array(img)
    rgb = a[..., :3].astype(np.int16)

    # Candidate background: near-white anywhere in the image.
    near_white = (np.abs(rgb - 255).max(axis=2) <= args.tol)

    # Keep only the components of that mask which touch the border. This is
    # what separates the margin and the gaps between panels from any white
    # inside a panel -- the latter is enclosed and never reaches an edge.
    lab, n = ndimage.label(near_white)
    if n == 0:
        print("No near-white background found; nothing to do.")
        return
    edge_labels = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    edge_labels.discard(0)
    background = np.isin(lab, list(edge_labels))

    if not args.keep_text:
        # Antialiased text leaves a light-grey fringe just outside the letters
        # that the near-white test misses, which reads as a dirty halo once the
        # page behind it shows through. Clear the fringe only where it is
        # adjacent to background already being cleared.
        light = (np.abs(rgb - 255).max(axis=2) <= 60)
        grown = ndimage.binary_dilation(background, iterations=2) & light
        background = background | grown

    a[..., 3] = np.where(background, 0, 255)

    out = Path(args.out) if args.out else src.with_name(f"{src.stem}_transparent.png")
    Image.fromarray(a, mode="RGBA").save(out)

    pct = 100.0 * background.mean()
    print(f"  {src.name}: {pct:.1f}% of pixels cleared")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
