"""
Stream a few BraTS cases out of the HuggingFace tars without downloading
the whole archive.

The dataset tars are ~32 GB each, but a smoke test only needs a handful of
cases. tarfile can read a non-seekable stream ("r|"), so we read the HTTP
response as it arrives, extract members until we have N complete cases,
then stop — which touches only the first few hundred MB of the archive.

Usage:
    python3 scripts/fetch_sample_cases.py --num_cases 6 --output_dir data/raw/brats2023

Which cases you get depends on the order they were written into the
archive, which is not sorted. That is fine for a smoke test — any N
complete cases exercise the same code paths — but it means this is not a
way to reconstruct a specific split.

Output matches what preprocess_brats.py expects:
    output_dir/
      BraTS-GLI-XXXXX-XXX/
        BraTS-GLI-XXXXX-XXX-t1c.nii.gz
        ... one file per modality, plus -seg.nii.gz
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

HF_URLS = {
    "brats2023": "https://huggingface.co/tom-ngh/brats-data/resolve/main/brats2023.tar",
    "brats2024": "https://huggingface.co/tom-ngh/brats-data/resolve/main/brats2024.tar",
}

# A complete case needs all four modalities plus the segmentation.
EXPECTED_PER_CASE = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", choices=sorted(HF_URLS), default="brats2023")
    p.add_argument("--num_cases", type=int, default=6,
                   help="Number of complete cases to pull.")
    p.add_argument("--output_dir", required=True,
                   help="Where to write the case folders.")
    p.add_argument("--url", default=None,
                   help="Override the archive URL (e.g. a local .tar path).")
    return p.parse_args()


def _case_name(member_path: str) -> str | None:
    """
    Case folder name from a tar member path.

    The archives carry a single top-level directory (run_a100.sh extracts
    them with --strip-components=1), so the case folder is the second
    component. Members that are not inside a case folder return None.
    """
    parts = [p for p in Path(member_path).parts if p not in (".", "/")]
    if len(parts) < 3:          # <top>/<case>/<file>
        return None
    # Skip AppleDouble sidecars and other dotfiles. They only appear in
    # archives built on macOS, but they would otherwise be counted as case
    # files and make an incomplete case look complete.
    if any(p.startswith("._") or p.startswith(".") for p in parts[1:]):
        return None
    return parts[1]


def main() -> None:
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    url = args.url or HF_URLS[args.dataset]
    print(f"Source     : {url}")
    print(f"Target     : {out}")
    print(f"Cases      : {args.num_cases}\n")

    if url.startswith(("http://", "https://")):
        stream = urllib.request.urlopen(url)
    else:
        stream = open(url, "rb")

    counts: dict[str, int] = {}
    order:  list[str] = []
    written = 0

    try:
        # "r|" is the streaming mode — no seeking, so it works on a socket.
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue

                case = _case_name(member.name)
                if case is None:
                    continue

                if case not in counts:
                    # Reached a case beyond the ones we want — everything
                    # earlier is already complete, so stop reading.
                    if len(order) >= args.num_cases:
                        break
                    counts[case] = 0
                    order.append(case)
                    print(f"  [{len(order)}/{args.num_cases}] {case}")

                src = tar.extractfile(member)
                if src is None:
                    continue

                dest = out / case / Path(member.name).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as fh:
                    shutil.copyfileobj(src, fh)

                counts[case] += 1
                written += 1
    finally:
        stream.close()

    complete   = [c for c in order if counts[c] >= EXPECTED_PER_CASE]
    incomplete = [c for c in order if counts[c] < EXPECTED_PER_CASE]

    print(f"\nWrote {written} files across {len(order)} case folders.")
    print(f"  complete   : {len(complete)}")
    if incomplete:
        # The last case can be cut short if the stream ended mid-case.
        print(f"  incomplete : {len(incomplete)}  {incomplete}")
        for c in incomplete:
            shutil.rmtree(out / c, ignore_errors=True)
        print("  (removed — preprocess_brats.py would skip them anyway)")

    if not complete:
        print("\nERROR: no complete cases fetched.", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. {len(complete)} usable cases in {out}")


if __name__ == "__main__":
    main()
