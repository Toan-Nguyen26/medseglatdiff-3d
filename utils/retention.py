"""
Bounded retention for periodic training artefacts.

Long runs write one checkpoint every `ckpt_every` steps and one
visualisation every `val_every` steps. At the scale these runs reach —
~200k steps for the ImageVAE, 1M for the diffusion UNet — that is a few
hundred .pth files and a few thousand PNGs per run, most of which nobody
ever opens.

`prune_oldest` keeps a rolling window of the N newest and deletes the
rest, so a run's artefact count is bounded no matter how long it runs.

The rolling window keeps recent history and discards early history. That
is the right trade for these files: periodic checkpoints exist to resume
from, and you resume from the latest one. Checkpoints you actually care
about keeping — best.pth and final.pth — are written under fixed names
that no glob here matches, so they are never candidates for deletion.
"""

from __future__ import annotations

from pathlib import Path


def prune_oldest(directory: str | Path, pattern: str, keep: int) -> int:
    """
    Delete all but the `keep` newest files matching `pattern` in `directory`.

    Returns the number of files deleted. Ordering is by modification time,
    not filename: the step counter in `step_{step}.pth` is not zero-padded,
    so `step_9000.pth` sorts after `step_11000.pth` lexically.

    `keep <= 0` disables pruning entirely and deletes nothing.
    """
    if keep <= 0:
        return 0

    directory = Path(directory)
    if not directory.is_dir():
        return 0

    files = sorted(
        (p for p in directory.glob(pattern) if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    if len(files) <= keep:
        return 0

    deleted = 0
    for stale in files[:-keep]:
        try:
            stale.unlink()
            deleted += 1
        except OSError:
            # A missing or locked file is not worth interrupting training for.
            pass
    return deleted
