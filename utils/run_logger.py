"""Lightweight run logging for multi-stage, multi-session training.

Designed for the constraints of this project: training happens across
many short, interrupted sessions (e.g. Kaggle's 12h/session, 30h/week
free-tier limits), across four sequential stages (Uni-Encoder pretrain,
mask VAE pretrain, frozen-encoder diffusion training, LLRD joint
fine-tune). Every checkpoint needs enough metadata attached that you can
tell, weeks later, exactly what produced it and what to resume from.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

LOGS_ROOT = Path(__file__).resolve().parents[1] / "logs"


def _git_commit_hash() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class CheckpointMeta:
    stage: str
    run_id: str
    step: int
    epoch: int
    parent_checkpoint: Optional[str]
    config_path: Optional[str]
    git_commit: Optional[str]
    seed: int
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, checkpoint_path: str | Path) -> Path:
        checkpoint_path = Path(checkpoint_path)
        meta_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(asdict(self), indent=2))
        return meta_path


class RunLogger:
    """
    Per-run logging: metrics CSV, free-text notes, and a config snapshot,
    under logs/runs/<run_id>/. Also appends a one-line summary to the
    git-tracked logs/EXPERIMENTS.md index so the full run history stays
    scannable without opening individual run folders.
    """

    def __init__(
        self,
        run_id: str,
        stage: str,
        config: Optional[dict[str, Any]] = None,
        logs_root: Path = LOGS_ROOT,
    ) -> None:
        self.run_id = run_id
        self.stage = stage
        self.run_dir = logs_root / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_path = self.run_dir / "metrics.csv"
        self._notes_path = self.run_dir / "notes.md"
        self._config_path = self.run_dir / "config.yaml"
        self._experiments_index = logs_root / "EXPERIMENTS.md"

        self._metrics_header_written = self._metrics_path.exists()
        self._csv_headers: dict[str, bool] = {}  # tracks header-written state per named CSV

        if config is not None:
            self._dump_config(config)

        self.note(f"Run started. stage={stage} git={_git_commit_hash()} "
                  f"python={sys.version.split()[0]} torch={torch.__version__}")

    def _dump_config(self, config: dict[str, Any]) -> None:
        try:
            import yaml

            self._config_path.write_text(yaml.dump(config, sort_keys=False))
        except ImportError:
            self._config_path.with_suffix(".json").write_text(json.dumps(config, indent=2))

    def log_metrics(self, step: int, *, csv: str = "metrics", **metrics: float) -> None:
        """Write a row to <csv>.csv under the run directory.

        Use different `csv` names to keep metric groups in separate files
        (e.g. csv="train" vs csv="val") so column headers never collide.
        """
        path = self.run_dir / f"{csv}.csv"
        is_new = not self._csv_headers.get(csv, path.exists())
        with path.open("a") as f:
            if is_new:
                f.write("timestamp,step," + ",".join(metrics.keys()) + "\n")
                self._csv_headers[csv] = True
            row = [str(time.time()), str(step)] + [str(v) for v in metrics.values()]
            f.write(",".join(row) + "\n")

    def note(self, text: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._notes_path.open("a") as f:
            f.write(f"- [{ts}] {text}\n")

    def save_checkpoint_meta(
        self,
        checkpoint_path: str | Path,
        *,
        step: int,
        epoch: int,
        parent_checkpoint: Optional[str] = None,
        seed: int = 0,
        extra: Optional[dict[str, Any]] = None,
    ) -> Path:
        meta = CheckpointMeta(
            stage=self.stage,
            run_id=self.run_id,
            step=step,
            epoch=epoch,
            parent_checkpoint=parent_checkpoint,
            config_path=str(self._config_path) if self._config_path.exists() else None,
            git_commit=_git_commit_hash(),
            seed=seed,
            extra=extra or {},
        )
        return meta.save(checkpoint_path)

    def append_to_experiments_index(self, summary: str) -> None:
        """
        Append one line to the committed logs/EXPERIMENTS.md master index.
        Call this once at the end of a run (or meaningful checkpoint) with
        a short human-readable result summary, e.g.:
            "Stage1 UniEncoderTiny, 40k steps, recon_loss=0.041"
        """
        date = time.strftime("%Y-%m-%d")
        line = f"| {date} | {self.run_id} | {self.stage} | {summary} |\n"
        if not self._experiments_index.exists():
            self._experiments_index.write_text(
                "| Date | Run ID | Stage | Summary |\n|---|---|---|---|\n"
            )
        with self._experiments_index.open("a") as f:
            f.write(line)


def new_run_id(stage: str) -> str:
    return f"{stage}_{time.strftime('%Y%m%d_%H%M%S')}"
