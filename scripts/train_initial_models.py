#!/usr/bin/env python
"""Convenience wrapper for the training pipeline.

Equivalent to::

    emo-train --start <2y-ago> --end <today> --artifacts-dir artifacts/models

but easier to invoke as a one-shot before the API is brought up for the
first time. All arguments are forwarded to ``emo-train`` after sensible
defaults are applied, so anything you pass on the command line wins.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

from energy_mix_optimizer.pipelines.train import main as train_main


def _default_args() -> list[str]:
    today = datetime.now(UTC).date()
    start = today - timedelta(days=365 * 2)
    return [
        "--start",
        start.isoformat(),
        "--end",
        today.isoformat(),
        "--artifacts-dir",
        "artifacts/models",
        "--cv-splits",
        "5",
    ]


if __name__ == "__main__":
    raise SystemExit(train_main(_default_args() + sys.argv[1:]))
