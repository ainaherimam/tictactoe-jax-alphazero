"""Lock-free, one-way metrics sink for the training loop.

`make_sink(run_dir)` returns an `on_metrics(gen, rec)` callback for train_loop. It
appends one JSON line per generation to `metrics.jsonl` and overwrites a small
`last.json` snapshot. Both are plain text writes of a dict of Python floats — no
locks, no device access, no blocking on the UI. The server tails these files; it
never writes them, so there is exactly one writer per file and no contention with
the trainer.
"""
from __future__ import annotations

import json
import os
import time


def make_sink(run_dir: str):
    os.makedirs(run_dir, exist_ok=True)
    jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    last_path = os.path.join(run_dir, "last.json")
    # line-buffered append so each generation's record is flushed promptly but the
    # write itself stays cheap (no fsync, no per-call open).
    f = open(jsonl_path, "a", buffering=1)

    def on_metrics(gen: int, rec: dict):
        rec = {**rec, "t": time.time()}
        f.write(json.dumps(rec) + "\n")
        # atomic-ish snapshot: write to temp then rename so a reader never sees half.
        tmp = last_path + ".tmp"
        with open(tmp, "w") as g:
            json.dump(rec, g)
        os.replace(tmp, last_path)

    on_metrics.close = f.close  # let the runner close the handle on shutdown
    return on_metrics
