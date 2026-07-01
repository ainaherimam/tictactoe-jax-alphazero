"""Eval subprocess entrypoint: `python -m jax_az.monitor.eval_runner <run_dir>`.

Reads `<run_dir>/eval_request.json` (written by the server), runs
`jax_az.eval_solver.evaluate` against the requested checkpoint(s), and writes the
results CSV (`<run_dir>/eval.csv`) plus a lifecycle file (`<run_dir>/eval.json`)
so the server can show eval status/progress without touching this process.

Imports jax (it IS the evaluator). The server must never import this module.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
import traceback


def _write_eval(run_dir: str, **fields):
    path = os.path.join(run_dir, "eval.json")
    cur = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                cur = json.load(f)
        except Exception:
            cur = {}
    cur.update(fields)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f, indent=2)
    os.replace(tmp, path)


def main(run_dir: str):
    with open(os.path.join(run_dir, "eval_request.json")) as f:
        req = json.load(f)
    out_csv = os.path.join(run_dir, "eval.csv")

    _write_eval(run_dir, state="running", pid=os.getpid(), started_at=time.time(),
                done=0, total=req.get("target_count", 0), last_gen=None,
                target=req.get("label", ""), csv="eval.csv",
                ended_at=None, error=None)

    # SIGTERM (eval stop button) -> mark stopped and exit cleanly.
    def _term(signum, frame):
        _write_eval(run_dir, state="stopped", ended_at=time.time())
        os._exit(0)
    signal.signal(signal.SIGTERM, _term)

    from jax_az.eval_solver import evaluate

    def on_ckpt(done, total, gen):
        _write_eval(run_dir, done=done, total=total, last_gen=gen)

    try:
        evaluate(req["checkpoint"], sims=int(req["sims"]),
                 per_group=int(req["per_group"]), pgn=bool(req.get("pgn", False)),
                 seed=int(req.get("seed", 0)), out_csv=out_csv, on_checkpoint=on_ckpt,
                 overrides={"per_group": int(req["per_group"])})
        _write_eval(run_dir, state="finished", ended_at=time.time())
    except Exception:
        _write_eval(run_dir, state="error", ended_at=time.time(),
                    error=traceback.format_exc())
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m jax_az.monitor.eval_runner <run_dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
