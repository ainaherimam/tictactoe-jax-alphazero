"""Trainer subprocess entrypoint: `python -m jax_az.monitor.runner <run_dir>`.

Reads `<run_dir>/config.json`, rebuilds the four real config objects, and runs
`train_loop` with a metrics sink. Lifecycle (state/pid/error) is written to
`<run_dir>/run.json` so the server can show status without touching this process.

Imports jax (it IS the trainer). The server must never import this module.
"""
from __future__ import annotations

import dataclasses
import json
import os
import signal
import sys
import time
import traceback


def _write_run(run_dir: str, **fields):
    path = os.path.join(run_dir, "run.json")
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


def _coerce(v, ftype):
    """Coerce a JSON form value (often a string) to the dataclass field's type.
    Empty string -> None for Optional fields. ftype may be a real type or a
    string annotation like 'Optional[int]' (PEP 563)."""
    t = ftype if isinstance(ftype, str) else str(ftype)
    # JSON has no tuples; tuple-typed fields (e.g. temp_schedule) arrive as lists.
    # SearchConfig is a hashable static jit arg, so nested lists must become tuples.
    if isinstance(v, list) and "tuple" in t:
        return tuple(tuple(x) if isinstance(x, list) else x for x in v)
    if not isinstance(v, str):
        return v
    optional = "Optional" in t or "None" in t
    if v == "" and optional or v.lower() in ("none", "null"):
        return None
    if "int" in t:
        return int(v)
    if "float" in t:
        return float(v)
    if "bool" in t:
        return v.lower() in ("1", "true", "yes")
    return v


def _build_configs(cfg: dict):
    """Reconstruct the dataclass/config objects from posted JSON, applying only
    keys that are real fields so an unknown key can never crash a run."""
    from jax_az.env import GameConfig
    from jax_az.search import SearchConfig
    from jax_az.train_jax import LoopConfig
    from src.models.alphazero_model import TrainingConfig

    def from_dataclass(klass, d):
        fields = {f.name: f.type for f in dataclasses.fields(klass)}
        kw = {k: _coerce(v, fields[k]) for k, v in (d or {}).items() if k in fields}
        return klass(**kw)

    game = from_dataclass(GameConfig, cfg.get("game"))
    search = from_dataclass(SearchConfig, cfg.get("search"))
    loop = from_dataclass(LoopConfig, cfg.get("loop"))

    train = TrainingConfig()  # plain class with class-level attrs
    for k, v in (cfg.get("train") or {}).items():
        if hasattr(train, k):
            setattr(train, k, v)
    return game, search, loop, train


def main(run_dir: str):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)

    _write_run(run_dir, state="starting", pid=os.getpid(), started_at=time.time())

    # SIGTERM (from the server's stop button) -> mark stopped and exit cleanly.
    def _term(signum, frame):
        _write_run(run_dir, state="stopped", ended_at=time.time())
        os._exit(0)
    signal.signal(signal.SIGTERM, _term)

    from jax_az.train_jax import train_loop
    from jax_az.monitor.sink import make_sink

    game, search, loop, train = _build_configs(cfg)
    # Isolate this run's checkpoints under the run dir unless the user set a path.
    if not (cfg.get("train") or {}).get("checkpoint_dir"):
        train.checkpoint_dir = os.path.join(run_dir, "checkpoints")

    sink = make_sink(run_dir)
    _write_run(run_dir, state="running")
    try:
        train_loop(seed=int(cfg.get("seed", 0)), cfg=train, loop=loop,
                   search_cfg=search, game=game, on_metrics=sink, run_dir=run_dir)
        _write_run(run_dir, state="finished", ended_at=time.time())
    except Exception:
        _write_run(run_dir, state="error", ended_at=time.time(),
                   error=traceback.format_exc())
        raise
    finally:
        sink.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m jax_az.monitor.runner <run_dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
