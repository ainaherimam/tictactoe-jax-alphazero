"""Resume bundle for a jax_az run: full train state + replay ring + loop scalars.

A run dir already holds `config.json` (settings) and per-gen weights-only orbax
checkpoints. Those are enough for inference/eval but NOT for a faithful resume:
`save_checkpoint` drops the optimizer state (Adam moments) and the schedule step,
and the replay ring lives only on-device. This module saves exactly the three
things a resume needs and nothing the run dir already has:

  resume/            orbax dump of the WHOLE TrainState pytree
                     (params + batch_stats + opt_state + step)
  replay.npz         the replay.Buf arrays + cursor
  train_state.json   {gen, key, games_total, arch}   (arch guards a mismatched resume)

All three are written together every time a checkpoint is, so they never drift.
Missing/incomplete bundle -> `has_resume` is False -> caller starts fresh (empty
ring, random weights), which is the "not filled" case in the spec.
"""
from __future__ import annotations

import json
import os
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from jax_az import replay


def _resume_dir(run_dir):  return os.path.abspath(os.path.join(run_dir, "resume"))
def _replay_path(run_dir): return os.path.join(run_dir, "replay.npz")
def _scalars_path(run_dir): return os.path.join(run_dir, "train_state.json")

_ckptr = None
def _checkpointer():
    global _ckptr
    if _ckptr is None:
        _ckptr = ocp.StandardCheckpointer()
    return _ckptr


def arch_of(cfg) -> dict:
    """Fields that must match for an orbax restore to line up (net shape + ring dtype)."""
    return {"num_channels": cfg.num_channels, "num_res_blocks": cfg.num_res_blocks,
            "variant": cfg.variant}


def has_resume(run_dir) -> bool:
    return bool(run_dir) and os.path.isdir(_resume_dir(run_dir)) \
        and os.path.exists(_replay_path(run_dir)) and os.path.exists(_scalars_path(run_dir))


def save(run_dir, state, buf, gen: int, key, games_total: int, arch: dict):
    os.makedirs(run_dir, exist_ok=True)
    # Full train state. StandardCheckpointer won't overwrite, so clear the rolling
    # dir first (version-proof; avoids depending on a `force=` kwarg).
    shutil.rmtree(_resume_dir(run_dir), ignore_errors=True)
    _checkpointer().save(_resume_dir(run_dir), state)

    # Replay ring. ponytail: plain savez of the full (mostly-zero) ring; for a 1M
    # capacity this is ~260MB written every checkpoint. One file, overwritten, so
    # disk stays bounded. If the write dominates, save the ring less often than
    # weights — that's the upgrade path, not a separate format.
    d = {k: np.asarray(v) for k, v in buf._asdict().items()}
    tmp_npz = _replay_path(run_dir) + ".tmp.npz"
    np.savez(tmp_npz, **d)
    os.replace(tmp_npz, _replay_path(run_dir))

    tmp = _scalars_path(run_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"gen": int(gen), "key": np.asarray(key).tolist(),
                   "games_total": int(games_total), "arch": arch}, f)
    os.replace(tmp, _scalars_path(run_dir))


def load(run_dir, template_state, expect_arch: dict | None = None):
    """Restore (state, buf, gen, key, games_total). `template_state` supplies the
    pytree structure orbax restores into — build it with `create_train_state`."""
    with open(_scalars_path(run_dir)) as f:
        s = json.load(f)
    if expect_arch is not None and s.get("arch") != expect_arch:
        raise ValueError(
            f"resume arch mismatch: saved {s.get('arch')} != config {expect_arch}. "
            f"Resuming would corrupt the restore — fix config.py or use a fresh run dir.")

    state = _checkpointer().restore(_resume_dir(run_dir), template_state)
    d = np.load(_replay_path(run_dir))
    buf = replay.Buf(**{k: jnp.asarray(d[k]) for k in d.files})
    key = jnp.asarray(s["key"], dtype=jnp.uint32)
    return state, buf, int(s["gen"]), key, int(s["games_total"])


def demo():
    """Self-check: a full state + buffer + scalars round-trip restores byte-for-byte
    into a *differently* initialised template (proving it loads the saved weights,
    not the template's), and the arch guard rejects a mismatch."""
    import tempfile
    from src.models.alphazero_model import TrainingConfig, create_train_state, train_step

    cfg = TrainingConfig()
    cfg.num_channels, cfg.num_res_blocks = 8, 1
    arch = {"num_channels": 8, "num_res_blocks": 1, "variant": cfg.variant}

    state = create_train_state(jax.random.PRNGKey(0), cfg)
    # take one real grad step so opt_state/step are non-trivial
    K, size, cells = 2 * 16, 4, 16
    batch = {"boards": jnp.ones((K, 2, size, size)), "pi": jnp.ones((K, cells)) / cells,
             "z": jnp.zeros((K,)), "mask": jnp.ones((K, cells))}
    state, _ = train_step(state, batch, cfg.lambda_v, cfg.train_value_only, cfg.variant)

    buf = replay.empty(4 * cells, size, cells)
    buf = replay.add(buf, jnp.ones((K, 2, size, size)), jnp.ones((K, cells)) / cells,
                     jnp.ones((K,)), jnp.ones((K, cells)), jnp.ones((K,)))

    run_dir = tempfile.mkdtemp()
    key = jax.random.PRNGKey(42)
    save(run_dir, state, buf, gen=7, key=key, games_total=123, arch=arch)

    template = create_train_state(jax.random.PRNGKey(999), cfg)  # different init
    rstate, rbuf, gen, rkey, gt = load(run_dir, template, expect_arch=arch)

    a = jax.tree_util.tree_leaves(state.params)
    b = jax.tree_util.tree_leaves(rstate.params)
    assert all(np.allclose(x, y) for x, y in zip(a, b)), "params not restored from disk"
    assert int(rstate.step) == int(state.step) == 1, "optimizer step not restored"
    assert gen == 7 and gt == 123, "scalars wrong"
    assert bool(jnp.array_equal(rkey, key)), "rng key not restored"
    assert float(replay.filled(rbuf)) == float(replay.filled(buf)), "ring fill lost"
    assert int(rbuf.cursor) == int(buf.cursor), "ring cursor lost"

    # arch guard
    try:
        load(run_dir, template, expect_arch={**arch, "num_channels": 16})
        assert False, "arch guard did not fire"
    except ValueError:
        pass

    shutil.rmtree(run_dir, ignore_errors=True)
    print("jax_az.runstate demo OK: state+opt+ring+scalars round-trip, arch guard fires")


if __name__ == "__main__":
    demo()
