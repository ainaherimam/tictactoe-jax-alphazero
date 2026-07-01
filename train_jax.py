"""Single-process AlphaZero loop on the JAX/mctx path (next_plan.md §5).

Replaces the C++ self-play + SHM ring + Python inference server + train.py quartet
with one process: per generation, self-play with the current weights fills the
on-device replay ring, then K gradient steps train on augmented samples drawn from
it. Everything inside the `for gen` body is device arrays end to end — `play_batch`,
`add`, `sample`, `augment`, `train_step` all jitted, no host transfer except scalar
metrics for logging.

Reuses `create_train_state` / `train_step` / `save_checkpoint` /
`compute_solver_metrics` / `TrainingConfig` from src.models.alphazero_model
**unchanged**. The only new glue is this loop. Run: `python -m jax_az.train_jax`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from jax_az.env import Env, GameConfig
from jax_az.model import make_model, make_eval_fn
from jax_az.search import Search, SearchConfig
from jax_az import selfplay, replay

from src.models.alphazero_model import (
    TrainingConfig, create_train_state, train_step, save_checkpoint,
    compute_solver_metrics,
)


@dataclass
class LoopConfig:
    """Loop-only knobs. Net / optimizer / checkpoint knobs stay in TrainingConfig."""
    num_generations: int = 1000
    games_per_gen: int = 256            # B parallel self-play games per generation
    replay_capacity: int = 1_000_000    # rounded up to a multiple of B*cells
    eval_batch_size: int = 512          # positions sampled for solver metrics


def _round_up(n: int, k: int) -> int:
    return ((n + k - 1) // k) * k


def train_loop(seed: int = 0, cfg: TrainingConfig | None = None,
               loop: LoopConfig | None = None,
               search_cfg: SearchConfig | None = None,
               game: GameConfig | None = None,
               on_metrics=None, run_dir: str | None = None):
    """Run the generation loop. `on_metrics(gen, dict)` is called when set (wire
    wandb here if wanted); otherwise metrics just print.

    `run_dir`: if it holds a resume bundle (full state + replay ring + scalars, see
    `jax_az.runstate`), the run continues from where that checkpoint left off —
    weights, optimizer moments, schedule step, replay contents, rng and gen counter.
    Otherwise (no bundle, or run_dir is None) the run starts fresh: random weights +
    empty ring. The bundle is (re)written to run_dir every time a checkpoint saves.

    # ponytail: wandb/W&B logging is plumbing, not the loop — left as a callback
    # hook (on_metrics) rather than wired in. Add the WandbLogger call site when a
    # run actually needs it.
    """
    cfg = cfg or TrainingConfig()
    loop = loop or LoopConfig()
    search_cfg = search_cfg or SearchConfig()
    game = game or GameConfig(4, 3, True)

    from jax_az import device
    print(f"[train_jax] {device.info()}")

    env = Env(game)
    B, cells = loop.games_per_gen, env.cells
    N = _round_up(loop.replay_capacity, B * cells)

    model = make_model(env, cfg.num_channels, cfg.num_res_blocks, cfg.variant)
    search = Search(env, make_eval_fn(model), search_cfg)

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    state = create_train_state(init_key, cfg)
    buf = replay.empty(N, env.size, cells)
    PERM = replay.build_perm(env.size)

    # Resume from run_dir if it holds a complete bundle; else fresh (random + empty,
    # the "checkpoint folder / buffer not filled" case). arch guard fails loud rather
    # than letting a mismatched config silently corrupt the restore.
    from jax_az import runstate
    arch = runstate.arch_of(cfg)
    start_gen, games_total = 0, 0
    if run_dir and runstate.has_resume(run_dir):
        state, buf, last_gen, key, games_total = runstate.load(run_dir, state, arch)
        start_gen = last_gen + 1
        print(f"[train_jax] resumed {run_dir} at gen {last_gen}: {games_total} games, "
              f"{float(replay.filled(buf)):.0f} positions in ring")

    # self-play params = current weights (a generation already lags). jit once.
    play = jax.jit(selfplay.play_batch, static_argnums=(2, 3, 4, 5))

    from src.core.solver.misere_solver import MisereSolver
    solver = MisereSolver()
    solver.solve()

    history = []  # (gen, last_loss, acc_nn) for the smoke gate / inspection
    for gen in range(start_gen, loop.num_generations):
        t0 = time.time()
        key, sk, ak = jax.random.split(key, 3)
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        block = play(variables, sk, env, search, B, search_cfg)
        buf = replay.add(buf, *block)
        games_total += B

        # filled is a device scalar; pull it once and reuse (the loop needs it for
        # the train gate anyway — no extra host sync added for monitoring).
        filled_f = float(replay.filled(buf))
        loss = None
        metrics = None
        if filled_f >= cfg.min_positions:
            for _ in range(cfg.steps_per_generation):
                key, s1, s2 = jax.random.split(key, 3)
                batch = replay.augment(replay.sample(buf, s1, cfg.batch_size), s2, PERM)
                state, metrics = train_step(state, batch, cfg.lambda_v,
                                            cfg.train_value_only, cfg.variant)
            loss = float(metrics['loss'])

        ckpt_saved = None
        if gen % cfg.save_every_n_gens == 0 and loss is not None:
            save_checkpoint(state, cfg.checkpoint_dir, step=gen)
            if run_dir:
                # full resume bundle alongside the weights-only checkpoint
                runstate.save(run_dir, state, buf, gen, key, games_total, arch)
            ckpt_saved = gen

        acc = None
        if gen % cfg.evaluate_every_n_gens == 0 and filled_f >= cfg.batch_size:
            acc = _solver_eval(state, buf, loop.eval_batch_size, solver, key)
            print(f"[gen {gen:5d}] loss={loss} solver_acc_nn={acc['policy_acc_nn']:.3f} "
                  f"acc_mcts={acc['policy_acc_mcts']:.3f}")

        # Per-generation monitor record. All values are host-side already; cursor is
        # derived from games_total (not pulled from device) to avoid an extra sync.
        if on_metrics:
            dt = time.time() - t0
            rec = {
                'gen': gen, 'dt': dt,
                'filled': filled_f, 'capacity': int(N),
                'cursor': (games_total * cells) % N,
                'games_total': games_total, 'games_per_gen': B,
                'training': metrics is not None,
                'loss': loss,
                'games_per_sec': (B / dt) if dt > 0 else 0.0,
                'checkpoint_saved': ckpt_saved,
            }
            if metrics is not None:
                # metrics are 0-dim arrays from the train step already materialised
                # on host (loss was just pulled); converting siblings is ~free.
                rec.update({k: float(v) for k, v in metrics.items()})
                rec['samples_per_sec'] = (
                    (cfg.batch_size * cfg.steps_per_generation) / dt) if dt > 0 else 0.0
            if acc is not None:
                rec['solver_acc_nn'] = float(acc['policy_acc_nn'])
                rec['solver_acc_mcts'] = float(acc['policy_acc_mcts'])
            on_metrics(gen, rec)

        history.append((gen, loss, None if acc is None else acc['policy_acc_nn']))

    return state, history


def _solver_eval(state, buf, bs, solver, key):
    """Raw (un-augmented) sample -> NN forward -> solver-agreement metrics."""
    batch = replay.sample(buf, key, bs)
    (p_pred_log, _), _ = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch['boards'], jnp.ones_like(batch['mask']), training=False, mutable=[])
    return compute_solver_metrics(
        boards=np.asarray(batch['boards']), pi_mcts=np.asarray(batch['pi']),
        p_pred_log=np.asarray(p_pred_log), solver=solver)


def demo():
    """Phase-4 gate (next_plan.md §6, test 5): a short run on 4x4 misère; loss
    trends down and solver agreement is sane. Small + CPU-friendly."""
    import shutil
    cfg = TrainingConfig()
    cfg.num_channels, cfg.num_res_blocks = 16, 2
    cfg.min_positions, cfg.batch_size = 256, 128
    cfg.steps_per_generation = 8
    cfg.evaluate_every_n_gens = 4
    cfg.save_every_n_gens = 10_000          # ~never; keep the smoke test out of repo checkpoints/
    cfg.checkpoint_dir = "/tmp/jax_az_smoke_ckpt"
    cfg.lr_warmup_steps = 20
    shutil.rmtree(cfg.checkpoint_dir, ignore_errors=True)  # smoke test reruns; orbax won't overwrite

    loop = LoopConfig(num_generations=30, games_per_gen=32,
                      replay_capacity=50_000, eval_batch_size=256)
    search_cfg = SearchConfig(num_simulations=24, temp_drop_ply=6)

    evals = []
    # on_metrics now fires every generation; solver acc is only present at eval gens.
    def _collect(g, m):
        if 'solver_acc_nn' in m:
            evals.append((g, m['loss'], m['solver_acc_nn']))
    state, history = train_loop(seed=0, cfg=cfg, loop=loop, search_cfg=search_cfg,
                                on_metrics=_collect)

    losses = [l for _, l, _ in history if l is not None]
    assert len(losses) > 6, "loop did not train"
    n = len(losses) // 3
    first, last = np.mean(losses[:n]), np.mean(losses[-n:])
    assert last < first, f"loss did not decrease: first={first:.3f} last={last:.3f}"

    accs = [a for _, _, a in evals]
    assert len(accs) >= 2, "no solver evals ran"
    # learning signal: agreement improved over the run, or is already decent
    assert accs[-1] >= accs[0] or accs[-1] > 0.3, \
        f"solver agreement did not rise: {accs[0]:.3f} -> {accs[-1]:.3f}"
    print(f"jax_az.train_jax demo OK: loss {first:.3f} -> {last:.3f}, "
          f"solver_acc_nn {accs[0]:.3f} -> {accs[-1]:.3f}")


if __name__ == "__main__":
    import argparse
    import os
    import sys

    current = os.environ.get("JAX_AZ_DEVICE", "auto")
    ap = argparse.ArgumentParser(description="JAX AlphaZero loop (smoke test).")
    ap.add_argument("--device", choices=["cpu", "gpu", "auto"], default=current,
                    help="CPU/GPU switch (= env JAX_AZ_DEVICE). Default: auto.")
    args, _ = ap.parse_known_args()
    # jax is already imported above with `current`; if a different device is asked,
    # re-exec once so JAX_AZ_DEVICE is set BEFORE jax initializes its backend.
    # ponytail: re-exec is the only way a CLI flag can beat import-time backend init.
    if args.device != current:
        os.environ["JAX_AZ_DEVICE"] = args.device
        os.execv(sys.executable, [sys.executable, "-m", "jax_az.train_jax", *sys.argv[1:]])
    demo()
