"""Search-free policy/value accuracy of the net vs the perfect Misère solver.

For a *solved* game the solver gives ground truth for EVERY position, so the
densest, cheapest, zero-variance signal is to score the raw network against it
with no MCTS at all — one batched forward over every reachable non-terminal
position (to-move perspective). See jax_az/eval.md §2(A).

Metrics (all in [0,1] unless noted, higher=better except regret/mse):
  * optimal_move_rate  — argmax(policy) is a solver-optimal move
  * policy_regret      — best_value - E_policy[value]; expected value lost to
                         the policy's move distribution (0 = perfect, lower=better)
  * value_class_acc    — net value, snapped to nearest of {-1,0,+1}, matches the
                         solver's win/draw/loss verdict
  * value_mse          — mean (net_value - solver_value)^2  (lower=better)

Optionally (search_step given) also reports `mcts_optimal_move_rate`: the same
optimal-move rate but after running MCTS on each position — the search-helps
delta (eval.md §2(C)). One batched search over all positions, far cheaper than
the per-ply game eval.

    python -m jax_az.eval_policy jax_az_runs/run-.../checkpoints/checkpoint_410
    python -m jax_az.eval_policy <dir-of-checkpoints>            # sweeps all
"""
from __future__ import annotations

import argparse
import collections

import jax
import jax.numpy as jnp
import numpy as np

from jax_az import features  # noqa: F401  (planes layout)
from jax_az.env import Env, GameConfig, State
from jax_az.eval_solver import (  # reuse: same arch resolution + checkpoint discovery
    FULL, CELLS, find_checkpoints, resolve_arch,
)
from src.core.solver.misere_solver import MisereSolver, canonicalize, has_line
from src.models.alphazero_model import load_checkpoint_for_inference

_BIN_CENTERS = np.array([-1.0, 0.0, 1.0])


def _popcount(x: int) -> int:
    return bin(x).count("1")


# ============================================================
# Enumerate every reachable non-terminal position, to-move perspective
# ============================================================
def enumerate_positions(solver: MisereSolver):
    """BFS the whole game tree; return [(own, opp), ...] for every non-terminal
    position from the to-move player's view, deduped under D4 symmetry. The net
    sees only (own, opp), so to-move perspective is the right unit."""
    seen = {canonicalize(0, 0)}
    out = []
    q = collections.deque([(0, 0)])
    while q:
        own, opp = q.popleft()
        out.append((own, opp))
        occ = own | opp
        for c in range(CELLS):
            if occ & (1 << c):
                continue
            nown = own | (1 << c)
            if has_line(nown):                  # mover completed a line -> terminal
                continue
            child = (opp, nown)                 # swap perspective
            if (child[0] | child[1]) == FULL:   # board full -> terminal (draw)
                continue
            key = canonicalize(*child)
            if key in seen:
                continue
            seen.add(key)
            q.append(child)
    return out


def build_targets(solver: MisereSolver, positions):
    """Solver ground truth as dense arrays. Returns dict of np arrays:
        own[N], opp[N] int32; legal[N,A] bool; sval[N,A] float (solver value per
        legal cell, illegal=0); opt[N,A] bool (cell is solver-optimal);
        pos_val[N] float in {-1,0,1} (to-move value)."""
    N = len(positions)
    own = np.array([p[0] for p in positions], np.int32)
    opp = np.array([p[1] for p in positions], np.int32)
    legal = np.zeros((N, CELLS), bool)
    sval = np.zeros((N, CELLS), np.float32)
    opt = np.zeros((N, CELLS), bool)
    pos_val = np.zeros(N, np.float32)
    for i, (bx, bo) in enumerate(positions):
        avs = solver.get_action_values(bx, bo, True)  # [(cell, value)] to-move view
        best = max(v for _, v in avs)
        pos_val[i] = best
        for cell, v in avs:
            legal[i, cell] = True
            sval[i, cell] = v
            opt[i, cell] = (v == best)
    return dict(own=own, opp=opp, legal=legal, sval=sval, opt=opt, pos_val=pos_val)


# ============================================================
# Metrics
# ============================================================
def policy_metrics(eval_fn, variables, env, t, chunk=8192) -> dict:
    """Net forward over all positions -> the four search-free metrics. Chunked so
    the activations of a wide ResNet over ~400k positions fit in GPU memory."""
    own, opp = t["own"], t["opp"]
    logits = np.empty((len(own), CELLS), np.float32)
    value = np.empty(len(own), np.float32)
    for i in range(0, len(own), chunk):
        planes = features.planes_batch(
            State(jnp.asarray(own[i:i+chunk]), jnp.asarray(opp[i:i+chunk])), env.size)
        lg, v = eval_fn(variables, planes)
        logits[i:i+chunk] = np.asarray(lg)
        value[i:i+chunk] = np.asarray(v)
    legal, sval, opt, pos_val = t["legal"], t["sval"], t["opt"], t["pos_val"]

    masked = np.where(legal, logits, -1e30)
    choice = masked.argmax(axis=1)
    optimal_move_rate = float(opt[np.arange(len(choice)), choice].mean())

    # softmax over legal moves only, then expected solver value under the policy
    ex = np.where(legal, np.exp(masked - masked.max(1, keepdims=True)), 0.0)
    policy = ex / ex.sum(1, keepdims=True)
    e_val = (policy * sval).sum(1)
    policy_regret = float((pos_val - e_val).mean())

    pred_class = _BIN_CENTERS[np.abs(value[:, None] - _BIN_CENTERS).argmin(1)]
    value_class_acc = float((pred_class == pos_val).mean())
    value_mse = float(((value - pos_val) ** 2).mean())

    return dict(positions=len(choice), optimal_move_rate=optimal_move_rate,
                policy_regret=policy_regret, value_class_acc=value_class_acc,
                value_mse=value_mse)


def mcts_optimal_move_rate(search_step, variables, env, t, chunk=4096) -> float:
    """Optimal-move rate after MCTS on each position (eval.md §2(C) delta).
    Chunked: a full MCTS tree per position over ~400k positions is memory-heavy."""
    own, opp = t["own"], t["opp"]
    actions = np.empty(len(own), np.int64)
    for i in range(0, len(own), chunk):
        a = search_step(variables, jax.random.PRNGKey(0),
                        State(jnp.asarray(own[i:i+chunk]), jnp.asarray(opp[i:i+chunk])))
        actions[i:i+chunk] = np.asarray(a)
    return float(t["opt"][np.arange(len(actions)), actions].mean())


# ============================================================
# Per-checkpoint driver (also callable from eval_solver.evaluate)
# ============================================================
def evaluate_policy(checkpoint, overrides=None, with_search=False, sims=100):
    overrides = overrides or {}
    ckpts = find_checkpoints(checkpoint)
    arch, game = resolve_arch(ckpts[0], overrides)
    env = Env(GameConfig(game["size"], game["win_length"], game["misere"]))

    print("Solving Misère TT...")
    solver = MisereSolver(); solver.solve()
    positions = enumerate_positions(solver)
    t = build_targets(solver, positions)
    print(f"Positions: {len(positions)}  Checkpoints: {len(ckpts)}\n")

    from jax_az import model as azmodel
    m = azmodel.make_model(env, arch["num_channels"], arch["num_res_blocks"], arch["variant"])
    eval_fn = azmodel.make_eval_fn(m)

    search_step = None
    if with_search:
        from jax_az.search import Search, SearchConfig
        cfg = SearchConfig(algorithm="muzero", num_simulations=sims,
                           temperature=0.0, dirichlet_fraction=0.0)
        search = Search(env, eval_fn, cfg)
        search_step = jax.jit(lambda v, k, s: search.run(v, k, s, config=cfg).action)

    rows = []
    for i, ckpt in enumerate(ckpts):
        gen = int(ckpt.name.split("_")[1])
        loaded = load_checkpoint_for_inference(
            str(ckpt), arch["num_channels"], arch["num_res_blocks"],
            env.num_actions, arch["variant"])
        variables = {"params": loaded["params"], "batch_stats": loaded["batch_stats"]}
        mtr = policy_metrics(eval_fn, variables, env, t)
        if with_search:
            mtr["mcts_optimal_move_rate"] = mcts_optimal_move_rate(
                search_step, variables, env, t)
        mtr["checkpoint"] = gen
        rows.append(mtr)
        extra = f"  mcts_opt={mtr['mcts_optimal_move_rate']:.3f}" if with_search else ""
        print(f"[{i+1}/{len(ckpts)}] ckpt {gen:>4}  "
              f"opt_move={mtr['optimal_move_rate']:.3f}  "
              f"regret={mtr['policy_regret']:.4f}  "
              f"val_acc={mtr['value_class_acc']:.3f}  val_mse={mtr['value_mse']:.4f}{extra}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Search-free policy/value eval vs solver")
    ap.add_argument("checkpoint", help="a checkpoint_N folder, or a dir of them")
    ap.add_argument("--with-search", action="store_true",
                    help="also report optimal-move rate after MCTS (search-helps delta)")
    ap.add_argument("--sims", type=int, default=100, help="MCTS sims for --with-search")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    a = ap.parse_args()
    evaluate_policy(a.checkpoint, with_search=a.with_search, sims=a.sims,
                    overrides={"variant": a.variant, "num_channels": a.channels,
                               "num_res_blocks": a.blocks})


def demo():
    """Self-check: enumeration covers the solver TT's non-terminal positions, and a
    perfect 'net' (logits = solver action values, value = solver value) scores a
    perfect 1.0 optimal-move rate / value accuracy / 0 regret."""
    env = Env(GameConfig(4, 3, True))
    solver = MisereSolver(); solver.solve()
    positions = enumerate_positions(solver)
    assert len(positions) > 100, len(positions)
    # all enumerated positions are non-terminal and have >=1 legal move
    for bx, bo in positions[:50]:
        assert (bx | bo) != FULL and not has_line(bx) and not has_line(bo)
    t = build_targets(solver, positions)

    # an oracle eval_fn: all policy mass on the solver-optimal moves, value = solver
    # position value. Must score perfectly (argmax optimal, 0 regret, exact value).
    def oracle(_vars, planes):
        logits = np.where(t["opt"], 10.0, -1e30).astype(np.float32)
        return jnp.asarray(logits), jnp.asarray(t["pos_val"])
    mtr = policy_metrics(oracle, {}, env, t, chunk=len(positions))  # oracle returns full-N
    assert mtr["optimal_move_rate"] == 1.0, mtr
    assert mtr["value_class_acc"] == 1.0, mtr
    assert abs(mtr["policy_regret"]) < 1e-6, mtr
    assert mtr["value_mse"] < 1e-12, mtr
    print(f"jax_az.eval_policy demo OK: {len(positions)} positions, oracle scores perfect "
          f"(opt=1.0 val_acc=1.0 regret=0)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()
