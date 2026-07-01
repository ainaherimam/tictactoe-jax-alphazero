"""JAX port of eval_against_solver.cpp — AlphaZero (X) vs the perfect Misère
solver (O), over the same 4-group board set, with the same blunder analysis.

Point it at a checkpoint folder; evaluation starts:

    python -m jax_az.eval_solver jax_az_runs/run-.../checkpoints/checkpoint_410
    python -m jax_az.eval_solver jax_az_runs/run-.../checkpoints   # sweeps all
    JAX_AZ_DEVICE=gpu python -m jax_az.eval_solver checkpoints/checkpoint_270 --pgn

What it mirrors from the C++ (executables/eval_against_solver.cpp +
generate_initial_boards.cpp):
  * positions: BFS-enumerate, dedup by D4 symmetry, classify each non-terminal
    by (first_mover, solver value) into 4 groups of `per_group` (O-wins skipped).
  * play: 1 game/board, AZ=X greedy MCTS (temp 0, no Dirichlet), solver=O perfect.
  * blunder: every AZ move co-analysed by solver.get_action_values; chosen < best
    => blunder, with WIN->DRAW / WIN->LOSS / DRAW->LOSS severity.
  * aggregate: theory-matched W/D/L -> Elo, per-group raw + theory score, blunder
    stats. Writes data/solver_eval_results_<ts>.csv (same columns) + stdout.

The speed win over the C++ thread pool: all boards run through ONE batched mctx
search per ply (vmapped on GPU); the solver opponent + blunder lookups run on host
via the existing pure-Python solver (src/core/solver/misere_solver.py) — O(1) TT
hits, a few thousand calls total, negligible next to the net.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as _dt
import json
import math
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_az import features  # noqa: F401  (kept: planes layout lives here)
from jax_az.env import Env, GameConfig, State
from jax_az.search import SearchConfig

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.core.solver.misere_solver import (  # noqa: E402
    MisereSolver, canonicalize, has_line,
)
from src.models.alphazero_model import load_checkpoint_for_inference  # noqa: E402

CELLS = 16
FULL = 0xFFFF
GROUP_NAMES = ["X_first_X_wins", "O_first_X_wins", "X_first_Draw", "O_first_Draw"]
X_WINS_GROUPS = (0, 1)  # theoretical outcome X wins; the rest are draws


@dataclasses.dataclass
class EvalConfig:
    """AZ-vs-solver evaluation knobs. Mirrored in jax_az/config.py (EVAL) and the
    monitor's DEFAULTS["eval"]; the eval is greedy (temperature 0, no Dirichlet)."""
    sims: int = 400          # MCTS simulations per AZ move
    per_group: int = 100     # eval boards per theory group (x4 groups)
    seed: int = 0            # PRNG seed for the batched search
    pgn: bool = False        # also write per-game annotated PGN files


# ============================================================
# Position generation — port of generate_initial_boards.cpp
# ============================================================
def _popcount(x: int) -> int:
    return bin(x).count("1")


def generate_positions(solver: MisereSolver, per_group: int = 100):
    """BFS from the empty board; classify non-terminal positions into 4 groups.

    Returns a flat list of (bx, bo, group) with up to `per_group` per group,
    canonicalised, lowest piece-count first (matches the C++ BFS order)."""
    def classify(is_x_turn: bool, value: int):
        if is_x_turn:
            return 0 if value > 0 else (2 if value == 0 else -1)
        return 1 if value < 0 else (3 if value == 0 else -1)

    groups = [[] for _ in range(4)]
    need = per_group * 4
    found = 0
    seen = set()
    cbx, cbo = canonicalize(0, 0)
    seen.add((cbx, cbo))
    q = collections.deque([(0, 0)])

    while q and found < need:
        bx, bo = q.popleft()
        nx, no = _popcount(bx), _popcount(bo)
        is_x_turn = nx == no
        depth = nx + no
        if depth > 0 and has_line(bo if is_x_turn else bx):
            continue                                  # terminal: skip
        if (bx | bo) == FULL:
            continue

        g = classify(is_x_turn, solver.get_position_value(bx, bo, is_x_turn))
        if g >= 0 and len(groups[g]) < per_group:
            cx, co = canonicalize(bx, bo)
            groups[g].append((cx, co))
            found += 1

        if found < need:
            occ = bx | bo
            for cell in range(CELLS):
                if occ & (1 << cell):
                    continue
                nbx, nbo = (bx | (1 << cell), bo) if is_x_turn else (bx, bo | (1 << cell))
                if has_line(nbx if is_x_turn else nbo):
                    continue                          # terminal child: skip
                key = canonicalize(nbx, nbo)
                if key in seen:
                    continue
                seen.add(key)
                q.append((nbx, nbo))

    return [(bx, bo, g) for g in range(4) for (bx, bo) in groups[g]]


# ============================================================
# Checkpoint discovery + arch resolution
# ============================================================
def find_checkpoints(path: str):
    """A `checkpoint_N` dir -> [that dir]; a dir of them -> all, sorted by N."""
    p = Path(path).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"not a directory: {p}")
    if p.name.startswith("checkpoint_"):
        return [p]
    subs = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")]
    if not subs:
        raise FileNotFoundError(f"no checkpoint_N folders under {p}")
    return sorted(subs, key=lambda d: int(d.name.split("_")[1]))


def resolve_arch(ckpt_dir: Path, overrides: dict):
    """num_channels/num_res_blocks/variant/game from the run's config.json (walk
    up from the checkpoint), with CLI overrides winning. Sensible defaults if none."""
    arch = {"num_channels": 64, "num_res_blocks": 4, "variant": "v1_scalar_mse"}
    game = {"size": 4, "win_length": 3, "misere": True}
    for parent in [ckpt_dir, *ckpt_dir.parents]:
        cfg_path = parent / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            tr = cfg.get("train", {})
            for k in arch:
                if tr.get(k) is not None:
                    arch[k] = tr[k]
            game.update({k: v for k, v in cfg.get("game", {}).items() if v is not None})
            print(f"[arch] from {cfg_path}: {arch}  game={game}")
            break
    else:
        print(f"[arch] no config.json found near {ckpt_dir}; defaults: {arch}")
    arch.update({k: v for k, v in overrides.items() if v is not None})
    return arch, game


# ============================================================
# Play all boards vs the solver (batched search, host solver)
# ============================================================
def _state_to_bxbo(own, opp, x_to_move):
    """to-move (own,opp) bitmasks -> (bx, bo) absolute boards."""
    return (own, opp) if x_to_move else (opp, own)


def cell_to_coord(cell: int) -> str:
    x, y = cell // 4, cell % 4
    return f"{4 - x}{chr(ord('a') + y)}"


def play_all(search_step, variables, env, boards, solver, rng):
    """Run one game per board in lockstep. Returns (winners, blunders, histories).

    winners[b] in {'X','O','draw'}; blunders[b] is a list of dicts; histories[b]
    is a list of (cell, is_blunder) for the PGN writer."""
    B = len(boards)
    own = np.zeros(B, np.int64)
    opp = np.zeros(B, np.int64)
    x_first = np.zeros(B, bool)
    for b, (bx, bo, _g) in enumerate(boards):
        xf = _popcount(bx) == _popcount(bo)
        own[b], opp[b] = (bx, bo) if xf else (bo, bx)
        x_first[b] = xf
    x_to_move = x_first.copy()

    done = np.zeros(B, bool)
    winners = [None] * B
    blunders = [[] for _ in range(B)]
    histories = [[] for _ in range(B)]
    analyzed = np.zeros(B, np.int64)

    # boards already terminal (shouldn't happen — generator skips them) -> draw
    for b in range(B):
        if (own[b] | opp[b]) == FULL:
            done[b], winners[b] = True, "draw"

    # Only boards with X (AZ) to move this ply need a search; the rest are
    # solver moves or finished and would be discarded. A non-done board has made
    # exactly `ply` moves, so X-to-move <=> x_first ^ (ply odd) — i.e. the live
    # X-to-move set is one parity cohort. Search just that cohort, padded to its
    # (fixed) size so jit compiles at most twice. ~2x fewer net forwards.
    cap_even = int(x_first.sum()) or 1          # X-to-move count on even plies
    cap_odd = int((~x_first).sum()) or 1        # ... on odd plies

    for ply in range(CELLS):
        if done.all():
            break
        idx = np.where(x_to_move & ~done)[0]    # boards needing an AZ move now
        actions = {}
        if idx.size:
            rng, k = jax.random.split(rng)
            cap = cap_even if ply % 2 == 0 else cap_odd
            pad = np.zeros(cap, np.int64)
            pad[:idx.size] = idx                # tail repeats board 0 (result ignored)
            acts = np.asarray(search_step(
                variables, k, State(jnp.asarray(own[pad], jnp.int32),
                                    jnp.asarray(opp[pad], jnp.int32))))
            actions = {int(idx[i]): int(acts[i]) for i in range(idx.size)}

        for b in range(B):
            if done[b]:
                continue
            bx, bo = _state_to_bxbo(int(own[b]), int(opp[b]), x_to_move[b])
            if x_to_move[b]:                                   # AZ (X) move
                cell = actions[b]
                avs = dict(solver.get_action_values(bx, bo, True))
                best = max(avs.values())
                chosen = avs.get(cell, best)
                analyzed[b] += 1
                is_blunder = chosen < best
                if is_blunder:
                    blunders[b].append({
                        "move": ply + 1, "bx": bx, "bo": bo, "cell": cell,
                        "chosen": chosen, "best": best,
                        "best_cells": [c for c, v in avs.items() if v == best],
                    })
            else:                                             # solver (O) move
                cell = solver.get_best_move(bx, bo, False)
                is_blunder = False
            histories[b].append((cell, is_blunder))

            # apply move: place in to-move's board, then swap perspective
            new_own = int(opp[b])
            new_opp = int(own[b]) | (1 << cell)
            own[b], opp[b] = new_own, new_opp
            mover_was_x = bool(x_to_move[b])
            x_to_move[b] = not x_to_move[b]

            if has_line(new_opp):                             # mover completed a line
                winners[b] = "O" if mover_was_x else "X"      # misère: mover LOSES
                done[b] = True
            elif (new_own | new_opp) == FULL:
                winners[b] = "draw"
                done[b] = True

    for b in range(B):                                        # safety: unfinished -> draw
        if winners[b] is None:
            winners[b] = "draw"
    return winners, blunders, histories, analyzed


# ============================================================
# Aggregation (theory-matched W/D/L -> Elo) + reporting
# ============================================================
def compute_elo(w, d, l):
    n = w + d + l
    if n == 0:
        return 0.0
    s = min(1 - 1e-6, max(1e-6, (w + 0.5 * d) / n))
    return 400.0 * math.log10(s / (1 - s))


def aggregate(boards, winners, blunders, analyzed):
    per_group = {g: dict(W=0, D=0, L=0, rawX=0, rawD=0, rawO=0) for g in range(4)}
    for b, (_bx, _bo, g) in enumerate(boards):
        xw = winners[b] == "X"
        dr = winners[b] == "draw"
        ow = winners[b] == "O"
        gg = per_group[g]
        gg["rawX"] += xw; gg["rawD"] += dr; gg["rawO"] += ow
        if g in X_WINS_GROUPS:
            gg["W"] += xw; gg["L"] += dr + ow
        else:
            gg["D"] += dr; gg["L"] += xw + ow

    W = sum(g["W"] for g in per_group.values())
    D = sum(g["D"] for g in per_group.values())
    L = sum(g["L"] for g in per_group.values())

    bl = dict(analyzed=int(analyzed.sum()), total=0, games=0,
              win_to_draw=0, win_to_loss=0, draw_to_loss=0)
    for blist in blunders:
        if blist:
            bl["games"] += 1
        for x in blist:
            bl["total"] += 1
            if x["best"] == 1 and x["chosen"] == 0:
                bl["win_to_draw"] += 1
            elif x["best"] == 1 and x["chosen"] <= -1:
                bl["win_to_loss"] += 1
            elif x["best"] == 0 and x["chosen"] <= -1:
                bl["draw_to_loss"] += 1
    return per_group, (W, D, L), bl


def report(gen, per_group, totals, bl):
    W, D, L = totals
    elo = compute_elo(W, D, L)
    rate = 100.0 * bl["total"] / bl["analyzed"] if bl["analyzed"] else 0.0
    print(f"  W={W}  D={D}  L={L}  Elo={elo:.1f}")
    print("  Per category (checkpoint=X, solver=O):")
    for g in range(4):
        gg = per_group[g]
        n = gg["W"] + gg["D"] + gg["L"]
        score = f"  theory_score={100*(gg['W']+0.5*gg['D'])/n:.1f}%" if n else ""
        print(f"    [{GROUP_NAMES[g]:<15}]  raw(X={gg['rawX']:>3} D={gg['rawD']:>3} "
              f"O={gg['rawO']:>3}){score}")
    print(f"  AZ moves analyzed: {bl['analyzed']}  blunders: {bl['total']} ({rate:.2f}%)"
          f"  games-with-blunder: {bl['games']}")
    print(f"  severity breakdown:  WIN->DRAW={bl['win_to_draw']}  "
          f"WIN->LOSS={bl['win_to_loss']}  DRAW->LOSS={bl['draw_to_loss']}")
    return elo


_CSV_HEADER = (
    "checkpoint,theory_wins,theory_draws,theory_losses,total_games,score,elo,"
    "az_moves_analyzed,total_blunders,blunder_rate_pct,games_with_blunder,"
    "win_to_draw,win_to_loss,draw_to_loss,"
    # search-free policy/value metrics vs solver (eval_policy), primary signal:
    "policy_positions,optimal_move_rate,policy_regret,value_class_acc,value_mse,"
    "xfirst_xwins_W,xfirst_xwins_D,xfirst_xwins_L,xfirst_xwins_rawX,xfirst_xwins_rawD,xfirst_xwins_rawO,"
    "ofirst_xwins_W,ofirst_xwins_D,ofirst_xwins_L,ofirst_xwins_rawX,ofirst_xwins_rawD,ofirst_xwins_rawO,"
    "xfirst_draw_W,xfirst_draw_D,xfirst_draw_L,xfirst_draw_rawX,xfirst_draw_rawD,xfirst_draw_rawO,"
    "ofirst_draw_W,ofirst_draw_D,ofirst_draw_L,ofirst_draw_rawX,ofirst_draw_rawD,ofirst_draw_rawO\n")


def csv_row(gen, per_group, totals, bl, elo, pm):
    W, D, L = totals
    n = W + D + L
    score = (W + 0.5 * D) / n if n else 0.0
    rate = 100.0 * bl["total"] / bl["analyzed"] if bl["analyzed"] else 0.0
    cells = [gen, W, D, L, n, f"{score:.6f}", f"{elo:.2f}", bl["analyzed"], bl["total"],
             f"{rate:.4f}", bl["games"], bl["win_to_draw"], bl["win_to_loss"], bl["draw_to_loss"],
             pm["positions"], f"{pm['optimal_move_rate']:.6f}", f"{pm['policy_regret']:.6f}",
             f"{pm['value_class_acc']:.6f}", f"{pm['value_mse']:.6f}"]
    for g in range(4):
        gg = per_group[g]
        cells += [gg["W"], gg["D"], gg["L"], gg["rawX"], gg["rawD"], gg["rawO"]]
    return ",".join(str(c) for c in cells) + "\n"


def write_pgn(out_dir, gen, boards, winners, blunders, histories, analyzed):
    """Per-game annotated files, mirroring the C++ PGN layout (move list with ??
    on blunders + a blunder section). Behind the --pgn switch."""
    root = Path(out_dir) / f"checkpoint_{gen}"
    for sub in ("P1_win_theoretical_board_start", "draw_theoretical_board_start"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for b, (bx, bo, g) in enumerate(boards):
        theory = "X_wins" if g in X_WINS_GROUPS else "draw"
        sub = "P1_win_theoretical_board_start" if g in X_WINS_GROUPS else "draw_theoretical_board_start"
        win = winners[b]
        tag = {"X": "Xwon", "O": "Owon", "draw": "draw"}[win]
        result = {"X": "1-0", "O": "0-1", "draw": "1/2-1/2"}[win]
        f = root / sub / f"game_{b:06d}_{GROUP_NAMES[g]}_board{b:03d}_{tag}.pgn"
        lines = [
            '[Event "AZ vs Solver (jax)"]', '[X "Checkpoint (AlphaZero)"]', '[O "Perfect Solver"]',
            f'[Group "{GROUP_NAMES[g]}"]', f'[Theory "{theory}"]', f'[Actual "{tag}"]',
            f'[InitialBx "0x{bx:x}"]', f'[InitialBo "0x{bo:x}"]',
            f'[AzMovesAnalyzed "{int(analyzed[b])}"]', f'[AzBlunders "{len(blunders[b])}"]',
            result, "",
            " ".join(f"{i+1}.{cell_to_coord(c)}{'??' if bd else ''}"
                     for i, (c, bd) in enumerate(histories[b])),
        ]
        if blunders[b]:
            lines += ["", "=== AZ Blunders ===",
                      "(Values from X's view: +1 win, 0 draw, -1 loss)", ""]
            for x in blunders[b]:
                best = ",".join(cell_to_coord(c) for c in x["best_cells"])
                lines.append(f"Move {x['move']}: played {cell_to_coord(x['cell'])} "
                             f"value={x['chosen']:+d} best={x['best']:+d} via [{best}]")
        f.write_text("\n".join(lines) + "\n")


# ============================================================
# Driver
# ============================================================
def evaluate(checkpoint, sims=400, per_group=100, pgn=False, overrides=None, seed=0,
             out_csv=None, on_checkpoint=None):
    """out_csv: write the results CSV here (default: data/solver_eval_results_jax_<ts>.csv).
    on_checkpoint(done, total, gen): called after each checkpoint is written (live progress)."""
    overrides = overrides or {}
    ckpts = find_checkpoints(checkpoint)
    arch, game = resolve_arch(ckpts[0], overrides)
    per_group = overrides.get("per_group") or per_group

    env = Env(GameConfig(game["size"], game["win_length"], game["misere"]))

    print("Solving Misère TT...")
    solver = MisereSolver()
    solver.solve()
    boards = generate_positions(solver, per_group)
    # search-free policy/value metric: ground truth over EVERY position, built once.
    from jax_az.eval_policy import enumerate_positions, build_targets, policy_metrics
    pm_targets = build_targets(solver, enumerate_positions(solver))
    print(f"Positions: {len(boards)} ({per_group}/group)  policy-eval positions: "
          f"{pm_targets['pos_val'].shape[0]}  Checkpoints: {len(ckpts)}  sims: {sims}\n")

    # mctx search: greedy (temp 0), no Dirichlet — pure exploitation for eval.
    from jax_az import model as azmodel
    from jax_az.search import Search
    eval_cfg = SearchConfig(algorithm="muzero", num_simulations=sims,
                            temperature=0.0, dirichlet_fraction=0.0)
    m = azmodel.make_model(env, arch["num_channels"], arch["num_res_blocks"], arch["variant"])
    eval_fn = azmodel.make_eval_fn(m)
    search = Search(env, eval_fn, eval_cfg)

    def _step(variables, rng, state):
        return search.run(variables, rng, state, config=eval_cfg).action
    search_step = jax.jit(_step)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_csv:
        csv_path = out_csv
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    else:
        os.makedirs("data", exist_ok=True)
        csv_path = f"data/solver_eval_results_jax_{ts}.csv"
    summary = []
    with open(csv_path, "w") as csv:
        csv.write(_CSV_HEADER)
        for i, ckpt in enumerate(ckpts):
            gen = int(ckpt.name.split("_")[1])
            print(f"[{i+1}/{len(ckpts)}] checkpoint {gen}")
            loaded = load_checkpoint_for_inference(
                str(ckpt), arch["num_channels"], arch["num_res_blocks"],
                env.num_actions, arch["variant"])
            variables = {"params": loaded["params"], "batch_stats": loaded["batch_stats"]}

            winners, blunders, histories, analyzed = play_all(
                search_step, variables, env, boards, solver, jax.random.PRNGKey(seed))
            per_grp, totals, bl = aggregate(boards, winners, blunders, analyzed)
            elo = report(gen, per_grp, totals, bl)
            pm = policy_metrics(eval_fn, variables, env, pm_targets)
            print(f"  policy(no search): optimal_move={pm['optimal_move_rate']:.3f}  "
                  f"regret={pm['policy_regret']:.4f}  value_acc={pm['value_class_acc']:.3f}  "
                  f"value_mse={pm['value_mse']:.4f}")
            csv.write(csv_row(gen, per_grp, totals, bl, elo, pm)); csv.flush()
            summary.append((gen, *totals, elo, bl["total"]))
            if pgn:
                write_pgn("eval_games", gen, boards, winners, blunders, histories, analyzed)
                print("  PGNs -> eval_games/checkpoint_%d/" % gen)
            if on_checkpoint:
                on_checkpoint(i + 1, len(ckpts), gen)
            print()

    print("Checkpoint    W    D    L     Elo  Blunders")
    for gen, W, D, L, elo, nbl in summary:
        print(f"{gen:>10}{W:>5}{D:>5}{L:>5}{elo:>8.1f}{nbl:>10}")
    print(f"\nResults: {csv_path}")
    return summary


def main():
    d = EvalConfig()
    ap = argparse.ArgumentParser(description="JAX AZ-vs-solver eval (port of eval_against_solver.cpp)")
    ap.add_argument("checkpoint", help="a checkpoint_N folder, or a dir of them (sweeps all)")
    ap.add_argument("--sims", type=int, default=d.sims, help="MCTS simulations per AZ move")
    ap.add_argument("--per-group", type=int, default=d.per_group, help="eval boards per group (x4 groups)")
    ap.add_argument("--pgn", action="store_true", help="also write per-game annotated PGN files")
    ap.add_argument("--variant", default=None, help="override net value-head variant")
    ap.add_argument("--channels", type=int, default=None, help="override num_channels")
    ap.add_argument("--blocks", type=int, default=None, help="override num_res_blocks")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    evaluate(a.checkpoint, sims=a.sims, per_group=a.per_group, pgn=a.pgn, seed=a.seed,
             overrides={"variant": a.variant, "num_channels": a.channels,
                        "num_res_blocks": a.blocks, "per_group": a.per_group})


def demo():
    """Self-check on a tiny random net: generation classifies correctly, a full
    eval runs end-to-end, and the solver as O is never beaten by a random AZ (it
    plays perfectly, so X never wins a draw/X-win board against it -> 0 theory
    wins is fine; the assert is that aggregation + Elo are well-formed)."""
    import tempfile
    from src.models.alphazero_model import (
        create_inference_state, save_checkpoint, TrainStateWithBatchStats, get_checkpointer)
    import optax

    env = Env(GameConfig(4, 3, True))
    solver = MisereSolver(); solver.solve()
    boards = generate_positions(solver, per_group=5)
    assert len(boards) == 20, f"expected 20 boards, got {len(boards)}"
    # every generated board is non-terminal and correctly grouped
    for bx, bo, g in boards:
        is_x = _popcount(bx) == _popcount(bo)
        v = solver.get_position_value(bx, bo, is_x)
        exp = (0 if v > 0 else 2) if is_x else (1 if v < 0 else 3)
        assert g == exp and v != (-1 if is_x else 1), (bx, bo, g, v)

    # save a tiny random checkpoint and run a full (small) eval through the driver
    inf = create_inference_state(jax.random.PRNGKey(0), 8, 1, env.num_actions, "v1_scalar_mse")
    state = TrainStateWithBatchStats.create(
        apply_fn=None, params=inf["params"], tx=optax.adam(1e-3), batch_stats=inf["batch_stats"])
    d = tempfile.mkdtemp()
    save_checkpoint(state, d, step=1)
    get_checkpointer().wait_until_finished()   # save is async; flush before we read it
    summary = evaluate(os.path.join(d, "checkpoint_1"), sims=8, per_group=5,
                       overrides={"variant": "v1_scalar_mse", "num_channels": 8, "num_res_blocks": 1})
    gen, W, D, L, elo, nbl = summary[0]
    assert W + D + L == 20, (W, D, L)
    print("jax_az.eval_solver demo OK: 20 boards classified, full eval ran, W+D+L=20")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()
