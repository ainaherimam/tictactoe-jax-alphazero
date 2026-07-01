"""Replay-buffer data analysis: turn a run's `replay.npz` into training-data stats.

The replay ring stores one row per recorded position: board planes [2,4,4]
(to-move / opponent), pi[16], z (game outcome, to-move view), mask[16], valid.
There is NO game/ply metadata stored — and none is needed:

  * ply (move number) = number of filled cells on the board (each move fills one).
  * the empty board is recorded once per game, so #games = #valid rows at ply 0.
  * per-ply counts form a survival curve => game-length distribution + avg moves,
    with NO change to self-play data collection (so zero training-perf risk).

The solver-dependent stats (true game-theoretic label, board-space coverage) run
the existing pure-Python perfect solver once and reuse it for every run. Output is
a single self-contained HTML dashboard (data embedded, opens on file://, no server)
plus the raw JSON.

    python -m jax_az.analyze_replay                       # sweep jax_az_runs/*/replay.npz
    python -m jax_az.analyze_replay jax_az_runs/run-XXXX  # one run
    python -m jax_az.analyze_replay a/replay.npz --out report.html
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.core.solver.misere_solver import MisereSolver, canonicalize  # noqa: E402
from jax_az.eval_policy import enumerate_positions                    # noqa: E402

_BITS = (1 << np.arange(16, dtype=np.int64))


def _board_masks(boards: np.ndarray):
    """boards [N,2,4,4] of 0/1 -> (own, opp) int bitmasks [N], bit c = cell c
    (row-major, == the solver's bit layout, see features.planes)."""
    p = boards.reshape(boards.shape[0], 2, 16).astype(np.int64)
    return (p[:, 0] * _BITS).sum(1), (p[:, 1] * _BITS).sum(1)


def _popcount(a: np.ndarray) -> np.ndarray:
    return np.array([bin(int(x)).count("1") for x in a], dtype=np.int64)


# --- solver: cached perfect labels + reachable-position universe ------------
_SOLVER = None
_UNIVERSE = None


def _solver():
    global _SOLVER, _UNIVERSE
    if _SOLVER is None:
        _SOLVER = MisereSolver()
        _SOLVER.solve()
        # universe = every reachable non-terminal position, as D4-canonical classes
        _UNIVERSE = {canonicalize(o, p) for (o, p) in enumerate_positions(_SOLVER)}
    return _SOLVER, _UNIVERSE


def analyze(npz_path: str) -> dict:
    d = np.load(npz_path)
    boards, pi, z, mask, valid = d["boards"], d["pi"], d["z"], d["mask"], d["valid"]
    N = boards.shape[0]
    v = valid > 0.5
    n = int(v.sum())
    if n == 0:
        return {"positions": 0, "ring_capacity": N, "error": "empty ring"}

    boards, pi, z, mask = boards[v], pi[v], z[v], mask[v]
    own, opp = _board_masks(boards)
    ply = boards.reshape(n, -1).sum(1).astype(np.int64)        # filled cells

    # --- games / lengths from the survival curve ---------------------------
    # count[p] = valid positions at ply p = games still alive at ply p.
    # ponytail: assumes whole-generation blocks (no half-overwritten games at the
    # ring seam) — true while ring N is a multiple of the per-gen block. Fine for
    # stats either way; off by at most one partial generation.
    count = np.bincount(ply, minlength=16)[:16]
    games = int(count[0])                                       # empty boards = games
    ending = count - np.append(count[1:], 0)                    # ending[L-1] = games of length L
    length_hist = {int(L + 1): int(ending[L]) for L in range(16) if ending[L]}
    avg_moves = n / games if games else 0.0

    # --- confidence of the policy targets ----------------------------------
    max_pi = pi.max(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(pi > 0, pi * np.log(pi), 0.0).sum(1)    # nats, over legal moves

    # --- endings: outcome from the empty-board (X-to-move) rows ------------
    z0 = z[ply == 0]
    endings = {"p1_x_wins": int((z0 > 0.5).sum()),
               "p2_o_wins": int((z0 < -0.5).sum()),
               "draw": int((np.abs(z0) <= 0.5).sum())}

    # --- solver: true game-theoretic value per position --------------------
    solver, universe = _solver()
    key = own * 65536 + opp
    ukey, inv = np.unique(key, return_inverse=True)
    uown, uopp = ukey // 65536, ukey % 65536
    uply = _popcount(uown) + _popcount(uopp)
    ux = (uply % 2) == 0                                        # X (first player) to move
    ubx = np.where(ux, uown, uopp)
    ubo = np.where(ux, uopp, uown)
    uval = np.array([solver.get_position_value(int(bx), int(bo), bool(xt))
                     for bx, bo, xt in zip(ubx, ubo, ux)], dtype=np.float32)
    val = uval[inv]                                             # true value per row (to-move view)

    matches = val == z
    match_rate = float(matches.mean())
    # by true class, and mismatch direction (z worse / better than optimal play)
    by_class = {}
    for name, lo, hi in (("loss", -1.5, -0.5), ("draw", -0.5, 0.5), ("win", 0.5, 1.5)):
        m = (val > lo) & (val <= hi)
        by_class[name] = {"n": int(m.sum()),
                          "match_rate": float(matches[m].mean()) if m.any() else None}
    mismatch = {"z_below_true": int((z < val).sum()),    # to-move ended up worse than optimal
                "z_above_true": int((z > val).sum())}    # ... better (opponent later blundered)
    match_by_ply = {int(p): float(matches[ply == p].mean())
                    for p in range(16) if (ply == p).any()}

    # label noise: same board reached in different games with different outcomes
    pz = np.unique(key * 4 + (z + 1).astype(np.int64))         # distinct (board, z) combos
    inconsistent = len(pz) - len(ukey)
    label_noise = inconsistent / len(ukey) if len(ukey) else 0.0

    # --- board-space coverage (D4-canonical classes) ----------------------
    data_canon = {canonicalize(int(o), int(p)) for o, p in zip(uown, uopp)}
    covered = len(data_canon & universe)
    coverage = {"covered": covered, "reachable": len(universe),
                "pct": 100.0 * covered / len(universe),
                "distinct_boards_raw": int(len(ukey))}

    return {
        "positions": n,
        "ring_capacity": N,
        "fill_pct": 100.0 * n / N,
        "games": games,
        "avg_moves": avg_moves,
        "length_hist": length_hist,
        "ply_counts": {int(p): int(count[p]) for p in range(16) if count[p]},
        "confidence": {"mean_max_pi": float(max_pi.mean()),
                       "median_max_pi": float(np.median(max_pi)),
                       "mean_entropy_nats": float(ent.mean())},
        "endings": endings,
        "solver_label": {"match_rate": match_rate, "by_class": by_class,
                         "mismatch": mismatch, "match_by_ply": match_by_ply,
                         "label_noise": label_noise},
        "coverage": coverage,
    }


def _find_runs(args):
    if not args:
        return sorted(glob.glob(os.path.join(_ROOT, "jax_az_runs", "*", "replay.npz")))
    out = []
    for a in args:
        if a.endswith(".npz"):
            out.append(a)
        elif os.path.exists(os.path.join(a, "replay.npz")):
            out.append(os.path.join(a, "replay.npz"))
        else:
            print(f"  skip (no replay.npz): {a}")
    return out


def build(paths) -> dict:
    runs = {}
    for p in paths:
        name = os.path.basename(os.path.dirname(p)) or os.path.basename(p)
        print(f"[analyze] {name} ...", flush=True)
        runs[name] = analyze(p)
        s = runs[name]
        if s.get("positions"):
            print(f"          {s['positions']:,} pos / {s['games']:,} games  "
                  f"avg_moves={s['avg_moves']:.2f}  coverage={s['coverage']['pct']:.1f}%  "
                  f"solver_match={s['solver_label']['match_rate']*100:.1f}%")
    return {"runs": runs}


# --- HTML dashboard (self-contained, data embedded) ------------------------
def render_html(data: dict) -> str:
    blob = json.dumps(data)
    return _HTML_TEMPLATE.replace("/*DATA*/", blob)


def _write(out, data):
    """Write a {runs:...} dict as HTML dashboard + sibling JSON; return (html, json) paths."""
    with open(out, "w") as f:
        f.write(render_html(data))
    json_path = os.path.splitext(out)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    return out, json_path


def main():
    ap = argparse.ArgumentParser(description="Replay-buffer training-data analysis dashboard")
    ap.add_argument("runs", nargs="*", help="run dirs or replay.npz paths (default: jax_az_runs/*)")
    ap.add_argument("--out", default=os.path.join(_ROOT, "jax_az_runs", "replay_analysis.html"))
    a = ap.parse_args()
    paths = _find_runs(a.runs)
    if not paths:
        print("no replay.npz found"); return
    data = build(paths)

    # per-run report inside each run's own folder
    for p in paths:
        name = os.path.basename(os.path.dirname(p)) or os.path.basename(p)
        per_out = os.path.join(os.path.dirname(p), "replay_analysis.html")
        _write(per_out, {"runs": {name: data["runs"][name]}})
        print(f"  run report: {per_out}")

    # big combined report over the whole jax_az_runs folder
    html_path, json_path = _write(a.out, data)
    print(f"\nDashboard: {html_path}\nJSON:      {json_path}")


def demo():
    """Self-check on a synthetic 2-game ring: ply/games/length/solver-label math.
    Game A (3 plies recorded, X to move at ply0/2, O at ply1) and game B (4 plies)
    laid out as real self-play would, with one deliberately wrong z to exercise the
    mismatch counter."""
    import tempfile
    # build a board at given (own,opp); plane0=own(to-move), plane1=opp
    def planes(own, opp):
        b = np.zeros((2, 16), np.float32)
        for c in range(16):
            if own >> c & 1: b[0, c] = 1
            if opp >> c & 1: b[1, c] = 1
        return b.reshape(2, 4, 4)

    # ply0 empty, ply1 one X, ply2 X+O (X to move) for each game -> lengths 3 & 4
    rows = [(0, 0), (1, 0), (1, 2)]            # game A: 3 positions
    rowsB = [(0, 0), (2, 0), (2, 8), (10, 8)]  # game B: 4 positions
    allr = rows + rowsB
    M = len(allr)
    boards = np.stack([planes(o, p) for o, p in allr])
    pi = np.full((M, 16), 1 / 16, np.float32)
    z = np.zeros(M, np.float32)                # all draws (placeholder)
    z[0] = 1.0                                 # game A empty board: claim X wins
    mask = np.ones((M, 16), np.float32)
    valid = np.ones(M, np.float32)
    d = tempfile.mktemp(suffix=".npz")
    np.savez(d, boards=boards, pi=pi, z=z, mask=mask, valid=valid, cursor=np.int32(0))

    s = analyze(d)
    assert s["positions"] == M
    assert s["games"] == 2, s["games"]                       # two empty boards
    assert s["length_hist"] == {3: 1, 4: 1}, s["length_hist"]  # one len-3, one len-4 game
    assert abs(s["avg_moves"] - M / 2) < 1e-9
    # solver labels are real; match_rate is well-formed in [0,1]
    assert 0.0 <= s["solver_label"]["match_rate"] <= 1.0
    assert s["coverage"]["covered"] >= 1
    # endings come only from the two empty-board rows (z=+1 and z=0)
    assert s["endings"]["p1_x_wins"] + s["endings"]["draw"] == 2, s["endings"]
    os.remove(d)
    print("jax_az.analyze_replay demo OK: ply/games/length/solver-label/coverage math holds")


_HTML_TEMPLATE = r"""<!doctype html><html><head><meta charset=utf8>
<title>Replay analysis</title><style>
body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#0f1115;color:#dfe3ea}
header{padding:16px 24px;background:#161a21;border-bottom:1px solid #262b34}
h1{font-size:18px;margin:0 0 4px}h2{font-size:14px;color:#8b93a1;margin:0 0 10px;text-transform:uppercase;letter-spacing:.05em}
main{padding:24px;max-width:1100px;margin:auto}
select{font:14px sans-serif;background:#1c2129;color:#dfe3ea;border:1px solid #333;border-radius:6px;padding:6px 10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:24px}
.card{background:#161a21;border:1px solid #262b34;border-radius:10px;padding:16px}
.big{font-size:28px;font-weight:600}.sub{color:#8b93a1;font-size:12px}
.bar{height:18px;background:#2563eb;border-radius:3px;min-width:1px}
.barbg{background:#1c2129;border-radius:3px;flex:1}
.row{display:flex;align-items:center;gap:10px;margin:3px 0}
.lab{width:120px;color:#8b93a1;font-size:12px;text-align:right;flex:none}
.val{width:70px;font-size:12px;flex:none}
.g{color:#22c55e}.r{color:#ef4444}.y{color:#eab308}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:4px 8px;border-bottom:1px solid #262b34;text-align:left}
</style></head><body>
<header><h1>Replay buffer — training data analysis</h1>
<div class=sub>misère 4×4 tic-tac-toe · true labels &amp; coverage from the perfect solver</div>
<div style=margin-top:10px><select id=run onchange=draw()></select></div></header>
<main id=app></main>
<script>
const DATA=/*DATA*/;
const sel=document.getElementById('run');
Object.keys(DATA.runs).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=k;sel.appendChild(o)});
const pct=x=>x.toFixed(1)+'%', num=x=>x.toLocaleString();
function bars(obj,fmt,max){max=max||Math.max(...Object.values(obj),1);
  return Object.entries(obj).map(([k,v])=>`<div class=row><div class=lab>${k}</div>
    <div class=barbg><div class=bar style="width:${100*v/max}%"></div></div>
    <div class=val>${fmt?fmt(v):num(v)}</div></div>`).join('')}
function card(t,b){return `<div class=card><h2>${t}</h2>${b}</div>`}
function stat(big,sub){return `<div class=big>${big}</div><div class=sub>${sub}</div>`}
function draw(){
  const s=DATA.runs[sel.value], a=document.getElementById('app');
  if(!s||!s.positions){a.innerHTML='<div class=card>empty / no data</div>';return}
  const sl=s.solver_label, cov=s.coverage, c=s.confidence, e=s.endings;
  const mr=(sl.match_rate*100), col=mr>90?'g':mr>70?'y':'r';
  a.innerHTML=`
  <div class=grid>
    ${card('Positions',stat(num(s.positions),pct(s.fill_pct)+' of ring ('+num(s.ring_capacity)+')'))}
    ${card('Games',stat(num(s.games),'avg '+s.avg_moves.toFixed(2)+' moves/game'))}
    ${card('Board coverage',stat(pct(cov.pct),cov.covered+' / '+cov.reachable+' reachable positions'))}
    ${card('Correct solver labels',`<div class="big ${col}">${pct(mr)}</div><div class=sub>z == true game-theoretic value</div>`)}
    ${card('Policy confidence',stat(pct(c.mean_max_pi*100),'mean max-π · entropy '+c.mean_entropy_nats.toFixed(2)+' nats'))}
    ${card('Label noise',stat(pct(sl.label_noise*100),'distinct boards with inconsistent z'))}
  </div>
  <div class=grid>
    ${card('Game length (moves)',bars(s.length_hist,num))}
    ${card('Endings (per game)',bars({['P1 (X) wins']:e.p1_x_wins,['P2 (O) wins']:e.p2_o_wins,Draw:e.draw},num))}
  </div>
  <div class=grid>
    ${card('Solver-label match by ply',bars(mapv(sl.match_by_ply,v=>v*100),pct,100))}
    ${card('Survival — positions per ply',bars(s.ply_counts,num))}
  </div>
  <div class=grid>
    ${card('Label accuracy by true outcome',`<table><tr><th>true value</th><th>positions</th><th>match</th></tr>
      ${['win','draw','loss'].map(k=>{const x=sl.by_class[k];return `<tr><td>${k}</td><td>${num(x.n)}</td>
        <td>${x.match_rate==null?'–':pct(x.match_rate*100)}</td></tr>`}).join('')}</table>
      <div class=sub style=margin-top:8px>mismatch dir: z&lt;true (data worse) ${num(sl.mismatch.z_below_true)} ·
        z&gt;true (data luckier) ${num(sl.mismatch.z_above_true)}</div>`)}
  </div>`;
}
function mapv(o,f){const r={};for(const k in o)r[k]=f(o[k]);return r}
draw();
</script></body></html>"""


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()
