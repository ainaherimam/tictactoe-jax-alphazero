# Evaluating the net against a solved game — review of `eval_solver.py`

## TL;DR

The current eval (greedy MCTS agent as X vs perfect solver as O, over 400 seeded
boards, with move-level blunder analysis) **is a legitimate agent-strength
metric, but it is the expensive way to learn the least.** For a *solved* game you
already own ground truth for **every** position; the cheapest, densest, lowest
variance signal is to score the network's policy/value **directly against the
solver with no search at all**. That metric is ~1000× cheaper, deterministic, and
contains strictly more information than the W/D/L→Elo outcome.

Recommendation: make a **search-free policy/value-vs-solver metric the primary
per-checkpoint eval**, and keep the MCTS-vs-solver game as a *secondary,
occasional* agent-level sanity check on fewer boards.

---

## 1. Is the current eval relevant?

Partially. Breaking it into its two outputs:

**Blunder analysis (good).** For each AZ (X) move it compares the chosen move's
solver value to the best solver value and records `WIN→DRAW / WIN→LOSS /
DRAW→LOSS`. This is the strongest part: a direct, move-level, ground-truth
measure of policy quality. Keep it.

**W/D/L → Elo outcome (weak / redundant).** Because O plays *perfectly* and AZ is
*greedy and deterministic*, each board produces exactly one fixed game, and its
outcome is **fully determined by the blunders AZ makes on the realized line**. So
the outcome is a lossy summary of data already computed in the blunder pass.
"Elo vs a perfect opponent" is also an awkward number — against perfection the
agent's score is bounded and the Elo mostly reflects blunder frequency, not a
rating you could compare across opponents.

**Coverage is thin and biased.** Each board contributes only the handful of
positions on *one* realized game line. The solver already gives ground truth for
**every** position in the tree — the eval pays for MCTS but throws almost all of
that ground truth away.

**It is structurally wasteful (see §3).** `play_all` runs the batched MCTS over
*all* boards every ply (`eval_solver.py:210`) but only consumes the result for
boards where X is to move; O-to-move and finished boards are searched and
discarded. Roughly two-thirds of the GPU search is wasted.

So: relevant as an *agent-level* check, but it is not the right *primary* metric
for a solved game, and its dominant cost buys its weakest output.

---

## 2. Better / complementary evaluations

The game is small and **solved**: `MisereSolver.solve()` already enumerates the
full transposition table (it prints `TT populated: N positions`). Use it.

### (A) Search-free policy & value accuracy vs solver — *make this primary*

One batched forward pass of the net over **all** canonical, non-terminal,
X-to-move positions in the solver TT. Per position, compare to solver ground
truth:

- **Optimal-move rate** — fraction where `argmax(policy)` is in the solver's
  optimal-action set. The headline number: "how often does the raw net pick a
  perfect move?"
- **Policy regret** — expected solver-value drop under the full policy
  distribution: `regret = best_value - Σ_a policy[a] · value[a]`. Penalises mass
  on bad moves, not just the argmax. A smooth, differentiable-feeling signal that
  moves every checkpoint.
- **Value sign-agreement + MSE** — does `sign(net_value)` match the solver's
  win/draw/loss verdict? Mean squared error against `{-1,0,+1}`. Tells you whether
  the value head actually knows who is winning.

Properties: deterministic, zero variance, **no MCTS**, every position
contributes (dense), and it answers the real question for a solved game — *has
the network learned it?* This is one forward pass over a few thousand positions:
milliseconds, vs minutes for the current eval.

### (B) Reachability-weighted optimal-move rate

Weight each position in (A) by how often self-play actually reaches it (estimate
from a self-play visit histogram, or uniform-by-depth as a cheap proxy). Keeps
the metric honest about positions that matter rather than rare deep lines.

### (C) Search-helps-or-hurts delta

Compute optimal-move rate from the **raw policy** vs after **MCTS** on the same
positions. If MCTS doesn't raise it (or lowers it), your search hyperparameters
(sims, c_puct, value scale) are mis-tuned — a bug class the outcome metric can't
surface.

### (D) Keep a *small* agent-level game eval (secondary)

The current MCTS-vs-solver game still has unique value: it exercises the full
deployed stack (search + temperature + masking + env stepping) end to end and
catches integration bugs (A) cannot. Demote it: run it on fewer boards
(e.g. per_group=25) and less often (every Nth checkpoint), not on every save.

**Suggested per-checkpoint dashboard:** optimal-move rate, policy regret, value
sign-agreement (all from (A)); plus the §3-optimised game eval + blunder stats
occasionally.

---

## 3. Speed / efficiency improvements for the existing game eval

Biggest win first.

1. **Adopt (A) as the routine metric and run the game eval rarely.** The fastest
   MCTS game is the one you don't run every checkpoint. (A) replaces the *signal*;
   the game eval becomes an occasional check. This alone removes most of the cost.

2. **Stop searching boards that don't need an AZ move this ply.** Today the search
   runs over all B boards every ply and discards O-to-move + finished boards
   (`eval_solver.py:204-213`). Gather the live, X-to-move boards into a
   **fixed-capacity padded batch** (size = B, pad with a dummy state), run search
   on that, scatter results back. Fixed shape ⇒ no jit recompiles, and the net
   forwards shrink to the boards that actually use the result. ~3× on the game
   eval.

3. **Cut eval sims.** `sims=400` is heavy for a 4×4 game the net largely
   memorises. Try 64–100, or switch to Gumbel with a small
   `max_num_considered_actions`. Validate the drop against metric (A) so you know
   you're not losing fidelity.

4. **Don't re-pay fixed costs per checkpoint in a sweep.** Positions are already
   generated once (good). Also build the solver once (already done). When sweeping
   N checkpoints, the only per-checkpoint work should be: load params + the (now
   smaller) batched search. The jit is already reused across checkpoints — keep it
   that way (don't let batch shape vary, per #2).

5. **Minor: reduce per-ply host↔device stalls.** `np.asarray(search_step(...))`
   forces a device sync every ply (16/checkpoint). With #2 the host solver loop is
   tiny (O(1) TT lookups), so this is minor — but if it shows up, overlap the host
   solver work with the next ply's search instead of blocking on each.

Net: #1 changes the regime (cheap dense metric every checkpoint); #2+#3 make the
remaining occasional game eval ~5–10× faster without changing what it measures.

---

## What's implemented (this change)

- **`jax_az/eval_policy.py`** — metric (A) + (C). `enumerate_positions(solver)`
  BFS-enumerates every reachable non-terminal position (to-move perspective,
  ~434k; the solver TT is alpha-beta-pruned so it's smaller). `policy_metrics`
  runs one batched net forward and returns `{optimal_move_rate, policy_regret,
  value_class_acc, value_mse}`. Standalone CLI:
  `python -m jax_az.eval_policy <ckpt|dir> [--with-search]`; `--with-search` adds
  `mcts_optimal_move_rate` (the §2(C) search-helps delta). `demo` self-checks an
  oracle scores perfectly.
- **Wired into the game eval** — `eval_solver.evaluate()` now builds the position
  targets once and writes the four policy metrics into the results CSV (new
  columns `optimal_move_rate, policy_regret, value_class_acc, value_mse`) and
  stdout for every checkpoint. The header-keyed monitor CSV parser ignores the
  extra columns, so nothing downstream breaks.
- **Game-eval speedup (§3 #2)** — `play_all` now searches only the live
  X-to-move cohort each ply (one parity group), padded to a fixed size so jit
  compiles at most twice. Deterministic eval ⇒ identical results, ~2× fewer net
  forwards. Solver-move and finished boards are no longer searched.

### Not done (deliberate, runtime knobs not code)

- **Lower eval sims (§3 #3)** — left as the existing `--sims` flag; changing the
  default would silently change every result. Validate a lower value against
  `eval_policy` before adopting.
- **Reachability weighting (§2 B)** — needs a self-play visit histogram (a
  separate pipeline) for marginal gain over the dense uniform metric; skipped.
- **Demote the game eval to occasional (§2 D, §3 #1)** — a scheduling choice in
  the monitor, not this module. The cheap `eval_policy` is what you'd run every
  checkpoint; run the game eval less often.
