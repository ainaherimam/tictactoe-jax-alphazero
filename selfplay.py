"""Batched self-play: B parallel games via lax.scan, producing training tuples.

Ports C++ `Game::play()` + `PositionPool::finalize_game` (next_plan.md §0, §3) to
one jitted scan. Each ply, BEFORE moving, we record the to-move player's
`(board, pi, mask)` — exactly C++'s `collect_position`. `z` is backfilled once,
vectorized, by `assign_z` (the misère/negamax sign flip — the load-bearing bit).

`play_batch` returns one flat `[B*cells, ...]` block (padding rows included) ready
for `replay.add`; padding rows carry `valid=0` and are never sampled.

Sign convention follows env.terminal_and_reward: `reward` at the terminating ply is
the *mover's* view (misère: completing a line -> -1). `z` for a position is +reward*
for the player who shares the terminal mover's parity, -reward* for the opponent.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from jax_az import features


class Rollout(NamedTuple):
    """Per-ply scan outputs, stacked along axis 0 = ply (length env.cells)."""
    board: jax.Array   # [cells, B, 2, size, size] f32 — to-move perspective
    pi: jax.Array      # [cells, B, A] f32 — mctx visit-count policy
    mask: jax.Array    # [cells, B, A] f32 — legal moves at that ply
    action: jax.Array  # [cells, B] int32 — sampled move
    reward: jax.Array  # [cells, B] f32 — mover's-view reward of this transition
    newly: jax.Array   # [cells, B] bool — game terminated AT this ply (once per game)
    valid: jax.Array   # [cells, B] f32 — 1 if game was live before this ply


def rollout(params, rng, env, search, B, search_cfg) -> Rollout:
    """Run B games to completion. Length-`cells` scan: a full board is always
    terminal, so no truncation case. Finished games are frozen (step is a no-op)
    so their board never fills past termination — search always has a legal move."""
    size, A = env.size, env.num_actions

    def ply(carry, t):
        state, done, rng = carry
        rng, k = jax.random.split(rng)
        out = search.run(params, k, state, ply=t, config=search_cfg)
        pi, action = out.action_weights, out.action.astype(jnp.int32)
        rec = (features.planes_batch(state, size),     # board, to-move view
               pi.astype(jnp.float32),
               env.legal_mask_batch(state).astype(jnp.float32),
               action,
               (~done).astype(jnp.float32))            # valid = was live this ply
        stepped = env.step_batch(state, action)
        nxt = State_where(done, state, stepped)         # freeze finished games
        nd, reward = env.terminal_and_reward_batch(nxt)
        newly = nd & ~done
        return (nxt, done | nd, rng), rec + (reward.astype(jnp.float32), newly)

    state = env.init_batch(B)
    done = jnp.zeros(B, bool)
    _, (board, pi, mask, action, valid, reward, newly) = lax.scan(
        ply, (state, done, rng), jnp.arange(env.cells, dtype=jnp.int32))
    return Rollout(board, pi, mask, action, reward, newly, valid)


def State_where(cond, a, b):
    """Pick env State `a` where cond[B] else `b`, field-wise (jit-friendly)."""
    c = cond
    return a.__class__(jnp.where(c, a.own, b.own), jnp.where(c, a.opp, b.opp))


def assign_z(reward, newly):
    """Backfill z[cells, B] from per-ply reward + the one `newly`-terminal ply.

    T = the ply that ended each game; r* = mover's-view reward there (misère: -1,
    draw: 0). Players alternate, so ply t belongs to parity t%2:
        z[t] = r*   if t%2 == T%2   (same player as the terminal mover)
             = -r*  otherwise        (the opponent)
             = 0    when r* == 0      (draw — falls out for free)
    This replaces PositionPool::finalize_game. The most likely place for a silent
    sign flip; gated by the z-parity test below."""
    cells = reward.shape[0]
    T = jnp.argmax(newly.astype(jnp.int32), axis=0)         # [B] terminal ply
    rstar = jnp.sum(reward * newly.astype(jnp.float32), axis=0)  # [B] (one nonzero)
    t_idx = jnp.arange(cells, dtype=jnp.int32)[:, None]     # [cells, 1]
    same = (t_idx % 2) == (T[None, :] % 2)                  # [cells, B]
    return jnp.where(same, rstar[None, :], -rstar[None, :])  # [cells, B]


def play_batch(params, rng, env, search, B, search_cfg):
    """B self-play games -> one flat `[B*cells, ...]` block for `replay.add`.

    Returns `(boards, pi, z, mask, valid)` — order matches `replay.add`'s signature.
    Padding plies (after a game ended) carry valid=0 and are dropped by sampling."""
    r = rollout(params, rng, env, search, B, search_cfg)
    z = assign_z(r.reward, r.newly)                         # [cells, B]
    flat = lambda a: a.reshape((-1,) + a.shape[2:])         # [cells, B, ...] -> [cells*B, ...]
    return flat(r.board), flat(r.pi), flat(z), flat(r.mask), flat(r.valid)


def demo():
    """Self-checks: z-sign parity (vs the C++ winner-parity rule) + well-formed
    tuples. The Phase-3a gate (next_plan.md §6, tests 1 & 2)."""
    from jax_az.env import Env, GameConfig
    from jax_az.model import make_az_search

    env = Env(GameConfig(4, 3, True))                       # misère 4x4
    search, variables = make_az_search(env, num_channels=8, num_res_blocks=1)
    B = 6
    r = rollout(variables, jax.random.PRNGKey(0), env, search, B, search.config)
    z = assign_z(r.reward, r.newly)

    valid = r.valid > 0.5
    newly = r.newly
    cells = env.cells

    # Every game terminates within `cells` plies (full board is terminal).
    assert bool(jnp.all(newly.sum(0) == 1)), "each game must terminate exactly once"

    # --- Test 1: z-sign parity vs an independent winner-parity derivation ----
    T = jnp.argmax(newly.astype(jnp.int32), axis=0)         # [B]
    rstar = jnp.sum(r.reward * newly.astype(jnp.float32), axis=0)  # [B]
    draw = rstar == 0.0
    # misère: the terminal mover (parity T%2) LOSES -> winner is the opponent.
    winner_parity = jnp.where(env.misere, (T % 2) ^ 1, T % 2)    # [B]
    t_idx = jnp.arange(cells, dtype=jnp.int32)[:, None]
    ref_z = jnp.where(draw[None, :], 0.0,
                      jnp.where((t_idx % 2) == winner_parity[None, :], 1.0, -1.0))
    assert bool(jnp.all(jnp.where(valid, z == ref_z, True))), "z sign mismatch!"

    # --- Test 2: well-formed tuples (for valid rows) ------------------------
    v = valid
    assert bool(jnp.all(jnp.where(v, jnp.abs(r.pi.sum(-1) - 1.0) < 1e-4, True))), "pi !~ 1"
    assert bool(jnp.all((r.mask == 0) | (r.mask == 1))), "mask not 0/1"
    chosen_legal = jnp.take_along_axis(r.mask, r.action[..., None], -1)[..., 0]
    assert bool(jnp.all(jnp.where(v, chosen_legal == 1.0, True))), "illegal move sampled"
    assert bool(jnp.all(jnp.where(v[..., None], jnp.isin(z[..., None], jnp.array([-1., 0., 1.])), True)))
    b = r.board
    assert bool(jnp.all((b == 0) | (b == 1))), "board not 0/1"
    assert bool(jnp.all(jnp.where(v, (b[:, :, 0] * b[:, :, 1]).reshape(cells, B, -1).sum(-1) == 0, True))), \
        "planes overlap"

    # play_batch returns a flat block of the right shape
    boards, pi, zf, mask, validf = play_batch(
        variables, jax.random.PRNGKey(1), env, search, B, search.config)
    assert boards.shape == (cells * B, 2, env.size, env.size)
    assert pi.shape == mask.shape == (cells * B, env.num_actions)
    assert zf.shape == validf.shape == (cells * B,)
    print(f"jax_az.selfplay demo OK: {B} games, z-sign parity + well-formed tuples")


if __name__ == "__main__":
    demo()
