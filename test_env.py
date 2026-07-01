"""Parity tests for the JAX env — the load-bearing safety net (plan.md §6).

Run:  .ven/bin/python jax_az/test_env.py

Checks, all assert-based (repo's existing __main__ test style):
  1. line masks: independent count + 4x4/k3 set matches the C++/solver LINE_MASKS
  2. parity: legal_mask / terminal / reward / step vs an independent numpy
     reference, over thousands of random *reachable* positions, every config
  3. solver anchor: env line detection == MisereSolver.has_line on 4x4 misère
  4. value sign: tiny mctx search on near-terminal positions agrees with a
     ground-truth oracle (catches the discount=-1 two-player flip, and that the
     misère/normal switch flips the value sign)
"""
import importlib.util
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_az.env import Env, GameConfig, State  # noqa: E402

CONFIGS = [
    GameConfig(3, 3, True),  GameConfig(3, 3, False),
    GameConfig(4, 3, True),  GameConfig(4, 3, False),
    GameConfig(4, 4, True),  GameConfig(4, 4, False),
    GameConfig(5, 3, True),  GameConfig(5, 3, False),
    GameConfig(5, 4, True),  GameConfig(5, 5, True),  GameConfig(5, 5, False),
]


# --------------------------------------------------------------------------- #
# Independent numpy reference (different algorithm than env's bitmask windows) #
# --------------------------------------------------------------------------- #
def ref_has_line(mask: int, size: int, k: int) -> bool:
    """True if `mask` has k-in-a-row. Coordinate walk over 4 directions —
    deliberately *not* the bitmask construction env uses, so a shared off-by-one
    can't hide."""
    occ = [(c // size, c % size) for c in range(size * size) if (mask >> c) & 1]
    occ_set = set(occ)
    for (dr, dc) in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for (r, c) in occ:
            if all((r + dr * i, c + dc * i) in occ_set for i in range(k)):
                return True
    return False


def ref_term(own: int, opp: int, cfg: GameConfig):
    """(done, reward) from the just-moved player's view — mover is `opp`."""
    size, k = cfg.size, cfg.win_length
    full = (1 << (size * size)) - 1
    made = ref_has_line(opp, size, k)
    is_full = (own | opp) == full
    done = made or is_full
    sign = -1.0 if cfg.misere else 1.0
    reward = sign if made else 0.0
    return done, reward


def ref_lines_set(size: int, k: int) -> set:
    """All line bitmasks from coordinate windows (independent of env._line_masks)."""
    out = set()
    for r in range(size):
        for c in range(size):
            for (dr, dc) in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + dr * i, c + dc * i) for i in range(k)]
                if all(0 <= rr < size and 0 <= cc < size for rr, cc in cells):
                    out.add(sum(1 << (rr * size + cc) for rr, cc in cells))
    return out


def random_reachable(cfg: GameConfig, n_games: int, seed: int):
    """Play random legal games; return (own[], opp[], action[], next_own[],
    next_opp[]) for every visited transition plus all visited states."""
    rng = np.random.default_rng(seed)
    cells = cfg.size * cfg.size
    owns, opps = [], []
    s_own, s_opp, s_act, s_nown, s_nopp = [], [], [], [], []
    for _ in range(n_games):
        own, opp = 0, 0
        owns.append(own); opps.append(opp)
        while True:
            done, _ = ref_term(own, opp, cfg)
            if done:
                break
            legal = [c for c in range(cells) if not ((own | opp) >> c) & 1]
            if not legal:
                break
            a = int(rng.choice(legal))
            n_own, n_opp = opp, own | (1 << a)   # place in own, then swap
            s_own.append(own); s_opp.append(opp); s_act.append(a)
            s_nown.append(n_own); s_nopp.append(n_opp)
            own, opp = n_own, n_opp
            owns.append(own); opps.append(opp)
    return (np.array(owns, np.int64), np.array(opps, np.int64),
            np.array(s_own, np.int64), np.array(s_opp, np.int64),
            np.array(s_act, np.int64), np.array(s_nown, np.int64),
            np.array(s_nopp, np.int64))


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_line_masks():
    for cfg in CONFIGS:
        env = Env(cfg)
        s, k = cfg.size, cfg.win_length
        w = s - k + 1
        expected_count = 2 * s * w + 2 * w * w  # rows+cols + 2 diagonals
        env_set = {int(x) for x in env.lines.tolist()}
        assert len(env.lines) == expected_count, (cfg, len(env.lines), expected_count)
        assert env_set == ref_lines_set(s, k), cfg
    # exact match against the committed C++/solver line table (4x4, k=3)
    solver = _load_solver()
    if solver is not None:
        env = Env(GameConfig(4, 3, True))
        assert {int(x) for x in env.lines.tolist()} == set(solver.LINE_MASKS)
        assert len(solver.LINE_MASKS) == 24
    print("  [ok] line masks: counts + sets match (solver-anchored on 4x4/k3)")


def test_parity():
    total = 0
    for cfg in CONFIGS:
        env = Env(cfg)
        owns, opps, s_own, s_opp, s_act, s_nown, s_nopp = random_reachable(
            cfg, n_games=400, seed=1234 + hash(cfg) % 1000)
        total += len(owns)

        # legal_mask + terminal + reward over every visited state, batched
        st = State(jnp.asarray(owns, jnp.int32), jnp.asarray(opps, jnp.int32))
        legal = np.asarray(env.legal_mask_batch(st))
        done, reward = env.terminal_and_reward_batch(st)
        done = np.asarray(done); reward = np.asarray(reward)

        cells = cfg.size * cfg.size
        for i in range(len(owns)):
            own, opp = int(owns[i]), int(opps[i])
            ref_legal = np.array([not ((own | opp) >> c) & 1 for c in range(cells)])
            assert np.array_equal(legal[i], ref_legal), (cfg, own, opp)
            rd, rr = ref_term(own, opp, cfg)
            assert bool(done[i]) == rd, (cfg, own, opp, done[i], rd)
            assert float(reward[i]) == rr, (cfg, own, opp, reward[i], rr)

        # step parity: env.step must reproduce the reference swap exactly
        if len(s_own):
            st2 = State(jnp.asarray(s_own, jnp.int32), jnp.asarray(s_opp, jnp.int32))
            ns = env.step_batch(st2, jnp.asarray(s_act, jnp.int32))
            assert np.array_equal(np.asarray(ns.own), s_nown.astype(np.int32)), cfg
            assert np.array_equal(np.asarray(ns.opp), s_nopp.astype(np.int32)), cfg
    print(f"  [ok] parity: {total} reachable states match numpy reference (all configs)")


def test_solver_anchor():
    solver = _load_solver()
    if solver is None:
        print("  [skip] solver anchor: misere_solver.py not importable")
        return
    cfg = GameConfig(4, 3, True)
    env = Env(cfg)
    rng = np.random.default_rng(7)
    n = 0
    for _ in range(2000):
        bits = int(rng.integers(0, 1 << 16))
        env_has = bool(env.has_line(jnp.int32(bits)))
        assert env_has == solver.has_line(bits), bits
        n += 1
    print(f"  [ok] solver anchor: env.has_line == solver.has_line on {n} random 4x4 boards")


def test_value_sign():
    import mctx  # noqa: F401
    from jax_az.search import Search
    cases = [GameConfig(4, 3, True), GameConfig(4, 3, False), GameConfig(3, 3, True)]
    for cfg in cases:
        env = Env(cfg)
        positions = _find_near_terminal(cfg, want=8, seed=99)
        assert positions, f"no near-terminal positions found for {cfg}"

        owns = jnp.asarray([p[0] for p in positions], jnp.int32)
        opps = jnp.asarray([p[1] for p in positions], jnp.int32)
        oracle = np.array([p[2] for p in positions])          # max achievable reward
        best_actions = [p[3] for p in positions]              # set of optimal actions
        st = State(owns, opps)

        def zero_eval(params, planes):
            b = planes.shape[0]
            return jnp.zeros((b, env.num_actions)), jnp.zeros((b,))

        search = Search(env, zero_eval)
        out = search.run_muzero(None, jax.random.PRNGKey(0), st,
                                num_simulations=2 * env.num_actions)
        chosen = np.asarray(out.action)
        root_val = np.asarray(out.search_tree.node_values[:, 0])

        for i in range(len(positions)):
            assert chosen[i] in best_actions[i], (cfg, i, chosen[i], best_actions[i])
            assert np.sign(round(root_val[i])) == np.sign(oracle[i]), (
                cfg, i, root_val[i], oracle[i])
    print("  [ok] value sign: mctx search picks optimal move + root value sign correct "
          "(misère and normal)")


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _find_near_terminal(cfg: GameConfig, want: int, seed: int):
    """Reachable, non-terminal positions where *every* legal move is immediately
    terminal. Oracle value (negamax, 1 ply) = max over moves of that move's
    reward; optimal action set = argmax. Independent of any network."""
    rng = np.random.default_rng(seed)
    cells = cfg.size * cfg.size
    found = {}
    for _ in range(40000):
        own, opp = 0, 0
        while True:
            if ref_term(own, opp, cfg)[0]:
                break
            legal = [c for c in range(cells) if not ((own | opp) >> c) & 1]
            rewards, all_term = [], True
            for a in legal:
                n_own, n_opp = opp, own | (1 << a)
                d, r = ref_term(n_own, n_opp, cfg)
                rewards.append(r)
                all_term &= d
            if all_term and legal:
                best = max(rewards)
                acts = {legal[j] for j in range(len(legal)) if rewards[j] == best}
                found[(own, opp)] = (own, opp, best, acts)
                break
            a = int(rng.choice(legal))
            own, opp = opp, own | (1 << a)
        if len(found) >= want:
            break
    return list(found.values())[:want]


_SOLVER = "uninit"
def _load_solver():
    """Load src/core/solver/misere_solver.py by path (avoids package-import fuss)."""
    global _SOLVER
    if _SOLVER != "uninit":
        return _SOLVER
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "src", "core", "solver", "misere_solver.py")
    try:
        spec = importlib.util.spec_from_file_location("misere_solver", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SOLVER = mod
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not load solver: {e}")
        _SOLVER = None
    return _SOLVER


if __name__ == "__main__":
    print("env parity tests")
    test_line_masks()
    test_parity()
    test_solver_anchor()
    test_value_sign()
    print("ALL TESTS PASSED")
