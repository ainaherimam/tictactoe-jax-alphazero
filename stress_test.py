"""Stress + adversarial tests for the JAX env — push every config through
large vmapped self-play and hunt for anything that can go wrong (plan.md §11).

Run:  .ven/bin/python jax_az/stress_test.py

What it hammers:
  * big batched lax.scan self-play (10k-100k parallel games) per config
  * invariants every ply: only legal moves taken, no cell double-occupied,
    masks stay in [0, 2^cells) (int32 overflow guard, esp. 5x5 = 25 bits),
    every game terminates, reward in {-1,0,+1}, at most one outcome per game
  * planes() always 0/1, the two planes disjoint, right shape, no NaN/inf
  * jit determinism (same key -> identical games)
  * mctx never returns an illegal action; action_weights finite, sum to 1,
    zero on illegal cells
  * hand-built edge cases: empty board, immediate terminal, draw, single move
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jax_az.env import Env, GameConfig, State  # noqa: E402
from jax_az import features                      # noqa: E402

# (config, batch). Smaller batch for bigger boards to keep CPU runtime sane.
WORKLOAD = [
    (GameConfig(3, 3, True), 100_000), (GameConfig(3, 3, False), 100_000),
    (GameConfig(4, 3, True), 50_000),  (GameConfig(4, 3, False), 50_000),
    (GameConfig(4, 4, True), 50_000),
    (GameConfig(5, 3, True), 20_000),  (GameConfig(5, 4, True), 20_000),
    (GameConfig(5, 5, True), 20_000),  (GameConfig(5, 5, False), 20_000),
]


def make_player(env: Env, max_plies: int):
    """jitted random-legal self-play for B games -> (final_state, infos)."""
    def play(key, B):
        state = env.init_batch(B)
        done0 = jnp.zeros((B,), bool)

        def body(carry, _):
            state, done, key = carry
            key, sub = random.split(key)
            legal = env.legal_mask_batch(state)                  # [B, A]
            a = random.categorical(sub, jnp.where(legal, 0.0, -1e9), axis=-1)
            ns = env.step_batch(state, a)
            # freeze finished games (don't advance them)
            nxt = State(jnp.where(done, state.own, ns.own),
                        jnp.where(done, state.opp, ns.opp))
            nd, reward = env.terminal_and_reward_batch(nxt)
            newly = nd & ~done
            chosen_legal = jnp.take_along_axis(legal, a[:, None], 1)[:, 0] | done
            info = dict(own=state.own, opp=state.opp,
                        chosen_legal=chosen_legal,
                        reward=jnp.where(newly, reward, 0.0),
                        newly=newly)
            return (nxt, done | nd, key), info

        (final, done_f, _), infos = lax.scan(
            body, (state, done0, key), None, length=max_plies)
        return final, done_f, infos
    return jax.jit(play, static_argnums=1)


def check_selfplay(cfg: GameConfig, B: int):
    env = Env(cfg)
    cells = env.cells
    play = make_player(env, max_plies=cells + 1)
    final, done_f, infos = play(random.PRNGKey(0), B)

    own = np.asarray(infos["own"]); opp = np.asarray(infos["opp"])  # [T, B]
    chosen_legal = np.asarray(infos["chosen_legal"])
    reward = np.asarray(infos["reward"])
    newly = np.asarray(infos["newly"])

    assert bool(jnp.all(done_f)), f"{cfg}: not all games terminated in {cells+1} plies"
    assert chosen_legal.all(), f"{cfg}: an illegal action was selected"
    assert np.all((own & opp) == 0), f"{cfg}: a cell was occupied by both players"
    assert own.min() >= 0 and opp.min() >= 0, f"{cfg}: negative mask (int32 overflow)"
    assert own.max() < (1 << cells) and opp.max() < (1 << cells), f"{cfg}: mask overflow"
    assert np.all(np.isin(reward, [-1.0, 0.0, 1.0])), f"{cfg}: reward outside {{-1,0,1}}"
    # at most one terminal transition recorded per game
    assert newly.sum(axis=0).max() <= 1, f"{cfg}: game ended more than once"

    # planes: 0/1, disjoint, right shape, finite
    pl = np.asarray(features.planes_batch(final, env.size))
    assert pl.shape == (B, 2, env.size, env.size), f"{cfg}: bad planes shape {pl.shape}"
    assert np.all(np.isin(pl, [0.0, 1.0])), f"{cfg}: planes not binary"
    assert np.all(pl[:, 0] + pl[:, 1] <= 1.0), f"{cfg}: planes overlap"
    assert np.all(np.isfinite(pl)), f"{cfg}: non-finite plane"

    # jit determinism: same key -> identical games
    f2, _, _ = play(random.PRNGKey(0), B)
    assert np.array_equal(np.asarray(final.own), np.asarray(f2.own)), f"{cfg}: nondeterministic"

    n_term = int(newly.sum())
    print(f"  [ok] {cfg!s:<42} B={B:>6}  plies<= {cells+1:>2}  terminals={n_term}")
    return final


def _sample_nonterminal(env: Env, key, B: int):
    """A batch of random reachable *non-terminal* states with >=1 legal move."""
    play = make_player(env, max_plies=max(1, env.cells // 2))
    _, _, infos = play(key, B)
    own = np.asarray(infos["own"])[-1]; opp = np.asarray(infos["opp"])[-1]
    st = State(jnp.asarray(own, jnp.int32), jnp.asarray(opp, jnp.int32))
    legal = np.asarray(env.legal_mask_batch(st))
    nonterminal = ~np.asarray(env.terminal_and_reward_batch(st)[0])
    keep = nonterminal & (legal.sum(1) > 0)
    return State(st.own[keep], st.opp[keep]), legal[keep]


def _assert_policy_out(out, legal: np.ndarray, label: str):
    """Invariants every mctx policy output must satisfy, for any hyperparameters."""
    actions = np.asarray(out.action)
    weights = np.asarray(out.action_weights)
    picked_legal = legal[np.arange(len(actions)), actions]
    assert picked_legal.all(), f"{label}: mctx picked an illegal action"
    assert np.all(np.isfinite(weights)), f"{label}: non-finite action_weights"
    assert np.allclose(weights.sum(1), 1.0, atol=1e-4), f"{label}: weights !sum to 1"
    assert np.all(weights[~legal] <= 1e-6), f"{label}: weight on illegal cell"


def check_mctx_legal(cfg: GameConfig, B: int = 4096, sims: int = 32):
    """mctx must never choose an occupied cell; weights finite, normalized, masked."""
    from jax_az.search import Search
    env = Env(cfg)
    st, legal = _sample_nonterminal(env, random.PRNGKey(1), B)
    if st.own.shape[0] == 0:
        print(f"  [skip] mctx-legal {cfg}: no non-terminal states sampled")
        return

    def zero_eval(params, planes):
        b = planes.shape[0]
        return jnp.zeros((b, env.num_actions)), jnp.zeros((b,))

    search = Search(env, zero_eval)
    out = search.run_muzero(None, random.PRNGKey(2), st, num_simulations=sims)
    _assert_policy_out(out, legal, str(cfg))
    print(f"  [ok] mctx legality {cfg!s:<38} states={st.own.shape[0]:>5} sims={sims}")


def check_az_net_search(cfg: GameConfig, B: int = 128):
    """The real AlphaZero net guiding mctx, swept across every search knob.

    Proves: (1) the net wires in as eval_fn with the right shapes; (2) every
    SearchConfig field is honored end-to-end (legal/finite/normalized output for
    each, and num_simulations is reflected in the tree's visit budget)."""
    from jax_az.model import make_model, init_variables, make_eval_fn
    from jax_az.search import Search, SearchConfig
    env = Env(cfg)
    st, legal = _sample_nonterminal(env, random.PRNGKey(7), B)
    if st.own.shape[0] == 0:
        print(f"  [skip] az-net {cfg}: no non-terminal states sampled")
        return

    # small net keeps the CPU grid fast; arch is identical to alphazero_model.py
    model = make_model(env, num_channels=16, num_res_blocks=2)
    variables = init_variables(random.PRNGKey(0), model, env)
    eval_fn = make_eval_fn(model)

    # eval_fn contract: log-prob policy [n, A] (rows sum to 1 under exp) + scalar value
    planes = features.planes_batch(st, env.size)
    logp, v = eval_fn(variables, planes)
    n = st.own.shape[0]
    assert logp.shape == (n, env.num_actions), f"{cfg}: bad policy shape {logp.shape}"
    assert v.shape == (n,), f"{cfg}: bad value shape {v.shape}"
    lp = np.asarray(logp); vv = np.asarray(v)
    assert np.all(np.isfinite(lp)) and np.all(np.isfinite(vv)), f"{cfg}: non-finite net output"
    assert np.allclose(np.exp(lp).sum(1), 1.0, atol=1e-4), f"{cfg}: policy not a distribution"
    assert np.all(np.abs(vv) <= 1.0 + 1e-4), f"{cfg}: value outside [-1, 1]"

    search = Search(env, eval_fn)
    rng = random.PRNGKey(3)
    A = env.num_actions

    # --- PUCT grid: every muzero knob exercised ---
    muzero_cfgs = [
        SearchConfig(num_simulations=2),
        SearchConfig(num_simulations=200, pb_c_init=2.5, pb_c_base=50_000.0),
        SearchConfig(dirichlet_alpha=0.03, dirichlet_fraction=0.5),
        SearchConfig(temperature=0.0),
        SearchConfig(temperature=2.0, qtransform_epsilon=1e-3),
        SearchConfig(max_depth=2),
    ]
    for c in muzero_cfgs:
        _assert_policy_out(search.run_muzero(variables, rng, st, config=c), legal,
                           f"{cfg} muzero {c}")

    # --- per-call override path + algorithm dispatch ---
    out = search.run(variables, rng, st, algorithm="muzero",
                     num_simulations=8, temperature=0.7)
    _assert_policy_out(out, legal, f"{cfg} run/override")

    # --- the num_simulations knob is genuinely wired through (visit budget) ---
    for sims in (7, 53):
        vc = np.asarray(search.run_muzero(
            variables, rng, st, num_simulations=sims).search_tree.summary().visit_counts)
        assert np.all(vc.sum(1) == sims), f"{cfg}: num_simulations={sims} not honored"

    print(f"  [ok] az-net search {cfg!s:<34} states={n:>4} "
          f"muzero×{len(muzero_cfgs)}")


def check_edge_cases():
    """Hand-built adversarial states — the cases a fuzzer rarely hits cleanly."""
    # empty board: never terminal, all cells legal
    env = Env(GameConfig(4, 3, True))
    s0 = env.init()
    assert not bool(env.terminal_and_reward(s0)[0])
    assert int(env.legal_mask(s0).sum()) == 16

    # immediate terminal: mover (opp) holds a 3-line -> done
    line = int(env.lines[0])
    s = State(jnp.int32(0), jnp.int32(line))
    done, r = env.terminal_and_reward(s)
    assert bool(done) and float(r) == -1.0, "misere: completing a line must lose (-1)"

    # normal mode flips the sign: completing a line wins (+1)
    envn = Env(GameConfig(4, 3, False))
    done, r = envn.terminal_and_reward(State(jnp.int32(0), jnp.int32(int(envn.lines[0]))))
    assert bool(done) and float(r) == 1.0, "normal: completing a line must win (+1)"

    # draw: board full, mover has no line -> done, reward 0
    full = (1 << env.cells) - 1
    sdraw = State(jnp.int32(full), jnp.int32(0))   # opp(=mover) empty -> no line, full
    done, r = env.terminal_and_reward(sdraw)
    assert bool(done) and float(r) == 0.0, "full board, no line -> draw (0)"

    # single legal move: 15 cells filled, exactly one empty
    s1 = State(jnp.int32(0), jnp.int32(full ^ (1 << 5)))
    lm = env.legal_mask(s1)
    assert int(lm.sum()) == 1 and bool(lm[5]), "single-move board"
    s1b = env.step(s1, jnp.int32(5))
    assert int((s1b.own | s1b.opp)) == full, "step into last cell fills board"

    # 5x5 = 25 bits: top corner cell (bit 24) must round-trip without overflow
    env5 = Env(GameConfig(5, 5, True))
    s5 = env5.step(env5.init(), jnp.int32(24))
    assert int(s5.opp) == (1 << 24) and int(s5.opp) > 0, "5x5 bit-24 overflow"
    print("  [ok] edge cases: empty / terminal / draw / single-move / 5x5 bit-24")


if __name__ == "__main__":
    print("stress test: vectorized self-play")
    for cfg, B in WORKLOAD:
        check_selfplay(cfg, B)
    print("stress test: mctx legality + masking")
    for cfg in (GameConfig(3, 3, True), GameConfig(4, 3, True),
                GameConfig(4, 4, True), GameConfig(5, 5, True)):
        check_mctx_legal(cfg)
    print("stress test: AlphaZero net + full hyperparameter sweep")
    for cfg in (GameConfig(3, 3, True), GameConfig(4, 3, True),
                GameConfig(4, 4, True), GameConfig(5, 5, True)):
        check_az_net_search(cfg)
    print("stress test: adversarial edge cases")
    check_edge_cases()
    print("ALL STRESS TESTS PASSED")
