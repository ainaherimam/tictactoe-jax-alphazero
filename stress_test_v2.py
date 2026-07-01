"""Stress test: search must run the *exact* AlphaZero v2 net (`v2_softmax_ce`).

Run:  .ven/bin/python jax_az/stress_test_v2.py

"v2" = the value head returns [B, 3] raw logits over bins {-1, 0, +1}; the MCTS
value readout is (softmax(logits) * bin_centers).sum(-1) — the same collapse
`alphazero_model.alphazero_loss` uses for its value-accuracy metric. This file
proves three things end to end:

  1. exact net: the search's evaluator is `src.models.alphazero_model.AlphaZeroNet`
     itself (not a copy) configured with variant "v2_softmax_ce".
  2. exact readout: eval_fn's scalar value == the v2 readout applied to the raw
     [B, 3] logits the net emits — bit-for-bit the training-time collapse.
  3. exact wiring under search: across the full muzero/gumbel knob grid the net
     guides mctx to legal, finite, normalized policies, value stays in [-1, 1].
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_az.env import Env, GameConfig                  # noqa: E402
from jax_az import features                              # noqa: E402
from jax_az.model import (                               # noqa: E402
    make_model, init_variables, make_eval_fn, make_az_search,
)
from jax_az.search import Search, SearchConfig          # noqa: E402
from jax_az.stress_test import _sample_nonterminal, _assert_policy_out  # noqa: E402
from src.models.alphazero_model import AlphaZeroNet     # noqa: E402

V2 = "v2_softmax_ce"
BIN_CENTERS = jnp.array([-1.0, 0.0, 1.0])
CFGS = (GameConfig(3, 3, True), GameConfig(4, 3, True),
        GameConfig(4, 4, True), GameConfig(5, 5, True))


def v2_readout(v_logits):
    """The canonical v2 value collapse, copied from alphazero_loss (L285)."""
    return (jax.nn.softmax(v_logits, axis=-1) * BIN_CENTERS).sum(-1)


def check_exact_net(cfg: GameConfig):
    """The thing the search runs IS AlphaZeroNet in its v2 configuration."""
    env = Env(cfg)
    model = make_model(env, num_channels=16, num_res_blocks=2)
    assert type(model) is AlphaZeroNet, f"{cfg}: not the real net: {type(model)}"
    assert model.variant == V2, f"{cfg}: search default variant is {model.variant!r}, want {V2!r}"
    assert model.num_actions == env.num_actions, f"{cfg}: action count mismatch"

    # make_az_search must also default to v2 (the one-liner users actually call).
    search, _ = make_az_search(env, num_channels=8, num_res_blocks=1)
    assert search.eval_fn.__module__.startswith("jax_az.model")
    print(f"  [ok] exact v2 net    {cfg!s:<34} {type(model).__name__}(variant={V2})")


def check_exact_readout(cfg: GameConfig, B: int = 256):
    """eval_fn's scalar value == v2 readout of the net's raw [B, 3] logits,
    and the value head genuinely emits 3 logits (not a scalar v1 head)."""
    env = Env(cfg)
    model = make_model(env, num_channels=16, num_res_blocks=2)
    variables = init_variables(random.PRNGKey(0), model, env)
    eval_fn = make_eval_fn(model)

    st, _ = _sample_nonterminal(env, random.PRNGKey(7), B)
    if st.own.shape[0] == 0:
        print(f"  [skip] readout {cfg}: no non-terminal states"); return
    planes = features.planes_batch(st, env.size)

    # Raw head output — must be [n, 3] logits (the v2 contract).
    mask = jnp.ones((planes.shape[0], env.num_actions), planes.dtype)
    _, v_raw = model.apply(variables, planes, mask, training=False)
    assert v_raw.ndim == 2 and v_raw.shape[-1] == 3, \
        f"{cfg}: v2 head must be [n,3] logits, got {v_raw.shape}"

    logp, v = eval_fn(variables, planes)
    n = st.own.shape[0]
    assert logp.shape == (n, env.num_actions) and v.shape == (n,), f"{cfg}: bad eval shapes"
    # exactness: search's value == the canonical training-time v2 collapse
    assert np.allclose(np.asarray(v), np.asarray(v2_readout(v_raw)), atol=0, rtol=0), \
        f"{cfg}: eval_fn value differs from v2 readout"
    assert np.all(np.isfinite(np.asarray(logp))) and np.all(np.isfinite(np.asarray(v)))
    assert np.allclose(np.exp(np.asarray(logp)).sum(1), 1.0, atol=1e-4), f"{cfg}: policy !distribution"
    assert np.all(np.abs(np.asarray(v)) <= 1.0 + 1e-6), f"{cfg}: v2 value outside [-1,1]"
    print(f"  [ok] exact readout   {cfg!s:<34} states={n:>4}  |v|<=1, value==softmax·bins")


def check_search_grid(cfg: GameConfig, B: int = 128):
    """v2 net guides mctx legally across the full hyperparameter grid."""
    env = Env(cfg)
    model = make_model(env, num_channels=16, num_res_blocks=2)
    variables = init_variables(random.PRNGKey(0), model, env)
    search = Search(env, make_eval_fn(model))
    st, legal = _sample_nonterminal(env, random.PRNGKey(7), B)
    if st.own.shape[0] == 0:
        print(f"  [skip] grid {cfg}: no non-terminal states"); return
    rng = random.PRNGKey(3)

    # muzero / PUCT only — the search algorithm this project uses.
    grid = [
        SearchConfig(num_simulations=2),
        SearchConfig(num_simulations=200, pb_c_init=2.5, temperature=0.0),
        SearchConfig(dirichlet_alpha=0.03, dirichlet_fraction=0.5, max_depth=2),
        SearchConfig(temperature=2.0, qtransform_epsilon=1e-3),
        SearchConfig(num_simulations=64, pb_c_base=50_000.0, temperature=0.7),
    ]
    for c in grid:
        _assert_policy_out(search.run_muzero(variables, rng, st, config=c), legal,
                           f"{cfg} muzero {c}")

    # num_simulations is genuinely threaded into the search tree's visit budget.
    for sims in (7, 53):
        vc = np.asarray(search.run_muzero(
            variables, rng, st, num_simulations=sims).search_tree.summary().visit_counts)
        assert np.all(vc.sum(1) == sims), f"{cfg}: num_simulations={sims} not honored"
    print(f"  [ok] v2 search grid  {cfg!s:<34} states={st.own.shape[0]:>4}  muzero×{len(grid)}")


if __name__ == "__main__":
    print("v2 stress test: search uses the exact AlphaZero v2 net")
    print("--- exact net identity ---")
    for cfg in CFGS:
        check_exact_net(cfg)
    print("--- exact v2 value readout ---")
    for cfg in CFGS:
        check_exact_readout(cfg)
    print("--- v2 net under full search grid ---")
    for cfg in CFGS:
        check_search_grid(cfg)
    print("ALL V2 STRESS TESTS PASSED")
