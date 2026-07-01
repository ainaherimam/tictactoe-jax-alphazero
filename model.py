"""The AlphaZero net as an mctx evaluator.

Reuses the *exact* training network — `AlphaZeroNet` from
`src/models/alphazero_model.py` — rather than a copy, so the search is guided by
the same architecture (and the same checkpoints) the training pipeline produces.

`make_eval_fn(model)` returns the `eval_fn(variables, planes) -> (logits, value)`
that `jax_az.search.Search` expects:
  * logits  [B, A]  — the policy head's log-probabilities (mctx softmaxes them,
                      and `softmax(log_softmax(x)) == softmax(x)`, so priors are exact)
  * value   [B]     — a scalar in [-1, 1], collapsed from whatever value-head
                      `variant` the net uses (mirrors the readout in
                      `alphazero_model.alphazero_loss`).

`variables` is the opaque pytree mctx threads through search: the full
`{'params':..., 'batch_stats':...}` dict from `init_variables` or an Orbax
checkpoint. Pass it straight to `Search.run*` as `params`.
"""
import os
import sys

import jax
import jax.numpy as jnp

# Reuse the real training model — import needs this package dir (which holds the
# vendored `src/` tree) on sys.path regardless of the caller's CWD.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.constants import INPUT_CHANNELS                          # noqa: E402
from src.models.alphazero_model import AlphaZeroNet               # noqa: E402
from jax_az import features                                       # noqa: E402

_BIN_CENTERS = jnp.array([-1.0, 0.0, 1.0])  # value bins {loss, draw, win}


def make_model(env, num_channels: int = 64, num_res_blocks: int = 4,
               variant: str = "v2_softmax_ce") -> AlphaZeroNet:
    """AlphaZeroNet sized for `env` (one action per cell). Arch knobs editable."""
    return AlphaZeroNet(
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        num_actions=env.num_actions,
        variant=variant,
    )


def init_variables(rng, model: AlphaZeroNet, env):
    """Fresh {'params', 'batch_stats'} for `model` on `env`-shaped input."""
    dummy_board = jnp.zeros((1, INPUT_CHANNELS, env.size, env.size))
    dummy_mask = jnp.ones((1, env.num_actions))
    return model.init(rng, dummy_board, dummy_mask, training=False)


def value_to_scalar(v: jax.Array, variant: str) -> jax.Array:
    """Collapse a value head's output to a scalar in [-1, 1] (per variant)."""
    if variant in ("v1_scalar_mse", "v6_scalar_l1", "v3_softmax_mse"):
        return v                                                  # already [B] scalar
    if variant == "v2_softmax_ce":
        return (jax.nn.softmax(v, axis=-1) * _BIN_CENTERS).sum(-1)
    if variant == "v4_tanh_per_bin_mse":
        return (v * _BIN_CENTERS).sum(-1) / 2.0
    if variant == "v5_independent_bce":
        return (jax.nn.sigmoid(v) * _BIN_CENTERS).sum(-1)
    raise ValueError(f"Unknown value-head variant: {variant}")


def make_eval_fn(model: AlphaZeroNet):
    """eval_fn(variables, planes) -> (log_prob_logits [B, A], value [B])."""
    A = model.num_actions
    variant = model.variant

    def eval_fn(variables, planes):
        # The net's `mask` arg is unused in its forward pass; pass ones.
        mask = jnp.ones((planes.shape[0], A), planes.dtype)
        logp, v = model.apply(variables, planes, mask, training=False)
        return logp, value_to_scalar(v, variant)

    return eval_fn


def make_az_search(env, num_channels: int = 64, num_res_blocks: int = 4,
                   variant: str = "v2_softmax_ce", rng=None, config=None):
    """One-liner: build net + eval_fn + Search. Returns (search, variables).

    `search.run(variables, rng, state)` is now an AlphaZero-guided mctx search.
    """
    from jax_az.search import Search, SearchConfig
    if rng is None:
        rng = jax.random.PRNGKey(0)
    model = make_model(env, num_channels, num_res_blocks, variant)
    variables = init_variables(rng, model, env)
    search = Search(env, make_eval_fn(model), config or SearchConfig())
    return search, variables


def demo():
    """Self-check: AZ-guided mctx returns a legal, normalized policy."""
    from jax_az.env import Env, GameConfig
    env = Env(GameConfig(4, 3, True))
    search, variables = make_az_search(env, num_channels=8, num_res_blocks=1)
    state = env.init_batch(3)
    out = search.run(variables, jax.random.PRNGKey(1), state, num_simulations=16)
    w = out.action_weights
    assert w.shape == (3, env.num_actions)
    assert bool(jnp.all(jnp.isfinite(w)))
    assert bool(jnp.allclose(w.sum(-1), 1.0, atol=1e-4))
    assert int(out.search_tree.summary().visit_counts.sum(-1)[0]) == 16
    print("jax_az.model demo: AZ-guided mctx OK (legal, normalized, 16 sims)")


if __name__ == "__main__":
    demo()
