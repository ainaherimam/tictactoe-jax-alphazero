"""mctx wiring with every search hyperparameter editable.

`Search(env, eval_fn, config)` runs batched MCTS via mctx. Every knob mctx
exposes lives in `SearchConfig` and can be set at construction OR overridden per
call (keyword args are applied on top of the config for that one call):

    cfg = SearchConfig(algorithm="muzero", num_simulations=200, pb_c_init=1.4)
    search = Search(env, eval_fn, cfg)
    out = search.run(params, rng, state)                  # uses cfg
    out = search.run(params, rng, state, temperature=0.0) # one-off override
    out = search.run_gumbel(params, rng, state, num_simulations=64)

`eval_fn(params, planes) -> (logits[B, A], value[B])` — exactly the shape the
AlphaZero net produces; wire the real net with `jax_az.model.make_eval_fn`. Any
callable of that shape works, so the env/masking is testable with a zero eval.

Two-player sign convention (plan.md §3): reward is from the acting player's view
(env.terminal_and_reward already returns the mover's reward); `discount = -1`
between plies flips to the opponent's frame (zero-sum negamax); `discount = 0` at
terminal so no value leaks past the end. Illegal actions are masked at the root
(`invalid_actions`) AND in every recurrent prior, so search never returns an
occupied cell.
"""
import functools
from dataclasses import dataclass, replace
from typing import Optional

import jax.numpy as jnp
import mctx

from jax_az import features

MASK_VALUE = -1e9  # logit for illegal actions


def scheduled_temperature(cfg: "SearchConfig", ply):
    """Self-play temperature for this `ply`.

    Multi-step schedule (`cfg.temp_schedule`) wins if set: a list of
    `[until_ply, temp]` breakpoints, e.g. `[[1, 2.0], [6, 1.0]]` means temp 2.0
    for plies < 1, temp 1.0 for plies < 6, then `cfg.temp_final` afterwards.
    Otherwise the single-drop `temp_drop_ply`/`temp_final` path; with neither set,
    constant `cfg.temperature`. `ply` may be a traced scalar (works in lax.scan)."""
    if cfg.temp_schedule:
        # build from the back: default temp_final, then each breakpoint overrides
        # it for plies below its bound (later/earlier bounds layered correctly).
        temp = jnp.asarray(cfg.temp_final, jnp.float32)
        for bound, t in reversed(cfg.temp_schedule):
            temp = jnp.where(ply < bound, jnp.float32(t), temp)
        return temp
    if cfg.temp_drop_ply is None:
        return cfg.temperature
    return jnp.where(ply < cfg.temp_drop_ply, cfg.temperature, cfg.temp_final)


@dataclass(frozen=True)
class SearchConfig:
    """All mctx search hyperparameters. Defaults match mctx's own defaults
    except `num_simulations` and `dirichlet_alpha` (tuned for this game)."""

    # --- which algorithm + shared budget ---
    algorithm: str = "muzero"           # "muzero" (PUCT) or "gumbel"
    num_simulations: int = 100
    max_depth: Optional[int] = None     # cap tree depth; None = unbounded

    # --- PUCT / muzero_policy ---
    pb_c_init: float = 1.25             # exploration constant
    pb_c_base: float = 19652.0          # exploration log-growth base
    dirichlet_alpha: float = 0.5        # root Dirichlet noise concentration
    dirichlet_fraction: float = 0.25    # weight of root noise vs prior
    temperature: float = 1.0            # action-sampling temperature (0 = argmax)
    temp_drop_ply: Optional[int] = None  # ply at which temperature drops to temp_final
    temp_final: float = 0.0             # temperature for plies past the last bound
    # multi-step schedule (wins over temp_drop_ply): ascending [until_ply, temp]
    # breakpoints, e.g. [[1, 2.0], [6, 1.0]] -> 2.0 for ply<1, 1.0 for ply<6, then temp_final
    temp_schedule: tuple = ()

    # --- Gumbel / gumbel_muzero_policy ---
    max_num_considered_actions: int = 16  # root actions Gumbel considers
    gumbel_scale: float = 1.0             # scale of the Gumbel noise

    # --- q-value transform (both algorithms) ---
    qtransform_epsilon: float = 1e-8    # numerical floor in the q-transform
    # qtransform_completed_by_mix_value (gumbel) only:
    value_scale: float = 0.1
    maxvisit_init: float = 50.0
    rescale_values: bool = True
    use_mixed_value: bool = True


class Search:
    def __init__(self, env, eval_fn, config: SearchConfig = SearchConfig()):
        self.env = env
        self.eval_fn = eval_fn
        self.size = env.size
        self.config = config

    # --- mctx callbacks -----------------------------------------------------

    def _priors(self, params, state):
        logits, value = self.eval_fn(params, features.planes_batch(state, self.size))
        legal = self.env.legal_mask_batch(state)              # [B, A] bool
        logits = jnp.where(legal, logits, MASK_VALUE)
        return logits, value

    def root_fn(self, params, rng, state) -> "mctx.RootFnOutput":
        logits, value = self._priors(params, state)
        return mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

    def recurrent_fn(self, params, rng, action, state):
        next_state = self.env.step_batch(state, action)
        done, reward = self.env.terminal_and_reward_batch(next_state)  # [B], [B]
        logits, value = self._priors(params, next_state)
        value = jnp.where(done, 0.0, value)                   # no value past terminal
        discount = jnp.where(done, 0.0, -1.0).astype(jnp.float32)
        out = mctx.RecurrentFnOutput(
            reward=reward.astype(jnp.float32),
            discount=discount,
            prior_logits=logits,
            value=value.astype(jnp.float32),
        )
        return out, next_state

    # --- run ----------------------------------------------------------------

    def _cfg(self, config: Optional[SearchConfig], overrides) -> SearchConfig:
        cfg = self.config if config is None else config
        return replace(cfg, **overrides) if overrides else cfg

    def run(self, params, rng, state, *, ply=None,
            config: Optional[SearchConfig] = None,
            **overrides) -> "mctx.PolicyOutput":
        """Dispatch on cfg.algorithm. Keyword overrides edit the config per call.
        Pass `ply` (move index) to apply the temperature schedule (temp_drop_ply)."""
        cfg = self._cfg(config, overrides)
        if ply is not None:
            cfg = replace(cfg, temperature=scheduled_temperature(cfg, ply))
        if cfg.algorithm == "gumbel":
            return self.run_gumbel(params, rng, state, config=cfg)
        if cfg.algorithm == "muzero":
            return self.run_muzero(params, rng, state, config=cfg)
        raise ValueError(f"unknown algorithm {cfg.algorithm!r} (muzero|gumbel)")

    def run_muzero(self, params, rng, state, *,
                   config: Optional[SearchConfig] = None, **overrides):
        """Vanilla PUCT + Dirichlet — the closest match to the old C++ self-play."""
        cfg = self._cfg(config, overrides)
        root = self.root_fn(params, rng, state)
        invalid = ~self.env.legal_mask_batch(state)
        qtransform = functools.partial(
            mctx.qtransform_by_parent_and_siblings, epsilon=cfg.qtransform_epsilon)
        return mctx.muzero_policy(
            params, rng, root, self.recurrent_fn,
            num_simulations=cfg.num_simulations,
            invalid_actions=invalid,
            max_depth=cfg.max_depth,
            qtransform=qtransform,
            dirichlet_fraction=cfg.dirichlet_fraction,
            dirichlet_alpha=cfg.dirichlet_alpha,
            pb_c_init=cfg.pb_c_init,
            pb_c_base=cfg.pb_c_base,
            temperature=cfg.temperature,
        )

    def run_gumbel(self, params, rng, state, *,
                   config: Optional[SearchConfig] = None, **overrides):
        cfg = self._cfg(config, overrides)
        root = self.root_fn(params, rng, state)
        invalid = ~self.env.legal_mask_batch(state)
        qtransform = functools.partial(
            mctx.qtransform_completed_by_mix_value,
            value_scale=cfg.value_scale,
            maxvisit_init=cfg.maxvisit_init,
            rescale_values=cfg.rescale_values,
            use_mixed_value=cfg.use_mixed_value,
            epsilon=cfg.qtransform_epsilon,
        )
        return mctx.gumbel_muzero_policy(
            params, rng, root, self.recurrent_fn,
            num_simulations=cfg.num_simulations,
            invalid_actions=invalid,
            max_depth=cfg.max_depth,
            qtransform=qtransform,
            max_num_considered_actions=cfg.max_num_considered_actions,
            gumbel_scale=cfg.gumbel_scale,
        )  # ponytail: this mctx's gumbel_muzero_policy has no `temperature` (gumbel uses gumbel_scale)
