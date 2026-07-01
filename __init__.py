"""Vectorized JAX environment for k-in-a-row Tic-Tac-Toe (normal or misère).

Ready for `mctx` batched MCTS. See env.py for the core, search.py for the
mctx wiring, and test_env.py / stress_test.py for the parity + stress checks.
"""
# The `src/` tree (real training model + solver) is vendored inside this package
# but imported as top-level `src.*`. Put this folder on sys.path so those resolve
# no matter the caller's CWD — this package is launched as `python -m jax_az.X`
# from the parent dir, which only puts the parent (for `jax_az`) on the path.
import os as _os, sys as _sys  # noqa: E402
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

# CPU/GPU switch (JAX_AZ_DEVICE) — must run BEFORE jax is imported below.
from jax_az.device import apply_from_env as _apply_device  # noqa: E402
_apply_device()

from jax_az.env import Env, GameConfig, State  # noqa: F401, E402
from jax_az.search import Search, SearchConfig  # noqa: F401

# Model helpers are lazy: jax_az.model imports the Flax net (flax/optax/orbax),
# which the env-only path doesn't need. Accessing them triggers the import.
_LAZY = {"make_az_search", "make_model", "make_eval_fn", "init_variables"}


def __getattr__(name):
    if name in _LAZY:
        import jax_az.model as _m
        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Env", "GameConfig", "State", "Search", "SearchConfig", *sorted(_LAZY)]
