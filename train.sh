#!/usr/bin/env bash
# Launch the AlphaZero training loop with the settings in jax_az/config.py.
# Force CPU/GPU with JAX_AZ_DEVICE, e.g. `JAX_AZ_DEVICE=gpu ./train.sh`.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # jax_az package dir
PY="$HERE/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
cd "$(dirname "$HERE")"                                 # parent, so `-m jax_az.*` resolves
exec "$PY" -m jax_az.config "$@"
