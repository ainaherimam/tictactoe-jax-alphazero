#!/usr/bin/env bash
# Run the env stress test. Extra args pass through, e.g. `./stress_test.sh --help`.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # jax_az package dir
PY="$HERE/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
cd "$(dirname "$HERE")"                                 # parent, so `-m jax_az.*` resolves
exec "$PY" -m jax_az.stress_test "$@"
