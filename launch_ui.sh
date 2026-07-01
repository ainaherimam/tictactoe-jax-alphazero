#!/usr/bin/env bash
# Launch the monitor + launcher web UI at http://localhost:8000 (override with PORT).
# Runs server.py as a FILE (stdlib only, stays jax-free); it spawns trainers as
# subprocesses using this same Python, so they get the venv's jax/flax/etc.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # jax_az package dir
PY="$HERE/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
PKG="$(dirname "$HERE")/jax_az"; [ -e "$PKG" ] || ln -s "$HERE" "$PKG"  # clone dir may be named otherwise
cd "$(dirname "$HERE")"                                 # parent, so the server's `-m jax_az.*` spawns resolve
exec "$PY" "$HERE/monitor/server.py" "$@"
