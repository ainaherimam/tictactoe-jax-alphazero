#!/usr/bin/env bash
# Setup for Google Colab (or any fresh machine).
#
# On Colab, run in a cell after cloning the repo *as* jax_az:
#   !git clone <url> jax_az && bash jax_az/setup.sh
#   %cd /content            # parent of the jax_az package
#   !python -m jax_az.stress_test        # or train_jax / eval_solver / ...
#
# The package is imported as `jax_az`, so you always launch from the PARENT of
# this folder. The vendored `src/` tree is put on sys.path by jax_az/__init__.py.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # the jax_az package dir

if [ -d /content ] || [ -n "${COLAB_GPU:-}" ]; then
    # Colab already ships jax/jaxlib(+CUDA) and numpy — keep them so GPU works.
    echo "[setup] Colab detected: keeping preinstalled jax, installing the rest."
    pip install -q flax==0.12.7 optax==0.2.8 orbax-checkpoint==0.12.1 mctx==0.0.71
else
    pip install -q -r "$HERE/requirements.txt"
fi

echo "[setup] done. Launch from the parent of this folder, e.g.:"
echo "    cd $(dirname "$HERE") && python -m jax_az.stress_test"
