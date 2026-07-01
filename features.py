"""State -> network planes. The only place the network touches env state.

planes(state, size) -> [2, size, size] float32 = [current player, opponent],
matching the existing Flax model's 2-plane input (src/models/alphazero_model.py).
"""
import jax
import jax.numpy as jnp


def planes(state, size: int) -> jax.Array:
    """[2, size, size] float32: plane 0 = own (to-move), plane 1 = opponent."""
    cells = size * size
    ix = jnp.arange(cells, dtype=jnp.int32)
    own = (state.own >> ix) & 1
    opp = (state.opp >> ix) & 1
    return jnp.stack([own, opp]).reshape(2, size, size).astype(jnp.float32)


def planes_batch(state, size: int) -> jax.Array:
    """[B, 2, size, size] float32 for a batch of states."""
    return jax.vmap(lambda s: planes(s, size))(state)
