"""Vectorized JAX env for k-in-a-row Tic-Tac-Toe — normal or misère, any size.

State = two int32 bitmasks *from the to-move player's perspective*:

    State(own, opp)   # bit c set  <=>  a piece on cell c (row-major: c = r*size + c)

`step` places a piece in `own` then swaps to `(opp, own|bit)`, so the returned
state is always "plane 0 = current player, plane 1 = opponent". Zero perspective
bookkeeping — the network input is the same shape every ply.

Sign / reward convention (the load-bearing subtlety, see plan.md §3, §11):
`terminal_and_reward(state)` reports the result of the move that *produced*
`state`, from the perspective of the player who just moved. That mover is `opp`
in the new state (the swap already happened). So:

    misère : completing a line  -> mover LOSES  -> reward -1
    normal : completing a line  -> mover WINS   -> reward +1
    board full, no line         -> draw         -> reward  0

int32 holds up to 31 bits; max board here is 5x5 = 25 cells, so no overflow and
the sign bit is never touched. Everything is integer bit-ops => jax.vmap-able.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


class State(NamedTuple):
    """Two bitmasks from the to-move player's view. Registered JAX pytree."""
    own: jax.Array  # int32, current player's pieces
    opp: jax.Array  # int32, opponent's pieces


@dataclass(frozen=True)
class GameConfig:
    size: int = 4          # board is size x size
    win_length: int = 3    # k-in-a-row to complete a line
    misere: bool = True    # True: completing a line loses; False: it wins

    def __post_init__(self):
        if not (3 <= self.size <= 5):
            raise ValueError(f"size must be 3..5, got {self.size}")
        if not (3 <= self.win_length <= self.size):
            raise ValueError(
                f"win_length must be 3..size ({self.size}), got {self.win_length}")


def _line_masks(size: int, k: int) -> list[int]:
    """Every k-in-a-row line as a bitmask. Four directions: →, ↓, ↘, ↙.

    Mirrors what board.cpp/misere_solver walk: row windows, col windows, and the
    two diagonals. Returns plain Python ints (one bitmask per line).
    """
    masks: list[int] = []
    def cell(r: int, c: int) -> int:
        return 1 << (r * size + c)
    for r in range(size):
        for c in range(size):
            if c + k <= size:                       # horizontal →
                masks.append(sum(cell(r, c + i) for i in range(k)))
            if r + k <= size:                       # vertical ↓
                masks.append(sum(cell(r + i, c) for i in range(k)))
            if r + k <= size and c + k <= size:     # diagonal ↘
                masks.append(sum(cell(r + i, c + i) for i in range(k)))
            if r + k <= size and c - k + 1 >= 0:    # anti-diagonal ↙
                masks.append(sum(cell(r + i, c - i) for i in range(k)))
    return masks


class Env:
    """Pure, vmappable env for one GameConfig. Methods are functions of `State`.

    Construct once (`Env(GameConfig(...))`); the line masks / constants are baked
    in as closed-over arrays, so every method jits and vmaps cleanly.
    """

    def __init__(self, config: GameConfig = GameConfig()):
        self.config = config
        self.size = config.size
        self.cells = config.size * config.size
        self.num_actions = self.cells
        self.misere = config.misere
        self.full_mask = jnp.int32((1 << self.cells) - 1)
        self.lines = jnp.asarray(
            _line_masks(config.size, config.win_length), dtype=jnp.int32)
        self._cell_ix = jnp.arange(self.cells, dtype=jnp.int32)
        # reward sign for the player who just completed a line
        self._win_sign = -1.0 if config.misere else 1.0

    # --- single-game primitives (vmap these for a batch) --------------------

    def init(self) -> State:
        z = jnp.int32(0)
        return State(z, z)

    def _bits(self, mask: jax.Array) -> jax.Array:
        """int32 mask -> [cells] int32 of 0/1, cell c in bit c."""
        return (mask >> self._cell_ix) & 1

    def legal_mask(self, state: State) -> jax.Array:
        """[cells] bool: True on empty cells."""
        return self._bits(state.own | state.opp) == 0

    def step(self, state: State, action: jax.Array) -> State:
        """Place current player's piece at `action`, then swap perspective.

        Assumes `action` is legal (an empty cell). Self-play / mctx never pass an
        occupied cell (masked by `legal_mask` / `invalid_actions`).
        """
        bit = jnp.int32(1) << action.astype(jnp.int32)
        return State(state.opp, state.own | bit)

    def has_line(self, mask: jax.Array) -> jax.Array:
        """bool scalar: does this player's bitmask contain a completed line?"""
        return jnp.any((mask & self.lines) == self.lines)

    def terminal_and_reward(self, state: State):
        """(done bool, reward float) for the move that produced `state`.

        Reward is from the just-moved player's view (that player is `opp` here):
        +/-1 if they completed a line (sign per misère/normal), 0 for a draw,
        0 while the game continues. Assumes reachable play, i.e. only the last
        mover can hold a fresh line.
        """
        mover = state.opp
        made_line = self.has_line(mover)
        full = (state.own | state.opp) == self.full_mask
        done = made_line | full
        reward = jnp.where(made_line, self._win_sign, 0.0).astype(jnp.float32)
        return done, reward

    # --- batched convenience (B parallel games) -----------------------------

    def init_batch(self, batch: int) -> State:
        z = jnp.zeros((batch,), dtype=jnp.int32)
        return State(z, z)

    def legal_mask_batch(self, state: State) -> jax.Array:
        return jax.vmap(self.legal_mask)(state)

    def step_batch(self, state: State, action: jax.Array) -> State:
        return jax.vmap(self.step)(state, action)

    def terminal_and_reward_batch(self, state: State):
        return jax.vmap(self.terminal_and_reward)(state)
