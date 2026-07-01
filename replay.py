"""On-device replay ring + symmetry augmentation (next_plan.md §2, §4, §5).

The C++ 25K-position SHM ring existed only because self-play and training were two
processes. In one JAX process that whole apparatus collapses to one fixed device
array + a write cursor. Games end at different plies, so the per-generation valid-row
count is dynamic — which fights jit's static shapes if you try to *compact*. So we
don't: `add` writes the whole `[B*cells, …]` block (padding included) and `sample`
draws with probability ∝ `valid`, so a padding row (valid=0) is structurally never
returned. The `valid` mask also doubles as the fill tracker — no separate size/n.

`augment` replaces the numpy per-sample `for`-loop in train.py (`augment_batch`):
the 8 square symmetries become a single precomputed `PERM[8, cells]` gather over the
cell axis, fully on-device. Same transform, no host round-trip, no Python loop.
"""
from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


# --- ring buffer -----------------------------------------------------------

class Buf(NamedTuple):
    """All device arrays. Carried as a pytree through the train loop (not mutated
    in place — jit-friendly)."""
    boards: jax.Array   # [N, 2, size, size] f32
    pi: jax.Array       # [N, cells] f32
    z: jax.Array        # [N] f32
    mask: jax.Array     # [N, cells] f32
    valid: jax.Array    # [N] f32 — 1 real, 0 padding/unfilled; also the fill tracker
    cursor: jax.Array   # [] int32


def empty(N: int, size: int, cells: int) -> Buf:
    """Empty ring. `valid` starts all-zero, so nothing is sampled until filled.
    `N` should be a multiple of the per-generation block (`B*cells`)."""
    z = lambda *s: jnp.zeros(s, jnp.float32)
    return Buf(z(N, 2, size, size), z(N, cells), z(N), z(N, cells), z(N), jnp.int32(0))


@jax.jit
def add(buf: Buf, boards, pi, z, mask, valid) -> Buf:
    """Write a `[K, …]` block (K = B*cells, constant) at the cursor. Takes padding
    rows as-is; they carry valid=0 and are never drawn. N a multiple of K means the
    block never splits the seam (the modulo handles it cleanly regardless)."""
    N = buf.boards.shape[0]
    K = boards.shape[0]
    idx = (buf.cursor + jnp.arange(K, dtype=jnp.int32)) % N
    setk = lambda a, v: a.at[idx].set(v.astype(a.dtype))
    return buf._replace(
        boards=setk(buf.boards, boards), pi=setk(buf.pi, pi), z=setk(buf.z, z),
        mask=setk(buf.mask, mask), valid=setk(buf.valid, valid),
        cursor=(buf.cursor + K) % N)


def filled(buf: Buf) -> jax.Array:
    """Number of real positions currently held (replaces C++ wait_for_data)."""
    return buf.valid.sum()


@functools.partial(jax.jit, static_argnums=2)
def sample(buf: Buf, key, bs: int) -> dict:
    """Draw `bs` rows with p ∝ valid — structurally cannot return a padding row.
    Caller gates on `filled(buf) >= min_positions` so `valid.sum() > 0`."""
    idx = jax.random.choice(key, buf.boards.shape[0], (bs,),
                            replace=True, p=buf.valid / buf.valid.sum())
    return {'boards': buf.boards[idx], 'pi': buf.pi[idx],
            'z': buf.z[idx], 'mask': buf.mask[idx]}


# --- symmetry augmentation -------------------------------------------------

def build_perm(size: int) -> jax.Array:
    """`PERM[8, cells]`: for each of the 8 square symmetries, output cell -> input
    cell. Built by applying the *same* transform (h-flip then rot90, the order
    augment_batch uses) to a 0..cells-1 index grid, so `take(a, PERM[s])` reproduces
    that transform on any cell-indexed array. sym = flip(>=4) composed with k=sym%4."""
    cells = size * size
    grid = np.arange(cells).reshape(size, size)
    perms = []
    for s in range(8):
        g = grid
        if s >= 4:
            g = np.flip(g, axis=-1)          # horizontal flip first (== augment_batch)
        g = np.rot90(g, k=s % 4, axes=(-2, -1))
        perms.append(g.reshape(-1))
    return jnp.asarray(np.stack(perms), dtype=jnp.int32)   # [8, cells]


def apply_syms(batch: dict, syms, PERM) -> dict:
    """Apply per-sample symmetry `syms[bs]` to a batch via gather. z is invariant
    under spatial transforms. Shapes: boards [bs,2,size,size], pi/mask [bs,cells]."""
    p = PERM[syms]                                          # [bs, cells]
    g = lambda a: jnp.take_along_axis(a, p, axis=1)         # gather cell axis
    bsh = batch['boards'].shape
    bflat = batch['boards'].reshape(bsh[0], bsh[1], -1)     # [bs, 2, cells]
    pb = jnp.broadcast_to(p[:, None, :], bflat.shape)
    boards = jnp.take_along_axis(bflat, pb, axis=2).reshape(bsh)
    return {'boards': boards, 'pi': g(batch['pi']), 'mask': g(batch['mask']),
            'z': batch['z']}


def augment(batch: dict, key, PERM) -> dict:
    """Random symmetry per sample — the on-device replacement for augment_batch."""
    bs = batch['z'].shape[0]
    syms = jax.random.randint(key, (bs,), 0, 8)
    return apply_syms(batch, syms, PERM)


# --- self-checks (next_plan.md §6, tests 3 & 4) ----------------------------

def _ref_augment_numpy(batch, syms, size):
    """Independent reference: exactly augment_batch's per-sample numpy transform
    (h-flip if sym>=4, then rot90 k=sym%4). Different code path than apply_syms."""
    bs = batch['z'].shape[0]
    boards = np.asarray(batch['boards']).copy()
    pi = np.asarray(batch['pi']).reshape(bs, size, size).copy()
    mask = np.asarray(batch['mask']).reshape(bs, size, size).copy()
    for i in range(bs):
        k, flip = int(syms[i]) % 4, int(syms[i]) >= 4
        if flip:
            boards[i] = np.flip(boards[i], axis=-1)
            pi[i] = np.flip(pi[i], axis=-1); mask[i] = np.flip(mask[i], axis=-1)
        if k:
            boards[i] = np.rot90(boards[i], k=k, axes=(-2, -1))
            pi[i] = np.rot90(pi[i], k=k, axes=(-2, -1))
            mask[i] = np.rot90(mask[i], k=k, axes=(-2, -1))
    return boards, pi.reshape(bs, -1), mask.reshape(bs, -1)


def demo():
    size, cells = 4, 16
    N, K, bs = 8 * cells, 2 * cells, 64        # N multiple of block K
    key = jax.random.PRNGKey(0)

    # --- Test 3: sampling never returns a padding row -----------------------
    buf = empty(N, size, cells)
    # block with HALF the rows padding (valid=0), distinct ids in z to trace rows
    k1, k2, k3 = jax.random.split(key, 3)
    valid_block = (jax.random.uniform(k1, (K,)) > 0.5).astype(jnp.float32)
    z_block = jnp.arange(K, dtype=jnp.float32) + 1.0       # nonzero, unique-ish ids
    blk = lambda: (jnp.zeros((K, 2, size, size)), jnp.zeros((K, cells)),
                   z_block, jnp.zeros((K, cells)), valid_block)
    # add enough blocks to wrap around the ring at least once, still partly unfilled
    for _ in range(5):
        buf = add(buf, *blk())
    drawn = sample(buf, k2, 4096)
    # every drawn row must correspond to a valid (non-padding) slot
    # reconstruct: a drawn z came from a valid row iff that slot's valid==1
    drawn_valid = jnp.isin(drawn['z'], z_block[valid_block > 0.5])
    assert bool(jnp.all(drawn_valid)), "sampled a padding row!"
    # 5 adds into a 4-block ring -> all N//K regions hold the (identical) block
    assert float(filled(buf)) == float(valid_block.sum()) * (N // K), "fill count wrong"
    assert int(buf.cursor) == (5 * K) % N

    # empty buffer holds nothing
    assert float(filled(empty(N, size, cells))) == 0.0

    # --- Test 4: PERM gather == numpy rot90/flip reference ------------------
    PERM = build_perm(size)
    assert PERM.shape == (8, cells)
    b = {'boards': jax.random.normal(k3, (10, 2, size, size)),
         'pi': jax.random.uniform(jax.random.fold_in(k3, 1), (10, cells)),
         'mask': (jax.random.uniform(jax.random.fold_in(k3, 2), (10, cells)) > 0.5).astype(jnp.float32),
         'z': jax.random.normal(jax.random.fold_in(k3, 3), (10,))}
    syms = np.array([0, 1, 2, 3, 4, 5, 6, 7, 3, 5])        # cover all 8
    got = apply_syms(b, jnp.asarray(syms), PERM)
    rb, rpi, rmask = _ref_augment_numpy(b, syms, size)
    assert np.allclose(np.asarray(got['boards']), rb, atol=1e-6), "boards != reference"
    assert np.allclose(np.asarray(got['pi']), rpi, atol=1e-6), "pi != reference"
    assert np.allclose(np.asarray(got['mask']), rmask, atol=1e-6), "mask != reference"
    assert np.allclose(np.asarray(got['z']), np.asarray(b['z'])), "z must be invariant"
    # identity symmetry is a no-op
    idg = apply_syms(b, jnp.zeros(10, jnp.int32), PERM)
    assert np.allclose(np.asarray(idg['pi']), np.asarray(b['pi'])), "sym 0 not identity"

    print("jax_az.replay demo OK: padding never sampled + augment == numpy reference")


if __name__ == "__main__":
    demo()
