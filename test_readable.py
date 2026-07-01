"""Readable smoke test: 1 generation of a couple of self-play games on the JAX path.

Shows the whole pipeline in plain text so you can eyeball that it's really running:
  1. self-play plays B games (random net), printed ply-by-ply as a 4x4 board with the
     recorded pi / z / mask and the literal NN input planes;
  2. those positions go into a small replay ring;
  3. a small batch is sampled + symmetry-augmented — exactly the dict fed to train_step.

Pure inspection, no asserts (the real gates live in selfplay.py / replay.py demos).
Run: `python -m jax_az.test_readable`  (add `--games N --sims M --ring R --batch K`).
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from jax_az.env import Env, GameConfig
from jax_az.model import make_az_search
from jax_az import selfplay, replay


def fmt_board(planes, ply: int) -> str:
    """planes = [2,4,4] to-move view (plane0=own, plane1=opp). Render an ABSOLUTE
    board with X = player 0 (first mover), O = player 1, so the game reads coherently
    even though 'own/opp' flips each ply."""
    own, opp = np.asarray(planes[0]), np.asarray(planes[1])
    x, o = (own, opp) if ply % 2 == 0 else (opp, own)   # player 0 == X throughout
    size = own.shape[0]
    rows = []
    for r in range(size):
        rows.append(" ".join("X" if x[r, c] else "O" if o[r, c] else "." for c in range(size)))
    return "\n".join("   " + row for row in rows)


def fmt_grid(vec, size: int, fmt="{:.2f}") -> str:
    a = np.asarray(vec).reshape(size, size)
    return "\n".join("   " + " ".join(fmt.format(v) for v in row) for row in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--sims", type=int, default=16, help="MCTS sims/move")
    ap.add_argument("--ring", type=int, default=64, help="replay ring capacity")
    ap.add_argument("--batch", type=int, default=4, help="sampled batch size")
    args = ap.parse_args()

    env = Env(GameConfig(4, 3, True))           # misère 4x4
    size, cells, B = env.size, env.cells, args.games
    search, variables = make_az_search(env, num_channels=8, num_res_blocks=1)
    import dataclasses
    scfg = dataclasses.replace(search.config, num_simulations=args.sims)

    print(f"=== 1 generation, {B} self-play games (random net, {args.sims} sims/move) ===\n")
    boards, pi, z, mask, valid = selfplay.play_batch(
        variables, jax.random.PRNGKey(0), env, search, B, scfg)
    # flat row for (ply t, game g) is t*B + g  (reshape was [cells, B, ...] -> [-1])
    boards, pi, z, mask, valid = map(np.asarray, (boards, pi, z, mask, valid))

    for g in range(B):
        print(f"\n{'='*48}\nGAME {g}\n{'='*48}")
        for t in range(cells):
            row = t * B + g
            if valid[row] < 0.5:                # game already ended — padding ply
                continue
            mv = int(np.asarray(pi[row]).argmax())
            print(f"\n-- ply {t}  (to move: {'X' if t % 2 == 0 else 'O'})  "
                  f"top move -> cell {mv} (r{mv // size},c{mv % size})  z={z[row]:+.0f} --")
            print("board:")
            print(fmt_board(boards[row], t))
            print("pi (MCTS visit policy):")
            print(fmt_grid(pi[row], size))
            print(f"legal mask: {mask[row].astype(int).tolist()}")
            print(f"raw NN input planes own/opp: own={boards[row,0].reshape(-1).astype(int).tolist()}")
            print(f"                             opp={boards[row,1].reshape(-1).astype(int).tolist()}")

    n_real = int(valid.sum())
    print(f"\n{'='*48}\nrecorded {n_real} real positions "
          f"({B*cells - n_real} padding rows, valid=0)\n{'='*48}")

    # --- ring + sampling: exactly what train_jax does before train_step ---------
    N = ((args.ring + B * cells - 1) // (B * cells)) * (B * cells)   # round up like the loop
    buf = replay.empty(N, size, cells)
    buf = replay.add(buf, *map(jnp.asarray, (boards, pi, z, mask, valid)))
    PERM = replay.build_perm(size)
    batch = replay.sample(buf, jax.random.PRNGKey(1), args.batch)
    batch = replay.augment(batch, jax.random.PRNGKey(2), PERM)

    print(f"\nring filled={int(replay.filled(buf))}/{N}; sampled+augmented batch of {args.batch} "
          f"(this dict is what train_step receives):")
    for i in range(args.batch):
        # augmented samples have no ply context, so render raw planes directly
        own = np.asarray(batch['boards'][i, 0]).astype(int)
        opp = np.asarray(batch['boards'][i, 1]).astype(int)
        rows = ["   " + " ".join("#" if own[r, c] else "o" if opp[r, c] else "." for c in range(size))
                for r in range(size)]
        print(f"\n  sample {i}:  z={float(batch['z'][i]):+.0f}   board(own=#,opp=o):")
        print("\n".join(rows))
        print(f"     pi={np.round(np.asarray(batch['pi'][i]), 2).tolist()}")
        print(f"     mask={np.asarray(batch['mask'][i]).astype(int).tolist()}")

    print("\nOK — self-play -> replay ring -> sampled batch all flowed.")


if __name__ == "__main__":
    main()
