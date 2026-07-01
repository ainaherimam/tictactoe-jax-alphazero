"""
Python port of MisereSolver — perfect Negamax solver for 4x4 Misère Tic-Tac-Toe.

Rules: A player LOSES if they complete a 3-in-a-row (horizontal, vertical, or diagonal).
       If the board fills with no line completed, it is a draw.

Board layout (bit indices):
  0  1  2  3
  4  5  6  7
  8  9 10 11
 12 13 14 15

Bit i corresponds to board cell (row=i//4, col=i%4).

Usage:
    solver = MisereSolver()
    solver.solve()   # populate TT once — call this at startup

    # Batch policy evaluation
    optimal_policy = solver.get_optimal_policy(bx, bo, is_x_turn)  # [16] float32

    # Batch utility
    from src.core.solver.misere_solver import boards_to_solver_policies
    solver_policies = boards_to_solver_policies(boards_np, solver)  # [B, 16]
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants — must match misere_solver.h
# ---------------------------------------------------------------------------

LINE_MASKS: List[int] = [
    # Rows (2 windows of 3 per row × 4 rows = 8)
    (1 << 0) | (1 << 1) | (1 << 2),     # row 0: cells 0,1,2
    (1 << 1) | (1 << 2) | (1 << 3),     # row 0: cells 1,2,3
    (1 << 4) | (1 << 5) | (1 << 6),     # row 1: cells 4,5,6
    (1 << 5) | (1 << 6) | (1 << 7),     # row 1: cells 5,6,7
    (1 << 8) | (1 << 9) | (1 << 10),    # row 2: cells 8,9,10
    (1 << 9) | (1 << 10) | (1 << 11),   # row 2: cells 9,10,11
    (1 << 12) | (1 << 13) | (1 << 14),  # row 3: cells 12,13,14
    (1 << 13) | (1 << 14) | (1 << 15),  # row 3: cells 13,14,15
    # Columns (2 windows of 3 per col × 4 cols = 8)
    (1 << 0) | (1 << 4) | (1 << 8),     # col 0: cells 0,4,8
    (1 << 4) | (1 << 8) | (1 << 12),    # col 0: cells 4,8,12
    (1 << 1) | (1 << 5) | (1 << 9),     # col 1: cells 1,5,9
    (1 << 5) | (1 << 9) | (1 << 13),    # col 1: cells 5,9,13
    (1 << 2) | (1 << 6) | (1 << 10),    # col 2: cells 2,6,10
    (1 << 6) | (1 << 10) | (1 << 14),   # col 2: cells 6,10,14
    (1 << 3) | (1 << 7) | (1 << 11),    # col 3: cells 3,7,11
    (1 << 7) | (1 << 11) | (1 << 15),   # col 3: cells 7,11,15
    # Diagonals (top-left to bottom-right)
    (1 << 0) | (1 << 5) | (1 << 10),    # cells 0,5,10
    (1 << 5) | (1 << 10) | (1 << 15),   # cells 5,10,15
    (1 << 1) | (1 << 6) | (1 << 11),    # cells 1,6,11
    (1 << 4) | (1 << 9) | (1 << 14),    # cells 4,9,14
    # Anti-diagonals (top-right to bottom-left)
    (1 << 3) | (1 << 6) | (1 << 9),     # cells 3,6,9
    (1 << 6) | (1 << 9) | (1 << 12),    # cells 6,9,12
    (1 << 2) | (1 << 5) | (1 << 8),     # cells 2,5,8
    (1 << 7) | (1 << 10) | (1 << 13),   # cells 7,10,13
]

# Symmetry permutation tables (D4 group — 4 rotations × 2 reflections)
SYM_PERM: List[List[int]] = [
    # 0: identity
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    # 1: rotate 90 CW
    [12,  8,  4,  0, 13,  9,  5,  1, 14, 10,  6,  2, 15, 11,  7,  3],
    # 2: rotate 180
    [15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0],
    # 3: rotate 270 CW
    [ 3,  7, 11, 15,  2,  6, 10, 14,  1,  5,  9, 13,  0,  4,  8, 12],
    # 4: reflect horizontal (flip rows)
    [12, 13, 14, 15,  8,  9, 10, 11,  4,  5,  6,  7,  0,  1,  2,  3],
    # 5: reflect vertical (flip columns)
    [ 3,  2,  1,  0,  7,  6,  5,  4, 11, 10,  9,  8, 15, 14, 13, 12],
    # 6: reflect main diagonal (transpose)
    [ 0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15],
    # 7: reflect anti-diagonal
    [15, 11,  7,  3, 14, 10,  6,  2, 13,  9,  5,  1, 12,  8,  4,  0],
]

# Move ordering: center cells first (better pruning), then edges, then corners
MOVE_ORDER: List[int] = [5, 6, 9, 10, 1, 2, 4, 7, 8, 11, 13, 14, 0, 3, 12, 15]

# Precomputed: for symmetry s and cell i, the transformed output bit
# _SYM_BIT[s][i] = (1 << SYM_PERM[s][i])
_SYM_BIT: List[List[int]] = [
    [1 << SYM_PERM[s][i] for i in range(16)]
    for s in range(8)
]


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def has_line(bits: int) -> bool:
    """Return True if `bits` contains any completed 3-in-a-row."""
    for mask in LINE_MASKS:
        if (bits & mask) == mask:
            return True
    return False


def transform_board(bits: int, sym: int) -> int:
    """Apply symmetry transform `sym` to a 16-bit board mask."""
    result = 0
    sym_bits = _SYM_BIT[sym]
    b = bits
    while b:
        lsb = b & (-b)            # isolate lowest set bit
        cell = lsb.bit_length() - 1
        result |= sym_bits[cell]
        b ^= lsb
    return result


def canonicalize(bx: int, bo: int) -> Tuple[int, int]:
    """Return the canonical (lexicographically smallest) form under 8 symmetries."""
    canon_bx, canon_bo = bx, bo
    for s in range(1, 8):
        tx = transform_board(bx, s)
        to = transform_board(bo, s)
        if tx < canon_bx or (tx == canon_bx and to < canon_bo):
            canon_bx, canon_bo = tx, to
    return canon_bx, canon_bo


def board_to_masks(board: np.ndarray) -> Tuple[int, int, bool]:
    """
    Convert a single board tensor [2, 4, 4] to (bx, bo, is_x_turn).

    Input encoding:
      board[0, r, c] = 1.0  →  current player has a piece at (r, c)
      board[1, r, c] = 1.0  →  opponent has a piece at (r, c)

    X always plays first, so:
      n_current == n_opponent  →  current player is X  (is_x_turn = True)
      n_opponent == n_current + 1  →  current player is O  (is_x_turn = False)
    """
    plane0 = board[0]  # current player, shape [4, 4]
    plane1 = board[1]  # opponent,       shape [4, 4]

    current_mask = 0
    opponent_mask = 0
    n_current = 0
    n_opponent = 0

    for r in range(4):
        for c in range(4):
            bit = 1 << (r * 4 + c)
            if plane0[r, c] > 0.5:
                current_mask |= bit
                n_current += 1
            if plane1[r, c] > 0.5:
                opponent_mask |= bit
                n_opponent += 1

    # Determine whose turn it is
    if n_current == n_opponent:
        # Equal pieces → X's turn; current player is X
        bx, bo, is_x_turn = current_mask, opponent_mask, True
    else:
        # X has one more piece → O's turn; current player is O
        bx, bo, is_x_turn = opponent_mask, current_mask, False

    return bx, bo, is_x_turn


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

# Transposition table flags (mirrors C++ misere_solver.h)
TT_EXACT = 0
TT_LOWER_BOUND = 1
TT_UPPER_BOUND = 2


class MisereSolver:
    """
    Perfect Negamax solver for 4x4 Misère Tic-Tac-Toe.

    Call solve() once at startup to populate the transposition table (TT).
    All subsequent queries are pure dictionary lookups — O(1) per position.

    TT maps (canon_bx, canon_bo) → (value, flag) where value ∈ {-1, 0, +1}
    from the player-to-move's perspective and flag is one of
    TT_EXACT / TT_LOWER_BOUND / TT_UPPER_BOUND (standard fail-soft
    alpha-beta semantics — matches the C++ implementation).
    """

    def __init__(self) -> None:
        # (canon_bx, canon_bo) -> (value, flag)
        self._tt: Dict[Tuple[int, int], Tuple[int, int]] = {}
        # (canon_bx, canon_bo) -> exact value, to-move perspective. Separate from
        # _tt because the alpha-beta _tt stores fail-soft *bounds*; reading those
        # back as exact (and re-searching them) corrupts results under heavy
        # querying. This cache only ever holds full-window EXACT values, so all
        # queries are pure, stable lookups.
        self._exact: Dict[Tuple[int, int], int] = {}
        self._solved = False

    # ---- Negamax --------------------------------------------------------------

    def _negamax(self, bx: int, bo: int, is_x_turn: bool,
                 alpha: int, beta: int, depth: int) -> int:
        # Terminal: the player who just moved completed a line → they lose
        # → current player wins.
        if depth > 0:
            last_bits = bo if is_x_turn else bx
            if has_line(last_bits):
                return +1

        occupied = bx | bo
        if occupied == 0xFFFF:
            return 0  # draw — full board, no line

        # Canonicalize and probe TT (flag-aware)
        cbx, cbo = canonicalize(bx, bo)
        key = (cbx, cbo)
        entry = self._tt.get(key)
        if entry is not None:
            value, flag = entry
            if flag == TT_EXACT:
                return value
            if flag == TT_LOWER_BOUND and value > alpha:
                alpha = value
            elif flag == TT_UPPER_BOUND and value < beta:
                beta = value
            if alpha >= beta:
                return value

        empty = (~occupied) & 0xFFFF
        best = -2
        orig_alpha = alpha

        for cell in MOVE_ORDER:
            if not (empty & (1 << cell)):
                continue

            if is_x_turn:
                new_bx = bx | (1 << cell)
                new_bo = bo
                our_bits = new_bx
            else:
                new_bx = bx
                new_bo = bo | (1 << cell)
                our_bits = new_bo

            # Placing here completes our own line → instant loss (-1).
            # Never improves alpha past -1, so no pruning needed.
            if has_line(our_bits):
                if -1 > best:
                    best = -1
                continue

            score = -self._negamax(new_bx, new_bo, not is_x_turn, -beta, -alpha, depth + 1)
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        if best == -2:
            best = -1  # all legal moves were immediate self-losses

        if best <= orig_alpha:
            flag = TT_UPPER_BOUND
        elif best >= beta:
            flag = TT_LOWER_BOUND
        else:
            flag = TT_EXACT
        self._tt[key] = (best, flag)
        return best

    def _exact_value(self, bx: int, bo: int, is_x_turn: bool) -> int:
        """Exact game value (to-move perspective), via a clean full-window,
        memoized negamax with NO alpha-beta pruning — so every stored entry is a
        true value, never a bound. Memoized in `self._exact` by canonical key, so
        the whole tree costs one DFS and all later queries are O(1) lookups."""
        # terminal: the player who just moved completed a line -> they lose (misère)
        # -> current player wins.
        if (bx | bo) and has_line(bo if is_x_turn else bx):
            return +1
        if (bx | bo) == 0xFFFF:
            return 0
        key = canonicalize(bx, bo)
        cached = self._exact.get(key)
        if cached is not None:
            return cached

        occupied = bx | bo
        best = -2
        for cell in range(16):
            if occupied & (1 << cell):
                continue
            if is_x_turn:
                new_bx, new_bo, ours = bx | (1 << cell), bo, bx | (1 << cell)
            else:
                new_bx, new_bo, ours = bx, bo | (1 << cell), bo | (1 << cell)
            if has_line(ours):                 # completing our own line = instant loss
                val = -1
            elif (new_bx | new_bo) == 0xFFFF:
                val = 0
            else:
                val = -self._exact_value(new_bx, new_bo, not is_x_turn)
            if val > best:
                best = val
        if best == -2:
            best = -1                          # no legal move (shouldn't reach here)
        self._exact[key] = best
        return best

    # ---- Public API -----------------------------------------------------------

    def solve(self) -> int:
        """
        Solve from the initial empty board and populate the full TT.
        Returns the game-theoretic value for the first player (X).
        Call this once at startup before any queries.
        """
        self._tt.clear()
        self._exact.clear()
        value = self._negamax(0, 0, True, -1, 1, 0)
        # Build the pure exact-value cache (one clean DFS) so every later query is
        # an O(1), corruption-free lookup. `_negamax`'s `_tt` keeps only the
        # fail-soft bounds it computed; queries no longer touch it.
        self._exact_value(0, 0, True)
        self._solved = True
        print(f"[MisereSolver] solved: {len(self._exact):,} exact positions, "
              f"root value = {value:+d}")
        return value

    def get_position_value(self, bx: int, bo: int, is_x_turn: bool) -> int:
        """
        Return the exact game-theoretic value from the current player's perspective.
        Call solve() first for O(1) TT hits; otherwise falls back to on-demand search.
        """
        # Terminal checks
        if bx | bo:
            last_bits = bo if is_x_turn else bx
            if has_line(last_bits):
                return +1
        if (bx | bo) == 0xFFFF:
            return 0

        # Pure, exact lookup (memoized, never a bound) — see `_exact_value`.
        return self._exact_value(bx, bo, is_x_turn)

    def get_action_values(
        self, bx: int, bo: int, is_x_turn: bool
    ) -> List[Tuple[int, int]]:
        """
        Return [(cell, value), ...] for every legal action, with value
        from the current player's perspective AFTER making that move:
        +1 = the move wins, 0 = draw, -1 = the move loses.

        A move that completes our own 3-in-a-row is an immediate self-loss (-1).
        """
        results: List[Tuple[int, int]] = []
        occupied = bx | bo
        for cell in range(16):
            if occupied & (1 << cell):
                continue

            if is_x_turn:
                new_bx = bx | (1 << cell)
                new_bo = bo
                our_bits = new_bx
            else:
                new_bx = bx
                new_bo = bo | (1 << cell)
                our_bits = new_bo

            if has_line(our_bits):
                results.append((cell, -1))
                continue
            if (new_bx | new_bo) == 0xFFFF:
                results.append((cell, 0))
                continue
            child_val = self.get_position_value(new_bx, new_bo, not is_x_turn)
            results.append((cell, -child_val))
        return results

    def get_best_move(self, bx: int, bo: int, is_x_turn: bool) -> int:
        """
        Return an optimal cell. When multiple moves share the best value,
        prefers non-self-loss moves (so a losing player doesn't end the game
        prematurely). Returns -1 if no legal move exists.
        """
        occupied = bx | bo
        pos_val = self.get_position_value(bx, bo, is_x_turn)

        fallback = -1
        for cell in MOVE_ORDER:
            if occupied & (1 << cell):
                continue

            if is_x_turn:
                new_bx = bx | (1 << cell)
                new_bo = bo
                our_bits = new_bx
            else:
                new_bx = bx
                new_bo = bo | (1 << cell)
                our_bits = new_bo

            is_self_loss = has_line(our_bits)
            if is_self_loss:
                child_val = -1
            elif (new_bx | new_bo) == 0xFFFF:
                child_val = 0
            else:
                child_val = -self.get_position_value(new_bx, new_bo, not is_x_turn)

            if child_val != pos_val:
                continue
            if not is_self_loss:
                return cell            # prefer non-self-loss
            if fallback == -1:
                fallback = cell        # remember in case all optimal moves self-loss
        return fallback

    def get_optimal_moves(self, bx: int, bo: int, is_x_turn: bool) -> List[int]:
        """
        Return all optimal move cell indices at the given position.
        A move is optimal if it achieves the game-theoretic value for
        the current player.
        """
        occupied = bx | bo
        pos_val = self.get_position_value(bx, bo, is_x_turn)
        optimal: List[int] = []

        for cell in range(16):
            if occupied & (1 << cell):
                continue

            if is_x_turn:
                new_bx = bx | (1 << cell)
                new_bo = bo
                our_bits = new_bx
            else:
                new_bx = bx
                new_bo = bo | (1 << cell)
                our_bits = new_bo

            # Skip immediate self-loss — never a sensible "optimal" move even
            # in a lost position (matches C++ get_optimal_moves).
            if has_line(our_bits):
                continue

            if (new_bx | new_bo) == 0xFFFF:
                child_val = 0
            else:
                child_val = -self.get_position_value(new_bx, new_bo, not is_x_turn)

            if child_val == pos_val:
                optimal.append(cell)

        return optimal

    def get_optimal_policy(self, bx: int, bo: int, is_x_turn: bool) -> np.ndarray:
        """
        Return a uniform distribution over all optimal moves as a [16] float32 array.
        """
        optimal = self.get_optimal_moves(bx, bo, is_x_turn)
        policy = np.zeros(16, dtype=np.float32)
        if optimal:
            prob = 1.0 / len(optimal)
            for cell in optimal:
                policy[cell] = prob
        return policy


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def boards_to_solver_policies(
    boards: np.ndarray,
    solver: MisereSolver,
) -> np.ndarray:
    """
    Compute optimal solver policies for a batch of board tensors.

    Args:
        boards: float32 array of shape [B, 2, 4, 4]
        solver:  pre-solved MisereSolver instance

    Returns:
        float32 array of shape [B, 16] — uniform distribution over optimal moves
    """
    B = boards.shape[0]
    policies = np.zeros((B, 16), dtype=np.float32)
    for i in range(B):
        bx, bo, is_x_turn = board_to_masks(boards[i])
        policies[i] = solver.get_optimal_policy(bx, bo, is_x_turn)
    return policies
