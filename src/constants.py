"""
Game and neural network constants — Python mirror of src/core/game/constants.h

ALL values here must stay in sync with constants.h.
"""

# ============================================================================
# BOARD DIMENSIONS
# ============================================================================

BOARD_SIZE   = 4
BOARD_HEIGHT = 4
BOARD_WIDTH  = 4
BOARD_CELLS  = BOARD_SIZE * BOARD_SIZE       # 16

# ============================================================================
# NEURAL NETWORK INPUT
# ============================================================================

INPUT_PLANES   = 2                           # current pieces, opponent pieces
INPUT_CHANNELS = INPUT_PLANES                # alias used in some modules
INPUT_SIZE     = INPUT_PLANES * BOARD_HEIGHT * BOARD_WIDTH  # 48 floats (flat)

# ============================================================================
# POLICY / ACTION SPACE
# ============================================================================

POLICY_SIZE = BOARD_CELLS                   # 16 — one action per board cell

# ============================================================================
# WIN CONDITION
# ============================================================================

WIN_LENGTH     = 3                           # consecutive pieces needed to win
HISTORY_LENGTH = 4                           # past board states stored in Board::history

# ============================================================================
# GAME LIMITS
# ============================================================================

MAX_GAME_MOVES = 20
