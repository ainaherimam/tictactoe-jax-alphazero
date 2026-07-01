"""ALL editable knobs for the JAX AlphaZero loop, in one place, by category.

Edit the values below, then launch:

    python -m jax_az.config                 # CPU/GPU = auto
    JAX_AZ_DEVICE=gpu python -m jax_az.config

Every field here IS the real dataclass field the loop reads (GameConfig /
SearchConfig / LoopConfig / TrainingConfig) — there is no shadow copy to keep in
sync. train_loop() already takes these four objects; this file just spells out
every knob so a whole run is configured in one edit.
"""
from jax_az.env import GameConfig
from jax_az.eval_solver import EvalConfig
from jax_az.search import SearchConfig
from jax_az.train_jax import LoopConfig, train_loop
from src.models.alphazero_model import TrainingConfig

SEED = 0

# Run directory for checkpoints + resume bundle (full state + replay ring + scalars).
# None: ephemeral run, nothing persisted beyond cfg.checkpoint_dir. Set a path and the
# run becomes resumable: relaunching with the same RUN_DIR continues where it stopped;
# an empty/absent dir starts fresh (random weights, empty ring).
RUN_DIR = None

# === GAME ==================================================================
GAME = GameConfig(
    size=4,            # board is size x size (3..5)
    win_length=3,      # k-in-a-row to complete a line (3..size)
    misere=True,       # True: completing a line LOSES; False: it wins
)

# === SEARCH — MCTS / mctx ==================================================
SEARCH = SearchConfig(
    algorithm="muzero",            # "muzero" (PUCT) | "gumbel"
    num_simulations=100,
    max_depth=None,                # cap tree depth; None = unbounded

    # PUCT / muzero_policy
    pb_c_init=1.25,                # exploration constant
    pb_c_base=19652.0,             # exploration log-growth base
    dirichlet_alpha=0.5,           # root Dirichlet noise concentration
    dirichlet_fraction=0.25,       # weight of root noise vs prior
    temperature=1.0,               # action-sampling temperature (0 = argmax)
    temp_drop_ply=None,            # ply at which temperature drops to temp_final
    temp_final=0.0,                # temperature for plies >= temp_drop_ply

    # Gumbel / gumbel_muzero_policy (only used when algorithm="gumbel")
    max_num_considered_actions=16,
    gumbel_scale=1.0,

    # q-value transform
    qtransform_epsilon=1e-8,
    value_scale=0.1,               # gumbel q-transform only
    maxvisit_init=50.0,            # gumbel q-transform only
    rescale_values=True,           # gumbel q-transform only
    use_mixed_value=True,          # gumbel q-transform only
)

# === SELF-PLAY + REPLAY RING ==============================================
LOOP = LoopConfig(
    num_generations=1000,
    games_per_gen=256,             # B parallel self-play games per generation
    replay_capacity=1_000_000,     # ring size (rounded up to a multiple of B*cells)
    eval_batch_size=512,           # positions sampled for solver metrics
)

# === NET ARCH + TRAINING + OPTIMIZER + CHECKPOINT/EVAL ====================
# TrainingConfig is a plain class (class-level attrs), so set fields by assignment.
TRAIN = TrainingConfig()

# --- net architecture (the loop builds the net from these) ---
TRAIN.num_channels = 64
TRAIN.num_res_blocks = 4
# value-head variant — pick one:
#   v1_scalar_mse        scalar tanh in [-1,1]; MSE loss
#   v2_softmax_ce        3 logits over {-1,0,+1}; softmax cross-entropy
#   v3_softmax_mse       scalar E[v] from a 3-bin softmax; MSE
#   v4_tanh_per_bin_mse  3 independent tanh values; per-bin MSE vs +/-1
#   v5_independent_bce   3 logits, independent Bernoullis; per-bin BCE
#   v6_scalar_l1         scalar tanh in [-1,1]; L1 loss
TRAIN.variant = "v1_scalar_mse"
TRAIN.train_value_only = False     # freeze policy head, train value+body only
TRAIN.lambda_v = 1.0               # value-loss weight

# --- data ---
TRAIN.min_positions = 1024         # wait for this many before training starts
TRAIN.batch_size = 128             # positions per gradient step
TRAIN.steps_per_generation = 10    # gradient steps per generation

# --- optimizer ---
TRAIN.learning_rate = 0.001
TRAIN.weight_decay = 1e-4
TRAIN.grad_clip = 1.0
TRAIN.lr_schedule = "cosine"       # "constant" | "cosine"
TRAIN.lr_warmup_steps = 100
# Cosine horizon must ≈ total grad steps (num_generations × steps_per_generation),
# else the schedule never anneals and LR stays at peak the whole run. 1500×10=15k.
TRAIN.lr_decay_steps = 15_000
TRAIN.lr_min = 1e-5

# --- checkpoint / eval ---
TRAIN.checkpoint_dir = "checkpoints"
TRAIN.save_every_n_gens = 10
TRAIN.evaluate_every_n_gens = 300

# === EVALUATION — AZ (X) vs the perfect Misère solver (O) =================
# Used by `python -m jax_az.eval_solver <checkpoint>` and the monitor's Eval tab.
# Eval is greedy: temperature 0, no Dirichlet noise (pure exploitation).
EVAL = EvalConfig(
    sims=400,          # MCTS simulations per AZ move (higher = stronger, slower)
    per_group=100,     # eval boards per theory group; 4 groups -> 4x this many games
    seed=0,            # PRNG seed for the batched search
    pgn=False,         # also write per-game annotated PGN files under eval_games/
)

# Catch a typo'd variant before a multi-hour launch wastes itself.
assert TRAIN.variant in {
    "v1_scalar_mse", "v2_softmax_ce", "v3_softmax_mse",
    "v4_tanh_per_bin_mse", "v5_independent_bce", "v6_scalar_l1",
}, f"unknown net variant: {TRAIN.variant!r}"
assert SEARCH.algorithm in {"muzero", "gumbel"}, \
    f"unknown search algorithm: {SEARCH.algorithm!r}"


def _dump_config_json(run_dir: str):
    """Persist these settings as <run_dir>/config.json, in the same schema the
    monitor server/runner reads. Written only on a fresh run (no config.json yet) so
    a resume keeps the settings it started with — edit/delete the file to change them."""
    import dataclasses, json, os
    path = os.path.join(run_dir, "config.json")
    if os.path.exists(path):
        return
    os.makedirs(run_dir, exist_ok=True)
    train = {k: getattr(TRAIN, k) for k in dir(TRAIN)
             if not k.startswith("_")
             and isinstance(getattr(TRAIN, k), (int, float, str, bool, type(None)))}
    cfg = {"game": dataclasses.asdict(GAME), "search": dataclasses.asdict(SEARCH),
           "loop": dataclasses.asdict(LOOP), "train": train,
           "seed": SEED, "device": os.environ.get("JAX_AZ_DEVICE", "auto")}
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] settings -> {path}")


if __name__ == "__main__":
    if RUN_DIR:
        _dump_config_json(RUN_DIR)
    train_loop(seed=SEED, cfg=TRAIN, loop=LOOP, search_cfg=SEARCH, game=GAME,
               run_dir=RUN_DIR)
