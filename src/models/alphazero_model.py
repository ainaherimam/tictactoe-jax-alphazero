"""
AlphaZero Neural Network Model - JAX/Flax Implementation
=========================================================
Contains all model-related components:
- Network architecture (ResNet-style)
- Loss function
- Training state management
- JIT-compiled training step
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training import train_state
from typing import Tuple, Any, Dict, Optional, TYPE_CHECKING
import orbax.checkpoint as ocp
from pathlib import Path

from src.constants import INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH, POLICY_SIZE

if TYPE_CHECKING:
    from src.core.solver.misere_solver import MisereSolver


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""
    num_channels: int
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        residual = x
        
        x = nn.Conv(self.num_channels, (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        x = nn.Conv(self.num_channels, (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        x = x + residual
        x = nn.relu(x)
        
        return x


VALUE_VARIANTS = (
    "v1_scalar_mse",
    "v2_softmax_ce",
    "v3_softmax_mse",
    "v4_tanh_per_bin_mse",
    "v5_independent_bce",
    "v6_scalar_l1",
)


class AlphaZeroNet(nn.Module):
    """AlphaZero neural network: board → (policy, value).

    Value head selected by `variant`:
      - v1_scalar_mse:        scalar tanh in [-1, 1]; MSE loss
      - v2_softmax_ce:        [B, 3] raw logits over {-1, 0, +1}; softmax CE loss
      - v3_softmax_mse:       scalar E[v] from a 3-bin softmax; MSE loss
      - v4_tanh_per_bin_mse:  [B, 3] independent tanh values; per-bin MSE vs ±1
      - v5_independent_bce:   [B, 3] raw logits for independent Bernoullis; per-bin BCE
      - v6_scalar_l1:         scalar tanh in [-1, 1]; L1 loss
    """
    num_channels: int = 64
    num_res_blocks: int = 4
    num_actions: int = 16
    variant: str = "v1_scalar_mse"

    @nn.compact
    def __call__(self, x, mask, training: bool = True):
        # Input arrives as NCHW [B, C, H, W]; Flax nn.Conv expects NHWC.
        x = jnp.transpose(x, (0, 2, 3, 1))

        # Input conv
        x = nn.Conv(self.num_channels, (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        # Residual tower
        for _ in range(self.num_res_blocks):
            x = ResidualBlock(self.num_channels)(x, training=training)

        # Policy head
        p = nn.Conv(2, (1, 1))(x)
        p = nn.BatchNorm(use_running_average=not training)(p)
        p = nn.relu(p)
        p = p.reshape((p.shape[0], -1))  # Flatten
        p = nn.Dense(self.num_actions)(p)

        # Apply log-softmax
        p = jax.nn.log_softmax(p, axis=-1)

        # Value head
        v = nn.Conv(1, (1, 1))(x)
        v = nn.BatchNorm(use_running_average=not training)(v)
        v = nn.relu(v)
        v = v.reshape((v.shape[0], -1))  # Flatten
        v = nn.Dense(64)(v)
        v = nn.relu(v)
        bin_centers = jnp.array([-1.0, 0.0, 1.0])

        if self.variant in ("v1_scalar_mse", "v6_scalar_l1"):
            # V1 (MSE) and V6 (L1) share the same scalar tanh head;
            # only the training loss differs.
            v = nn.Dense(1)(v)
            v = jnp.tanh(v).squeeze(-1)                     # [B] scalar in [-1, 1]

        elif self.variant == "v2_softmax_ce":
            # V2: return RAW LOGITS over the three bins.
            # Loss: optax.softmax_cross_entropy on these logits.
            # MCTS readout: scalar = (softmax(logits) * bin_centers).sum(-1)
            v = nn.Dense(3)(v)                              # [B, 3] logits

        elif self.variant == "v3_softmax_mse":
            # V3: softmax + scalar E[v] readout, trained with MSE on E[v].
            v_logits = nn.Dense(3)(v)
            v_probs = jax.nn.softmax(v_logits, axis=-1)     # [B, 3], Σ = 1
            v = (v_probs * bin_centers).sum(-1)             # [B] in [-1, 1]

        elif self.variant == "v4_tanh_per_bin_mse":
            # V4 (revised): 3 independent tanh outputs, no normalisation across bins.
            # Trained per-bin with MSE against signed indicators ∈ {-1, +1}, so every
            # logit (including the draw bin) receives a real, symmetric gradient.
            # Head returns [B, 3]; loss applies per-bin MSE; MCTS reads out a scalar
            # as (tanh_values * bin_centers).sum(-1) / 2.
            v_logits = nn.Dense(3)(v)
            v = jnp.tanh(v_logits)                          # [B, 3] in [-1, 1]

        elif self.variant == "v5_independent_bce":
            # V5: return RAW LOGITS for three INDEPENDENT Bernoullis (one per bin).
            # Loss: optax.sigmoid_binary_cross_entropy per bin, summed over bins.
            # MCTS readout: scalar = (sigmoid(logits) * bin_centers).sum(-1) ∈ [-1, 1]
            v = nn.Dense(3)(v)                              # [B, 3] logits

        else:
            raise ValueError(f"Unknown value-head variant: {self.variant}")

        return p, v


# ============================================================================
# TRAINING STATE
# ============================================================================

class TrainStateWithBatchStats(train_state.TrainState):
    """Extended training state with batch statistics for batch norm."""
    batch_stats: Any


# ============================================================================
# BODY-PARAMETER IDENTIFICATION (for grad-norm diagnostics)
# ============================================================================

# Body = input conv/BN + the residual tower. Everything else (policy head
# Conv_1/BN_1/Dense_0 and value head Conv_2/BN_2/Dense_1/Dense_2[/Dense_3]) is
# considered a head. This mirrors the layer order in AlphaZeroNet.__call__.
def _is_body_path(path) -> bool:
    if not path:
        return False
    top = path[0]
    key = top.key if hasattr(top, "key") else top
    if not isinstance(key, str):
        return False
    return key.startswith("ResidualBlock") or key in ("Conv_0", "BatchNorm_0")


def _body_grad_norm(grads) -> jnp.ndarray:
    """L2 norm of the gradient restricted to body parameters."""
    masked = jax.tree_util.tree_map_with_path(
        lambda path, g: g if _is_body_path(path) else jnp.zeros_like(g),
        grads,
    )
    return optax.global_norm(masked)


# ============================================================================
# LOSS FUNCTION
# ============================================================================


def scalar_to_categorical(z):
    """Map z ∈ {-1, 0, +1} to one-hot over bins [loss, draw, win]."""
    bin_idx = (z + 1).astype(jnp.int32)              # {-1,0,+1} → {0,1,2}
    return jax.nn.one_hot(bin_idx, num_classes=3)     # [B, 3]


def _compute_value_loss(v_pred, z_target, variant: str):
    """Variant-specific value loss. Pure function of (pred, target, variant)."""
    z_float = z_target.astype(jnp.float32)
    if variant == "v1_scalar_mse":
        return jnp.mean((v_pred - z_float) ** 2)
    if variant == "v6_scalar_l1":
        return jnp.mean(jnp.abs(v_pred - z_float))
    if variant == "v2_softmax_ce":
        z_onehot = scalar_to_categorical(z_target)
        return optax.softmax_cross_entropy(logits=v_pred, labels=z_onehot).mean()
    if variant == "v3_softmax_mse":
        return jnp.mean((v_pred - z_float) ** 2)
    if variant == "v4_tanh_per_bin_mse":
        z_onehot = scalar_to_categorical(z_target)
        target = 2.0 * z_onehot - 1.0
        return ((v_pred - target) ** 2).sum(axis=-1).mean()
    if variant == "v5_independent_bce":
        z_onehot = scalar_to_categorical(z_target)
        per_bin_bce = optax.sigmoid_binary_cross_entropy(
            logits=v_pred, labels=z_onehot
        )
        return per_bin_bce.sum(axis=-1).mean()
    raise ValueError(f"Unknown value loss variant: {variant}")


def alphazero_loss(params, state, batch, lambda_v: float = 1.0,
                   train_value_only: bool = False,
                   variant: str = "v1_scalar_mse"):
    """
    Compute AlphaZero loss: cross-entropy(policy) + value loss.

    The value-loss formula depends on `variant` (see AlphaZeroNet).

    When train_value_only=True, only value_loss contributes to the gradient.
    Policy head parameters receive zero gradient and remain at their initial
    (random) values, while the shared body and value head are trained normally.

    Returns:
        (total_loss, (metrics_dict, new_batch_stats))
    """
    boards = batch['boards']  # [B, 2, 4, 4]
    pi_target = batch['pi']   # [B, 16]
    z_target = batch['z']     # [B]  values in {-1, 0, +1}
    mask = batch['mask']      # [B, 16]

    # Forward pass
    (p_pred, v_pred), updates = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        boards,
        mask,
        training=True,
        mutable=['batch_stats']
    )
    # p_pred: [B, 16] log-probs
    # v_pred: shape depends on variant — [B] scalar, or [B, 3]

    # Policy loss: cross-entropy with target distribution
    policy_loss = -jnp.sum(pi_target * p_pred, axis=-1).mean()

    # Sanity-check head shape against variant expectations.
    if variant in ("v2_softmax_ce", "v5_independent_bce", "v4_tanh_per_bin_mse"):
        assert v_pred.ndim == 2 and v_pred.shape[-1] == 3, \
            f"{variant} expects [B, 3], got {v_pred.shape}"
    else:
        assert v_pred.ndim == 1, \
            f"{variant} expects [B] scalar, got {v_pred.shape}"

    value_loss = _compute_value_loss(v_pred, z_target, variant)

    # λ_v rescales the value loss so its magnitude is comparable to the policy
    # loss across variants. Calibrated per-variant in train.py.
    effective_value_loss = lambda_v * value_loss

    # Total loss — when train_value_only, drop policy gradient entirely.
    # policy_loss is still computed above for metric reporting.
    total_loss = effective_value_loss if train_value_only else (
        policy_loss + effective_value_loss
    )

    # Policy accuracy: pred top-1 == target top-1
    top1_target = jnp.argmax(pi_target, axis=-1)
    top1_pred   = jnp.argmax(p_pred,   axis=-1)
    policy_top1_acc = jnp.mean(top1_pred == top1_target)

    # Value accuracy — collapse each variant's head to a scalar in [-1, 1],
    # then compare sign (with a 0.33 dead-zone for draws) against z_target.
    bin_centers_arr = jnp.array([-1.0, 0.0, 1.0])
    if variant == "v2_softmax_ce":
        v_scalar = (jax.nn.softmax(v_pred, axis=-1) * bin_centers_arr).sum(-1)
    elif variant == "v4_tanh_per_bin_mse":
        v_scalar = (v_pred * bin_centers_arr).sum(-1) / 2.0
    elif variant == "v5_independent_bce":
        v_scalar = (jax.nn.sigmoid(v_pred) * bin_centers_arr).sum(-1)
    else:
        v_scalar = v_pred
    v_pred_sign = jnp.where(jnp.abs(v_scalar) < 0.33, 0.0, jnp.sign(v_scalar))
    value_accuracy = jnp.mean(v_pred_sign == jnp.sign(z_target))

    # Metrics
    metrics = {
        'loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'effective_value_loss': effective_value_loss,
        'policy_to_value_ratio': policy_loss / (effective_value_loss + 1e-8),
        'lambda_v': jnp.asarray(lambda_v, dtype=jnp.float32),
        'policy_entropy': -jnp.sum(jnp.exp(p_pred) * p_pred, axis=-1).mean(),
        'value_accuracy': value_accuracy,
        'policy_top1_acc': policy_top1_acc,   # pred top-1 == target top-1
    }

    return total_loss, (metrics, updates['batch_stats'])

# ============================================================================
# TRAINING STEP (JIT-COMPILED)
# ============================================================================

@functools.partial(jax.jit, static_argnums=(3, 4))
def train_step(state, batch, lambda_v, train_value_only: bool = False,
               variant: str = "v1_scalar_mse"):
    """
    Single gradient step.

    Args:
        state: Training state with params, optimizer state, batch_stats
        batch: Dictionary with 'boards', 'pi', 'z', 'mask'
        lambda_v: Scalar weight on value loss (runtime, not static — change it
            without triggering recompilation).
        train_value_only: If True, only the value head (and shared body) are
            trained; policy head parameters receive zero gradient and remain
            at their initial random values.
        variant: Value-head variant string (see AlphaZeroNet).

    Returns:
        (updated_state, metrics_dict)
    """
    grad_fn = jax.value_and_grad(alphazero_loss, has_aux=True)
    (loss, (metrics, new_batch_stats)), grads = grad_fn(
        state.params, state, batch, lambda_v, train_value_only, variant
    )

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)

    metrics['grad_norm'] = optax.global_norm(grads)
    metrics['body_grad_norm'] = _body_grad_norm(grads)

    return state, metrics


# ----------------------------------------------------------------------------
# Per-head body-gradient diagnostics
# ----------------------------------------------------------------------------
# Two extra backward passes — one for policy_loss alone, one for
# (λ_v · value_loss) alone — so we can measure each head's gradient signal
# into the shared body. Only call on logging steps; the extra cost is ~2×
# a normal backward, but it is the only direct check that λ_v is doing its
# job of equalising value-vs-policy influence on the body across variants.
@functools.partial(jax.jit, static_argnums=(3,))
def compute_grad_diagnostics(state, batch, lambda_v,
                             variant: str = "v1_scalar_mse"):
    """Body-grad norms for policy-only and value-only losses (no state update)."""

    def _policy_only(params):
        (p_pred, _), _ = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['boards'], batch['mask'], training=True,
            mutable=['batch_stats'],
        )
        return -jnp.sum(batch['pi'] * p_pred, axis=-1).mean()

    def _value_only(params):
        (_, v_pred), _ = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['boards'], batch['mask'], training=True,
            mutable=['batch_stats'],
        )
        return lambda_v * _compute_value_loss(v_pred, batch['z'], variant)

    grads_p = jax.grad(_policy_only)(state.params)
    grads_v = jax.grad(_value_only)(state.params)

    p_body = _body_grad_norm(grads_p)
    v_body = _body_grad_norm(grads_v)

    return {
        'policy_only_body_grad_norm': p_body,
        'value_only_body_grad_norm':  v_body,
        'value_to_policy_body_grad_ratio': v_body / (p_body + 1e-8),
    }


# ============================================================================
# SOLVER-BASED POLICY ACCURACY METRICS
# ============================================================================

def compute_solver_metrics(
    boards: np.ndarray,
    pi_mcts: np.ndarray,
    p_pred_log: np.ndarray,
    solver: "MisereSolver",
) -> Dict[str, float]:
    """
    Measure how well MCTS targets and NN predictions agree with the perfect solver.

    For each board position the solver returns all optimal moves (moves that
    achieve the game-theoretic value).  We then check whether the greedy
    action from each policy falls among those optimal moves.

    Args:
        boards:      [B, 2, 4, 4]  float32 board planes (current, opponent)
        pi_mcts:     [B, 16]       MCTS visit-count policy (the training target π)
        p_pred_log:  [B, 16]       NN log-probabilities (output of the network)
        solver:      pre-solved MisereSolver instance (call solver.solve() once first)

    Returns:
        Dict with:
          'policy_acc_mcts' — fraction of positions where MCTS top-1 is solver-optimal
          'policy_acc_nn'   — fraction of positions where NN  top-1 is solver-optimal
    """
    from src.core.solver.misere_solver import board_to_masks

    boards_np   = np.asarray(boards)
    pi_np       = np.asarray(pi_mcts)
    p_pred_np   = np.asarray(p_pred_log)

    B = boards_np.shape[0]
    mcts_correct = 0
    nn_correct   = 0
    valid_count  = 0

    for i in range(B):
        bx, bo, is_x_turn = board_to_masks(boards_np[i])
        optimal = set(solver.get_optimal_moves(bx, bo, is_x_turn))
        if not optimal:
            continue

        valid_count += 1

        if int(np.argmax(pi_np[i])) in optimal:
            mcts_correct += 1

        # Mask occupied cells before argmax: NN log-softmax is over all 16
        # cells, so the raw argmax can land on an occupied cell.
        occupied = bx | bo
        nn_logits = p_pred_np[i].copy()
        for c in range(16):
            if occupied & (1 << c):
                nn_logits[c] = -np.inf
        if int(np.argmax(nn_logits)) in optimal:
            nn_correct += 1

    if valid_count == 0:
        return {"policy_acc_mcts": 0.0, "policy_acc_nn": 0.0}

    return {
        "policy_acc_mcts": mcts_correct / valid_count,
        "policy_acc_nn":   nn_correct   / valid_count,
    }


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

class TrainingConfig:
    """Training hyperparameters and settings."""
    
    # Data settings
    min_positions: int = 1024           # Wait for this many before starting
    batch_size: int = 128               # Positions per gradient step
    steps_per_generation: int = 10      # Steps to run when new data arrives
    
    # Model architecture
    num_channels: int = 64              # Residual block channels
    num_res_blocks: int = 4             # Number of residual blocks
    
    # Optimization
    learning_rate: float = 0.001        # Initial learning rate
    weight_decay: float = 1e-4          # L2 regularization
    grad_clip: float = 1.0              # Gradient clipping threshold
    
    # Learning rate schedule
    lr_schedule: str = "cosine"         # "constant" or "cosine"
    lr_warmup_steps: int = 100          # Warmup steps for cosine schedule
    lr_decay_steps: int = 1_000_000     # Cosine decay horizon (post-warmup)
    lr_min: float = 1e-5                # Minimum LR for cosine schedule
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_gens: int = 10        # Save checkpoint every N generations
    evaluate_every_n_gens: int = 300
    train_every_n_gens: int = 100
    
    # Training mode
    train_value_only: bool = False       # If True, freeze policy head (value head + body only)
    variant: str = "v1_scalar_mse"       # Value-head variant; see AlphaZeroNet.VALUE_VARIANTS

    # Value-loss weighting (fairness across variants).
    # Currently fixed at 1.0; per-variant calibration may be added later.
    lambda_v: float = 1.0

    # Logging
    log_every_n_steps: int = 10         # Print metrics every N steps
    verbose: bool = True                # Detailed logging


def create_inference_state(rng, num_channels: int = 64, num_res_blocks: int = 4,
                           num_actions: int = POLICY_SIZE,
                           variant: str = "v1_scalar_mse"):
    """
    Initialize model for inference only (no optimizer).

    Args:
        rng: JAX random key
        num_channels: Number of channels in residual blocks
        num_res_blocks: Number of residual blocks
        num_actions: Number of possible actions (policy size)

    Returns:
        Dictionary with 'params', 'batch_stats', and 'apply_fn'
    """
    # Create model
    model = AlphaZeroNet(
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        num_actions=num_actions,
        variant=variant,
    )

    # Initialize with dummy input
    dummy_board = jnp.zeros((1, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
    dummy_mask = jnp.ones((1, num_actions))

    variables = model.init(rng, dummy_board, dummy_mask, training=False)

    return {
        'params': variables['params'],
        'batch_stats': variables['batch_stats'],
        'apply_fn': model.apply,
    }


def create_train_state(rng, config: TrainingConfig):
    """
    Initialize model and optimizer.
    
    Args:
        rng: JAX random key
        config: Training configuration
    
    Returns:
        Initialized training state with params, optimizer, and batch_stats
    """
    
    # Create model
    model = AlphaZeroNet(
        num_channels=config.num_channels,
        num_res_blocks=config.num_res_blocks,
        num_actions=POLICY_SIZE,
        variant=config.variant,
    )

    # Initialize with dummy input
    dummy_board = jnp.zeros((1, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
    dummy_mask = jnp.ones((1, POLICY_SIZE))
    
    variables = model.init(rng, dummy_board, dummy_mask, training=False)
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    # Create optimizer with weight decay and gradient clipping
    if config.lr_schedule == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.lr_warmup_steps,
            decay_steps=config.lr_warmup_steps + config.lr_decay_steps,
            end_value=config.lr_min,
        )
    else:
        schedule = config.learning_rate
    
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    )
    
    # Create training state with batch_stats
    state = TrainStateWithBatchStats.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )
    
    return state


# ============================================================================
# CHECKPOINTING (ORBAX)
# ============================================================================

# Global checkpointer instance (reused across saves/loads)
_checkpointer = None

def get_checkpointer():
    """Get or create the global checkpointer instance."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = ocp.StandardCheckpointer()
    return _checkpointer


def save_checkpoint(state, checkpoint_dir: str, step: Optional[int] = None):
    """
    Save model weights using Orbax (params and batch_stats only).
    
    Args:
        state: Training state
        checkpoint_dir: Directory to save checkpoints
        step: Training step number (uses state.step if None)
    """

    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Determine checkpoint path
    step = int(step if step is not None else state.step)
    checkpoint_path = checkpoint_dir / f"checkpoint_{step}"

    # Only save weights (params and batch_stats)
    checkpoint_data = {
        'params': state.params,
        'batch_stats': state.batch_stats,
    }

    checkpointer = get_checkpointer()
    checkpointer.save(checkpoint_path, checkpoint_data)

    print(f"[Checkpoint] ✓ Saved weights to checkpoint_{step}")



def load_checkpoint(state, checkpoint_path: str, learning_rate: float = 1e-3, 
                    weight_decay: float = 1e-4, grad_clip: float = 1.0) -> 'TrainStateWithBatchStats':
    """
    Load model weights from Orbax checkpoint and create fresh optimizer.
    
    Args:
        state: Template training state (for structure)
        checkpoint_path: Path to checkpoint directory
        learning_rate: Learning rate for new optimizer
        weight_decay: Weight decay for new optimizer
        grad_clip: Gradient clipping threshold for new optimizer
    
    Returns:
        New training state with loaded weights and fresh optimizer
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    print(f"[Checkpoint] Loading weights from {checkpoint_path}")
    
    # Load checkpoint
    checkpointer = get_checkpointer()
    restored = checkpointer.restore(checkpoint_path)
    
    # Create fresh optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    )
    
    # Create new training state with loaded weights
    state = TrainStateWithBatchStats.create(
        apply_fn=state.apply_fn,
        params=restored['params'],
        tx=tx,
        batch_stats=restored.get('batch_stats', {}),
    )
    
    print(f"[Checkpoint] ✓ Loaded weights (created fresh optimizer)")
    return state


def load_checkpoint_for_inference(
    checkpoint_path: str,
    num_channels: int = 64,
    num_res_blocks: int = 4,
    num_actions: int = 16,
    variant: str = "v1_scalar_mse",
) -> Dict:
    """
    Load checkpoint for inference only (no optimizer).
    
    This loads only the model weights (params and batch_stats).
    
    Args:
        checkpoint_path: Path to checkpoint directory
        num_channels: Number of channels in model
        num_res_blocks: Number of residual blocks  
        num_actions: Number of possible actions
    
    Returns:
        Dictionary with 'params', 'batch_stats', 'apply_fn'
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    print(f"[Checkpoint] Loading inference weights from {checkpoint_path}")

    # Build a target tree on *local* devices so Orbax restores onto whatever
    # devices we actually have. Without a target it rebuilds the saved sharding,
    # which fails ("Device cuda:0 not found") when a GPU-saved checkpoint is
    # loaded on a host without that GPU (e.g. CPU inference server).
    target = create_inference_state(
        jax.random.PRNGKey(0), num_channels, num_res_blocks, num_actions, variant
    )

    # Load checkpoint directly with Orbax
    checkpointer = get_checkpointer()
    restored = checkpointer.restore(
        checkpoint_path,
        target={'params': target['params'], 'batch_stats': target['batch_stats']},
    )

    # Create model for apply_fn
    model = AlphaZeroNet(
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        num_actions=num_actions,
        variant=variant,
    )

    print(f"[Checkpoint] ✓ Loaded inference weights")
    
    return {
        'params': restored['params'],
        'batch_stats': restored.get('batch_stats', {}),
        'apply_fn': model.apply,
    }