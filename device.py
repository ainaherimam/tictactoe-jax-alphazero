"""CPU/GPU switch for the JAX path — a thin name over JAX's native JAX_PLATFORMS.

The switch is the env var **JAX_AZ_DEVICE = cpu | gpu | auto** (default auto). It is
applied (mapped to JAX_PLATFORMS) at `jax_az` import, *before* jax loads its backend,
so every entry point honors it uniformly with no per-script wiring:

    JAX_AZ_DEVICE=gpu  python -m jax_az.train_jax
    JAX_AZ_DEVICE=cpu  python -m jax_az.selfplay
    python -m jax_az.train_jax --device gpu      # CLI flag re-execs to apply early

`auto` leaves JAX to pick a GPU if a CUDA backend is present, else CPU. An explicit
JAX_PLATFORMS always wins (we only `setdefault`). There is no custom device code —
JAX already has the switch; this just gives it a friendly name and applies it once.
"""
import os

# jax backend name for each requested device (gpu -> the cuda backend)
_PLATFORM = {"cpu": "cpu", "gpu": "cuda", "cuda": "cuda", "tpu": "tpu", "auto": ""}


def apply_from_env() -> None:
    """Map JAX_AZ_DEVICE -> JAX_PLATFORMS before jax is imported. Idempotent."""
    dev = os.environ.get("JAX_AZ_DEVICE", "auto").strip().lower()
    if dev not in _PLATFORM:
        raise ValueError(f"JAX_AZ_DEVICE must be cpu|gpu|auto, got {dev!r}")
    plat = _PLATFORM[dev]
    if plat:  # cpu/gpu -> force that backend; auto -> let JAX choose
        os.environ.setdefault("JAX_PLATFORMS", plat)


def set_device(dev: str) -> None:
    """Programmatic switch. Must run BEFORE the first `import jax` to take effect."""
    os.environ["JAX_AZ_DEVICE"] = dev
    apply_from_env()


def info() -> str:
    """One-line report of the live JAX backend (imports jax lazily)."""
    import jax
    return (f"jax {jax.__version__} | backend={jax.default_backend()} "
            f"| devices={jax.devices()}")


if __name__ == "__main__":
    # `python -m jax_az.device` -> show what device the switch resolved to.
    print(info())
