"""Web monitor + launcher for the jax_az AlphaZero loop.

Decoupled by design: the trainer (runner.py) is a subprocess that only *appends*
metrics to files; the server (server.py) only *reads* them and never imports jax.
See monitor_plan.md.
"""
