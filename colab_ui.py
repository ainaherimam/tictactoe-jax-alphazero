"""Launch the monitor + launcher web UI on Google Colab.

In a Colab cell, after `bash jax_az/setup.sh`:

    from jax_az.colab_ui import launch_ui
    launch_ui()                      # opens the dashboard in a new browser tab

Colab can't reach localhost:8000 directly, so we start launch_ui.sh in the
background (server stays jax-free, spawns trainers as subprocesses) and expose
the port with google.colab.output. The UI code itself is unchanged.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def launch_ui(port: int = 8000, wait: float = 3.0, iframe: bool = False):
    """Start the UI server in the background and open it. Returns the Popen."""
    proc = subprocess.Popen(
        ["bash", str(HERE / "launch_ui.sh")],
        env={**os.environ, "PORT": str(port)},
    )
    time.sleep(wait)                       # let the server bind before we open it
    from google.colab import output        # only exists on Colab
    if iframe:
        output.serve_kernel_port_as_iframe(port)
    else:
        output.serve_kernel_port_as_window(port)
    print(f"[colab] monitor UI on port {port} (pid {proc.pid})")
    return proc
