"""Monitor + launcher web server. Stdlib only — never imports jax.

    python jax_az/monitor/server.py            # http://localhost:8000

Run it as a FILE, not `-m jax_az.monitor.server`: the `-m` form would import the
jax_az package (whose __init__ imports the JAX env), pulling jax into this process
and letting it preallocate GPU memory against the trainer. As a plain script it
imports only stdlib and spawns the trainer in a separate process.

It serves the dashboard, launches a trainer as a subprocess
(`python -m jax_az.monitor.runner <run_dir>`), and reads each run's files to report
status. It only ever *reads* metrics.jsonl / last.json / run.json and *lists* the
checkpoints dir, so it cannot block or slow the trainer.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PKG_DIR = Path(__file__).resolve().parents[1]   # the jax_az package dir (holds src/, runs)
LAUNCH_DIR = PKG_DIR.parent                      # parent, so `-m jax_az.*` resolves
RUNS = PKG_DIR / "jax_az_runs"
STATIC = Path(__file__).resolve().parent / "static"

# Form defaults — mirror jax_az/config.py. Kept as a plain dict so the server stays
# jax-free; the runner maps these onto the real dataclass fields.
DEFAULTS = {
    "seed": 0,
    "device": "auto",                  # cpu | gpu | auto
    "game": {"size": 4, "win_length": 3, "misere": True},
    "search": {
        "algorithm": "muzero", "num_simulations": 100, "max_depth": None,
        "pb_c_init": 1.25, "pb_c_base": 19652.0,
        "dirichlet_alpha": 0.5, "dirichlet_fraction": 0.25,
        "temperature": 1.0, "temp_drop_ply": None, "temp_final": 0.0,
        "temp_schedule": [],  # [[until_ply, temp], ...]; wins over temp_drop_ply

        "max_num_considered_actions": 16, "gumbel_scale": 1.0,
    },
    "loop": {
        "num_generations": 1000, "games_per_gen": 256,
        "replay_capacity": 1_000_000, "eval_batch_size": 512,
    },
    "train": {
        "num_channels": 64, "num_res_blocks": 4, "variant": "v1_scalar_mse",
        "train_value_only": False, "lambda_v": 1.0,
        "min_positions": 1024, "batch_size": 128, "steps_per_generation": 10,
        "learning_rate": 0.001, "weight_decay": 1e-4, "grad_clip": 1.0,
        "lr_schedule": "cosine", "lr_warmup_steps": 100,
        "lr_decay_steps": 1_000_000, "lr_min": 1e-5,
        "save_every_n_gens": 10, "evaluate_every_n_gens": 300,
    },
    # AZ-vs-solver eval knobs — mirrors jax_az/config.py EVAL (EvalConfig).
    "eval": {
        "sims": 400, "per_group": 100, "seed": 0, "pgn": False,
    },
}


def _reap():
    """Reap any finished child (trainer/eval) so it doesn't linger as a zombie.
    Best-effort; ECHILD just means we have no children to reap right now."""
    try:
        while os.waitpid(-1, os.WNOHANG)[0] != 0:
            pass
    except ChildProcessError:
        pass


def _alive(pid):
    if not pid:
        return False
    _reap()                      # clear zombies first, else os.kill 'succeeds' on them
    try:
        pid = int(pid)
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    # A child we spawned but never wait()ed becomes a zombie after it exits; os.kill
    # still returns ok on it. Treat zombie/dead (Linux /proc state) as not alive.
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().rpartition(")")[2].split()[0]
        if state in ("Z", "X", "x"):
            return False
    except (OSError, IndexError):
        pass
    return True


def _read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _tail_metrics(run_dir: Path, since: int):
    """Return metrics records with index >= since. A half-written trailing line
    (writer mid-append) just fails json.loads and is skipped until next poll."""
    path = run_dir / "metrics.jsonl"
    out = []
    if not path.exists():
        return out, since
    with open(path) as f:
        lines = f.readlines()
    n = len(lines)
    for line in lines[since:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break   # partial trailing line; stop here, pick it up next poll
    return out, n


def _checkpoints(run_dir: Path):
    ck = run_dir / "checkpoints"
    if not ck.exists():
        return []
    gens = []
    for d in ck.iterdir():
        name = d.name
        if d.is_dir() and name.startswith("checkpoint_"):
            try:
                gens.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return sorted(gens)


def _run_summary(run_dir: Path):
    run = _read_json(run_dir / "run.json", {}) or {}
    last = _read_json(run_dir / "last.json", {}) or {}
    alive = _alive(run.get("pid"))
    state = run.get("state", "unknown")
    if state == "running" and not alive:
        state = "dead"   # process vanished without writing finished/error
    return {
        "run_id": run_dir.name,
        "state": state,
        "alive": alive,
        "pid": run.get("pid"),
        "started_at": run.get("started_at"),
        "ended_at": run.get("ended_at"),
        "error": run.get("error"),
        "config": run.get("config", _read_json(run_dir / "config.json", {})),
        "last": last,
        "checkpoints": _checkpoints(run_dir),
        "eval_elo": _best_eval_elo(run_dir),
    }


def _best_eval_elo(run_dir: Path):
    rows = _parse_eval_csv(run_dir / "eval.csv")
    elos = [r["elo"] for r in rows if isinstance(r.get("elo"), (int, float))]
    return max(elos) if elos else None


def _spawn(run_dir: Path, cfg: dict, resume: bool = False):
    """Start the trainer subprocess on `run_dir` and (re)write run.json. The runner
    auto-resumes from run_dir's bundle if one is present, so resume is just a respawn."""
    env = dict(os.environ)
    env["JAX_AZ_DEVICE"] = str(cfg.get("device", "auto"))
    env["PYTHONUNBUFFERED"] = "1"
    log = open(run_dir / "stdout.log", "a" if resume else "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "jax_az.monitor.runner", str(run_dir)],
        cwd=str(LAUNCH_DIR), env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    run = _read_json(run_dir / "run.json", {}) or {}
    run.update({"state": "starting", "pid": proc.pid, "started_at": time.time(),
                "config": cfg})
    run.pop("ended_at", None); run.pop("error", None)   # stale from the previous life
    with open(run_dir / "run.json", "w") as f:
        json.dump(run, f, indent=2)
    return proc.pid


def _launch(cfg: dict):
    RUNS.mkdir(exist_ok=True)
    # Optional user name -> folder; slugified, timestamp-suffixed on collision.
    name = str(cfg.pop("name", "") or "").strip()
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    if slug:
        run_id = slug if not (RUNS / slug).exists() else \
            f"{slug}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    else:
        run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    _spawn(run_dir, cfg)
    return run_id


def _resume(run_id: str):
    """Relaunch a stopped/finished/dead run from its saved bundle (same dir, same
    config). No-op (returns id) if it's still alive; None if the run dir is gone."""
    run_dir = RUNS / run_id
    if not run_dir.exists():
        return None
    run = _read_json(run_dir / "run.json", {}) or {}
    if _alive(run.get("pid")):
        return run_id
    cfg = _read_json(run_dir / "config.json", {}) or {}
    _spawn(run_dir, cfg, resume=True)
    return run_id


def _parse_eval_csv(path: Path):
    """Parse the eval results CSV into a list of typed dicts. A half-written
    trailing line (wrong column count) is skipped — picked up on the next poll."""
    if not path.exists():
        return []
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return []
    if len(lines) < 2:
        return []
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        vals = line.split(",")
        if len(vals) != len(header):
            continue
        rec = {}
        for k, v in zip(header, vals):
            try:
                rec[k] = int(v)
            except ValueError:
                try:
                    rec[k] = float(v)
                except ValueError:
                    rec[k] = v
        rows.append(rec)
    return rows


def _eval_summary(run_dir: Path):
    ev = _read_json(run_dir / "eval.json", {}) or {}
    state = ev.get("state", "none")
    alive = _alive(ev.get("pid"))
    if state == "running" and not alive:
        state = "dead"
    return {
        "state": state, "alive": alive,
        "done": ev.get("done", 0), "total": ev.get("total", 0),
        "last_gen": ev.get("last_gen"), "target": ev.get("target"),
        "started_at": ev.get("started_at"), "ended_at": ev.get("ended_at"),
        "error": ev.get("error"),
        "rows": _parse_eval_csv(run_dir / "eval.csv"),
    }


def _spawn_eval(run_dir: Path, req: dict):
    """Start the eval subprocess on `run_dir` (writes eval.csv + eval.json)."""
    run = _read_json(run_dir / "run.json", {}) or {}
    env = dict(os.environ)
    env["JAX_AZ_DEVICE"] = str((run.get("config") or {}).get("device", "auto"))
    env["PYTHONUNBUFFERED"] = "1"
    with open(run_dir / "eval_request.json", "w") as f:
        json.dump(req, f, indent=2)
    log = open(run_dir / "eval_stdout.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "jax_az.monitor.eval_runner", str(run_dir)],
        cwd=str(LAUNCH_DIR), env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    return proc.pid


def _spawn_analyze(run_dir: Path):
    """Run analyze_replay on this run's replay.npz -> replay_analysis.html in its dir.
    Pure-numpy + solver, no training GPU use, so force CPU to stay off the trainer."""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONUNBUFFERED"] = "1"
    log = open(run_dir / "analyze_stdout.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "jax_az.analyze_replay", str(run_dir)],
        cwd=str(LAUNCH_DIR), env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    with open(run_dir / "analyze.json", "w") as f:
        json.dump({"pid": proc.pid, "started_at": time.time()}, f)
    return proc.pid


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass   # quiet; this is a dev tool

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            html = (STATIC / "index.html").read_text()
            return self._send(200, html, "text/html; charset=utf-8")
        if u.path == "/api/defaults":
            return self._send(200, DEFAULTS)
        if u.path == "/api/runs":
            runs = []
            if RUNS.exists():
                for d in sorted(RUNS.iterdir(), reverse=True):
                    if d.is_dir():
                        runs.append(_run_summary(d))
            return self._send(200, runs)
        if u.path == "/api/run":
            rid = (q.get("id") or [""])[0]
            d = RUNS / rid
            if not d.exists():
                return self._send(404, {"error": "no such run"})
            return self._send(200, _run_summary(d))
        if u.path == "/api/metrics":
            rid = (q.get("id") or [""])[0]
            since = int((q.get("since") or ["0"])[0])
            d = RUNS / rid
            if not d.exists():
                return self._send(404, {"error": "no such run"})
            recs, n = _tail_metrics(d, since)
            return self._send(200, {"records": recs, "next": n})
        if u.path == "/api/log":
            rid = (q.get("id") or [""])[0]
            p = RUNS / rid / "stdout.log"
            text = p.read_text()[-8000:] if p.exists() else ""
            return self._send(200, text, "text/plain; charset=utf-8")
        if u.path == "/api/eval":
            rid = (q.get("id") or [""])[0]
            d = RUNS / rid
            if not d.exists():
                return self._send(404, {"error": "no such run"})
            return self._send(200, _eval_summary(d))
        if u.path == "/api/eval_log":
            rid = (q.get("id") or [""])[0]
            p = RUNS / rid / "eval_stdout.log"
            text = p.read_text()[-8000:] if p.exists() else ""
            return self._send(200, text, "text/plain; charset=utf-8")
        if u.path == "/api/analyze":                       # status of the replay analysis
            rid = (q.get("id") or [""])[0]
            d = RUNS / rid
            if not d.exists():
                return self._send(404, {"error": "no such run"})
            st = _read_json(d / "analyze.json", {}) or {}
            report = d / "replay_analysis.html"
            return self._send(200, {"running": _alive(st.get("pid")),
                                    "has_report": report.exists(),
                                    "mtime": report.stat().st_mtime if report.exists() else None})
        if u.path == "/api/replay":                        # serve the generated report HTML
            rid = (q.get("id") or [""])[0]
            p = RUNS / rid / "replay_analysis.html"
            if not p.exists():
                return self._send(404, "no replay analysis yet", "text/plain; charset=utf-8")
            return self._send(200, p.read_text(), "text/html; charset=utf-8")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if u.path == "/api/launch":
            run_id = _launch(body)
            return self._send(200, {"run_id": run_id})
        if u.path == "/api/stop":
            rid = body.get("id", "")
            run = _read_json(RUNS / rid / "run.json", {}) or {}
            pid = run.get("pid")
            if _alive(pid):
                os.kill(int(pid), signal.SIGTERM)
            return self._send(200, {"stopped": rid})
        if u.path == "/api/resume":
            out = _resume(body.get("id", ""))
            if not out:
                return self._send(404, {"error": "no such run"})
            return self._send(200, {"resumed": out})
        if u.path == "/api/eval":
            rid = body.get("id", "")
            d = RUNS / rid
            if not d.exists():
                return self._send(404, {"error": "no such run"})
            ev = _read_json(d / "eval.json", {}) or {}
            if _alive(ev.get("pid")):
                return self._send(409, {"error": "eval already running"})
            gens = _checkpoints(d)
            if not gens:
                return self._send(400, {"error": "no checkpoints to evaluate"})
            target = str(body.get("target", "all"))
            ckdir = d / "checkpoints"
            if target == "all":
                checkpoint, total, label = str(ckdir), len(gens), "whole run"
            else:
                g = int(target)
                checkpoint, total, label = str(ckdir / f"checkpoint_{g}"), 1, f"checkpoint_{g}"
            req = {
                "checkpoint": checkpoint, "target_count": total, "label": label,
                "sims": int(body.get("sims", DEFAULTS["eval"]["sims"])),
                "per_group": int(body.get("per_group", DEFAULTS["eval"]["per_group"])),
                "seed": int(body.get("seed", DEFAULTS["eval"]["seed"])),
                "pgn": bool(body.get("pgn", DEFAULTS["eval"]["pgn"])),
            }
            pid = _spawn_eval(d, req)
            return self._send(200, {"started": rid, "pid": pid, "label": label})
        if u.path == "/api/analyze":
            rid = body.get("id", "")
            d = RUNS / rid
            if not d.exists():
                return self._send(404, {"error": "no such run"})
            st = _read_json(d / "analyze.json", {}) or {}
            if _alive(st.get("pid")):
                return self._send(409, {"error": "analysis already running"})
            if not (d / "replay.npz").exists():
                return self._send(400, {"error": "no replay.npz for this run"})
            return self._send(200, {"started": rid, "pid": _spawn_analyze(d)})
        if u.path == "/api/eval_stop":
            rid = body.get("id", "")
            ev = _read_json(RUNS / rid / "eval.json", {}) or {}
            pid = ev.get("pid")
            if _alive(pid):
                os.kill(int(pid), signal.SIGTERM)
            return self._send(200, {"stopped": rid})
        return self._send(404, {"error": "not found"})


def main():
    port = int(os.environ.get("PORT", "8000"))
    RUNS.mkdir(exist_ok=True)
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[monitor] serving http://localhost:{port}  (runs in {RUNS})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[monitor] bye")


if __name__ == "__main__":
    main()
