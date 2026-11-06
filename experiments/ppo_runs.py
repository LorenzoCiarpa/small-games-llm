"""
Starts, checks and stops the PPO trainings of all the games, in the background: the processes survive the end of the
SSH session, and each one writes to experiments/<game>/ppo/results/<name>/.

    uv sync --group ppo --group analysis                 # once, before anything else
    uv run python experiments/ppo_runs.py list            # the planned runs
    uv run python experiments/ppo_runs.py start           # starts all the unfinished ones (resumes the interrupted ones)
    uv run python experiments/ppo_runs.py start breakout/ppo_ent05 seaquest/ppo_ent05
    uv run python experiments/ppo_runs.py status          # how far along each one is
    uv run python experiments/ppo_runs.py stop            # SIGTERM: they save their state and exit; `start` resumes them

Each run is `experiments/<game>/ppo/train.py --name <name> --resume ...`, so:
- every 5M steps it saves the full state (state.pt) and the weights (checkpoint_<N>M.json);
- `stop` (or a reboot of the machine) loses at most the work done since the last save;
- `start` resumes from there, with the same result as a run that was never interrupted;
- at the end it writes weights/<game>/<name>.json, results/<name>/final.pt and the evaluation results/<name>/eval.json.

The runs use the .venv Python directly (not `uv run`), so the `stop` signal reaches the training process only once.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"

MINATAR_STEPS = 200_000_000
# name -> (game, arguments of train.py). The base configuration is the CONFIG of experiments/<game>/ppo/train.py.
RUNS = {}
for _g in ("breakout", "seaquest", "asterix"):
    for _tag, _ent in (("ent01", "0.01"), ("ent05", "0.05"), ("ent10", "0.1")):
        RUNS[f"{_g}/ppo_{_tag}"] = (_g, ["--total-steps", str(MINATAR_STEPS), "--set", f"ent_coef={_ent}"])
RUNS["low_tide/ppo_100m"] = ("low_tide", ["--total-steps", "100000000"])

# PPO starting from the hand-written networks (experiments/ppo_warm_start/): the configuration of each game's best
# run from scratch, only the starting point of the actor changes. On Low Tide the hand-written networks have a
# different architecture, so there is also the run from scratch with the same architecture (ppo_arch_<net>).
for _g in ("breakout", "seaquest", "asterix"):
    RUNS[f"{_g}/ppo_from_astra"] = (_g, ["--total-steps", str(MINATAR_STEPS), "--set", "ent_coef=0.05",
                                         "--set", f"init=weights/{_g}/paper_astra.json"])
LOW_TIDE_ARCH = {"gpt": ["--set", "hidden=43,61,53", "--set", "activation=relu"],
                 "claude": ["--set", "hidden=46,36", "--set", "activations=relu,tanh"]}
for _net, _arch in LOW_TIDE_ARCH.items():
    RUNS[f"low_tide/ppo_from_{_net}"] = ("low_tide", ["--total-steps", "100000000", *_arch,
                                                      "--set", f"init=weights/low_tide/{_net}.json"])
    RUNS[f"low_tide/ppo_arch_{_net}"] = ("low_tide", ["--total-steps", "100000000", *_arch])
    # the same start, but the first 2M steps train only the critic (the actor stays the hand-written network)
    RUNS[f"low_tide/ppo_from_{_net}_cw"] = ("low_tide", [*RUNS[f"low_tide/ppo_from_{_net}"][1],
                                                         "--set", "critic_warmup=2000000"])


def run_dir(run):
    game, name = run.split("/")
    return ROOT / "experiments" / game / "ppo" / "results" / name


def alive(run):
    pid_file = run_dir(run) / "pid"
    if not pid_file.is_file():
        return None
    pid = int(pid_file.read_text())
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except FileNotFoundError:
        return None
    return pid if "train.py" in cmd and run.split("/")[1] in cmd else None


def last_line(path):
    if not path.is_file():
        return ""
    lines = [l for l in path.read_text(errors="replace").splitlines() if l.strip()]
    return lines[-1] if lines else ""


def start(runs, eval_workers):
    if subprocess.run([str(PY), "-c", "import torch, sklearn"], capture_output=True).returncode != 0:
        sys.exit("torch or scikit-learn is missing from .venv: run `uv sync --group ppo --group analysis` first")
    for run in runs:
        game, args = RUNS[run]
        d = run_dir(run)
        if alive(run):
            print(f"{run}: already running (pid {alive(run)})")
            continue
        if (d / "eval.json").is_file():
            print(f"{run}: finished, skipping")
            continue
        d.mkdir(parents=True, exist_ok=True)
        cmd = [str(PY), str(ROOT / "experiments" / game / "ppo" / "train.py"), "--name", run.split("/")[1],
               "--resume", "--eval-workers", str(eval_workers), *args]
        log = open(d / "console.log", "a")
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        log.flush()
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, env=env)
        (d / "pid").write_text(str(p.pid))
        print(f"{run}: started (pid {p.pid}), log in {(d / 'console.log').relative_to(ROOT)}")


def status(runs):
    for run in runs:
        d = run_dir(run)
        pid = alive(run)
        if (d / "eval.json").is_file():
            state = "finished"
        elif pid:
            state = f"running (pid {pid})"
        elif (d / "state.pt").is_file():
            state = "stopped, resumable"
        else:
            state = "not started" if not d.exists() else "stopped"
        print(f"{run:22s} {state:22s} {last_line(d / 'console.log')[:110]}")


def stop(runs):
    for run in runs:
        pid = alive(run)
        if pid:
            os.kill(pid, signal.SIGTERM)
            print(f"{run}: SIGTERM to {pid} (it saves its state at the end of the current update)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "start", "status", "stop"])
    ap.add_argument("runs", nargs="*", help="game/name (default: all)")
    ap.add_argument("--eval-workers", type=int, default=1)
    a = ap.parse_args()
    bad = [r for r in a.runs if r not in RUNS]
    if bad:
        sys.exit(f"unknown runs: {bad}; available: {', '.join(RUNS)}")
    runs = a.runs or list(RUNS)
    if a.command == "list":
        for r in runs:
            print(r, " ".join(RUNS[r][1]))
    elif a.command == "start":
        start(runs, a.eval_workers)
    elif a.command == "status":
        status(runs)
    else:
        stop(runs)
