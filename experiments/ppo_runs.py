"""
Lancia, controlla e ferma gli addestramenti PPO di tutti i giochi, in background: i processi sopravvivono alla
chiusura della sessione SSH, e ognuno scrive in experiments/<gioco>/ppo/results/<nome>/.

    uv sync --group ppo --group analysis                 # una volta, prima di tutto
    uv run python experiments/ppo_runs.py list            # le corse previste
    uv run python experiments/ppo_runs.py start           # lancia tutte quelle non finite (riprende le interrotte)
    uv run python experiments/ppo_runs.py start breakout/ppo_ent05 seaquest/ppo_ent05
    uv run python experiments/ppo_runs.py status          # a che punto è ognuna
    uv run python experiments/ppo_runs.py stop            # SIGTERM: salvano lo stato ed escono; `start` le riprende

Ogni corsa è `experiments/<gioco>/ppo/train.py --name <nome> --resume ...`, quindi:
- ogni 5M passi salva lo stato completo (state.pt) e i pesi (checkpoint_<N>M.json);
- `stop` (o un riavvio della macchina) perde al massimo il lavoro dall'ultimo salvataggio;
- `start` riparte da lì, con lo stesso risultato di una corsa mai interrotta;
- alla fine scrive weights/<gioco>/<nome>.json, results/<nome>/final.pt e la valutazione results/<nome>/eval.json.

Le corse girano con il Python di .venv direttamente (non con `uv run`), così il segnale di `stop` arriva una sola
volta al processo che addestra.
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
# nome -> (gioco, argomenti di train.py). La configurazione base è il CONFIG di experiments/<gioco>/ppo/train.py.
RUNS = {}
for _g in ("breakout", "seaquest", "asterix"):
    for _tag, _ent in (("ent01", "0.01"), ("ent05", "0.05"), ("ent10", "0.1")):
        RUNS[f"{_g}/ppo_{_tag}"] = (_g, ["--total-steps", str(MINATAR_STEPS), "--set", f"ent_coef={_ent}"])
RUNS["low_tide/ppo_100m"] = ("low_tide", ["--total-steps", "100000000"])


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
        sys.exit("manca torch o scikit-learn nel .venv: esegui prima `uv sync --group ppo --group analysis`")
    for run in runs:
        game, args = RUNS[run]
        d = run_dir(run)
        if alive(run):
            print(f"{run}: già in corsa (pid {alive(run)})")
            continue
        if (d / "eval.json").is_file():
            print(f"{run}: finita, salto")
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
        print(f"{run}: lanciata (pid {p.pid}), log in {(d / 'console.log').relative_to(ROOT)}")


def status(runs):
    for run in runs:
        d = run_dir(run)
        pid = alive(run)
        if (d / "eval.json").is_file():
            state = "finita"
        elif pid:
            state = f"in corsa (pid {pid})"
        elif (d / "state.pt").is_file():
            state = "ferma, riprendibile"
        else:
            state = "non lanciata" if not d.exists() else "ferma"
        print(f"{run:22s} {state:22s} {last_line(d / 'console.log')[:110]}")


def stop(runs):
    for run in runs:
        pid = alive(run)
        if pid:
            os.kill(pid, signal.SIGTERM)
            print(f"{run}: SIGTERM a {pid} (salva lo stato alla fine dell'aggiornamento in corso)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "start", "status", "stop"])
    ap.add_argument("runs", nargs="*", help="gioco/nome (default: tutte)")
    ap.add_argument("--eval-workers", type=int, default=1)
    a = ap.parse_args()
    bad = [r for r in a.runs if r not in RUNS]
    if bad:
        sys.exit(f"corse sconosciute: {bad}; disponibili: {', '.join(RUNS)}")
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
