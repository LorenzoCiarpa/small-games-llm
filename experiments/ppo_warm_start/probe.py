"""
Il gradiente di PPO a pesi fissi: quanto spinge l'addestramento in un certo punto dello spazio dei pesi?

Per ogni rete (scritta a mano, iniziale casuale di PPO, PPO a metà e a fine addestramento) gira PPO con la
configurazione della migliore corsa da zero del gioco, ma con l'attore fermo per tutta la corsa
(critic_warmup = total_steps): il critico impara il valore di quella policy, e ogni 5 aggiornamenti il log
registra il gradiente dell'attore (norma per minibatch, norma sul batch intero, parte policy-gradient e parte
entropia), senza mai applicarlo. Il gradiente con il critico già adattato è la media dell'ultimo quarto del log
(analyze.py); la prima riga è quello che vede PPO al primo passo, con il critico ancora casuale.

    uv run --group ppo python experiments/ppo_warm_start/probe.py                  # tutte le reti che esistono
    uv run --group ppo python experiments/ppo_warm_start/probe.py --only low_tide/gpt breakout/astra
    uv run --group ppo python experiments/ppo_warm_start/probe.py --steps 500000   # più veloce, critico meno adattato

Scrive results/probe/<gioco>/<rete>/train_log.csv; salta le reti già misurate (`--force` per rifarle). Le reti
finali delle corse che partono dai pesi scritti a mano si misurano quando quelle corse sono finite.
"""

import argparse
import importlib.util
import shutil
from dataclasses import replace
from pathlib import Path

from small_games.ppo import train

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results" / "probe"
LOW_TIDE_ARCH = {"gpt": dict(hidden=(43, 61, 53), activation="relu"),
                 "claude": dict(hidden=(46, 36), activations=("relu", "tanh")),
                 "64x64": dict(hidden=(64, 64), activation="tanh")}


def base_config(game):
    """The CONFIG of experiments/<game>/ppo/train.py (for Low Tide, the 100M-step run uses it unchanged)."""
    spec = importlib.util.spec_from_file_location(f"train_{game}", ROOT / "experiments" / game / "ppo" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG


def probes():
    """(game, label) -> (config changes, weight file or None for the random init of PPO)."""
    out = {}
    for g in ("breakout", "seaquest", "asterix"):
        res = f"experiments/{g}/ppo/results"
        out[g, "astra"] = ({}, f"weights/{g}/paper_astra.json")
        out[g, "random"] = ({}, None)
        out[g, "ppo_5M"] = ({}, f"{res}/ppo_ent05/checkpoint_0005M.json")
        out[g, "ppo_50M"] = ({}, f"{res}/ppo_ent05/checkpoint_0050M.json")
        out[g, "ppo_final"] = ({}, f"weights/{g}/ppo_ent05.json")
        out[g, "from_astra_final"] = ({}, f"weights/{g}/ppo_from_astra.json")
    res = "experiments/low_tide/ppo/results"
    for net in ("gpt", "claude"):
        arch = LOW_TIDE_ARCH[net]
        out["low_tide", net] = (arch, f"weights/low_tide/{net}.json")
        out["low_tide", f"random_arch_{net}"] = (arch, None)
        out["low_tide", f"from_{net}_final"] = (arch, f"weights/low_tide/ppo_from_{net}.json")
        out["low_tide", f"from_{net}_cw_final"] = (arch, f"weights/low_tide/ppo_from_{net}_cw.json")
        out["low_tide", f"arch_{net}_final"] = (arch, f"weights/low_tide/ppo_arch_{net}.json")
    arch = LOW_TIDE_ARCH["64x64"]
    out["low_tide", "random_64x64"] = (arch, None)
    out["low_tide", "ppo_100m_5M"] = (arch, f"{res}/ppo_100m/checkpoint_0005M.json")
    out["low_tide", "ppo_100m_50M"] = (arch, f"{res}/ppo_100m/checkpoint_0050M.json")
    out["low_tide", "ppo_100m_final"] = (arch, "weights/low_tide/ppo_100m.json")
    return out


def run(game, label, changes, init, steps, force):
    d = OUT / game / label
    if (d / "done").is_file() and not force:
        print(f"{game}/{label}: già misurata")
        return
    if init is not None and not (ROOT / init).is_file():
        print(f"{game}/{label}: manca {init}, salto")
        return
    cfg = replace(base_config(game), **changes, init=init, total_steps=steps, critic_warmup=steps,
                  log_every=5, save_every=10 ** 12)
    print(f"{game}/{label}: {init or 'iniziale casuale'}", flush=True)
    if d.exists():
        shutil.rmtree(d)
    train(cfg, d / "actor.json", d)
    for f in ("actor.json", "final.pt", "checkpoint_0000M.json"):     # the actor did not move: keep only the log
        (d / f).unlink(missing_ok=True)
    (d / "done").write_text(f"{init}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="gioco/rete (default: tutte)")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    todo = probes()
    keys = [tuple(k.split("/")) for k in args.only] if args.only else list(todo)
    for k in keys:
        run(*k, *todo[k], args.steps, args.force)
