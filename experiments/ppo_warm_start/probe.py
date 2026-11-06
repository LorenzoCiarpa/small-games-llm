"""
PPO's gradient at fixed weights: how hard does training push at a given point of weight space?

For each network (hand-written, PPO's random initialization, PPO halfway through and at the end of training) runs PPO
with the configuration of the game's best run from scratch, but with the actor frozen for the whole run
(critic_warmup = total_steps): the critic learns the value of that policy, and every 5 updates the log records the
actor's gradient (norm per minibatch, norm on the full batch, policy-gradient part and entropy part), without ever
applying it. The gradient with the critic already adapted is the mean of the last quarter of the log (analyze.py);
the first row is what PPO sees at its first step, with the critic still random.

    uv run --group ppo python experiments/ppo_warm_start/probe.py                  # all the networks that exist
    uv run --group ppo python experiments/ppo_warm_start/probe.py --only low_tide/gpt breakout/astra
    uv run --group ppo python experiments/ppo_warm_start/probe.py --steps 500000   # faster, critic less adapted

Writes results/probe/<game>/<network>/train_log.csv; skips the networks already measured (`--force` to redo them).
The final networks of the runs that start from the hand-written weights are measured once those runs have finished.
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
        print(f"{game}/{label}: already measured")
        return
    if init is not None and not (ROOT / init).is_file():
        print(f"{game}/{label}: {init} is missing, skipping")
        return
    cfg = replace(base_config(game), **changes, init=init, total_steps=steps, critic_warmup=steps,
                  log_every=5, save_every=10 ** 12)
    print(f"{game}/{label}: {init or 'random initialization'}", flush=True)
    if d.exists():
        shutil.rmtree(d)
    train(cfg, d / "actor.json", d)
    for f in ("actor.json", "final.pt", "checkpoint_0000M.json"):     # the actor did not move: keep only the log
        (d / f).unlink(missing_ok=True)
    (d / "done").write_text(f"{init}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="game/network (default: all)")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    todo = probes()
    keys = [tuple(k.split("/")) for k in args.only] if args.only else list(todo)
    for k in keys:
        run(*k, *todo[k], args.steps, args.force)
