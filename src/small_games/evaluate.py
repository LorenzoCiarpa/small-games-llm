"""
Score weight files and baselines on any registered game.

    uv run evaluate --list
    uv run evaluate low_tide weights/low_tide/claude.json
    uv run evaluate low_tide claude gpt             # names of files in weights/low_tide/
    uv run evaluate breakout paper_astra --baseline random
    uv run evaluate low_tide --baseline heuristic --baseline astra_code

The scoring protocol (episodes, seeds, sticky actions, step cap) is the game's
own (see `Game.protocol`); the options below override it. Episode k uses seed
seed0 + k, for the game and (with `policy_rng`) for the policy's sampling: the
same command always gives the same numbers, whatever the number of workers,
and `play <game> --seed <seed0 + k>` replays episode k.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np

from .core import Game, get_game, list_games
from .policy import MLPPolicy, available_weights, baselines, games_overview, load_mlp, policy_rng


def run_episodes(game_name, policy, seeds, sticky_prob, max_steps):
    """Play one episode per seed; returns one dict per episode."""
    game = get_game(game_name)
    env = game.make_env(sticky_prob=sticky_prob, max_steps=max_steps)
    out = []
    for seed in seeds:
        rng = policy_rng(seed)
        obs, info = env.reset(seed=seed)
        total, terminated, truncated = 0.0, False, False
        while not (terminated or truncated):
            obs, r, terminated, truncated, info = env.step(policy(obs, rng))
            total += r
        ep = {"return": total, "length": env.t, "truncated": bool(truncated and not terminated)}
        for k in game.episode_stats:
            ep[k] = float(info[k])
        out.append(ep)
    return out


def _single_thread_blas():
    # the matrices are tiny: one BLAS thread per worker avoids oversubscription
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def evaluate(game: Game, policy, episodes=None, seed0=None, sticky_prob=None, max_steps=None,
             workers=1, progress=False):
    proto = game.protocol
    episodes = proto.episodes if episodes is None else episodes
    seed0 = proto.seed0 if seed0 is None else seed0
    sticky_prob = proto.sticky_prob if sticky_prob is None else sticky_prob
    max_steps = proto.max_steps if max_steps is None else max_steps
    seeds = list(range(seed0, seed0 + episodes))

    t0 = time.time()
    if workers <= 1:
        results = run_episodes(game.name, policy, seeds, sticky_prob, max_steps)
    else:
        _single_thread_blas()
        size = max(1, math.ceil(episodes / (workers * 8)))
        chunks = [seeds[i:i + size] for i in range(0, episodes, size)]
        by_start, done = {}, 0
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
            futs = {ex.submit(run_episodes, game.name, policy, c, sticky_prob, max_steps): c[0]
                    for c in chunks}
            for fut in as_completed(futs):
                by_start[futs[fut]] = fut.result()
                done += len(by_start[futs[fut]])
                if progress:
                    print(f"\r  {done}/{episodes} episodes", end="", file=sys.stderr, flush=True)
        if progress:
            print("\r" + " " * 40 + "\r", end="", file=sys.stderr, flush=True)
        results = [ep for start in sorted(by_start) for ep in by_start[start]]

    R = np.array([ep["return"] for ep in results])
    summary = {
        "mean_return": float(R.mean()),
        "stderr": float(R.std(ddof=1) / math.sqrt(len(R))) if len(R) > 1 else 0.0,
        "min_return": float(R.min()),
        "max_return": float(R.max()),
        "mean_length": float(np.mean([ep["length"] for ep in results])),
        "truncated_rate": float(np.mean([ep["truncated"] for ep in results])),
    }
    for k in game.episode_stats:
        summary[f"mean_{k}"] = float(np.mean([ep[k] for ep in results]))
    summary["protocol"] = {"episodes": episodes, "seed0": seed0, "sticky_prob": sticky_prob,
                           "max_steps": max_steps}
    summary["seconds"] = round(time.time() - t0, 1)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Score weight files and baselines on a game.",
        epilog="games:\n" + games_overview() + "\n\n--list describes every game and baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", nargs="?", help="the game (see below)")
    ap.add_argument("weights", nargs="*", help="weight files, or names of files in weights/<game>/")
    ap.add_argument("--list", action="store_true", help="describe the games and their baselines, then exit")
    ap.add_argument("--baseline", action="append", default=[],
                    help="a baseline of the game (see below; repeatable)")
    ap.add_argument("--episodes", type=int, help="override the game's number of episodes")
    ap.add_argument("--seed0", type=int, help="override the first seed")
    ap.add_argument("--sticky", type=float, help="override the sticky-action probability")
    ap.add_argument("--max-steps", type=int, help="override the step cap")
    ap.add_argument("--greedy", action="store_true", help="argmax instead of sampling (extra info)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="parallel processes")
    args = ap.parse_args(argv)

    if args.list:
        print(games_overview(verbose=True))
        return 0
    if not args.game:
        ap.error(f"choose a game: {', '.join(list_games())}")
    try:
        game = get_game(args.game)
    except KeyError as e:
        ap.error(e.args[0])

    known = baselines(game)
    targets = []
    for name in args.baseline:
        if name not in known:
            ap.error(f"{game.name}: unknown baseline {name!r}; available: {', '.join(known)}")
        targets.append((f"baseline {name}", known[name]))
    for arg in args.weights:
        try:
            path, mlp = load_mlp(game, arg)
        except (FileNotFoundError, ValueError) as e:
            ap.error(f"{arg}: {e}")
        print(f"{path}: {mlp.describe(game.obs_dim)}")
        targets.append((str(path), MLPPolicy(mlp, args.greedy)))
    if not targets:
        names = ", ".join(available_weights(game)) or "none"
        ap.error(f"give weight files or --baseline (weights/{game.name}/: {names}; "
                 f"baselines: {', '.join(known)})")

    mode = "greedy" if args.greedy else "sampled"
    rows = []
    for label, policy in targets:
        res = evaluate(game, policy, args.episodes, args.seed0, args.sticky, args.max_steps,
                       workers=args.workers, progress=sys.stderr.isatty())
        print(f"{game.name} | {label} ({mode}): {json.dumps(res, indent=2)}", flush=True)
        rows.append((label, res))

    if len(rows) > 1:
        print(f"\n{game.name}, {rows[0][1]['protocol']['episodes']} episodes ({mode})")
        width = max(len(label) for label, _ in rows)
        for label, res in sorted(rows, key=lambda x: -x[1]["mean_return"]):
            print(f"  {label:{width}s}  {res['mean_return']:9.2f} ± {res['stderr']:.2f}"
                  f"   max {res['max_return']:g}   length {res['mean_length']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
