"""
Verifies bfs_exact.json (see README.md): the network must make the same move as `heuristic_policy` in every state.

The network exceeds the game's limits (3 layers of 64), so `uv run evaluate` does not load it. Here it is read with
the repo's own `load_weights`, with only the limits on layers and units removed, and evaluated with the reference
forward pass (`ReferenceMLP`, dense float64).

1. Rewrites the weights with build.py and checks that they match the file.
2. Plays the protocol episodes (seeds 10000.., sticky 0.1, actions sampled from the softmax, like `evaluate`).
   In every state it compares the network's move with the heuristic's and measures the logit margin.
   It plays the same episodes with the heuristic and compares the scores episode by episode.
3. Purpose-built states, some far from those of the game, to check distances that do not occur in the episodes:
   half random (up to 45% jellyfish, 1-3 shells), a quarter jellyfish mazes and a quarter serpentines (corridors up
   to ~60 steps long), with agent and shell in random cells.

    uv run python experiments/low_tide/bfs_exact/check.py [--episodes 500] [--random 20000]
"""

import argparse
import collections
import dataclasses
import json
import multiprocessing as mp
import time

import numpy as np

from build import K_ALL, OUT, build
from small_games.core import get_game
from small_games.games.low_tide.baselines import _decode, heuristic_policy
from small_games.games.low_tide.env import (CH_JELLY, CH_SHELL, H, MAX_INTERVAL, MOVES, OBS_DIM, IDX_COUNTDOWN,
                                            IDX_RISING, LowTideEnv, W)
from small_games.policy import policy_rng, sample_action
from small_games.weights import ReferenceMLP, load_weights

GAME = get_game("low_tide")
MLP = None


def load(path):
    """The repo's load_weights, with the layer and unit limits removed (activations and shapes are still checked)."""
    net = dataclasses.replace(GAME.net, max_hidden_layers=10**6, max_hidden_units=10**6)
    return ReferenceMLP(load_weights(path, dataclasses.replace(GAME, net=net)))


def bfs_depth(obs):
    """Distance from the agent of the shell found by heuristic_policy (None if there is none or if it moves up)."""
    g, ar, ac, level, rising, countdown, stun = _decode(obs)
    deepest = H - level - 1 - ((1 if countdown > 2 else 2) if rising else 0)
    if ar > deepest:
        return None
    blocked = g[CH_JELLY] > 0.5
    if rising:
        near = np.zeros_like(blocked)
        near[1:] |= blocked[:-1]; near[:-1] |= blocked[1:]
        near[:, 1:] |= blocked[:, :-1]; near[:, :-1] |= blocked[:, 1:]
        blocked = blocked | near
    dist = {(ar, ac): 0}
    q = collections.deque([(ar, ac)])
    while q:
        cur = q.popleft()
        if g[CH_SHELL][cur] > 0.5 and cur != (ar, ac):
            return dist[cur]
        for dr, dc in MOVES[1:]:
            n = (cur[0] + dr, cur[1] + dc)
            if 0 <= n[0] <= deepest and 0 <= n[1] < W and n not in dist and not blocked[n]:
                dist[n] = dist[cur] + 1
                q.append(n)
    return None


def margin(logits):
    s = np.sort(logits)
    return s[-1] - s[-2]


def episode(seed):
    """One protocol episode with the network: (score, states, differing moves, sampled ≠ argmax, min margin, max d)."""
    env = LowTideEnv(sticky_prob=GAME.protocol.sticky_prob, max_steps=GAME.protocol.max_steps)
    obs, _ = env.reset(seed=seed)
    rng = policy_rng(seed)
    total, n, diff, resampled, min_margin, dmax, done = 0.0, 0, 0, 0, np.inf, 0, False
    while not done:
        logits = MLP.logits(obs)
        a = sample_action(logits, rng)
        n += 1
        diff += int(np.argmax(logits)) != heuristic_policy(obs, None)
        resampled += a != int(np.argmax(logits))
        min_margin = min(min_margin, margin(logits))
        dmax = max(dmax, bfs_depth(obs) or 0)
        obs, rew, term, trunc, _ = env.step(a)
        total += rew
        done = term or trunc
    return total, n, diff, resampled, min_margin, dmax


def heuristic_episode(seed):
    env = LowTideEnv(sticky_prob=GAME.protocol.sticky_prob, max_steps=GAME.protocol.max_steps)
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, rew, term, trunc, _ = env.step(heuristic_policy(obs, None))
        total += rew
        done = term or trunc
    return total


def random_state(rng):
    """Any valid state: level 0..7, tide rising or falling, up to 45% jellyfish, 1-3 shells."""
    level = int(rng.integers(0, 8))
    rising = bool(rng.random() < 0.5) or level == 0
    countdown = int(rng.integers(1, MAX_INTERVAL + 1))
    dry = H - level
    grid = np.zeros((4, H, W), dtype=np.float32)
    grid[1, dry:, :] = 1
    p_jelly = rng.uniform(0, 0.45)
    jelly = rng.random((dry, W)) < p_jelly
    shell = np.zeros((dry, W), dtype=bool)
    free = np.argwhere(~jelly)
    for i in rng.choice(len(free), size=min(len(free), int(rng.integers(1, 4))), replace=False):
        shell[tuple(free[i])] = True
    empty = np.argwhere(~jelly & ~shell)
    ar, ac = empty[rng.integers(len(empty))]
    grid[0, ar, ac] = 1
    grid[2, :dry] = shell
    grid[3, :dry] = jelly
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[:400] = grid.ravel()
    obs[IDX_RISING] = 1.0 if rising else 0.0
    obs[IDX_COUNTDOWN] = countdown / MAX_INTERVAL
    return obs


def maze_state(rng, kind):
    """Calm tide (level 0 or 1, falling): jellyfish walls forming a maze or a serpentine, one shell."""
    level = int(rng.integers(0, 2))
    dry = H - level
    jelly = np.ones((H, W), dtype=bool)
    if kind == "serpentine":
        jelly[0::2, :] = False
        for i, r in enumerate(range(1, dry, 2)):
            jelly[r, W - 1 if i % 2 == 0 else 0] = False
    else:  # perfect maze (DFS) on a 5x5 grid of cells in the even rows and columns
        n = 5
        seen, stack = {(0, 0)}, [(0, 0)]
        jelly[0, 0] = False
        while stack:
            r, c = stack[-1]
            nxt = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                   if 0 <= r + dr < n and 0 <= c + dc < n and (r + dr, c + dc) not in seen]
            if not nxt:
                stack.pop()
                continue
            nr, nc = nxt[rng.integers(len(nxt))]
            jelly[2 * nr, 2 * nc] = jelly[r + nr, c + nc] = False
            seen.add((nr, nc))
            stack.append((nr, nc))
    jelly[dry:] = False
    free = np.argwhere(~jelly[:dry])
    (ar, ac), (sr, sc) = free[rng.choice(len(free), size=2, replace=False)]
    grid = np.zeros((4, H, W), dtype=np.float32)
    grid[0, ar, ac] = 1
    grid[1, dry:, :] = 1
    grid[2, sr, sc] = 1
    grid[3] = jelly
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[:400] = grid.ravel()
    obs[IDX_COUNTDOWN] = int(rng.integers(1, MAX_INTERVAL + 1)) / MAX_INTERVAL
    return obs


def random_batch(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    diff, min_margin, depths = 0, np.inf, []
    for i in range(n):
        obs = random_state(rng) if i % 4 < 2 else maze_state(rng, "serpentine" if i % 4 == 2 else "maze")
        logits = MLP.logits(obs)
        diff += int(np.argmax(logits)) != heuristic_policy(obs, None)
        min_margin = min(min_margin, margin(logits))
        depths.append(bfs_depth(obs) or 0)
    return diff, min_margin, depths


def main():
    global MLP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--random", type=int, default=20000)
    args = ap.parse_args()
    t0 = time.time()

    data = json.load(open(OUT))
    print(f"weights rewritten from the formulas equal to the file: {data['layers'] == [L.to_json() for L in build(K_ALL)]}")
    MLP = load(OUT)
    sizes = [Wm.shape[0] for Wm, _, _ in MLP.layers[:-1]]
    print(f"network: {len(sizes)} hidden layers, {sum(sizes)} neurons, {MLP.n_nonzero()} non-zero weights "
          f"out of {MLP.n_params()}")

    seeds = [GAME.protocol.seed0 + i for i in range(args.episodes)]
    with mp.get_context("fork").Pool(8) as pool:
        res = pool.map(episode, seeds, chunksize=1)
        heur = np.array(pool.map(heuristic_episode, seeds, chunksize=4))
        chunks = [(i, args.random // 40) for i in range(40)]
        rnd = pool.map(random_batch, chunks, chunksize=1)

    net = np.array([r[0] for r in res])
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    print(f"\nprotocol episodes: {args.episodes}, states {sum(r[1] for r in res)}")
    print(f"  moves different from the heuristic: {sum(r[2] for r in res)}")
    print(f"  sampled action ≠ argmax: {sum(r[3] for r in res)}; smallest logit margin: {min(r[4] for r in res):.0f}")
    print(f"  network score {net.mean():.2f} ± {se(net):.2f}, heuristic {heur.mean():.2f} ± {se(heur):.2f}, "
          f"episodes with a different score: {int(np.sum(net != heur))}")
    dmax = max(r[5] for r in res)
    print(f"  largest distance of the shell found by the BFS: {dmax} "
          f"(for these states K = {dmax - 1} steps, {dmax - 1 + 6} layers, would be enough)")

    depths = np.concatenate([r[2] for r in rnd])
    print(f"\nconstructed states (random, mazes, serpentines): {len(depths)}")
    print(f"  moves different from the heuristic: {sum(r[0] for r in rnd)}; smallest margin: {min(r[1] for r in rnd):.0f}")
    print(f"  shell distance: median {np.median(depths[depths > 0]):.0f}, largest {depths.max()}, "
          f"states with distance > {dmax}: {int(np.sum(depths > dmax))}")
    print(f"\n({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
