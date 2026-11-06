"""
Verifies weights/low_tide/bfs_net.json (see README.md) and compares it with the BFS heuristic.

1. Rewrites the weights from the formulas of build.py and checks that they match the file.
2. Plays the protocol episodes with the network (seeds 10000.., sticky 0.1, actions sampled from the softmax, like
   `uv run evaluate low_tide bfs_net`). In every state it compares the network with the specification of build.py:
   chosen move and potentials per direction.
3. In the same states it asks the heuristic (`heuristic_policy`) what it would do. When the two moves differ, it
   checks whether the network's move is still a step of a shortest path toward a shell, with the rules of the BFS
   (jellyfish, and cells next to jellyfish when the tide is rising, blocked; rows below the last safe one excluded).

    uv run python experiments/low_tide/bfs_net/check.py [--episodes 500]
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from build import JELLY, OUT, build, decode, policy_logits, potentials
from small_games.core import get_game
from small_games.games.low_tide.baselines import heuristic_policy
from small_games.games.low_tide.env import MOVES, LowTideEnv
from small_games.policy import policy_rng, sample_action
from small_games.weights import load_weights

ACTIONS = ("noop", "up", "down", "left", "right")


def forward(layers, obs):
    x, out = np.asarray(obs, dtype=np.float64), []
    for W, b, act in layers:
        x = W @ x + b
        x = np.maximum(x, 0.0) if act == "relu" else np.tanh(x) if act == "tanh" else x
        out.append(x)
    return out


def bfs_distances(obs):
    """Distance from every cell to the nearest shell, with the rules of heuristic_policy (None if outside)."""
    g, r, c, level, rising, countdown = decode(obs)
    deepest = 9 - level - (((1 if countdown > 2 else 2)) if rising else 0)
    blocked = g[3] > 0.5
    if rising:
        near = np.zeros_like(blocked)
        near[1:] |= blocked[:-1]; near[:-1] |= blocked[1:]
        near[:, 1:] |= blocked[:, :-1]; near[:, :-1] |= blocked[:, 1:]
        blocked = blocked | near
        blocked[r, c] = False
    dist = np.full((10, 10), -1)
    q = collections.deque()
    for R, C in np.argwhere(g[2] > 0.5):
        if R <= deepest and not blocked[R, C]:
            dist[R, C] = 0
            q.append((R, C))
    while q:
        R, C = q.popleft()
        for dr, dc in MOVES[1:]:
            nr, nc = R + dr, C + dc
            if 0 <= nr <= deepest and 0 <= nc < 10 and dist[nr, nc] < 0 and (not blocked[nr, nc] or (nr, nc) == (r, c)):
                dist[nr, nc] = dist[R, C] + 1
                q.append((nr, nc))
    return dist, (r, c), deepest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=500)
    args = ap.parse_args()
    game = get_game("low_tide")

    # 1. the weights in the file are those of the formulas
    on_disk = json.load(open(OUT))["layers"]
    built = build()
    print(f"weights rewritten from the formulas equal to the file: {on_disk == [L.to_json() for L in built]}")
    names = built[2].names
    groups = [[i for i, s in enumerate(names) if s.startswith(p)] for p in ("up,", "down,", "left,", "right,")]
    veto_groups = [[i for i, s in enumerate(names) if s.startswith(f"jellyfish {p}")]
                   for p in ("above", "below", "on the left", "on the right")]

    layers = load_weights(OUT, game)
    n = diff_spec = zero_bad = veto_bad = 0
    max_rel = 0.0
    same, other = 0, collections.Counter()
    returns = []
    for ep in range(args.episodes):
        seed = game.protocol.seed0 + ep
        env = LowTideEnv(sticky_prob=game.protocol.sticky_prob, max_steps=game.protocol.max_steps)
        obs, _ = env.reset(seed=seed)
        rng = policy_rng(seed)
        total, done = 0.0, False
        while not done:
            acts = forward(layers, obs)
            logits = acts[-1]
            spec = policy_logits(obs)
            n += 1
            a_net, a_spec = int(np.argmax(logits)), int(np.argmax(spec))
            diff_spec += a_net != a_spec
            # 2. potentials per direction: sums of the "up/down/left/right" units of the third layer
            h3 = acts[2]
            pot_net = np.array([h3[idx].sum() for idx in groups])
            pot_ref, veto_ref, _ = potentials(obs, jelly_value=JELLY)
            nz = pot_ref > 0
            if nz.any():
                max_rel = max(max_rel, float(np.max(np.abs(pot_net[nz] / pot_ref[nz] - 1))))
            zero_bad += int(np.any(pot_net[~nz] != 0))
            veto_net = np.array([h3[idx].sum() > 0 for idx in veto_groups])
            veto_bad += int(np.any(veto_net != veto_ref))
            g, r, c, level, rising, countdown = decode(obs)
            # 3. comparison with the BFS
            a_bfs = heuristic_policy(obs, None)
            if a_net == a_bfs:
                same += 1
            else:
                dist, (ar, ac), deepest = bfs_distances(obs)
                d0 = dist[ar, ac]
                dr, dc = MOVES[a_net]
                nr, nc = ar + dr, ac + dc
                ok_cell = 0 <= nr < 10 and 0 <= nc < 10
                if ar > deepest:
                    kind = "tide: different rules"
                elif d0 < 0:
                    kind = "no shell reachable for the BFS"
                elif ok_cell and dist[nr, nc] >= 0 and dist[nr, nc] == d0 - 1:
                    kind = "another step of a shortest path"
                elif a_net == 0 or not ok_cell:
                    kind = "stays still"
                elif g[3, nr, nc] > 0.5:
                    kind = "steps into a jellyfish"
                else:
                    kind = "a longer step (another shell, or the jellyfish avoided another way)"
                other[kind] += 1
            a = sample_action(logits, rng)
            obs, rew, term, trunc, _ = env.step(a)
            total += rew
            done = term or trunc
        returns.append(total)

    returns = np.array(returns)
    print(f"states: {n} ({args.episodes} episodes); score {returns.mean():.2f} ± {returns.std(ddof=1) / np.sqrt(len(returns)):.2f}")
    print(f"network ≠ specification (argmax): {diff_spec}")
    print(f"potentials by direction, largest relative error against the formula: {max_rel:.1e}; "
          f"states with a potential that should be 0 and is not: {zero_bad}")
    print(f"states with a veto (adjacent jellyfish) different from the formula: {veto_bad}")
    print(f"same move as the BFS heuristic: {same / n:.1%}")
    print("when the move differs:")
    for k, v in other.most_common():
        print(f"  {v / n:6.1%} of the states  {k}")


if __name__ == "__main__":
    main()
