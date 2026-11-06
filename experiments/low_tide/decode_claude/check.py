"""
Checks the hand reading of weights/low_tide/claude.json (see README.md).

For every state visited while playing with the network it recomputes, directly from the game state and without using
the weights, the 46 neurons of the first layer and the 5 logits as the README describes them, and compares them with
the real forward pass.

    uv run python experiments/low_tide/decode_claude/check.py [--episodes 200]
"""

import argparse
import collections
from pathlib import Path

import numpy as np

from small_games.core import get_game
from small_games.games.low_tide.env import LowTideEnv
from small_games.weights import load_weights

ROOT = Path(__file__).resolve().parents[3]
ACTIONS = ("no-op", "up", "down", "left", "right")


def forward(layers, obs):
    x, out = np.asarray(obs, dtype=np.float64), []
    for W, b, act in layers:
        x = W @ x + b
        if act == "relu":
            x = np.maximum(x, 0.0)
        elif act == "tanh":
            x = np.tanh(x)
        out.append(x)
    return out


def pm(b):
    """[x] of the README: +1 if true, -1 if false (a saturated tanh neuron)."""
    return 1.0 if b else -1.0


def decode(env):
    """First layer and logits rebuilt from the rules of the README, starting from the game state."""
    r, c, L = env.agent_r, env.agent_c, env.level
    rising = env.direction > 0
    m = 9 - r - L                       # dry rows between the agent and the water
    shore = 9 - L                       # shoreline: last dry row
    shell, jelly = env.shell, env.jelly
    shell_l, shell_r, shell_u = shell[r, :c].any(), shell[r, c + 1:].any(), shell[3:r, c].any()
    jelly_l = m == 0 and c >= 1 and jelly[shore, c - 1]
    jelly_r = m == 0 and c <= 8 and jelly[shore, c + 1]
    jelly_d = jelly[shore, c]

    h = np.zeros(46)
    h[0] = m
    h[1] = rising
    h[2 + c] = shell_u                  # 2..11  shell above, in my column
    if 3 <= r <= 9:
        h[12 + r - 3] = shell_l         # 12..18 shell to the left, in my row
        h[19 + r - 3] = shell_r         # 19..25 shell to the right, in my row
    h[26 + c] = 1                       # 26..35 agent's column
    for k in range(10):                 # 36..45 jellyfish on the shoreline, column k
        h[36 + k] = shore >= 3 and jelly[shore, k]

    logits = np.array([
        0.0,
        23.5 + 4.5 * pm(rising and shell_u) + 20 * pm(rising and m <= 1),
        -10 + 10.5 * pm(not rising and m >= 1) - 20 * pm(rising and m <= 2) - 2.5 * pm(m == 0)
        + 3 * pm(rising and m >= 3) - 30 * jelly_d,
        7 + 6 * pm(shell_l) - 30 * jelly_l,
        8.5 + 7.5 * pm(shell_r) - 30 * jelly_r,
    ])
    return h, logits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=10000)
    args = ap.parse_args()

    layers = load_weights(ROOT / "weights/low_tide/claude.json", get_game("low_tide"))
    rng = np.random.default_rng(0)
    n = bad_h = bad_arg = ties = 0
    max_diff = 0.0
    cases = collections.Counter()

    for ep in range(args.episodes):
        env = LowTideEnv()
        obs, _ = env.reset(seed=args.seed0 + ep)
        done = False
        while not done:
            h_true, _, logit_true = forward(layers, obs)
            h_mine, logit_mine = decode(env)
            n += 1
            if not np.array_equal(h_true > 0, h_mine > 0) or h_true[0] != h_mine[0]:
                bad_h += 1
            top2 = np.sort(logit_mine)[-2:]
            if top2[1] - top2[0] <= 0.5:
                ties += 1
            elif np.argmax(logit_true) != np.argmax(logit_mine):
                bad_arg += 1
            max_diff = max(max_diff, float(np.abs(logit_true - logit_mine).max()))

            m = 9 - env.agent_r - env.level
            key = ("incoming" if env.direction > 0 else "outgoing", "≥3" if m >= 3 else str(m))
            cases[key, ACTIONS[int(np.argmax(logit_true))]] += 1

            p = np.exp(logit_true - logit_true.max())
            obs, _, term, trunc, _ = env.step(rng.choice(5, p=p / p.sum()))
            done = term or trunc

    print(f"states visited: {n} ({args.episodes} episodes)")
    print(f"first layer, states in which the reading (on/off and value of h0) is wrong: {bad_h}")
    print(f"logits, maximum difference: {max_diff:.3f}")
    print(f"move with the highest logit differs, excluding near-ties < 0.5: {bad_arg}  (near-ties: {ties})")
    print("\nargmax of the network by situation (tide, m):")
    totals = collections.Counter()
    for (key, _), v in cases.items():
        totals[key] += v
    for key in sorted(totals):
        row = " ".join(f"{a}:{cases[key, a] / totals[key]:4.0%}" for a in ACTIONS)
        print(f"  {key[0]:6s} m={key[1]:2s} {totals[key] / n:5.1%} of states | {row}")


if __name__ == "__main__":
    main()
