"""
Checks the hand reading of weights/low_tide/gpt.json (see README.md).

For every state visited while playing with the network it recomputes, directly from the game state and without using
the weights, the 43 neurons of the first layer and the 5 logits as the README describes them, and compares them with
the real forward pass.

    uv run python experiments/low_tide/decode_gpt/check.py [--episodes 200]
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
WINDOW = [(R, C) for R in range(3, 9) for C in range(6, 10)]   # the window the network looks at: rows 3-8, columns 6-9


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


def relu(x):
    return max(0.0, x)


def decode(env):
    """First layer and logits rebuilt from the rules of the README, starting from the game state."""
    r, c, L = env.agent_r, env.agent_c, env.level
    rising = env.direction > 0
    m = 9 - r - L
    shell, jelly = env.shell, env.jelly

    h = np.zeros(43)
    for i, (R, C) in enumerate(WINDOW):            # 0..23: 2 shell, 1 empty, 0 jellyfish
        h[i] = 1 + shell[R, C] - jelly[R, C]
    for k in range(7):                             # 24..30: ramps on the agent's row
        h[24 + k] = relu(r - 2 - k)
    for k in range(4):                             # 31..34: ramps on the agent's column
        h[31 + k] = relu(c - 5 - k)
    h[35] = L + 10 * rising                        # level + 10 if rising
    h[36] = relu(2 - r)                            # 2 in row 0, 1 in row 1
    for k in range(5):                             # 37..41: rising and level >= 5, 4, 3, 2, 1
        h[37 + k] = rising and L >= 5 - k
    h[42] = rising

    # Manhattan distance from the agent's position "clamped" to rows >= 2, columns >= 5
    re, ce = max(r, 2), max(c, 5)
    attr = collections.Counter()                   # attraction of the shells, by direction
    jel = collections.Counter()                    # adjacent jellyfish, by direction
    for R, C in WINDOW:
        d = abs(re - R) + abs(ce - C)
        if shell[R, C] and not (rising and 9 - R - L <= 1):
            a = relu(6 - d)
            attr["up"] += a * (R < r); attr["down"] += a * (R > r)
            attr["left"] += a * (C < c); attr["right"] += a * (C > c)
        if jelly[R, C]:
            j = relu(2 - d)
            jel["up"] += j * (R < r); jel["down"] += j * (R > r)
            jel["left"] += j * (C < c); jel["right"] += j * (C > c)

    danger1 = relu(2 - m) if rising else 0.0       # rising and m <= 1
    danger2 = relu(3 - m) if rising else 0.0       # rising and m <= 2
    logits = np.array([
        -4.0,
        -12 + 12 * (r >= 4) + 0.75 * (r >= 7) + 12 * (r == 9)
        + 0.9 * attr["up"] - 25 * jel["up"] + 1000 * danger1,
        12.75 - 12 * (r >= 3) - 0.75 * (r >= 6) - 12 * (r >= 8)
        + 0.9 * attr["down"] - 25 * jel["down"] - 1000 * danger2,
        -12 + 12 * (c >= 7) + 0.75 * (c == 9) + 0.9 * attr["left"] - 25 * jel["left"],
        12.75 - 12 * (c >= 6) - 0.75 * relu(c - 7) - 11.25 * relu(c - 8) + 0.9 * attr["right"] - 25 * jel["right"],
    ])
    return h, logits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=10000)
    args = ap.parse_args()

    layers = load_weights(ROOT / "weights/low_tide/gpt.json", get_game("low_tide"))
    rng = np.random.default_rng(0)
    n = bad_h = bad_arg = ties = 0
    max_diff = 0.0
    cols = collections.Counter()
    rows = collections.Counter()
    shells_all = shells_win = 0

    for ep in range(args.episodes):
        env = LowTideEnv()
        obs, _ = env.reset(seed=args.seed0 + ep)
        done = False
        while not done:
            h_true, _, _, logit_true = forward(layers, obs)
            h_mine, logit_mine = decode(env)
            n += 1
            if not np.allclose(h_true, h_mine):
                bad_h += 1
            top2 = np.sort(logit_mine)[-2:]
            if top2[1] - top2[0] <= 0.5:
                ties += 1
            elif np.argmax(logit_true) != np.argmax(logit_mine):
                bad_arg += 1
            max_diff = max(max_diff, float(np.abs(logit_true - logit_mine).max()))
            cols[env.agent_c] += 1
            rows[env.agent_r] += 1
            shells_all += int(env.shell.sum())
            shells_win += int(env.shell[3:9, 6:10].sum())

            p = np.exp(logit_true - logit_true.max())
            obs, _, term, trunc, _ = env.step(rng.choice(5, p=p / p.sum()))
            done = term or trunc

    print(f"states visited: {n} ({args.episodes} episodes)")
    print(f"first layer, states in which the reading is wrong: {bad_h}")
    print(f"logits, maximum difference: {max_diff:.2e}")
    print(f"move with the highest logit differs, excluding near-ties < 0.5: {bad_arg}  (near-ties: {ties})")
    print("agent's column:", " ".join(f"{k}:{cols[k] / n:.1%}" for k in range(10)))
    print("agent's row:   ", " ".join(f"{k}:{rows[k] / n:.1%}" for k in range(10)))
    print(f"shells on the grid inside the window: {shells_win / max(shells_all, 1):.1%}")


if __name__ == "__main__":
    main()
