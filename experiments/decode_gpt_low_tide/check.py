"""
Verifica la lettura a mano di weights/low_tide/gpt.json (vedi README.md).

Per ogni stato visitato giocando con la rete ricalcola, direttamente dallo stato del gioco e senza usare i pesi,
i 43 neuroni del primo strato e i 5 logit come li descrive il README, e li confronta con il forward pass vero.

    uv run python experiments/decode_gpt_low_tide/check.py [--episodes 200]
"""

import argparse
import collections
from pathlib import Path

import numpy as np

from small_games.core import get_game
from small_games.games.low_tide.env import LowTideEnv
from small_games.weights import load_weights

ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ("fermo", "su", "giu", "sx", "dx")
WINDOW = [(R, C) for R in range(3, 9) for C in range(6, 10)]   # la finestra che la rete guarda: righe 3-8, colonne 6-9


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
    """Primo strato e logit ricostruiti dalle regole del README, a partire dallo stato del gioco."""
    r, c, L = env.agent_r, env.agent_c, env.level
    rising = env.direction > 0
    m = 9 - r - L
    shell, jelly = env.shell, env.jelly

    h = np.zeros(43)
    for i, (R, C) in enumerate(WINDOW):            # 0..23: 2 conchiglia, 1 vuota, 0 medusa
        h[i] = 1 + shell[R, C] - jelly[R, C]
    for k in range(7):                             # 24..30: rampe sulla riga dell'agente
        h[24 + k] = relu(r - 2 - k)
    for k in range(4):                             # 31..34: rampe sulla colonna dell'agente
        h[31 + k] = relu(c - 5 - k)
    h[35] = L + 10 * rising                        # livello + 10 se sale
    h[36] = relu(2 - r)                            # 2 in riga 0, 1 in riga 1
    for k in range(5):                             # 37..41: sale e livello >= 5, 4, 3, 2, 1
        h[37 + k] = rising and L >= 5 - k
    h[42] = rising

    # distanza di Manhattan dalla posizione dell'agente "schiacciata" in righe >= 2, colonne >= 5
    re, ce = max(r, 2), max(c, 5)
    attr = collections.Counter()                   # attrazione delle conchiglie, per direzione
    jel = collections.Counter()                    # meduse adiacenti, per direzione
    for R, C in WINDOW:
        d = abs(re - R) + abs(ce - C)
        if shell[R, C] and not (rising and 9 - R - L <= 1):
            a = relu(6 - d)
            attr["su"] += a * (R < r); attr["giu"] += a * (R > r)
            attr["sx"] += a * (C < c); attr["dx"] += a * (C > c)
        if jelly[R, C]:
            j = relu(2 - d)
            jel["su"] += j * (R < r); jel["giu"] += j * (R > r)
            jel["sx"] += j * (C < c); jel["dx"] += j * (C > c)

    danger1 = relu(2 - m) if rising else 0.0       # sale e m <= 1
    danger2 = relu(3 - m) if rising else 0.0       # sale e m <= 2
    logits = np.array([
        -4.0,
        -12 + 12 * (r >= 4) + 0.75 * (r >= 7) + 12 * (r == 9)
        + 0.9 * attr["su"] - 25 * jel["su"] + 1000 * danger1,
        12.75 - 12 * (r >= 3) - 0.75 * (r >= 6) - 12 * (r >= 8)
        + 0.9 * attr["giu"] - 25 * jel["giu"] - 1000 * danger2,
        -12 + 12 * (c >= 7) + 0.75 * (c == 9) + 0.9 * attr["sx"] - 25 * jel["sx"],
        12.75 - 12 * (c >= 6) - 0.75 * relu(c - 7) - 11.25 * relu(c - 8) + 0.9 * attr["dx"] - 25 * jel["dx"],
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

    print(f"stati visitati: {n} ({args.episodes} episodi)")
    print(f"primo strato, stati in cui la lettura sbaglia: {bad_h}")
    print(f"logit, differenza massima: {max_diff:.2e}")
    print(f"azione (argmax) diversa, esclusi i quasi-pareggi < 0.5: {bad_arg}  (quasi-pareggi: {ties})")
    print("colonna dell'agente:", " ".join(f"{k}:{cols[k] / n:.1%}" for k in range(10)))
    print("riga dell'agente:   ", " ".join(f"{k}:{rows[k] / n:.1%}" for k in range(10)))
    print(f"conchiglie sulla griglia dentro la finestra: {shells_win / max(shells_all, 1):.1%}")


if __name__ == "__main__":
    main()
