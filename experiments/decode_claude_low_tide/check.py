"""
Verifica la lettura a mano di weights/low_tide/claude.json (vedi README.md).

Per ogni stato visitato giocando con la rete ricalcola, direttamente dallo stato del gioco e senza usare i pesi,
i 46 neuroni del primo strato e i 5 logit come li descrive il README, e li confronta con il forward pass vero.

    uv run python experiments/decode_claude_low_tide/check.py [--episodes 200]
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
    """[x] del README: +1 se vero, -1 se falso (un neurone tanh saturo)."""
    return 1.0 if b else -1.0


def decode(env):
    """Primo strato e logit ricostruiti dalle regole del README, a partire dallo stato del gioco."""
    r, c, L = env.agent_r, env.agent_c, env.level
    rising = env.direction > 0
    m = 9 - r - L                       # righe asciutte tra agente e acqua
    shore = 9 - L                       # battigia: ultima riga asciutta
    shell, jelly = env.shell, env.jelly
    shell_l, shell_r, shell_u = shell[r, :c].any(), shell[r, c + 1:].any(), shell[3:r, c].any()
    jelly_l = m == 0 and c >= 1 and jelly[shore, c - 1]
    jelly_r = m == 0 and c <= 8 and jelly[shore, c + 1]
    jelly_d = jelly[shore, c]

    h = np.zeros(46)
    h[0] = m
    h[1] = rising
    h[2 + c] = shell_u                  # 2..11  conchiglia sopra, nella mia colonna
    if 3 <= r <= 9:
        h[12 + r - 3] = shell_l         # 12..18 conchiglia a sinistra, nella mia riga
        h[19 + r - 3] = shell_r         # 19..25 conchiglia a destra, nella mia riga
    h[26 + c] = 1                       # 26..35 colonna dell'agente
    for k in range(10):                 # 36..45 medusa sulla battigia, colonna k
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
            key = ("sale" if env.direction > 0 else "scende", "≥3" if m >= 3 else str(m))
            cases[key, ACTIONS[int(np.argmax(logit_true))]] += 1

            p = np.exp(logit_true - logit_true.max())
            obs, _, term, trunc, _ = env.step(rng.choice(5, p=p / p.sum()))
            done = term or trunc

    print(f"stati visitati: {n} ({args.episodes} episodi)")
    print(f"primo strato, stati in cui la lettura (acceso/spento e valore di h0) sbaglia: {bad_h}")
    print(f"logit, differenza massima: {max_diff:.3f}")
    print(f"azione (argmax) diversa, esclusi i quasi-pareggi < 0.5: {bad_arg}  (quasi-pareggi: {ties})")
    print("\nargmax della rete per situazione (marea, m):")
    totals = collections.Counter()
    for (key, _), v in cases.items():
        totals[key] += v
    for key in sorted(totals):
        row = " ".join(f"{a}:{cases[key, a] / totals[key]:4.0%}" for a in ACTIONS)
        print(f"  {key[0]:6s} m={key[1]:2s} {totals[key] / n:5.1%} degli stati | {row}")


if __name__ == "__main__":
    main()
