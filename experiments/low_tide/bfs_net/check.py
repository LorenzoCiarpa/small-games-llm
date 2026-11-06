"""
Verifica weights/low_tide/bfs_net.json (vedi README.md) e lo confronta con l'euristica BFS.

1. Riscrive i pesi dalle formule di build.py e controlla che coincidano con il file.
2. Gioca gli episodi del protocollo con la rete (seed 10000.., sticky 0.1, azioni campionate dalla softmax, come
   `uv run evaluate low_tide bfs_net`). In ogni stato confronta la rete con la specifica di build.py: mossa scelta e
   potenziali per direzione.
3. Negli stessi stati chiede all'euristica (`heuristic_policy`) cosa farebbe. Quando le due mosse sono diverse,
   guarda se quella della rete è comunque un passo di un cammino minimo verso una conchiglia, con le regole della
   BFS (meduse e celle vicine alle meduse bloccate quando la marea sale, righe sotto l'ultima sicura escluse).

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

ACTIONS = ("fermo", "su", "giù", "sinistra", "destra")


def forward(layers, obs):
    x, out = np.asarray(obs, dtype=np.float64), []
    for W, b, act in layers:
        x = W @ x + b
        x = np.maximum(x, 0.0) if act == "relu" else np.tanh(x) if act == "tanh" else x
        out.append(x)
    return out


def bfs_distances(obs):
    """Distanza da ogni cella alla conchiglia più vicina, con le regole di heuristic_policy (None se fuori)."""
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

    # 1. i pesi del file sono quelli delle formule
    on_disk = json.load(open(OUT))["layers"]
    built = build()
    print(f"pesi riscritti dalle formule uguali al file: {on_disk == [L.to_json() for L in built]}")
    names = built[2].names
    groups = [[i for i, s in enumerate(names) if s.startswith(p)] for p in ("su,", "giù,", "sinistra,", "destra,")]
    veto_groups = [[i for i, s in enumerate(names) if s.startswith(f"medusa {p}")]
                   for p in ("sopra", "sotto", "a sinistra", "a destra")]

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
            # 2. potenziali per direzione: somme delle unità "su/giù/sinistra/destra" del terzo strato
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
            # 3. confronto con la BFS
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
                    kind = "marea: regole diverse"
                elif d0 < 0:
                    kind = "nessuna conchiglia raggiungibile per la BFS"
                elif ok_cell and dist[nr, nc] >= 0 and dist[nr, nc] == d0 - 1:
                    kind = "un altro passo di un cammino minimo"
                elif a_net == 0 or not ok_cell:
                    kind = "resta fermo"
                elif g[3, nr, nc] > 0.5:
                    kind = "entra in una medusa"
                else:
                    kind = "un passo più lungo (altra conchiglia, o la medusa aggirata in altro modo)"
                other[kind] += 1
            a = sample_action(logits, rng)
            obs, rew, term, trunc, _ = env.step(a)
            total += rew
            done = term or trunc
        returns.append(total)

    returns = np.array(returns)
    print(f"stati: {n} ({args.episodes} episodi); punteggio {returns.mean():.2f} ± {returns.std(ddof=1) / np.sqrt(len(returns)):.2f}")
    print(f"rete ≠ specifica (argmax): {diff_spec}")
    print(f"potenziali per direzione, errore relativo massimo rispetto alla formula: {max_rel:.1e}; "
          f"stati con un potenziale che dovrebbe essere 0 e non lo è: {zero_bad}")
    print(f"stati con un veto (medusa adiacente) diverso dalla formula: {veto_bad}")
    print(f"stessa mossa dell'euristica BFS: {same / n:.1%}")
    print("quando la mossa è diversa:")
    for k, v in other.most_common():
        print(f"  {v / n:6.1%} degli stati  {k}")


if __name__ == "__main__":
    main()
