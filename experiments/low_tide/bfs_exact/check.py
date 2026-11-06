"""
Verifica bfs_exact.json (vedi README.md): la rete deve fare la stessa mossa di `heuristic_policy` in ogni stato.

La rete supera i limiti del gioco (3 strati da 64), quindi `uv run evaluate` non la carica. Qui la si legge con lo
stesso `load_weights` del repo, solo con i limiti di strati e unità tolti, e si usa il forward di riferimento
(`ReferenceMLP`, float64 denso).

1. Riscrive i pesi con build.py e controlla che coincidano con il file.
2. Gioca gli episodi del protocollo (seed 10000.., sticky 0.1, azioni campionate dalla softmax, come `evaluate`).
   In ogni stato confronta la mossa della rete con quella dell'euristica e misura il margine dei logit.
   Gioca gli stessi episodi con l'euristica e confronta i punteggi episodio per episodio.
3. Stati costruiti apposta, anche lontani da quelli del gioco, per controllare distanze che negli episodi non
   capitano: metà casuali (fino al 45% di meduse, 1-3 conchiglie), un quarto labirinti di meduse e un quarto
   serpentine (corridoi lunghi fino a ~60 passi), con agente e conchiglia in celle a caso.

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
    """load_weights del repo, con i limiti di strati e unità tolti (attivazioni e forma restano controllate)."""
    net = dataclasses.replace(GAME.net, max_hidden_layers=10**6, max_hidden_units=10**6)
    return ReferenceMLP(load_weights(path, dataclasses.replace(GAME, net=net)))


def bfs_depth(obs):
    """Distanza dall'agente della conchiglia trovata da heuristic_policy (None se non ce n'è o se sale)."""
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
    """Un episodio del protocollo con la rete: (punteggio, stati, mosse diverse, campionata ≠ argmax, margine min, d max)."""
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
    """Uno stato valido qualsiasi: livello 0..7, marea che sale o scende, meduse fino al 45%, 1-3 conchiglie."""
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
    """Marea ferma (livello 0 o 1, scende): pareti di meduse a labirinto o a serpentina, una conchiglia."""
    level = int(rng.integers(0, 2))
    dry = H - level
    jelly = np.ones((H, W), dtype=bool)
    if kind == "serpentina":
        jelly[0::2, :] = False
        for i, r in enumerate(range(1, dry, 2)):
            jelly[r, W - 1 if i % 2 == 0 else 0] = False
    else:  # labirinto perfetto (DFS) su una griglia 5x5 di celle nelle righe e colonne pari
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
        obs = random_state(rng) if i % 4 < 2 else maze_state(rng, "serpentina" if i % 4 == 2 else "labirinto")
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
    print(f"pesi riscritti dalle formule uguali al file: {data['layers'] == [L.to_json() for L in build(K_ALL)]}")
    MLP = load(OUT)
    sizes = [Wm.shape[0] for Wm, _, _ in MLP.layers[:-1]]
    print(f"rete: {len(sizes)} strati nascosti, {sum(sizes)} neuroni, {MLP.n_nonzero()} pesi diversi da zero "
          f"su {MLP.n_params()}")

    seeds = [GAME.protocol.seed0 + i for i in range(args.episodes)]
    with mp.get_context("fork").Pool(8) as pool:
        res = pool.map(episode, seeds, chunksize=1)
        heur = np.array(pool.map(heuristic_episode, seeds, chunksize=4))
        chunks = [(i, args.random // 40) for i in range(40)]
        rnd = pool.map(random_batch, chunks, chunksize=1)

    net = np.array([r[0] for r in res])
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    print(f"\nepisodi del protocollo: {args.episodes}, stati {sum(r[1] for r in res)}")
    print(f"  mosse diverse dall'euristica: {sum(r[2] for r in res)}")
    print(f"  azione campionata ≠ argmax: {sum(r[3] for r in res)}; margine minimo dei logit: {min(r[4] for r in res):.0f}")
    print(f"  punteggio rete {net.mean():.2f} ± {se(net):.2f}, euristica {heur.mean():.2f} ± {se(heur):.2f}, "
          f"episodi con punteggio diverso: {int(np.sum(net != heur))}")
    dmax = max(r[5] for r in res)
    print(f"  distanza massima della conchiglia trovata dalla BFS: {dmax} "
          f"(per questi stati basterebbero K = {dmax - 1} passi, {dmax - 1 + 6} strati)")

    depths = np.concatenate([r[2] for r in rnd])
    print(f"\nstati costruiti (casuali, labirinti, serpentine): {len(depths)}")
    print(f"  mosse diverse dall'euristica: {sum(r[0] for r in rnd)}; margine minimo: {min(r[1] for r in rnd):.0f}")
    print(f"  distanza della conchiglia: mediana {np.median(depths[depths > 0]):.0f}, massima {depths.max()}, "
          f"stati con distanza > {dmax}: {int(np.sum(depths > dmax))}")
    print(f"\n({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
