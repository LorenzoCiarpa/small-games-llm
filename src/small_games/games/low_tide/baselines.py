"""
Hand-written baselines for Low Tide.
DO NOT give this file to the model that writes the weights: it contains a
hand-written heuristic.
"""

import collections

import numpy as np

from .env import CH_JELLY, CH_SHELL, CH_WATER, H, IDX_COUNTDOWN, IDX_RISING, IDX_STUN, MAX_INTERVAL, MOVES, STUN_STEPS, W


def safe_policy(obs, rng):
    """Walk to row 0 and stay there: never drowns, never scores."""
    return 1


def _decode(obs):
    g = obs[:400].reshape(4, H, W)
    ar, ac = map(int, np.argwhere(g[0] > 0.5)[0])
    level = int(g[CH_WATER].sum() // W)
    rising = obs[IDX_RISING] > 0.5
    countdown = int(round(obs[IDX_COUNTDOWN] * MAX_INTERVAL))
    stun = int(round(obs[IDX_STUN] * STUN_STEPS))
    return g, ar, ac, level, rising, countdown, stun


def heuristic_policy(obs, rng):
    """BFS to the nearest shell, avoiding jellyfish and rows about to flood."""
    g, ar, ac, level, rising, countdown, stun = _decode(obs)
    deepest_ok = H - level - 1                  # lowest dry row
    if rising:
        deepest_ok -= 1 if countdown > 2 else 2 # keep a margin from rising water
    if ar > deepest_ok:
        return 1
    blocked = g[CH_JELLY] > 0.5
    if rising:                                  # a sting near the water is lethal:
        near = np.zeros_like(blocked)           # also avoid cells next to jellyfish
        near[1:] |= blocked[:-1]; near[:-1] |= blocked[1:]
        near[:, 1:] |= blocked[:, :-1]; near[:, :-1] |= blocked[:, 1:]
        blocked = blocked | near
        blocked[ar, ac] = False
    start = (ar, ac)
    prev = {start: None}
    q = collections.deque([start])
    goal = None
    while q:
        cur = q.popleft()
        if g[CH_SHELL][cur] > 0.5 and cur != start:
            goal = cur
            break
        for a in (1, 2, 3, 4):
            dr, dc = MOVES[a]
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr <= deepest_ok and 0 <= nc < W and (nr, nc) not in prev and not blocked[nr, nc]:
                prev[(nr, nc)] = (cur, a)
                q.append((nr, nc))
    if goal is None:
        # wait just above the waterline for the next receding row
        wait_row = max(0, deepest_ok - 1)
        return 2 if ar < wait_row else (1 if ar > wait_row else 0)
    node = goal
    while prev[node][0] != start:
        node = prev[node][0]
    return prev[node][1]
