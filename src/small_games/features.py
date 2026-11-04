"""
Interpretable features of each game, read from the observation alone (what the network sees), for analysing
policies: correlating hidden units with game quantities and distilling a policy into a small decision tree.

    FEATURES[game] = (names, fn)      fn(obs) -> np.ndarray of len(names), float

The features are a reasonable vocabulary for describing a strategy (positions, directions, distances, gauges),
not a complete description of the state. A missing object is encoded with NONE = 99 for distances, so that a tree
can separate "absent" from any real distance with a single threshold.
"""

from __future__ import annotations

import numpy as np

NONE = 99.0


def _grid(obs):
    return np.asarray(obs[:600]).reshape(10, 10, 6) > 0.5


def _cells(mask):
    return [tuple(int(v) for v in rc) for rc in np.argwhere(mask)]


# --------------------------------------------------------------------------------------------- breakout
BREAKOUT = ("paddle_x", "ball_x", "ball_y", "ball_dx", "ball_dy", "ball_rel_x", "landing_rel_x", "bricks")


def _landing(bx, by, dx, dy):
    """Column where the ball enters row 9 if nothing is in the way (MinAtar's wall clamping included)."""
    if dx == 0 and dy == 0:
        return bx
    x, y = bx, by
    for _ in range(40):
        nx, ny = x + dx, y + dy
        if nx < 0 or nx > 9:
            nx = min(max(nx, 0), 9)
            dx = -dx
        if ny < 0:
            ny = 0
            dy = -dy
        elif ny >= 9:
            return nx
        x, y = nx, ny
    return x


def breakout(obs):
    g = _grid(obs)
    p = np.flatnonzero(g[9, :, 0])
    paddle = float(p[0]) if len(p) else 4.0
    ball = _cells(g[:, :, 3])
    trail = _cells(g[:, :, 4])
    if not ball:
        return np.array([paddle, paddle, 9, 0, 0, 0, 0, g[:, :, 2].sum()], dtype=float)
    by, bx = ball[0]
    dx = dy = 0
    if trail:
        ty, tx = trail[0]
        dx, dy = int(np.sign(bx - tx)), int(np.sign(by - ty))
    land = _landing(bx, by, dx, dy)
    return np.array([paddle, bx, by, dx, dy, bx - paddle, land - paddle, g[:, :, 2].sum()], dtype=float)


# --------------------------------------------------------------------------------------------- asterix
OFFSETS = (-2, -1, 0, 1, 2)                       # rows relative to the player / sub
_REL = {-2: "up2", -1: "up1", 0: "same", 1: "down1", 2: "down2"}

ASTERIX = ("player_x", "player_y",
           *(f"enemy_eta_{_REL[k]}" for k in OFFSETS),   # steps (cells) before an enemy in that row reaches my column
           *(f"gold_eta_{_REL[k]}" for k in OFFSETS),    # same for a treasure; NONE if nothing is coming
           "gold_dx", "gold_dy", "enemies")


def _dir_from_trail(g, y, x):
    """+1 if the object at (y, x) moves right (trail on its left), -1 if left, 0 if unknown."""
    if x > 0 and g[y, x - 1, 4]:
        return 1
    if x < 9 and g[y, x + 1, 4]:
        return -1
    return 1 if x == 0 else -1 if x == 9 else 0


def _eta(xs, row, g, col):
    """Smallest distance to `col` of the objects at columns xs of `row` that are moving toward it (or on it)."""
    best = NONE
    for x in xs:
        dx = col - int(x)
        if dx == 0 or _dir_from_trail(g, row, int(x)) == np.sign(dx):
            best = min(best, float(abs(dx)))
    return best


def asterix(obs):
    g = _grid(obs)
    pl = _cells(g[:, :, 0])
    py, px = pl[0] if pl else (5, 5)
    enemy, gold = g[:, :, 1], g[:, :, 2]
    out = [px, py]
    for mask in (enemy, gold):
        for k in OFFSETS:
            row = py + k
            out.append(_eta(np.flatnonzero(mask[row]), row, g, px) if 1 <= row <= 8 else NONE)
    golds = _cells(gold)
    if golds:
        gy, gx = min(golds, key=lambda c: abs(c[0] - py) + abs(c[1] - px))
        out += [gx - px, gy - py]
    else:
        out += [NONE, NONE]
    out.append(float(enemy.sum()))
    return np.array(out, dtype=float)


# --------------------------------------------------------------------------------------------- seaquest
SEAQUEST = ("sub_x", "sub_y", "facing", "oxygen", "divers",
            *(f"enemy_eta_{_REL[k]}" for k in OFFSETS),   # enemies (fish, subs) coming toward the sub's column
            *(f"enemy_near_{_REL[k]}" for k in OFFSETS),  # nearest enemy in that row, any direction
            *(f"diver_near_{_REL[k]}" for k in OFFSETS),  # nearest diver in that row
            "target_ahead", "bullet_near", "diver_dx", "diver_dy", "torpedo_same_row", "enemies")


def seaquest(obs):
    g = _grid(obs)
    P, T = g[:9, :, 0], g[:9, :, 4]
    front = _cells(P & ~T) or _cells(P)
    back = _cells(P & T)
    sy, sx = front[0] if front else (0, 5)
    facing = int(np.sign(sx - back[0][1])) if back else 0
    row9 = g[9, :, 5]
    oxygen = int(np.argmin(row9)) if not row9.all() else 10          # leading ones from col 0
    divers = 0
    for c in range(8, -1, -1):                                      # trailing ones ending at col 8
        if not row9[c]:
            break
        divers += 1
    divers = min(divers, 6)
    enemy = g[:9, :, 1] & g[:9, :, 2]
    bullet = g[:9, :, 1] & ~g[:9, :, 2]
    diver = g[:9, :, 2] & ~g[:9, :, 1]

    def near(mask, row):
        if not 0 <= row <= 8:
            return NONE
        xs = np.flatnonzero(mask[row])
        return float(np.min(np.abs(xs - sx))) if len(xs) else NONE

    out = [sx, sy, facing, oxygen, divers]
    out += [_eta(np.flatnonzero(enemy[sy + k]), sy + k, g, sx) if 0 <= sy + k <= 8 else NONE for k in OFFSETS]
    out += [near(enemy, sy + k) for k in OFFSETS]
    out += [near(diver, sy + k) for k in OFFSETS]
    xs = np.flatnonzero(enemy[sy]) if sy <= 8 else []
    ahead = [abs(int(x) - sx) for x in xs if facing != 0 and np.sign(int(x) - sx) == facing]
    out.append(float(min(ahead)) if ahead else NONE)
    out.append(near(bullet, sy))
    dv = _cells(diver)
    if dv:
        dy_, dx_ = min(((y - sy, x - sx) for y, x in dv), key=lambda t: abs(t[0]) + abs(t[1]))
    else:
        dy_, dx_ = NONE, NONE
    out += [dx_, dy_, float(g[sy, :, 3].any()) if sy <= 8 else 0.0, float(enemy.sum())]
    return np.array(out, dtype=float)


# --------------------------------------------------------------------------------------------- low tide
LOW_TIDE = ("row", "col", "level", "rising", "countdown", "stun", "margin",
            "shells_up", "shells_down", "shells_left", "shells_right",
            "shell_dx", "shell_dy", "jelly_up", "jelly_down", "jelly_left", "jelly_right")


def low_tide(obs):
    o = np.asarray(obs)
    planes = o[:400].reshape(4, 10, 10) > 0.5
    ag = np.argwhere(planes[0])
    r, c = (int(ag[0][0]), int(ag[0][1])) if len(ag) else (1, 0)
    L = int(planes[1][:, 0].sum())
    sh, je = planes[2], planes[3]
    shells = np.argwhere(sh)
    if len(shells):
        k = np.argmin(np.abs(shells[:, 0] - r) + np.abs(shells[:, 1] - c))
        sdy, sdx = float(shells[k][0] - r), float(shells[k][1] - c)
    else:
        sdy = sdx = NONE

    def jel(rr, cc):
        return float(0 <= rr <= 9 and 0 <= cc <= 9 and je[rr, cc])

    return np.array([r, c, L, o[400], round(o[401] * 5), round(o[402] * 3), 9 - r - L,
                     sh[:r, c].sum(), sh[r + 1:, c].sum(), sh[r, :c].sum(), sh[r, c + 1:].sum(),
                     sdx, sdy, jel(r - 1, c), jel(r + 1, c), jel(r, c - 1), jel(r, c + 1)], dtype=float)


FEATURES = {
    "breakout": (BREAKOUT, breakout),
    "asterix": (ASTERIX, asterix),
    "seaquest": (SEAQUEST, seaquest),
    "low_tide": (LOW_TIDE, low_tide),
}
