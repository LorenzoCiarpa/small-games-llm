The policy simulates the exact ball physics and searches five paddle bounces ahead for productive shots. It penalizes risky trajectories and uses dynamic programming to choose paddle movements under sticky actions. Ambiguous ball directions are considered separately. All per-call state is discarded.

```python
import numpy as np
from collections import deque

_GAMMA = 0.99
_DISCOUNT = tuple(_GAMMA ** i for i in range(1024))
_FULL = (1 << 30) - 1
_SAFETY = 1000.0
_DEPTH = 5
_MAX_DP = 24

_CELL_BITS = tuple(
    (1 << (i - 10)) if 10 <= i < 40 else 0
    for i in range(100)
)
_BRICK_WEIGHTS = np.array(
    [float(1 << i) for i in range(30)], dtype=np.float64
)

_POS = np.arange(10, dtype=np.intp)
_MOVE = np.column_stack((_POS, np.maximum(0, _POS - 1),
                         np.minimum(9, _POS + 1)))
_COLS = np.arange(3, dtype=np.intp)
_ACTIONS = (0, 1, 3)


def _make_tables():
    visits = []
    endings = []

    # Collision-free trajectories from each upward paddle departure.
    for state in range(20):
        x = state // 2
        dx = 1 if state & 1 else -1
        y, dy, t = 8, -1, 0
        path = []

        while True:
            t += 1
            nx, ny = x + dx, y + dy

            if nx < 0:
                nx, dx = 0, 1
            elif nx > 9:
                nx, dx = 9, -1

            if ny < 0:
                ny, dy = 0, 1
            elif ny == 9:
                s0 = 2 * nx + int(dx > 0)
                s1 = s0 if x == nx else 2 * nx + int(dx < 0)
                endings.append((t, x, nx, s0, s1))
                break
            elif 1 <= ny <= 3:
                bit = 1 << ((ny - 1) * 10 + nx)
                # State immediately after a first brick collision.
                path.append((t, bit, nx, y, dx, -dy))

            x, y = nx, ny

        visits.append(tuple(path))

    # Exact shortest time to each individual brick on an otherwise
    # empty board, allowing paddle steering at intervening returns.
    nearest = []
    for start in range(20):
        hops = [999] * 20
        hops[start] = 0
        queue = deque([start])

        while queue:
            s = queue.popleft()
            for nxt in endings[s][3:5]:
                if hops[nxt] == 999:
                    hops[nxt] = hops[s] + 1
                    queue.append(nxt)

        distances = [100000] * 30
        for s in range(20):
            if hops[s] == 999:
                continue
            for t, bit, _, _, _, _ in visits[s]:
                k = bit.bit_length() - 1
                distances[k] = min(distances[k], 18 * hops[s] + t)

        grouped = {}
        for k, distance in enumerate(distances):
            if distance < 100000:
                grouped[distance] = grouped.get(distance, 0) | (1 << k)

        nearest.append(tuple(sorted(grouped.items())))

    # Optimal catch probability, conservatively taking the worst
    # previous executed action. Catch cells are adjacent or identical.
    survival = np.ones((10, 10, _MAX_DP + 1, 10), dtype=np.float64)
    for lo in range(10):
        for hi in range(lo, min(10, lo + 2)):
            value = np.zeros((10, 3), dtype=np.float64)
            value[lo, :] = 1.0
            value[hi, :] = 1.0

            for t in range(_MAX_DP + 1):
                probabilities = value.min(axis=1)
                survival[lo, hi, t, :] = probabilities
                survival[hi, lo, t, :] = probabilities
                if t != _MAX_DP:
                    executed = value[_MOVE, _COLS]
                    value = (
                        0.9 * executed.max(axis=1)[:, None]
                        + 0.1 * executed
                    )

    return tuple(visits), tuple(endings), tuple(nearest), survival


_VISITS, _ENDINGS, _NEAREST, _SURVIVAL = _make_tables()


def _fly(x, y, dx, dy, strike, bricks):
    """Advance to the next paddle interaction, without choosing a catch."""
    t = 0
    reward = 0.0

    while True:
        t += 1
        nx, ny = x + dx, y + dy

        if nx < 0:
            nx, dx = 0, 1
        elif nx > 9:
            nx, dx = 9, -1

        if ny < 0:
            ny, dy = 0, 1
            strike = False

        elif ny == 9:
            if bricks == 0:
                bricks = _FULL
            s0 = 2 * nx + int(dx > 0)
            s1 = s0 if x == nx else 2 * nx + int(dx < 0)
            return t, x, nx, s0, s1, bricks, reward

        else:
            bit = _CELL_BITS[ny * 10 + nx]
            if bricks & bit:
                if not strike:
                    bricks ^= bit
                    ny = y
                    dy = -dy
                    reward += _DISCOUNT[t - 1]
                strike = True
            else:
                strike = False

        x, y = nx, ny


def policy(obs, rng):
    board = np.asarray(obs).reshape(100, 6)
    paddle = int(np.argmax(board[90:100, 0]))
    ball = int(np.argmax(board[:, 3]))
    trail = int(np.argmax(board[:, 4]))
    bricks = int(board[10:40, 2] @ _BRICK_WEIGHTS)

    y, x = divmod(ball, 10)
    ty, tx = divmod(trail, 10)

    if y >= 9:
        return 0

    if x > tx:
        dx = 1
    elif x < tx:
        dx = -1
    else:
        dx = 1 if x < 5 else -1

    # Each hypothesis contains the current vertical direction and
    # the game's hidden consecutive-brick-strike flag.
    if y != ty:
        hypotheses = [
            (1 if y > ty else -1, bool(bricks & _CELL_BITS[ball]))
        ]
    elif y == 8:
        if paddle != tx:
            dx = -dx
        hypotheses = [(-1, False)]
    elif y == 3 and x == tx and bricks == _FULL:
        hypotheses = [(1, False)]  # Episode initialization.
    elif y == 0:
        hypotheses = [(1, False)]  # Ceiling reflection.
        if not (bricks & (1 << x)):
            hypotheses.append((-1, True))
    elif y == 2:
        hypotheses = []
        if not (bricks & (1 << x)):
            hypotheses.append((1, True))
        if not (bricks & (1 << (20 + x))):
            hypotheses.append((-1, True))
        if not hypotheses:
            hypotheses = [(1, True), (-1, True)]
    elif y == 1:
        hypotheses = [(-1, True)]
    else:
        hypotheses = [(1, True)]

    flight_cache = {}
    value_cache = {}

    def launch_flight(state, mask):
        key = (state, mask)
        cached = flight_cache.get(key)
        if cached is not None:
            return cached

        # Skip directly to the first brick interaction.
        for t, bit, bx, by, bdx, bdy in _VISITS[state]:
            if mask & bit:
                rest = _fly(bx, by, bdx, bdy, True, mask ^ bit)
                rt, c0, c1, s0, s1, next_mask, reward = rest
                result = (
                    t + rt, c0, c1, s0, s1, next_mask,
                    _DISCOUNT[t - 1] + _DISCOUNT[t] * reward
                )
                flight_cache[key] = result
                return result

        t, c0, c1, s0, s1 = _ENDINGS[state]
        result = (t, c0, c1, s0, s1, mask or _FULL, 0.0)
        flight_cache[key] = result
        return result

    def shot_value(state, pos, mask, depth):
        key = (state, pos, mask, depth)
        cached = value_cache.get(key)
        if cached is not None:
            return cached

        if depth == 0:
            result = 0.0
            for distance, bits in _NEAREST[state]:
                if mask & bits:
                    result = _DISCOUNT[distance - 1]
                    break
        else:
            t, c0, c1, s0, s1, next_mask, reward = launch_flight(
                state, mask
            )
            best = shot_value(s0, c0, next_mask, depth - 1)
            if c0 != c1:
                best = max(
                    best, shot_value(s1, c1, next_mask, depth - 1)
                )

            survival = float(
                _SURVIVAL[c0, c1, min(t, _MAX_DP), pos]
            )
            risk_cost = _SAFETY * max(0.0, 1.0 - survival)
            result = reward + _DISCOUNT[t] * (best - risk_cost)

        value_cache[key] = result
        return result

    action_values = np.zeros(3, dtype=np.float64)
    distance_hint = np.zeros(3, dtype=np.float64)

    for dy, strike in hypotheses:
        t, c0, c1, s0, s1, next_mask, _ = _fly(
            x, y, dx, dy, strike, bricks
        )

        v0 = shot_value(s0, c0, next_mask, _DEPTH)
        v1 = v0 if c0 == c1 else shot_value(
            s1, c1, next_mask, _DEPTH
        )

        terminal = np.zeros((10, 3), dtype=np.float64)
        terminal[c0, :] = _SAFETY + v0
        terminal[c1, :] = _SAFETY + v1

        # V[p, previous_action]. The sticky contribution is independent
        # of the requested action, so the optimal request does not
        # require remembering the previous action.
        value = terminal
        for _ in range(min(t, _MAX_DP) - 1):
            executed = value[_MOVE, _COLS]
            value = (
                0.9 * executed.max(axis=1)[:, None]
                + 0.1 * executed
            )

        action_values += value[_MOVE[paddle], _COLS]

        if abs(v0 - v1) <= 1e-10:
            target = c0 if abs(paddle - c0) <= abs(paddle - c1) else c1
        else:
            target = c0 if v0 > v1 else c1
        distance_hint += np.abs(_MOVE[paddle] - target)

    # When numerical values tie, move into position early rather
    # than spending otherwise useful movement slack.
    best_value = float(action_values.max())
    selected = 0
    best_hint = float("inf")
    for a in range(3):
        if action_values[a] >= best_value - 1e-9:
            hint = float(distance_hint[a])
            if hint < best_hint:
                selected = a
                best_hint = hint

    return _ACTIONS[selected]
```