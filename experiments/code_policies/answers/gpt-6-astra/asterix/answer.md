I use short-horizon planning to intercept treasure under both fast and slow movement models. A separate conservative safety search accounts for sticky actions, uncertain movement timing, and possible new spawns. The policy is stateless and favors central positions when opportunities are otherwise similar.

```python
import numpy as np


def _make_tables():
    cells = np.arange(80)
    x = cells % 10
    y = cells // 10

    dest = np.stack(
        (
            cells,
            y * 10 + np.maximum(x - 1, 0),
            np.maximum(y - 1, 0) * 10 + x,
            y * 10 + np.minimum(x + 1, 9),
            np.minimum(y + 1, 7) * 10 + x,
        ),
        axis=1,
    )
    actions = np.arange(5)
    repeated = [dest]
    for _ in range(7):
        repeated.append(dest[repeated[-1], actions])

    center_cost = (
        0.008 * (x - 4.5) ** 2
        + 0.002 * (y - 3.5) ** 2
    )
    return dest, actions, np.stack(repeated), center_cost


_DEST, _ACTIONS, _REPEATED, _CENTER_COST = _make_tables()
_ROWS = np.arange(8)
_COLS = np.arange(10)
_T = np.arange(1, 10)
_PHASES = np.arange(3)

_SLOW_PRE = ((_T[:, None] - 1 + _PHASES) // 3)[:, :, None, None]
_SLOW_POST = ((_T[:, None] + _PHASES) // 3)[:, :, None, None]

_REPEAT_T = np.arange(1, 9)[:, None]
_STICKY_TAIL = (0.1 ** np.arange(8))[:, None]


def policy(obs, rng):
    board = np.asarray(obs).reshape(10, 10, 6)

    player = int(np.argmax(board[:, :, 0]))
    py, px = divmod(player, 10)
    py = min(8, max(1, py))
    p = (py - 1) * 10 + px

    lanes = board[1:9]
    occupied = lanes[:, :, 1] + lanes[:, :, 2]
    ex = occupied.argmax(axis=1)
    exists = occupied[_ROWS, ex] > 0.5
    enemy = lanes[_ROWS, ex, 1] > 0.5
    gold = lanes[_ROWS, ex, 2] > 0.5

    left_trail = (
        (ex > 0)
        & (lanes[_ROWS, np.maximum(ex - 1, 0), 4] > 0.5)
    )
    right_trail = (
        (ex < 9)
        & (lanes[_ROWS, np.minimum(ex + 1, 9), 4] > 0.5)
    )
    direction = np.where(
        left_trail, 1,
        np.where(right_trail, -1, np.where(ex == 0, 1, -1))
    )

    progress = np.where(direction == 1, ex, 9 - ex)
    fresh = bool(np.any(exists & (progress == 0)))
    count = int(exists.sum())

    # Signed horizontal displacement from each entity.
    travel = (_COLS[None, :] - ex[:, None]) * direction[:, None]

    # Safety envelope: every possible position before/after movement,
    # allowing movement periods from one through five frames.
    st = _T[:6, None, None]
    robust = (
        enemy[None, :, None]
        & (travel[None, :, :] >= ((st - 1) // 5))
        & (travel[None, :, :] <= st)
    ).reshape(6, 80)

    if np.any(enemy):
        bad_dest = robust[:, _DEST]
        risk = np.zeros((80, 5), dtype=np.float64)

        # State includes the previous executed action. If q[a] is the
        # outcome of executing a, sticky control gives
        # 0.9 * min(q) + 0.1 * q[previous_action].
        for t in range(5, -1, -1):
            q = np.where(
                bad_dest[t],
                1.0,
                risk[_DEST, _ACTIONS],
            )
            risk = 0.9 * q.min(axis=1, keepdims=True) + 0.1 * q

        known_risk = q[p]
    else:
        known_risk = np.zeros(5, dtype=np.float64)

    # Unseen spawns matter particularly during a run of sticky repeats.
    # Empty lanes can spawn immediately. Gold lanes may become empty
    # after collection; enemy lanes must first clear the board.
    available = np.where(
        exists,
        np.where(gold, 2, 11 - progress),
        1,
    )

    empty_count = 8 - count
    spawn_scale = 0.25 if fresh else (1.0 if count >= 6 else 0.5)
    spawn_probability = spawn_scale / (
        3.0 * np.maximum(1, empty_count + exists.astype(np.int64))
    )

    paths = _REPEATED[:, p, :]
    path_rows = paths // 10
    path_x = paths % 10

    left_arrival = available[path_rows] + np.maximum(path_x, 1) - 1
    right_arrival = (
        available[path_rows] + np.maximum(9 - path_x, 1) - 1
    )

    possible_spawn_hits = (
        (_REPEAT_T >= left_arrival).astype(np.float64)
        + (_REPEAT_T >= right_arrival)
    )
    spawn_risk = np.sum(
        _STICKY_TAIL
        * spawn_probability[path_rows]
        * possible_spawn_hits,
        axis=0,
    )

    if np.any(gold):
        # Two interception models: movement every frame, and movement
        # every three frames with uniformly unknown phase.
        tt = _T[:, None, None]
        fast_contact = (
            (travel[None, :, :] >= tt - 1)
            & (travel[None, :, :] <= tt)
        )

        slow_contact = (
            (travel[None, None, :, :] >= _SLOW_PRE)
            & (travel[None, None, :, :] <= _SLOW_POST)
        ).mean(axis=1)

        rewards = np.stack(
            (
                fast_contact * gold[None, :, None],
                slow_contact * gold[None, :, None],
            ),
            axis=1,
        ).reshape(9, 2, 80)

        dangers = np.stack(
            (
                fast_contact & enemy[None, :, None],
                (slow_contact > 0) & enemy[None, :, None],
            ),
            axis=1,
        ).reshape(9, 2, 80)

        reward_at_dest = rewards[:, :, _DEST]
        danger_at_dest = dangers[:, :, _DEST]
        value = np.zeros((2, 80, 5), dtype=np.float64)

        # Treasure is an absorbing success, preventing repeated credit
        # for waiting on the same predicted treasure.
        for t in range(8, -1, -1):
            g = reward_at_dest[t]
            continuation = value[:, _DEST, _ACTIONS]
            q = g + (1.0 - g) * 0.95 * continuation
            q[danger_at_dest[t]] = 0.0
            value = (
                0.9 * q.max(axis=2, keepdims=True)
                + 0.1 * q
            )

        fast_weight = 0.15 if fresh else (0.85 if count >= 6 else 0.65)
        gain = (
            fast_weight * q[0, p]
            + (1.0 - fast_weight) * q[1, p]
        )
    else:
        gain = np.zeros(5, dtype=np.float64)

    scores = (
        gain
        - 1000.0 * (known_risk + spawn_risk)
        - _CENTER_COST[_DEST[p]]
    )

    # Never deliberately enter a known immediate collision if a
    # collision-free action is available.
    immediately_bad = robust[0, _DEST[p]]
    if not np.all(immediately_bad):
        scores[immediately_bad] = -1.0e12

    return int(np.argmax(scores))
```