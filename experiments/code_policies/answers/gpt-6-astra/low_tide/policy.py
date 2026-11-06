import heapq
import numpy as np


# Immutable geometry and heuristic tables.
_STEP = tuple(
    (
        p,
        p - 10 if p >= 10 else p,
        p + 10 if p < 90 else p,
        p - 1 if p % 10 else p,
        p + 1 if p % 10 < 9 else p,
    )
    for p in range(100)
)

_NEIGH = tuple(
    tuple(q for q in moves[1:] if q != p)
    for p, moves in enumerate(_STEP)
)

_DIST = tuple(
    tuple(
        abs(p // 10 - q // 10) + abs(p % 10 - q % 10)
        for q in range(100)
    )
    for p in range(100)
)

_DISCOUNT = tuple(0.93 ** t for t in range(64))
_HINT_WEIGHTS = (0.85, 0.35, 0.15)
_OVERSHOOT_COST = (0.16, 0.016, 0.0016)

_FRONT_VALUE = tuple(
    tuple(
        0.18 * 0.90 ** (
            2.0 * abs(p // 10 - row)
            + 0.65 * abs(p % 10 - 4.5)
        )
        for p in range(100)
    )
    for row in range(10)
)


def _escape_action(pos, jelly, latest, dry_end):
    """Best emergency direction, also useful for setting actions during stun."""
    best_action = 0
    best_key = (-1000000, -1000000, -1000000)

    for action, q in enumerate(_STEP[pos]):
        if q >= dry_end:
            continue
        if action and q == pos:
            continue

        key = (
            latest[q] - (3 if jelly[q] else 0),
            -int(bool(jelly[q])),
            -q // 10,
        )
        if key > best_key:
            best_key = key
            best_action = action

    return best_action


def policy(obs, rng):
    pos = int(np.argmax(obs[:100]))
    level = int(obs[100:200:10].sum())
    bottom = 9 - level
    dry_end = (bottom + 1) * 10

    rising = bool(obs[400] > 0.5)
    countdown = max(1, min(5, int(float(obs[401]) * 5 + 0.5)))
    stun = int(float(obs[402]) * 3 + 0.5)

    jelly = obs[300:400].tolist()
    shells = np.flatnonzero(obs[200:300] > 0.5).tolist()

    # Earliest possible flooding, assuming three-step future tide intervals.
    # Occupancy limits leave four steps before a cell floods.
    row_limit = [-10000] * 10
    row_limit[0:3] = [10000, 10000, 10000]

    low_target = max(0, min(2, level - 1))
    for row in range(3, bottom + 1):
        if rising:
            flood = countdown + 3 * (bottom - row)
        else:
            flood = countdown + 3 * (
                level + 9 - row - 2 * low_target
            )
        row_limit[row] = flood - 4

    # Backward latest-arrival search: every accepted cell must retain an
    # escape path to permanently dry land, without intentionally stinging.
    latest = [-10000] * 100
    latest[:30] = [10000] * 30
    heap = [(-10000, p) for p in range(20, 30)]
    heapq.heapify(heap)

    while heap:
        negative_value, p = heapq.heappop(heap)
        value = -negative_value
        if value != latest[p]:
            continue

        for q in _NEIGH[p]:
            if q >= dry_end or jelly[q]:
                continue
            candidate = min(row_limit[q // 10], value - 1)
            if candidate > latest[q]:
                latest[q] = candidate
                heapq.heappush(heap, (-candidate, q))

    # Escape values after an accidental jellyfish encounter has finished.
    for p in range(30, dry_end):
        if jelly[p]:
            escape = -10000
            for q in _NEIGH[p]:
                if q < dry_end and not jelly[q]:
                    escape = max(escape, latest[q] - 1)
            latest[p] = min(row_limit[p // 10], escape)

    emergency = _escape_action(pos, jelly, latest, dry_end)
    if stun:
        return int(emergency)

    # Each transition includes a deadline valid even if its direction is
    # repeated several times before the next intended turn succeeds.
    moves = [[] for _ in range(100)]
    for p in range(dry_end):
        if jelly[p]:
            continue

        options = [(0, p, latest[p], 0.0)]
        for action in range(1, 5):
            q = _STEP[p][action]
            if q == p or q >= dry_end or jelly[q]:
                continue

            bound = latest[q]
            penalty = 0.0
            z = q

            for k in range(3):
                zz = _STEP[z][action]
                if zz == z or zz >= dry_end:
                    break
                if jelly[zz]:
                    bound = min(bound, latest[zz] - 3)
                    penalty = _OVERSHOOT_COST[k]
                    break
                bound = min(bound, latest[zz])
                z = zz

            # Near a possible low-tide reversal, an accidental downward
            # continuation could enter newly spawned, previously unseen jelly.
            if (
                not rising
                and level <= 3
                and action == 2
                and q // 10 >= bottom - 1
            ):
                bound = min(bound, countdown - 2)

            options.append((action, q, bound, penalty))
        moves[p] = options

    # Earliest safe reachability. Among equally short paths prefer less
    # exposure to accidental stings.
    distance = [-1] * 100
    path_risk = [10000.0] * 100
    first_action = [0] * 100
    distance[pos] = 0
    path_risk[pos] = 0.0
    reachable = [pos]

    for p in reachable:
        arrival = distance[p] + 1
        for action, q, bound, penalty in moves[p]:
            if q == p or arrival > bound:
                continue

            risk = path_risk[p] + penalty
            first = action if p == pos else first_action[p]

            if distance[q] < 0:
                distance[q] = arrival
                path_risk[q] = risk
                first_action[q] = first
                reachable.append(q)
            elif distance[q] == arrival and risk < path_risk[q]:
                path_risk[q] = risk
                first_action[q] = first

    goals = [p for p in shells if distance[p] >= 0 and p != pos]

    if len(goals) == 1:
        goal = goals[0]
        if rising or level <= 1 or distance[goal] <= 2:
            return int(first_action[goal])

    # Preferred waiting shoreline at each planning time.
    max_time = max(12, max(distance))
    frontier = []
    for t in range(max_time + 1):
        row = bottom
        while row > 2 and row_limit[row] < t + 2:
            row -= 1
        frontier.append(row)

    if not goals:
        best_cost = float("inf")
        answer = emergency

        for p in reachable:
            t = distance[p]
            if latest[p] < t + 1:
                continue

            cost = (
                2.0 * abs(p // 10 - frontier[t])
                + 0.65 * abs(p % 10 - 4.5)
                + 0.12 * t
                + 0.8 * path_risk[p]
            )
            if cost < best_cost:
                best_cost = cost
                answer = first_action[p] if t else 0

        return int(answer)

    bit_at = [0] * 100
    for i, goal in enumerate(goals):
        bit_at[goal] = 1 << i

    # Optimistic remaining-shell hints, with collection deadlines.
    near = [[] for _ in range(100)]
    for p in reachable:
        ordered = sorted(goals, key=lambda g: _DIST[p][g])
        near[p] = [
            (
                bit_at[g],
                _DISCOUNT[_DIST[p][g]],
                latest[g] - _DIST[p][g],
            )
            for g in ordered
        ]

    # One-time expected value of being ready for the next uncovered row.
    # Its value is reduced when that row might immediately start flooding.
    fresh_value = [0.0] * 10
    if not rising:
        if level >= 4:
            amount = 0.85
        elif level == 3:
            amount = 0.60
        elif level == 2:
            amount = 0.475
        else:
            amount = 0.10

        fresh_value = [
            amount * 0.78 ** max(0, bottom - row)
            for row in range(10)
        ]

    nearest_distance = min(distance[g] for g in goals)
    horizon = 12 if len(goals) >= 6 or nearest_distance > 6 else 10
    width = 16

    # State: (ranking value, collected reward value, position, mask, first).
    beam = [(0.0, 0.0, pos, 0, 0)]
    answer = emergency

    for t in range(1, horizon + 1):
        discount = _DISCOUNT[t - 1]
        spawn_now = not rising and t == countdown
        pool = {}

        for _, value, p, mask, first in beam:
            for action, q, bound, penalty in moves[p]:
                if t > bound:
                    continue

                bit = bit_at[q]
                gain = 1.0 if bit and not (mask & bit) else 0.0
                if spawn_now:
                    gain += fresh_value[q // 10]

                new_value = value + discount * (gain - penalty)
                new_mask = mask | bit
                new_first = action if t == 1 else first
                key = (new_mask << 7) | q

                old = pool.get(key)
                if old is None or new_value > old[0]:
                    pool[key] = (new_value, q, new_mask, new_first)

        if not pool:
            break

        candidates = []
        front_values = _FRONT_VALUE[frontier[t]]

        for value, p, mask, first in pool.values():
            hint = front_values[p]
            count = 0

            for bit, proximity, deadline in near[p]:
                if mask & bit or t > deadline:
                    continue
                hint += _HINT_WEIGHTS[count] * proximity
                count += 1
                if count == 3:
                    break

            rank = value + _DISCOUNT[t] * hint
            candidates.append((rank, value, p, mask, first))

        candidates.sort(reverse=True)
        beam = candidates[:width]
        answer = beam[0][4]

    return int(answer)
