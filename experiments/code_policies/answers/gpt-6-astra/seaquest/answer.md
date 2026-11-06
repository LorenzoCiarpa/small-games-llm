The policy decodes the merged gauges conservatively, plans diver interceptions and firing positions, and uses a short-horizon collision model to select actions. It accounts for uncertain movement directions, incoming bullets, oxygen, and the risk of repeated sticky actions.

```python
import numpy as np


# Fixed movement tables. States include position and horizontal orientation.
_P = np.arange(90, dtype=np.intp)
_X = _P % 10
_Y = _P // 10
_MOVE = np.stack((
    _P,
    10 * _Y + np.maximum(_X - 1, 0),
    10 * np.maximum(_Y - 1, 0) + _X,
    10 * _Y + np.minimum(_X + 1, 9),
    10 * np.minimum(_Y + 1, 8) + _X,
))

_S = np.arange(180, dtype=np.intp)
_SP = _S % 90
_OR = (_S // 90) * 90
_NEXT = np.stack((
    _S,
    _MOVE[1, _SP],
    _MOVE[2, _SP] + _OR,
    _MOVE[3, _SP] + 90,
    _MOVE[4, _SP] + _OR,
))
_NPOS = _NEXT % 90
_REPEAT = np.stack([_NEXT[a, _NEXT[a]] % 90 for a in range(5)])
_EDGE = ((_X == 0) | (_X == 9)) & (_Y > 0)


def _direction(p, trail):
    x = p % 10
    left = x > 0 and bool(trail[p - 1])
    right = x < 9 and bool(trail[p + 1])
    if left != right:
        return 1 if left else -1
    if x == 0:
        return 1
    if x == 9:
        return -1
    return 0


def policy(obs, rng):
    z = np.asarray(obs).reshape(100, 6) > 0.5
    players = np.flatnonzero(z[:90, 0])
    if len(players) == 0:
        return 0

    danger = z[:90, 1]
    reward = z[:90, 2]
    trail = z[:90, 4]
    bodies = [int(p) for p in np.flatnonzero(danger & reward)]
    divers = [int(p) for p in np.flatnonzero(reward & ~danger)]
    bullets = [int(p) for p in np.flatnonzero(danger & ~reward)]
    friendly = [int(p) for p in np.flatnonzero(z[:90, 3])]

    # Usually the back is the only player cell carrying a trail.
    ambiguous = False
    if len(players) == 1:
        p = int(players[0])
        other = p
        facing = -1
    else:
        a, b = int(players[0]), int(players[1])
        if trail[a] != trail[b]:
            p = b if trail[a] else a
        elif danger[a] != danger[b]:
            p = b if danger[a] else a
        else:
            ambiguous = True
            p = a
            row = a // 10
            fs = [q % 10 for q in friendly if q // 10 == row]
            if fs and min(fs) > b % 10:
                p = b
            elif not (fs and max(fs) < a % 10):
                def support(q):
                    x = q % 10
                    return (
                        int(x > 0 and reward[q - 1]) +
                        int(x < 9 and reward[q + 1])
                    )
                if support(b) > support(a):
                    p = b
        other = b if p == a else a
        facing = 1 if p % 10 > other % 10 else -1
        if abs(p % 10 - other % 10) == 9:
            facing = 1 if p % 10 == 0 else -1

    px, py = p % 10, p // 10
    state = p + (90 if facing == 1 else 0)
    alternate = (
        other + (90 if facing == -1 else 0)
        if ambiguous else state
    )

    # A gap separates the oxygen prefix from the diver suffix.
    gauge = z[90:100, 5]
    prefix = 0
    while prefix < 10 and gauge[prefix]:
        prefix += 1

    if prefix < 9:
        count = 0
        j = 8
        while j >= 0 and gauge[j]:
            count += 1
            j -= 1
        count = min(count, 6)
        oxygen = 20 * prefix + 9
    else:
        count = -1                  # Unknown, not necessarily empty.
        oxygen = 120

    full_here = (
        not ambiguous and bool(reward[p]) and not bool(danger[p])
    )
    has_diver = count > 0 or full_here
    crowded = len(bodies) + len(bullets) >= 12
    returning = (
        py > 0 and (
            (count > 0 and prefix <= 1) or
            (count > 0 and prefix <= 2 and py >= 6 and crowded) or
            count == 6 or
            (full_here and py == 1)
        )
    )
    urgent = count == 0 and prefix <= 2

    if py == 0 and not bodies and not divers:
        return 0

    # Enemies in a row normally share a direction. Pool trail evidence.
    votes = [0] * 9
    raw_dirs = []
    for q in bodies:
        d = _direction(q, trail)
        raw_dirs.append(d)
        votes[q // 10] += d
    row_dir = [(v > 0) - (v < 0) for v in votes]
    enemies = []
    for q, d in zip(bodies, raw_dirs):
        d = row_dir[q // 10] or d
        enemies.append((q % 10, q // 10, d, q))

    # Estimate which enemies already have a friendly bullet approaching.
    covered = [0.0] * len(enemies)
    for q in friendly:
        bx, by = q % 10, q // 10
        same_row = [(i, e) for i, e in enumerate(enemies) if e[1] == by]
        if not same_row:
            continue
        if by == py and bx != px:
            bd = 1 if bx > px else -1
            confidence = 0.92
        else:
            _, nearest = min(same_row, key=lambda ie: abs(ie[1][0] - bx))
            bd = 1 if nearest[0] > bx else -1
            confidence = 0.82

        best_i = -1
        best_distance = 100
        for i, e in same_row:
            distance = bd * (e[0] - bx)
            if 0 <= distance < best_distance:
                best_distance = distance
                best_i = i
        if best_i >= 0:
            covered[best_i] = max(covered[best_i], confidence)

    # Collision probabilities for the next three updates. Body movement
    # phases and enemy types are hidden, so retain conservative alternatives.
    risk = np.zeros((3, 90), dtype=np.float64)
    forward = (
        (0.32, 0.008),
        (0.58, 0.042, 0.008),
        (0.82, 0.055, 0.040, 0.008),
    )
    for ex, ey, d, q in enemies:
        risk[0, q] += 1.0
        risk[1, q] += 0.8
        risk[2, q] += 0.6
        directions = ((d, 1.0),) if d else ((-1, 0.5), (1, 0.5))
        for dd, probability in directions:
            for t, weights in enumerate(forward):
                for k, weight in enumerate(weights, 1):
                    xx = ex + dd * k
                    if 0 <= xx < 10:
                        risk[t, ey * 10 + xx] += probability * weight

    for q in bullets:
        bx, by = q % 10, q // 10
        d = row_dir[by]
        directions = (
            ((d, 0.97), (-d, 0.03))
            if d else ((-1, 0.5), (1, 0.5))
        )
        risk[0, q] += 1.0
        for dd, probability in directions:
            for t in range(3):
                if t:
                    xx = bx + dd * t
                    if 0 <= xx < 10:
                        risk[t, by * 10 + xx] += probability
                xx = bx + dd * (t + 1)
                if 0 <= xx < 10:
                    risk[t, by * 10 + xx] += probability

    np.minimum(risk, 1.0, out=risk)
    hazard = 80.0 * risk
    edge_cost = min(1.0, 0.35 + 0.035 * max(0, len(bodies) - 4))
    hazard[:, _EDGE] += edge_cost

    if py > 0 and not has_diver:
        hazard[:, :10] += 1000.0
    elif py > 0 and not returning:
        hazard[:, :10] += 0.45

    step = 0.25 if returning else (0.19 if urgent else 0.085)
    travel = step + np.minimum(1.8, 1.4 * risk[0] + 0.5 * risk[1])
    travel[_EDGE] += 0.15
    if py > 0 and not has_diver:
        travel[:10] += 1000.0
    elif py > 0 and not returning:
        travel[:10] += 0.6

    # Rewards seed a shortest-path potential over position AND orientation.
    goals = np.full(180, -100.0, dtype=np.float64)
    if returning:
        goals[1:9] = 4.0
        goals[91:99] = 4.0
    else:
        for q in (44, 45):
            goals[q] = goals[q + 90] = 0.16

        if count == 0:
            diver_value = 4.3 if urgent else 2.65
        elif count == 1:
            diver_value = 1.95
        elif count == 2:
            diver_value = 1.25
        elif count >= 3:
            diver_value = 0.65
        else:
            diver_value = 1.20

        if not full_here and count != 6:
            for q in divers:
                dx, dy = q % 10, q // 10
                dd = _direction(q, trail)
                directions = (dd,) if dd else (-1, 1)
                for d in directions:
                    # Candidate interception cells, including waiting time.
                    for k in range(5):
                        xx = dx + d * k
                        if not 0 <= xx < 10:
                            break
                        arrival = abs(px - xx) + abs(py - dy)
                        leaves = 6 * k + 3
                        if arrival > leaves + 2:
                            continue
                        enters = max(0, 6 * k - 3)
                        wait = max(0, enters - arrival)
                        value = (
                            diver_value - step * wait - 0.035 * k
                            - 0.04 * max(0, arrival - leaves)
                        )
                        if urgent:
                            value -= 0.025 * dy
                            if arrival + wait + dy + 3 > oxygen:
                                value -= 1.0
                        target = dy * 10 + xx
                        if value > goals[target]:
                            goals[target] = value
                        if value > goals[target + 90]:
                            goals[target + 90] = value

        # Only correctly oriented positions receive firing-goal value.
        for i, (ex, ey, d, q) in enumerate(enemies):
            base = 1.65 * (1.0 - covered[i])
            if base < 0.12:
                continue
            remaining = (9 - ex if d == 1 else ex) if d else 5
            lifetime = 4.0 * remaining + 2.3
            for xx in range(1, 9):
                if xx == ex:
                    continue
                distance = abs(ex - xx)
                shooting_dir = 1 if ex > xx else -1
                in_front = d == 0 or (xx - ex) * d > 0
                if in_front and distance < 3:
                    continue

                setup = abs(px - xx) + abs(py - ey)
                flight = distance / (1.0 - 0.23 * shooting_dir * d)
                chance = min(1.0, lifetime / (setup + flight + 0.5))
                value = base * chance - 0.025 * distance
                target = ey * 10 + xx + (90 if shooting_dir == 1 else 0)
                if value > goals[target]:
                    goals[target] = value

    # Approximate weighted shortest paths, including the cost of unsafe
    # follow-through after a horizontal/vertical movement.
    path_cost = travel[_NPOS]
    path_cost = path_cost + 0.20 * np.minimum(1.5, 2.0 * risk[1, _REPEAT])
    field = goals.copy()
    for _ in range(18):
        propagated = np.max(
            field[_NEXT[1:]] - path_cost[1:], axis=0
        )
        field = np.maximum(goals, propagated)

    # Short-horizon escape planning. Mixing in the path potential at each
    # stage prevents waiting merely because a goal is within the horizon.
    value = field
    for t in (2, 1):
        continuation = np.max(
            value[_NEXT] - hazard[t, _NPOS], axis=0
        )
        value = 0.45 * field + 0.55 * continuation
    endpoint = 0.45 * field + 0.55 * value

    scores = []
    for action in range(6):
        movement = action if action < 5 else 0
        ns = int(_NEXT[movement, state])
        alt_ns = int(_NEXT[movement, alternate])
        np1, np2 = ns % 90, alt_ns % 90
        repeat1 = int(_REPEAT[movement, state])
        repeat2 = int(_REPEAT[movement, alternate])

        now_risk = max(hazard[0, np1], hazard[0, np2])
        repeat_risk = max(hazard[1, repeat1], hazard[1, repeat2])

        # Picking up a row-one diver makes a repeated upward action safe.
        if (
            action == 2 and py == 2 and not ambiguous
            and np1 in divers and not has_diver
        ):
            repeat_risk = 0.0

        score = (
            0.5 * (endpoint[ns] + endpoint[alt_ns])
            - now_risk - 0.10 * repeat_risk
        )
        if action == 0:
            score -= 0.012
        elif action == 5:
            score -= 0.035
        else:
            score -= 0.003
        scores.append(float(score))

    # Fire only along a useful ray; bullets already committed to a target
    # suppress redundant shots. Cooldown is estimated from nearby bullets.
    shot_target = -1
    nearest = 100
    for i, (ex, ey, d, q) in enumerate(enemies):
        distance = facing * (ex - px)
        if ey == py and 0 < distance < nearest and covered[i] < 0.75:
            nearest = distance
            shot_target = i

    if shot_target >= 0:
        ex, ey, d, q = enemies[shot_target]
        remaining = (9 - ex if d == 1 else ex) if d else 5
        lifetime = 4.0 * remaining + 2.3
        flight = nearest / (1.0 - 0.23 * facing * d)
        hit_chance = min(1.0, lifetime / (flight + 0.5))
        readiness = 1.0
        if any(
            abs(q % 10 - px) + abs(q // 10 - py) <= 3
            for q in friendly
        ):
            readiness = 0.45
        if ambiguous:
            readiness *= 0.1
        shot_value = 0.04 if returning else (0.10 if urgent else 0.68)
        scores[5] += (
            shot_value * readiness * hit_chance
            * (1.0 - covered[shot_target])
        )

    return int(max(range(6), key=scores.__getitem__))
```