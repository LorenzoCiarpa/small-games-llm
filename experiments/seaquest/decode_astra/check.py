"""
Checks the hand reading of weights/seaquest/paper_astra.json (see README.md).

Plays the same episodes as `uv run evaluate seaquest paper_astra` (seeds 10000.., sticky 0.1, actions sampled
from the softmax with the same generator). In every state it recomputes, directly from the game state (positions of
the submarine, fish, enemy submarines, enemy torpedoes, divers, oxygen) and without using the weights:

  * the 32 neurons of the first layer, with the formulas of section 4 of the README;
  * the 6 logits, with the formulas of section 6 (value of the rows);
  * the action chosen by the simplified rule of section 7 (only true-or-false conditions);

and compares them with the real forward pass. Then it summarizes the behaviour (rows, deaths, points).

    uv run python experiments/seaquest/decode_astra/check.py [--episodes 200]      # ~1.5 min
    uv run python experiments/seaquest/decode_astra/check.py --ablations          # families of neurons switched off
    uv run python experiments/seaquest/decode_astra/check.py --examples           # the situations of section 6
"""

import argparse
import collections
from pathlib import Path

import numpy as np

from small_games.core import get_game
from small_games.policy import policy_rng, sample_action
from small_games.weights import load_weights

ROOT = Path(__file__).resolve().parents[3]
ACTIONS = ("no-op", "left", "up", "right", "down", "fire")
NOOP, LEFT, UP, RIGHT, DOWN, FIRE = range(6)
T = np.tanh

# First layer, "nearby danger" (h16-h23): contribution of one object, by column 0..9
NEAR_ENEMY = np.array([0, 0, 0.2, 0.4, 6, 7, 6, 2, 0.2, 0])         # fish or enemy submarine
NEAR_BULLET = np.array([0, 0, 2, 4, 7, 7, 7, 5, 2, 0])              # enemy torpedo
NEAR_DIVER = np.array([0, 0, -1.8, -3.6, -1, 0, -1, -3, -1.8, 0])   # diver to rescue
# Second layer: attraction by distance in rows d = |row of the object − arrival row|
A = [0.6, 0.45, 0.34, 0.255, 0.19, 0.145, 0.11, 0.08]               # divers          (≈ 0.6·0.75^d)
C = [0.35, 0.12, 0.04, 0.01]                                        # left targets    (≈ 0.35·3^−d)


def forward(layers, obs):
    x, out = np.asarray(obs, dtype=np.float64), []
    for W, b, act in layers:
        x = W @ x + b
        if act == "tanh":
            x = np.tanh(x)
        out.append(x)
    return out


def grid(g):
    """The cells occupied by enemies (fish and submarines), enemy torpedoes and divers, from the game state."""
    enemy, bullet, diver = (np.zeros((10, 10), bool) for _ in range(3))
    for x, y, *_ in g.e_fish + g.e_subs:
        enemy[y, x] = True
    for x, y, *_ in g.e_bullets:
        bullet[y, x] = True
    for x, y, *_ in g.divers:
        diver[y, x] = True
    return enemy, bullet, diver


def layer1(g):
    """The first layer rebuilt from the game state (section 4): the vector of the 32 neurons."""
    enemy, bullet, diver = grid(g)
    danger, reward = enemy | bullet, enemy | diver          # the two channels of the observation (they are ORs)
    h = np.empty(32)
    for r in range(9):                                      # h0-h8: the submarine is in row r
        h[r] = T(6.0) if g.sub_y == r else T(-6.0)
    for j in range(2, 9):                                   # h9-h15: row j has more divers than torpedoes
        n = (reward[j] & ~danger[j]).sum() - (danger[j] & ~reward[j]).sum()
        h[9 + j - 2] = T(8 * n - 4)
    for j in range(1, 9):                                   # h16-h23: danger near columns 4-6, row j
        both, only_b, only_d = danger[j] & reward[j], danger[j] & ~reward[j], reward[j] & ~danger[j]
        s = NEAR_ENEMY[both].sum() + NEAR_BULLET[only_b].sum() + NEAR_DIVER[only_d].sum()
        h[16 + j - 1] = T(s - 3)
    o2_cell2 = g.oxygen * 10 // 200 >= 3                    # h24: oxygen < 60 and at least one diver on board
    cell8 = g.diver_count >= 1 or g.oxygen * 10 // 200 >= 9
    h[24] = T(6 * cell8 - 6 * o2_cell2 - 3)
    for j in range(2, 9):                                   # h25-h31: something in columns 0-3, row j
        h[25 + j - 2] = T(6 * danger[j, :4].sum() - 3)
    return h


def unpack(h):
    """The neurons of the first layer as fractions x = (h + 1) / 2 in [0, 1], indexed by row."""
    x = (h + 1) / 2
    div = {j: x[9 + j - 2] for j in range(2, 9)}
    near = {j: x[16 + j - 1] for j in range(1, 9)}
    left = {j: x[25 + j - 2] for j in range(2, 9)}
    return div, near, left, x[24]


def value(k, div, left, near):
    """Value of row k (section 5): attraction of divers and targets, minus the nearby danger in that row."""
    v = -0.70
    v += 2 * sum(A[abs(j - k)] * div[j] for j in div)
    v += 2 * sum(C[abs(j - k)] * left[j] for j in left if abs(j - k) < len(C))
    v -= 6 * near.get(k, 0.0)
    return v


def logits(g, h):
    """The 6 logits with the formulas of section 6, starting from the rebuilt first layer."""
    div, near, left, low = unpack(h)
    r = g.sub_y
    lo = 2 * low - 1                                        # [low oxygen and divers on board] as ±1
    if r == 0:                                              # at the surface
        fire = 1.8
    else:                                                   # stays in row r (and fires)
        fire = 31.8 + 30 * T(value(r, div, left, near))
    if r == 0:
        up = 0.0
    elif r == 1:
        up = 28 + 40 * T(6 * lo) + 12 * T(3 * lo)          # 80 if it has to surface, −24 otherwise
    elif r == 2:                                            # toward row 1: forbidden unless it has to surface
        up = 18 + 12 * T(3 * lo) + 30 * T(value(1, div, left, near) - 8.0 + 9.5 * low)
    else:                                                   # toward row r − 1
        up = 30 + 30 * T(value(r - 1, div, left, near) - 0.6 * near.get(r - 2, 0.0) + 1.5 * low)
    if r == 8:
        down = 0.0
    else:                                                   # toward row r + 1
        down = 30 + 30 * T(value(r + 1, div, left, near) - 0.6 * near.get(r + 2, 0.0))
        if r == 1:
            down += 4 - 4 * T(3 * lo)
    return np.array([-100, -100, up, -100, down, fire], dtype=np.float64)


def rule(g):
    """The simplified rule of section 7: only true-or-false conditions, no weights."""
    enemy, bullet, diver = grid(g)
    r = g.sub_y
    low = g.oxygen < 60 and g.diver_count >= 1
    danger = {j for j in range(1, 9) if enemy[j, 4:7].any() or bullet[j, 3:8].any()}
    weak = {j for j in range(1, 9) if enemy[j, 7] or bullet[j, 2] or bullet[j, 8]} - danger
    if r == 0:
        return FIRE if 1 in danger else DOWN               # waits at the surface, then dives
    if r == 1 and low:
        return UP                                          # surfaces

    rows_div = [j for j in range(2, 9) if diver[j].any()]
    rows_tgt = [j for j in range(2, 9) if (enemy[j, :4] | bullet[j, :4]).any()]

    def val(k):
        v = sum(1.2 * 0.75 ** abs(j - k) for j in rows_div)
        v += sum(0.7 * 3.0 ** -abs(j - k) for j in rows_tgt if abs(j - k) <= 3)
        return v - 6 * (k in danger) - 0.7 * (k in weak)

    cand = {FIRE: val(r)}
    if r >= 3 or low:
        cand[UP] = val(r - 1) - 0.6 * (r - 2 in danger) + 1.5 * low
    if r <= 7:
        cand[DOWN] = val(r + 1) - 0.6 * (r + 2 in danger)
    best = max(cand.values())
    if r == 1 and cand[DOWN] >= best - 0.3:                # from row 1, going down wins the ties
        return DOWN
    return FIRE if cand[FIRE] >= best - 0.1 else max(cand, key=cand.get)


def death_cause(g, prev, where, sticky, h_before, h_now, lg_now):
    """Why the episode ended: (what hit the submarine, how it ended up there).

    h_now and lg_now are the network's first layer and logits in the last state, h_before those of the step before."""
    y0, dc0, _, o0 = prev
    if g.oxygen < 0 or o0 <= 0:                             # even if it surfaces right at the last step
        return "", "oxygen ran out"
    if g.sub_y == 0 and dc0 == 0:
        return "", "sticky: surfaced with no divers on board" if sticky else "surfaced with no divers on board"
    sub = (g.sub_x, g.sub_y)
    who = "?"
    for kind, objs in (("enemy", g.e_fish + g.e_subs), ("enemy torpedo", g.e_bullets)):
        for o in objs:                                      # after the collision an object can still take one step
            if (o[0], o[1]) == sub or where.get(id(o)) == sub:
                who = f"{kind} from the {'left' if o[2] else 'right'}"
    near = 16 + g.sub_y - 1                                 # the "nearby danger" neuron of the fatal row
    if sticky:
        if g.sub_y == y0:
            return who, "sticky: held still"
        seen = h_before is not None and h_before[near] > 0
        return who, "sticky: moved, danger already on" if seen else "sticky: moved, danger not on"
    if y0 == 0:
        return who, "dive from the surface (softmax)"
    if h_now[near] <= 0:                                   # was there also a diver in the row (h9-h15)?
        diver = 2 <= g.sub_y <= 8 and h_now[9 + g.sub_y - 2] > 0
        return who, "danger not on, diver in the row" if diver else "danger not on"
    if max(lg_now[UP], lg_now[DOWN], lg_now[FIRE]) < 10:   # a safe move is worth at least 11.7
        return who, "trapped: no safe move"
    return who, "other"


def play(layers, episodes, seed0, verify=True):
    env = get_game("seaquest").make_env()
    st, by_row, deaths, surf = (collections.Counter() for _ in range(4))
    max_diff = 0.0
    scores = []
    for ep in range(episodes):
        seed = seed0 + ep
        rng = policy_rng(seed)
        obs, info = env.reset(seed=seed)
        g = env.env.env                                     # the MinAtar game, with its state
        h_before, done = None, False
        while not done:
            h_true, _, logit_true = forward(layers, obs)
            st["n"] += 1
            st["col5"] += g.sub_x == 5 and not g.sub_or
            a_true = int(np.argmax(logit_true))
            by_row[g.sub_y, ACTIONS[a_true]] += 1
            if verify:
                h_mine = layer1(g)
                st["bad_h"] += not np.allclose(h_true, h_mine, atol=1e-9)
                st["bad_onoff"] += not np.array_equal(h_true > 0, h_mine > 0)
                lg = logits(g, h_mine)
                max_diff = max(max_diff, float(np.abs(lg - logit_true).max()))
                top2 = np.sort(lg)[-2:]
                if top2[1] - top2[0] <= 0.5:
                    st["ties"] += 1
                elif np.argmax(lg) != a_true:
                    st["bad_arg"] += 1
                a_rule = rule(g)
                st["rule_same"] += a_rule == a_true
                top2 = np.sort(logit_true)[-2:]
                if top2[1] - top2[0] > 3:
                    st["clear"] += 1
                    st["rule_same_clear"] += a_rule == a_true

            low = g.oxygen < 60 and g.diver_count >= 1
            st["low"] += low
            st["low_up"] += low and a_true == UP
            st["div_spawn"] += g.d_spawn_timer == 0         # a diver appears at this step
            st["div_seen"] += sum(1 for d in g.divers if 2 <= d[1] <= 8)
            st["div_away"] += sum(1 for d in g.divers if 2 <= d[1] <= 8 and (d[0] < 5) != d[2] and d[0] != 5)
            prev = (g.sub_y, g.diver_count, g.surface, g.oxygen)
            where = {id(o): (o[0], o[1]) for o in g.e_fish + g.e_subs + g.e_bullets}
            a = sample_action(logit_true, rng)
            st["not_argmax"] += a != a_true
            obs, rwd, term, trunc, info = env.step(a)
            if g.sub_y > 0:
                st["div_taken"] += max(0, g.diver_count - prev[1])
            elif not prev[2]:                               # it has just surfaced
                surf[min(prev[1], 6)] += 1
                if prev[1] == 6:
                    st["bonus"] += prev[3] * 10 // 200
            if term:
                cause = death_cause(g, prev, where, env.last_action != a, h_before, h_true, logit_true)
                deaths[cause] += 1
                st["trap_edge"] += cause[1].startswith("trapped") and prev[0] in (1, 2, 8)  # starting row
            h_before = h_true
            done = term or trunc
        scores.append(info["score"])
        st["steps"] += info["t"]
    return st, by_row, deaths, surf, max_diff, np.array(scores)


def pinned_off(layers, units):
    """Copy of the network with the first-layer neurons in `units` fixed at −1 (off)."""
    (W1, b1, a1), *rest = layers
    W1, b1 = W1.copy(), b1.copy()
    W1[units] = 0
    b1[units] = -50
    return [(W1, b1, a1), *rest]


def observation(sub_row, fish=(), bullets=(), divers=(), oxygen=200, divers_on_board=0):
    """A hand-built observation: submarine in column 5 facing left, objects at (row, column)."""
    o = np.zeros(600)

    def put(r, c, ch):
        o[(r * 10 + c) * 6 + ch] = 1

    put(sub_row, 5, 0), put(sub_row, 6, 0), put(sub_row, 6, 4)     # bow, stern, trail of the stern
    for r, c in fish:
        put(r, c, 1), put(r, c, 2)
    for r, c in bullets:
        put(r, c, 1)
    for r, c in divers:
        put(r, c, 2)
    for c in range(oxygen * 10 // 200):
        put(9, c, 5)
    for c in range(9 - divers_on_board, 9):
        put(9, c, 5)
    return o


EXAMPLES = [
    ("empty sea, row 5", dict(sub_row=5)),
    ("diver in my row", dict(sub_row=5, divers=[(5, 1)])),
    ("diver one row below", dict(sub_row=5, divers=[(6, 1)])),
    ("diver three rows below", dict(sub_row=5, divers=[(8, 1)])),
    ("left target in my row, (5,2)", dict(sub_row=5, fish=[(5, 2)])),
    ("left target one row above, (4,2)", dict(sub_row=5, fish=[(4, 2)])),
    ("enemy at (5,4), in front of the bow", dict(sub_row=5, fish=[(5, 4)])),
    ("enemy at (5,6), on the stern", dict(sub_row=5, fish=[(5, 6)])),
    ("enemy at (5,7)", dict(sub_row=5, fish=[(5, 7)])),
    ("enemy torpedo at (5,3)", dict(sub_row=5, bullets=[(5, 3)])),
    ("enemy torpedo at (5,8)", dict(sub_row=5, bullets=[(5, 8)])),
    ("enemy at (6,5), one row below", dict(sub_row=5, fish=[(6, 5)])),
    ("enemy at (7,5), two rows below", dict(sub_row=5, fish=[(7, 5)])),
    ("enemy at (6,6) and divers at (6,3), (6,8)", dict(sub_row=5, fish=[(6, 6)], divers=[(6, 3), (6, 8)])),
    ("diver (7,1), target (4,2), enemy (6,6)", dict(sub_row=5, fish=[(4, 2), (6, 6)], divers=[(7, 1)])),
    ("oxygen 50, 2 divers on board", dict(sub_row=5, oxygen=50, divers_on_board=2)),
    ("oxygen 50, no divers on board", dict(sub_row=5, oxygen=50)),
    ("oxygen 50, 2 divers, enemy at (4,5)", dict(sub_row=5, fish=[(4, 5)], oxygen=50, divers_on_board=2)),
    ("row 2", dict(sub_row=2)),
    ("row 1", dict(sub_row=1)),
    ("row 1, oxygen 50, 2 divers", dict(sub_row=1, oxygen=50, divers_on_board=2)),
    ("row 0 (surface)", dict(sub_row=0)),
    ("row 0, enemy at (1,5)", dict(sub_row=0, fish=[(1, 5)])),
    ("row 8", dict(sub_row=8)),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--ablations", action="store_true", help="score with some families of neurons switched off")
    ap.add_argument("--examples", action="store_true", help="logits of the real network in hand-built situations")
    args = ap.parse_args()

    layers = load_weights(ROOT / "weights/seaquest/paper_astra.json", get_game("seaquest"))
    eps = args.episodes

    if args.examples:
        print(f"{'situation (row 5 unless stated)':42s} {'up':>6s} {'down':>6s} {'fire':>6s}   p(up, down, fire)")
        for name, kw in EXAMPLES:
            lg = forward(layers, observation(**kw))[-1]
            p = np.exp(lg - lg.max())
            p /= p.sum()
            print(f"{name:42s} {lg[UP]:6.1f} {lg[DOWN]:6.1f} {lg[FIRE]:6.1f}   {p[UP]:.2f} {p[DOWN]:.2f} {p[FIRE]:.2f}")
        return

    if args.ablations:
        variants = {
            "full network": [],
            "without divers (h9-h15)": list(range(9, 16)),
            "without nearby danger (h16-h23)": list(range(16, 24)),
            "without surfacing (h24)": [24],
            "without left targets (h25-h31)": list(range(25, 32)),
            "without divers, targets and danger": list(range(9, 24)) + list(range(25, 32)),
        }
        for name, units in variants.items():
            st, _, deaths, surf, _, scores = play(pinned_off(layers, units), eps, args.seed0, verify=False)
            se = scores.std(ddof=1) / np.sqrt(len(scores))
            oxy = deaths["", "oxygen ran out"]
            print(f"{name:34s} {scores.mean():6.1f} ± {se:3.1f}   steps/episode {st['steps'] / eps:5.0f}"
                  f"   surfacings/episode {sum(surf.values()) / eps:4.1f}   deaths from oxygen {oxy / eps:4.0%}")
        return

    st, by_row, deaths, surf, max_diff, scores = play(layers, eps, args.seed0)
    n = st["n"]
    se = scores.std(ddof=1) / np.sqrt(len(scores))
    print(f"states visited: {n} ({eps} episodes), score {scores.mean():.1f} ± {se:.1f}")
    print(f"submarine in column 5, facing left: {st['col5'] / n:.2%} of states")
    print(f"first layer, states where the reconstruction differs (> 1e-9): {st['bad_h']}"
          f"  (different on/off: {st['bad_onoff']})")
    print(f"logits, maximum difference: {max_diff:.3f}")
    print(f"move with the highest logit differs, excluding near-ties < 0.5: {st['bad_arg']}"
          f"  (near-ties: {st['ties']})")
    print(f"simplified rule, same move as the network: {st['rule_same'] / n:.1%} of states; "
          f"{st['rule_same_clear'] / max(st['clear'], 1):.1%} where the network has a margin > 3 "
          f"({st['clear'] / n:.0%})")

    print("\nargmax of the network by submarine row:")
    for r in range(9):
        tot = sum(by_row[r, a] for a in ACTIONS)
        row = " ".join(f"{a}:{by_row[r, a] / tot:4.0%}" for a in ("up", "down", "fire"))
        print(f"  row {r}: {tot / n:5.1%} of states | {row}")
    print(f"sampled action different from the argmax: {st['not_argmax'] / n:.1%} of steps")
    print(f"oxygen < 60 with at least one diver on board: {st['low'] / n:.1%} of states, argmax up in "
          f"{st['low_up'] / max(st['low'], 1):.0%} of these")

    print(f"\ndeaths ({eps} episodes):")
    hows = collections.Counter()
    for (who, how), v in deaths.items():
        hows[how] += v
    for how, v in hows.most_common():
        whos = sorted(((w, c) for (w, h), c in deaths.items() if h == how and w), key=lambda t: -t[1])
        print(f"  {how:46s} {v:4d} {v / eps:4.0%}  " + ", ".join(f"{w} {c}" for w, c in whos))
    print(f"  (trapped in rows 1, 2 and 8: {st['trap_edge']})")
    print(f"\nsurfacings per episode: {sum(surf.values()) / eps:.1f}; divers on board when it surfaces:",
          " ".join(f"{k}:{surf[k]}" for k in sorted(surf)))
    print(f"divers that appeared per episode {st['div_spawn'] / eps:.1f}, collected {st['div_taken'] / eps:.1f}; "
          f"of the visible divers (rows 2-8), {st['div_away'] / max(st['div_seen'], 1):.0%} "
          f"have already passed column 5")
    print(f"points per episode {scores.mean():.1f}: enemies hit {(scores.sum() - st['bonus']) / eps:.1f}, "
          f"surfacings with 6 divers {st['bonus'] / eps:.1f}; steps per episode {st['steps'] / eps:.0f}")


if __name__ == "__main__":
    main()
