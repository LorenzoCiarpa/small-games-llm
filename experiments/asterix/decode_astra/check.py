"""
Checks the hand reading of weights/asterix/paper_astra.json (see README.md).

Plays episodes with the network as `evaluate` does (softmax, sticky 0.1, same seeds and same generator). In every
state it recomputes, directly from the game state and without using the weights:

  - the pre-activations of the 32 neurons of the first layer (section 4 of the README);
  - the state of the 32 neurons of the second layer: off, halfway, on (section 5);
  - the 6 logits with the formulas of section 6;
  - the logits of the simplified rule of section 7;

and compares them with the real forward pass. With --rule N it also plays N episodes with the rule in place of the
network, with --variants N it plays N episodes with each of the network variants of section 9 (one piece removed).

    uv run python experiments/asterix/decode_astra/check.py                  # ~40 s
    uv run python experiments/asterix/decode_astra/check.py --rule 1000      # + ~40 s
    uv run python experiments/asterix/decode_astra/check.py --variants 1000  # + ~3 min
"""

import argparse
import collections
import copy
from pathlib import Path

import numpy as np

from small_games.core import get_game
from small_games.policy import policy_rng, sample_action
from small_games.weights import load_weights

ROOT = Path(__file__).resolve().parents[3]
NOOP, UP, DOWN = 0, 2, 4
MOVES = {NOOP: "no-op", UP: "up", DOWN: "down"}


def forward(layers, obs):
    """The pre-activations of every layer (for the last one, the logits)."""
    x, out = np.asarray(obs, dtype=np.float64), []
    for W, b, act in layers:
        z = W @ x + b
        out.append(z)
        x = np.tanh(z) if act == "tanh" else z
    return out


# ---------------------------------------------------------------- the game state

def entities_by_row(g):
    """row -> (column, moving right, is a treasure). At most one entity per row, in rows 1-8."""
    return {e[1]: (e[0], e[2], e[3]) for e in g.entities if e is not None}


def arrival(x, right):
    """Cells left to column 5 for an entity that is moving toward it (0: it is already in column 5).
    None if it is moving away from column 5."""
    if x == 5:
        return 0
    if (x < 5) == right:
        return abs(x - 5)
    return None


def trail(x, right):
    """The column of the trail: the cell the entity comes from."""
    c = x - 1 if right else x + 1
    return c if 0 <= c <= 9 else None


def incoming(g):
    """row -> d, for the enemies and for the treasures that are moving toward column 5."""
    enemies, treasures = {}, {}
    for k, (x, right, is_treasure) in entities_by_row(g).items():
        d = arrival(x, right)
        if d is not None:
            (treasures if is_treasure else enemies)[k] = d
    return enemies, treasures


# ---------------------------------------------------------------- section 4: first layer

ENEMY_NEAR = {0: 6, 1: 6, 2: 5}            # h0-h15, rows above and below: enemy incoming at d cells
ENEMY_MINE = {0: 6, 1: 6, 2: 4, 3: 3}      # h16-h23, my row
# side effects of the trail (section 4.5): the trail of a treasure in the danger neurons,
# that of an enemy in the treasure neurons
TRAIL_NEAR = {4: -5, 5: -6, 6: -5}
TRAIL_MINE = {3: -3, 4: -4, 5: -9, 6: -4, 7: -3}
TRAIL_TREASURE = {1: -8, 2: -8, 3: -16, 4: -16, 5: -24, 6: -16, 7: -16, 8: -8, 9: -8}


def guard_above(k, r):
    """Weight of the player in row r in the neuron 'danger in row k, above me'."""
    if k == 1:
        return {1: 8, 2: 6, 3: 3}.get(r, 0)
    return {k + 1: 4, k + 2: 3}.get(r, 0)


def guard_below(k, r):
    if k == 8:
        return {8: 8, 7: 6, 6: 3}.get(r, 0)
    return {k - 1: 4, k - 2: 3}.get(r, 0)


def guard_mine(k, r):
    return (7 if k in (1, 8) else 4) if r == k else 0


def treasure_term(k, r):
    """The player's term in the treasure neuron of row k: + if the treasure is above, − if it is below."""
    n = abs(r - k)
    t = 0.0 if n == 0 else 0.8 if n == 1 else 1.2 + 0.1 * n
    return t if r >= k else -t


def first_layer(g):
    """The 32 pre-activations of the first layer, from the game state."""
    r, ent = g.player_y, entities_by_row(g)
    pre = np.full(32, -8.0)
    for k in range(1, 9):
        near = mine = treasure = 0.0
        if k in ent:
            x, right, is_treasure = ent[k]
            d, c = arrival(x, right), trail(x, right)
            if is_treasure:
                treasure = 8.0 if d is not None else 0.0
                near, mine = TRAIL_NEAR.get(c, 0), TRAIL_MINE.get(c, 0)
            else:
                if d is not None:
                    near, mine = ENEMY_NEAR.get(d, 0), ENEMY_MINE.get(d, 0)
                treasure = TRAIL_TREASURE.get(c, 0)
        pre[k - 1] += near + guard_above(k, r)          # h0-h7:   danger above
        pre[7 + k] += near + guard_below(k, r)          # h8-h15:  danger below
        pre[15 + k] += mine + guard_mine(k, r)          # h16-h23: danger in my row
        pre[23 + k] += treasure + treasure_term(k, r)   # h24-h31: treasure
    return pre


# ---------------------------------------------------------------- section 6: the formulas

def formula_logits(pre):
    """The logits of section 6, from the first layer."""
    h = np.tanh(pre)
    a_up, a_down, a_noop = (1 + h[0:8]).sum(), (1 + h[8:16]).sum(), (1 + h[16:24]).sum()
    t = h[24:32]                                         # treasure in rows 1..8
    A, B = np.tanh(21.6 + 24 * t), np.tanh(8 + 20 * t)
    C, D = np.tanh(-8 + 20 * t), np.tanh(-21.6 + 24 * t)
    lo = np.full(6, -1000.0)
    lo[NOOP] = 17 - 80 * np.tanh(a_noop - 1) + 16 * (B[:7] - C[:7]).sum() + 16 * B[7]
    lo[UP] = 0.15 - 80 * np.tanh(a_up - 1) + 6 * (C[:7] - D[:7]).sum()
    lo[DOWN] = -0.69 - 80 * np.tanh(a_down - 1) + 6 * (A[1:] - B[1:]).sum()
    return lo


# ---------------------------------------------------------------- section 7: the rule

ATTRACTION = {1: 12.0, 2: 8.0, 3: 5.3, 4: 3.1, 5: 1.8, 6: 1.1, 7: 0.8}   # treasure incoming n rows away


def treasure_trail_column(g, r):
    """Column of the trail of the treasure in my row (None if there is none)."""
    e = entities_by_row(g).get(r)
    return trail(e[0], e[1]) if e is not None and e[2] else None


def alarms(g):
    """Alarm of no-op, up and down (table of section 5)."""
    r = g.player_y
    enemies, _ = incoming(g)
    c = treasure_trail_column(g, r) if r in (1, 8) else None

    def move_alarm(direction):
        t1, t2 = r + direction, r + 2 * direction
        if not 1 <= t1 <= 8:                             # against the wall: the move leaves you where you are
            if c is not None and 4 <= c <= 6:            # edge effect of the trail of a treasure
                return 0.0
            d = enemies.get(r)
            return 2.0 if d is not None and d <= 2 else 1.0
        a = 0.0
        d = enemies.get(t1)
        if t1 in (1, 8):                                 # toward an edge row
            a += 2.0 if d is not None and d <= 2 else 0.036
        elif d is not None and d <= 2:
            a += 1.96 if d <= 1 else 1.76
        d = enemies.get(t2) if 1 <= t2 <= 8 else None
        if d is not None and d <= 2:
            a += 1.76 if d <= 1 else 1.0
        return a

    d = enemies.get(r)
    if r in (1, 8):
        if d is not None and d <= 3:
            noop = 2.0 if d <= 2 else 1.96
        else:
            noop = 0.0 if c is not None and 3 <= c <= 7 else 0.24
    else:
        noop = {0: 1.96, 1: 1.96, 2: 1.0, 3: 0.24}.get(d, 0.0)
    return noop, move_alarm(-1), move_alarm(+1)


def rule(g):
    """The logits of the simplified rule of section 7."""
    r = g.player_y
    _, treasures = incoming(g)
    a_noop, a_up, a_down = alarms(g)
    lo = np.full(6, -1000.0)
    lo[NOOP] = 1 - 80 * np.tanh(a_noop - 1) + 32 * (r in treasures)
    lo[UP] = 0.15 - 80 * np.tanh(a_up - 1) + sum(ATTRACTION[r - k] for k in treasures if k < r)
    lo[DOWN] = -80 * np.tanh(a_down - 1) + sum(ATTRACTION[k - r] for k in treasures if k > r)
    return lo


# ---------------------------------------------------------------- section 5: the second layer

def alarm_state(a):
    """The expected state of g0, g1, g2 = tanh(alarm − 1): -1 off, 0 halfway, +1 on."""
    return -1 if a < 0.5 else (0 if a <= 1.5 else 1)


def real_state(v):
    return -1 if v < -0.3 else (0 if v <= 0.3 else 1)


def second_layer(g):
    """Expected state of the 32 neurons of the second layer (g3-g31 only the sign: +1 on, -1 off)."""
    r = g.player_y
    _, treasures = incoming(g)
    a_noop, a_up, a_down = alarms(g)
    st = np.full(32, -1)
    st[0], st[1], st[2] = alarm_state(a_up), alarm_state(a_down), alarm_state(a_noop)
    for k in range(1, 9):
        s = r - k if k in treasures else None            # + treasure above, − treasure below, 0 in my row
        on = lambda threshold: 1 if s is not None and s >= threshold else -1
        if k >= 2:
            st[4 * k - 2] = on(-2)                       # A: treasure above, in my row or at most 2 below
        st[4 * k - 1] = on(0)                            # B: treasure above or in my row
        if k <= 7:
            st[4 * k] = on(1)                            # C: treasure above
            st[4 * k + 1] = on(3)                        # D: treasure above, at least 3 rows away
    return st


def real_second_layer(g_true):
    st = np.where(g_true > 0, 1, -1)
    st[:3] = [real_state(v) for v in g_true[:3]]
    return st


# ---------------------------------------------------------------- the situations

SITUATIONS = ("no move without an alarm", "enemy in my row", "treasure in my row", "edge row",
              "treasure in another row", "nothing incoming")


def situation(g):
    """The first one that applies, in this order."""
    r = g.player_y
    enemies, treasures = incoming(g)
    if min(alarms(g)) >= 0.5:
        return SITUATIONS[0]
    if r in enemies and enemies[r] <= 3:
        return SITUATIONS[1]
    if r in treasures:
        return SITUATIONS[2]
    if r in (1, 8):
        return SITUATIONS[3]
    if treasures:
        return SITUATIONS[4]
    return SITUATIONS[5]


def survives(g, a):
    gg = copy.deepcopy(g)
    gg.act(a)
    return not gg.terminal


def softmax(lo):
    p = np.exp(lo - lo.max())
    return p / p.sum()


def play(game, logit, episodes, seed0):
    """Mean score, standard error and mean episode length of a policy, with the protocol of
    `evaluate`. logit(env, obs) -> the 6 logits."""
    ret, lengths = [], []
    for ep in range(episodes):
        seed = seed0 + ep
        env, rng = game.make_env(), policy_rng(seed)
        obs, _ = env.reset(seed=seed)
        done, tot = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(sample_action(logit(env, obs), rng))
            tot += r
            done = term or trunc
        ret.append(tot)
        lengths.append(env.t)
    ret = np.array(ret)
    return ret.mean(), ret.std(ddof=1) / np.sqrt(len(ret)), np.mean(lengths)


def variants(layers):
    """The network with one piece removed (section 9). The indices are those of sections 4-6."""
    (W1, b1, f1), (W2, b2, f2), (W3, b3, f3) = layers
    ALARM_OFF = 60.93                    # -80·tanh(-1): what an alarm that is off is worth
    TREASURE = slice(3, 32)              # g3-g31

    def network(W1n=W1, W3n=W3, b3n=b3):
        return [(W1n, b1, f1), (W2, b2, f2), (W3n, b3n, f3)]

    out = {}
    W, b = W3.copy(), b3.copy()
    W[:, TREASURE] = 0
    b[NOOP], b[DOWN] = 1.0, 0.0          # the values that the removed parts gave without treasures
    out["without treasures (g3-g31 do not reach the logits)"] = network(W3n=W, b3n=b)
    W, b = W3.copy(), b3.copy()
    W[NOOP, TREASURE] = 0
    b[NOOP] = 1.0
    out["without the +32 to no-op for a treasure in my row"] = network(W3n=W, b3n=b)
    W, b = W3.copy(), b3.copy()
    W[[UP, DOWN], TREASURE] = 0
    b[DOWN] = 0.0
    out["without attraction to treasures above and below"] = network(W3n=W, b3n=b)
    W = W1.copy()
    player = np.arange(0, 600, 6)        # channel 0, the player
    for i in range(16):                  # h0-h15: remove the player 2 rows from the row looked at
        W[i, player[W1[i, player] == 3]] = 0
    out["above and below look only at the destination row"] = network(W1n=W)
    W, b = W3.copy(), b3.copy()
    W[:, 0:3] = 0
    b[[NOOP, UP, DOWN]] += ALARM_OFF
    out["without alarms (g0-g2 as if they were off)"] = network(W3n=W, b3n=b)
    W, b = W3.copy(), b3.copy()
    W[NOOP, 2] = 0
    b[NOOP] += ALARM_OFF
    out["without the no-op alarm (g2)"] = network(W3n=W, b3n=b)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--rule", type=int, default=0, help="episodes to play with the rule of section 7")
    ap.add_argument("--variants", type=int, default=0, help="episodes to play with each variant of section 9")
    args = ap.parse_args()

    game = get_game("asterix")
    layers = load_weights(ROOT / "weights/asterix/paper_astra.json", game)
    n = off_column = bad_h = bad_g = bad_arg = ties = edge_effect = 0
    max_h = max_f = max_r = tv = 0.0
    bad_g_neurons = collections.Counter()
    cases = collections.Counter()
    ties_sit = collections.Counter()
    nothing_blocked = 0
    prob = collections.defaultdict(lambda: np.zeros(3))
    row_counts = collections.Counter()
    deaths = collections.Counter()
    into_enemy = collections.Counter()  # deaths entering a row with an enemy already in column 5
    death_rows = collections.Counter()
    returns = []
    treasures_spawned = 0

    for ep in range(args.episodes):
        seed = args.seed0 + ep
        env, rng = game.make_env(), policy_rng(seed)
        obs, _ = env.reset(seed=seed)
        g = env.env.env
        seen = list(g.entities)
        done, tot = False, 0.0
        while not done:
            h_pre, g_pre, logit_true = forward(layers, obs)
            g_true = np.tanh(g_pre)
            n += 1
            off_column += g.player_x != 5
            row_counts[g.player_y] += 1

            pre = first_layer(g)
            dh = float(np.abs(pre - h_pre).max())
            max_h = max(max_h, dh)
            bad_h += dh > 1e-9

            diff = second_layer(g) != real_second_layer(g_true)
            bad_g += diff.any()
            for i in np.nonzero(diff)[0]:
                bad_g_neurons[i] += 1

            max_f = max(max_f, float(np.abs(formula_logits(pre) - logit_true).max()))

            lo = rule(g)
            max_r = max(max_r, float(np.abs(lo - logit_true).max()))
            tv += 0.5 * np.abs(softmax(lo) - softmax(logit_true)).sum()
            top2 = np.sort(lo)[-2:]
            sit = situation(g)
            if top2[1] - top2[0] < 1.0:
                ties += 1
                ties_sit[sit] += 1
            elif np.argmax(lo) != np.argmax(logit_true):
                bad_arg += 1
            c = treasure_trail_column(g, g.player_y)
            if g.player_y in (1, 8) and c is not None and 3 <= c <= 7:
                edge_effect += arrival(*entities_by_row(g)[g.player_y][:2]) is None  # treasure moving away

            cases[sit] += 1
            if sit == SITUATIONS[-1]:
                nothing_blocked += max(alarms(g)[1:]) >= 0.5
            prob[sit] += softmax(logit_true)[[NOOP, UP, DOWN]]
            trapped = sit == SITUATIONS[0]
            survivals = tuple(survives(g, a) for a in (NOOP, UP, DOWN)) if trapped else None
            player_row = g.player_y
            enemies_before = incoming(g)[0]

            a = sample_action(logit_true, rng)
            obs, r, term, trunc, _ = env.step(a)
            tot += r
            done = term or trunc
            treasures_spawned += sum(1 for i, e in enumerate(g.entities)
                               if e is not None and e[3] and e is not seen[i])
            seen = list(g.entities)
            if term:
                death_rows[player_row] += 1
                if env.last_action != a:
                    cause = "sticky: the environment repeats the previous move"
                elif trapped and any(survivals):
                    cause = "no move without an alarm, but one survived"
                elif trapped:
                    cause = "no move without an alarm, and none survived"
                else:
                    cause = "other"
                deaths[cause, player_row in (1, 8)] += 1
                executed = env.last_action
                dest = player_row - 1 if executed == UP else player_row + 1 if executed == DOWN else player_row
                if dest != player_row and 1 <= dest <= 8 and enemies_before.get(dest) == 0:
                    into_enemy[cause] += 1
        returns.append(tot)

    print(f"states visited: {n} ({args.episodes} episodes); network score {np.mean(returns):.1f}")
    print(f"states with the player outside column 5: {off_column}")
    print(f"first layer, states in which a pre-activation differs from the reading: {bad_h} "
          f"(maximum deviation {max_h:.1e})")
    print(f"second layer, states in which the state of a neuron differs: {bad_g} {dict(bad_g_neurons)}")
    print(f"logits with the formulas of section 6, maximum difference: {max_f:.1e}")
    print(f"rule of section 7: maximum difference of the logits {max_r:.2f}, "
          f"mean distance between the softmaxes {tv / n:.4f}")
    print(f"  move with the highest logit differs, excluding near-ties < 1: {bad_arg}  (near-ties: {ties})")
    print("  near-ties by situation:", dict(ties_sit.most_common()))
    print(f"  states with the edge effect of a treasure's trail: {edge_effect}")

    print("\nmean probability of the network's moves, by situation:")
    for s in SITUATIONS:
        p = prob[s] / max(cases[s], 1)
        row = " ".join(f"{MOVES[a]}:{p[i]:4.0%}" for i, a in enumerate(MOVES))
        print(f"  {s:25s} {cases[s] / n:6.1%} of states | {row}")
    print(f"  in '{SITUATIONS[-1]}', states in which up or down has an alarm >= 0.5: "
          f"{nothing_blocked / max(cases[SITUATIONS[-1]], 1):.0%}")

    print("\nplayer's row:", " ".join(f"{k}:{row_counts[k] / n:.1%}" for k in range(1, 9)))
    print(f"treasures: collected {sum(returns):.0f} out of {treasures_spawned} that appeared "
          f"({sum(returns) / treasures_spawned:.1%})")
    print(f"deaths ({sum(deaths.values())}), in rows 1-8 and of these in the edge rows 1 and 8:")
    for cause in sorted({c for c, _ in deaths}, key=lambda c: -deaths[c, False] - deaths[c, True]):
        print(f"  {cause}: {deaths[cause, False] + deaths[cause, True]} (edge {deaths[cause, True]}; "
              f"entering a row with an enemy already in column 5: {into_enemy[cause]})")
    print("  row of death:", " ".join(f"{k}:{death_rows[k]}" for k in range(1, 9)))

    if args.rule:
        m, se, t = play(game, lambda env, obs: rule(env.env.env), args.rule, args.seed0)
        print(f"\nthe rule of section 7 in place of the network, {args.rule} episodes: {m:.1f} ± {se:.1f}"
              f" (mean length {t:.0f} steps)")
    if args.variants:
        print(f"\nvariants of the network, {args.variants} episodes:")
        networks = {"the network as it is": layers, **variants(layers)}
        for name, var in networks.items():
            m, se, t = play(game, lambda env, obs: forward(var, obs)[-1], args.variants, args.seed0)
            print(f"  {name:50s} {m:5.1f} ± {se:.1f}   mean length {t:4.0f} steps")


if __name__ == "__main__":
    main()
