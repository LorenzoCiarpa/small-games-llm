"""
Checks the hand reading of weights/breakout/paper_astra.json (see README.md).

First it rewrites all the weights from the formulas of the README and compares them with the file. Then it plays the
same episodes as `uv run evaluate breakout paper_astra --episodes N` (seeds 10000.., sticky 0.1, actions sampled from
the softmax): in every state it recomputes, from the game state and without using the weights, the neurons of the two
hidden layers and the 6 logits as the README describes them, and compares them with the real forward pass. Finally
it measures how correct the network's prediction is and why it loses the ball.

    uv run python experiments/breakout/decode_astra/check.py [--episodes 200] [--sticky 0.1]
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
# h0..h3: one ball direction each, (dx, dy) with dy = +1 downward
DIRS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
DIR_NAMES = ("down-right", "down-left", "up-right", "up-left")
MINATAR_DIR = {0: (-1, -1), 1: (1, -1), 2: (1, 1), 3: (-1, 1)}   # MinAtar's ball_dir -> (dx, dy)
MOVE = (0, -1, 0, 1, 0, 0)                                        # paddle displacement per action


def forward(layers, obs):
    x, out = np.asarray(obs, dtype=np.float64), []
    for W, b, act in layers:
        x = W @ x + b
        if act == "tanh":
            x = np.tanh(x)
        out.append(x)
    return out


def pm(b):
    """[x] of the README: +1 if true, -1 if false (a saturated tanh neuron)."""
    return 1.0 if b else -1.0


def step_x(x, dx):
    """One horizontal step with MinAtar's walls: outside the grid the ball stays on the edge and bounces."""
    nx = x + dx
    if nx < 0:
        return 0, 1
    if nx > 9:
        return 9, -1
    return nx, dx


def predicted_flight(r, c, dx, dy):
    """The landing column that the network assigns to the ball in (r, c) with direction (dx, dy) (section 4 of the
    README), and the steps the ball takes, according to the same model, to reach row 8 on the way down.

    Walls and ceiling as in MinAtar. If it is going up, the ball bounces as soon as the row above is a brick row
    (1-3), as if all the bricks were there; then it goes down with no obstacles. The column is the one it is in
    when it reaches row 8; if it is going down and is already in rows 6-8, the midpoint between the two columns that
    catch it (the one in row 8 and the next one). In row 9 (ball lost) the column itself.
    """
    if r == 9:
        return float(c), 0
    x, y, s, going_down = c, r, 0, dy > 0
    while dy < 0:
        x, dx = step_x(x, dx)
        s += 1
        if y - 1 < 0 or 1 <= y - 1 <= 3:
            dy = 1                      # ceiling or brick: it stays in the row, the vertical direction reverses
        else:
            y -= 1
    while y < 8:
        x, dx = step_x(x, dx)
        y += 1
        s += 1
    if going_down and r >= 6:
        return (x + step_x(x, dx)[0]) / 2, s
    return float(x), s


# TARGET[k, r, c]: predicted column for the ball in (r, c) with direction DIRS[k]; STEPS: predicted steps
_TABLE = np.array([[[predicted_flight(r, c, *d) for c in range(10)] for r in range(10)] for d in DIRS])
TARGET, STEPS = _TABLE[..., 0], _TABLE[..., 1].astype(int)


def rebuilt_layers():
    """The network's weights rewritten from the formulas of the README (sections 4-6): they check that the reading
    reproduces the file weight by weight, also in the cells that the ball never visits during play."""
    cell = lambda r, c, ch: (r * 10 + c) * 6 + ch
    W0, b0 = np.zeros((32, 600)), np.zeros(32)
    for k in range(4):                                  # h0..h3: 0.02·T on the ball, −0.02·p on the paddle
        for r in range(10):
            for c in range(10):
                W0[k, cell(r, c, 3)] = 0.02 * TARGET[k, r, c]
        for p in range(10):
            W0[k, cell(9, p, 0)] = -0.02 * p
    for r in range(10):
        for c in range(10):                             # h4: ruler on the columns, h5: ruler on the rows
            W0[4, cell(r, c, 3)] = {0: 10, 9: 170}.get(c, 20 * c)
            W0[4, cell(r, c, 4)] = -20 * c
            W0[5, cell(r, c, 3)] = 70 if r == 8 else 10 * r
            W0[5, cell(r, c, 4)] = -10 * r
    b0[5] = 5
    W1, b1 = np.zeros((32, 32)), np.zeros(32)
    for k, (dx, dy) in enumerate(DIRS):                 # g(2k): to the left, g(2k+1): to the right
        for side, sign in ((0, -1), (1, 1)):
            W1[2 * k + side, [k, 4, 5]] = 1000 * sign, 200 * dx, 200 * dy
            b1[2 * k + side] = -415
    W2, b2 = np.zeros((6, 32)), np.array([0.0, 90, -100, 90, -100, -100])
    W2[1, [0, 2, 4, 6]] = 30
    W2[3, [1, 3, 5, 7]] = 30
    return [(W0, b0), (W1, b1), (W2, b2)]


def read_direction(e):
    """The direction as the network reads it: from the ball and its trail (previous cell)."""
    right = e.ball_x > e.last_x or e.ball_x == e.last_x == 0
    down = e.ball_y > e.last_y or (e.ball_y == e.last_y and e.ball_y != 8)
    return right, down


def decode(e):
    """Hidden neurons and logits rebuilt from the rules of the README, starting from the game state."""
    right, down = read_direction(e)
    k = DIRS.index((1 if right else -1, 1 if down else -1))
    p = e.pos
    T = TARGET[:, e.ball_y, e.ball_x]
    h = np.zeros(32)
    h[:4] = np.tanh(0.02 * (T - p))             # h0..h3: (predicted column - paddle) / 50
    h[4], h[5] = pm(right), pm(down)            # h4: going right, h5: going down
    delta = T[k] - p
    go_left, go_right = delta <= -1, delta >= 1
    g = np.zeros(32)
    g[:8] = -1.0
    g[2 * k], g[2 * k + 1] = pm(go_left), pm(go_right)
    logits = np.array([0.0, 30 * pm(go_left), -100.0, 30 * pm(go_right), -100.0, -100.0])
    return h, g, logits, k, T[k]


def true_flight(e):
    """Steps left until the ball reaches row 8 on the way down, and the columns in which the paddle catches it.

    Simulates the rules of MinAtar (walls, ceiling, real bricks) without the paddle, on a copy of the bricks.
    """
    bx, by, d, strike = e.ball_x, e.ball_y, e.ball_dir, e.strike
    bricks = e.brick_map.copy()
    for s in range(400):
        dx, dy = MINATAR_DIR[d]
        if by == 8 and dy > 0:
            return s, {bx, min(9, max(0, bx + dx))}
        nx, ny, hit = bx + dx, by + dy, False
        if nx < 0 or nx > 9:
            nx = min(9, max(0, nx))
            d = [1, 0, 3, 2][d]
        if ny < 0:
            ny = 0
            d = [3, 2, 1, 0][d]
        elif bricks[ny, nx] == 1:
            hit = True
            if not strike:
                strike = True
                bricks[ny, nx] = 0
                ny = by
                d = [3, 2, 1, 0][d]
        if not hit:
            strike = False
        bx, by = nx, ny
    raise RuntimeError("the ball does not come back down")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--sticky", type=float, default=0.1)
    args = ap.parse_args()

    game = get_game("breakout")
    layers = load_weights(ROOT / "weights/breakout/paper_astra.json", game)

    # 0. static check: the weights rewritten from the formulas of the README against the file
    n_par = n_diff = n_nonzero = 0
    for (W, b, _), (W_r, b_r) in zip(layers, rebuilt_layers()):
        n_par += W.size + b.size
        n_diff += int(np.sum(np.abs(W - W_r) > 1e-9) + np.sum(np.abs(b - b_r) > 1e-9))
        n_nonzero += int(np.count_nonzero(W_r) + np.count_nonzero(b_r))
    print(f"weights rewritten from the formulas of the README: {n_nonzero} nonzero; "
          f"parameters different from the file: {n_diff} out of {n_par}\n")

    n = bad_h = bad_dir = bad_zero = bad_g = bad_arg = ties = 0
    max_dh = max_dl = 0.0
    dir_err, by_action = collections.Counter(), collections.Counter()
    pred_n, pred_ok = collections.Counter(), collections.Counter()
    shorter = 0
    scores, lengths, cut = [], [], []
    flights = flights_2sticky = 0
    margin_all, margin_miss, sticky_miss = collections.Counter(), collections.Counter(), collections.Counter()
    margin1_dist = collections.Counter()
    miss_dir_err = 0
    miss_kind = collections.Counter()
    wrong_phase = collections.Counter()
    flights_wrong = 0

    for ep in range(args.episodes):
        seed = args.seed0 + ep
        env = game.make_env(sticky_prob=args.sticky)
        rng = policy_rng(seed)
        obs, _ = env.reset(seed=seed)
        e = env.env.env                                   # MinAtar's Breakout: ball, trail, paddle, bricks
        done, score, flight = False, 0.0, None
        while not done:
            # 1. the reading against the real network
            h_true, g_true, logit_true = forward(layers, obs)
            h_mine, g_mine, logit_mine, k, t = decode(e)
            n += 1
            dh = float(np.abs(h_true[:4] - h_mine[:4]).max())
            max_dh = max(max_dh, dh)
            bad_h += dh > 1e-9
            bad_dir += not np.array_equal(np.sign(h_true[4:6]), h_mine[4:6])
            bad_zero += bool(np.any(h_true[6:] != 0) or np.any(g_true[8:] != 0))
            bad_g += not np.array_equal(np.sign(g_true[:8]), g_mine[:8])
            top2 = np.sort(logit_mine)[-2:]
            if top2[1] - top2[0] <= 0.5:
                ties += 1
            elif np.argmax(logit_true) != np.argmax(logit_mine):
                bad_arg += 1
            max_dl = max(max_dl, float(np.abs(logit_true - logit_mine).max()))
            by_action[int(np.argmax(logit_true))] += 1

            # 2. the flights: from the start of the game, or from the ball just returned (ball and trail in row 8),
            # to the next bounce
            steps, catch = true_flight(e)
            if flight is None or e.ball_y == e.last_y == 8:
                dist = min(abs(e.pos - x) for x in catch)
                flight = {"margin": steps + 1 - dist, "sticky": 0, "dir_err": 0, "wrong": 0}
                flights += 1
                margin_all[flight["margin"]] += 1
                if flight["margin"] == 1:
                    margin1_dist[dist] += 1

            # 3. the direction read by the network against the real one of the game
            if MINATAR_DIR[e.ball_dir] != DIRS[k]:
                flight["dir_err"] += 1
                if e.ball_y == e.last_y == 8:
                    dir_err["bounce off the edge of the paddle"] += 1
                elif e.ball_y == e.last_y:
                    dir_err["brick hit from above"] += 1
                else:
                    dir_err["other"] += 1

            # 4. the prediction: are the columns where the network stays still columns that really catch the ball?
            phase = "up" if DIRS[k][1] < 0 else ("down, rows 0-2" if e.ball_y <= 2 else "down, rows 3-8")
            pred_n[phase] += 1
            pred_ok[phase] += {x for x in range(10) if abs(t - x) < 1} <= catch
            if DIRS[k][1] < 0 and MINATAR_DIR[e.ball_dir] == DIRS[k]:
                shorter += steps < STEPS[k, e.ball_y, e.ball_x]   # is the real flight shorter than the predicted one?

            a = sample_action(logit_true, rng)
            # 5. does the network's choice bring the paddle closer to the real catch columns (or keep it there)?
            d_now = min(abs(e.pos - x) for x in catch)
            d_next = min(abs(min(9, max(0, e.pos + MOVE[a])) - x) for x in catch)
            if d_next > max(d_now - 1, 0):
                flights_wrong += flight["wrong"] == 0
                flight["wrong"] += 1
                wrong_phase[phase] += 1
            obs, r, term, trunc, _ = env.step(a)
            if env.last_action != a:
                flight["sticky"] += 1                   # the environment repeated an action other than the chosen one
                flights_2sticky += flight["sticky"] == 2
            score += r
            done = term or trunc
            if term:
                margin_miss[flight["margin"]] += 1
                sticky_miss[flight["sticky"]] += 1
                miss_dir_err += flight["dir_err"] > 0
                if flight["wrong"]:
                    miss_kind["the network made wrong moves"] += 1
                elif flight["margin"] == 1:
                    miss_kind["margin 1, all of the network's choices correct"] += 1
                else:
                    miss_kind["margin > 1, all of the network's choices correct"] += 1
        scores.append(score)
        lengths.append(env.t)
        cut.append(not term)               # reached the step limit

    scores = np.array(scores)
    se = scores.std(ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
    print(f"states visited: {n} ({args.episodes} episodes, sticky {args.sticky})")
    cut = np.array(cut, dtype=bool)
    print(f"score: {scores.mean():.2f} ± {se:.2f}, mean length {np.mean(lengths):.0f}, "
          f"episodes that reached the limit of 10000 steps: {cut.sum()}")
    if cut.any() and (~cut).any():
        print(f"  mean score: {scores[cut].mean():.1f} in the episodes that reached the limit, "
              f"{scores[~cut].mean():.1f} in those that ended with the ball lost")
    print("\nthe reading against the network")
    print(f"  h0..h3, states where the value is wrong (> 1e-9): {bad_h}  (maximum difference {max_dh:.1e})")
    print(f"  h4, h5 (direction), states where the sign is wrong: {bad_dir}")
    print(f"  unused neurons (h6..h31, g8..g31) different from 0: {bad_zero}")
    print(f"  g0..g7, states where on/off is wrong: {bad_g}")
    print(f"  logits, maximum difference: {max_dl:.4f}")
    print(f"  move with the highest logit differs, excluding near-ties < 0.5: {bad_arg}  (near-ties: {ties})")
    print("  argmax of the network:", " ".join(f"{ACTIONS[a]}:{by_action[a] / n:.1%}" for a in range(6)))

    print("\nthe network against the game")
    n_err = sum(dir_err.values())
    print(f"  direction read different from the real one: {n_err} states ({n_err / n:.2%})", dict(dir_err))
    print("  correct prediction (the columns where it stays still catch the ball), by phase:")
    for ph in ("up", "down, rows 0-2", "down, rows 3-8"):
        print(f"    {ph:18s} {pred_ok[ph]:8d} / {pred_n[ph]:8d}  ({pred_ok[ph] / max(pred_n[ph], 1):.1%})")
    print(f"  ball going up: states where the real flight is shorter than the predicted one: {shorter}")

    misses = sum(margin_miss.values())
    print(f"\nflights (from paddle to paddle): {flights}, lost balls: {misses}")
    print("  margin at takeoff = available actions - columns to cover:")
    groups = [("1", lambda m: m == 1), ("2-4", lambda m: 2 <= m <= 4), ("5-9", lambda m: 5 <= m <= 9),
              ("≥ 10", lambda m: m >= 10), ("≤ 0", lambda m: m <= 0)]
    for name, f in groups:
        a = sum(v for m, v in margin_all.items() if f(m))
        b = sum(v for m, v in margin_miss.items() if f(m))
        if a:
            print(f"    margin {name:4s}: {a:7d} flights ({a / flights:5.1%}), "
                  f"lost balls {b:4d} ({b / a:.2%} of the flights)")
    print(f"  flights with margin 1, by columns to cover: {dict(sorted(margin1_dist.items()))}")
    print(f"  lost balls by sticky replacements during the flight: {dict(sorted(sticky_miss.items()))}")
    print(f"  flights with at least 2 sticky replacements: {flights_2sticky} ({flights_2sticky / flights:.1%})")
    print(f"  lost balls in flights with at least one state with the direction misread: {miss_dir_err}")
    print("  wrong moves of the network (they do not bring the paddle closer to the real catch columns), by phase:",
          dict(wrong_phase))
    print(f"  flights with at least one wrong move of the network: {flights_wrong} ({flights_wrong / flights:.1%})")
    print("  lost balls by kind of flight:", dict(miss_kind))


if __name__ == "__main__":
    main()
