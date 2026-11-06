"""
The Low Tide heuristic written as the weights of an MLP: builds weights/low_tide/bfs_net.json.

The heuristic (`heuristic_policy` in src/small_games/games/low_tide/baselines.py) uses a BFS to look for the nearest
shell, avoiding the jellyfish and the rows about to be flooded. An exact BFS does not fit in a network with 3 layers
of 64 units (see README.md). This network computes a potential-based version of the same algorithm:

- **shells:** each one is worth 6^(18 − d), with d the Manhattan distance from the agent. The shells of each
  direction are summed: up (rows higher up), down (rows lower down, but allowed), left and right (columns, any
  row). The direction with the highest sum wins: the nearest shell weighs more than all the farther ones together,
  unless there are 6 or more shells one step farther. On ties, the BFS preference: up, down, left, right.
- **jellyfish:** never step into a cell with a jellyfish. With that direction removed, the next one wins.
- **tide:** as in the heuristic. If the agent is below the last safe row it moves up; it does not go down below that
  row; if there is no shell it waits one row above.

`policy_logits(obs)` is the specification, written directly on the observation. `build()` writes the weights that
compute it; check.py verifies that network and specification choose the same move.

    uv run python experiments/low_tide/bfs_net/build.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "weights" / "low_tide" / "bfs_net.json"

H = W = 10
ROWS = range(3, 9)               # rows with shells and jellyfish seen by the network (row 9 is never allowed)
BASE, T = 6, 18                  # a shell at distance d is worth BASE^(T-d)
P1 = BASE ** (T - 1)             # value of an adjacent shell: the unit of measure of the potentials
DELTA = 0.02                     # tie preference: up ×(1+3δ), down ×(1+2δ), left ×(1+δ)
LAMBDA = 1.4e16                  # logit scale: even at distance 17 ties are broken by ~100
W_VETO, W_FORCE = 1e3 * LAMBDA, 1e6 * LAMBDA
PHI = 30.0                       # logit of waiting: softmax picks it with probability ~1, yet it stays below every potential
JELLY = 0.5                      # value (in P1 units) of an adjacent jellyfish
IDX_AGENT, IDX_WATER, IDX_SHELL, IDX_JELLY, IDX_RISING, IDX_COUNTDOWN = 0, 100, 200, 300, 400, 401


def pot(d):
    return BASE ** (T - d) if 1 <= d < T else 0


# ------------------------------------------------------------------------------------------------ specification
def decode(obs):
    g = np.asarray(obs[:400]).reshape(4, H, W)
    r, c = map(int, np.argwhere(g[0] > 0.5)[0])
    level = int(round(g[1, :, 0].sum()))
    rising = obs[IDX_RISING] > 0.5
    countdown = int(round(obs[IDX_COUNTDOWN] * 5))
    return g, r, c, level, rising, countdown


def potentials(obs, jelly_value=0.0):
    """Potentials (up, down, left, right) in P1 units, the jellyfish directly adjacent to the agent, and σ.

    With jelly_value=JELLY the potentials are those of the network's direction units, in which an adjacent jellyfish
    enters as a value 0.5 that the veto then removes; the specification uses them with jelly_value=0.
    """
    g, r, c, level, rising, countdown = decode(obs)
    margin = (1 + (countdown <= 2)) if rising else 0
    deepest = 9 - level - margin            # last allowed row
    sigma = deepest - r                     # allowed rows below the agent (< 0: it must move up)
    p = np.zeros(4)
    for R in ROWS:
        for C in range(W):
            d = abs(R - r) + abs(C - c)
            v = pot(d) / P1 if g[2, R, C] > 0.5 else jelly_value if (g[3, R, C] > 0.5 and d == 1) else 0.0
            p += v * np.array([R < r, r < R <= deepest, C < c, C > c])

    def jelly(R, C):
        return R in ROWS and 0 <= C < W and g[3, R, C] > 0.5

    veto = np.array([jelly(r - 1, c), jelly(r + 1, c), jelly(r, c - 1), jelly(r, c + 1)])
    return p, veto, sigma


def policy_logits(obs):
    """The 5 logits of the specification (noop, up, down, left, right), from the observation."""
    (up, down, left, right), (j_up, j_down, j_left, j_right), sigma = potentials(obs)
    lg = np.zeros(5)
    lg[1] = LAMBDA * (1 + 3 * DELTA) * up - W_VETO * j_up + W_FORCE * (sigma < 0) + PHI * (sigma <= 0)
    lg[2] = LAMBDA * (1 + 2 * DELTA) * down - W_VETO * j_down - W_FORCE * (sigma <= 0) + PHI * (sigma >= 2)
    lg[3] = LAMBDA * (1 + DELTA) * left - W_VETO * j_left
    lg[4] = LAMBDA * right - W_VETO * j_right
    lg[0] = PHI * (sigma == 1)
    return lg


def spec_policy(obs, rng=None):
    return int(np.argmax(policy_logits(obs)))


# ------------------------------------------------------------------------------------------------ weights
class Layer:
    def __init__(self, n_in, activation):
        self.n_in, self.activation, self.rows, self.names = n_in, activation, [], []

    def unit(self, name, weights, bias=0.0):
        """Adds a neuron: weights = {input index: weight}. Returns its index."""
        self.rows.append(({int(k): float(v) for k, v in weights.items() if v != 0}, float(bias)))
        self.names.append(name)
        return len(self.rows) - 1

    def to_json(self):
        return {"in": self.n_in, "out": len(self.rows), "activation": self.activation,
                "W": {"sparse": [[i, j, w] for i, (ws, _) in enumerate(self.rows) for j, w in sorted(ws.items())]},
                "b": {"sparse": [[i, b] for i, (_, b) in enumerate(self.rows) if b != 0]}}


def build():
    # ---- layer 1 (relu): one shell or jellyfish per cell, plus position and tide -----------------------------
    L1 = Layer(403, "relu")
    cell = {}
    for R in ROWS:
        for C in range(W):
            w = {IDX_SHELL + 10 * R + C: P1, IDX_JELLY + 10 * R + C: P1 * JELLY}
            for ar in range(H):                       # the agent at (ar, ac): pot(d) − P1, always ≤ 0
                for ac in range(W):
                    w[IDX_AGENT + 10 * ar + ac] = pot(abs(R - ar) + abs(C - ac)) - P1
            cell[R, C] = L1.unit(f"cell ({R},{C})", w)
    row = L1.unit("agent's row r", {IDX_AGENT + p: p // 10 for p in range(100)})
    col = L1.unit("agent's column c", {IDX_AGENT + p: p % 10 for p in range(100)})
    m = L1.unit("m = 9 − r − L", {**{IDX_AGENT + p: -(p // 10) for p in range(100)},
                                  **{IDX_WATER + 10 * R: -1 for R in range(3, 10)}}, 9)
    v = L1.unit("margin ≈ 0/1/2", {IDX_RISING: 2, IDX_COUNTDOWN: -1})

    # ---- layer 2 (tanh): potentials per row and column, adjacent jellyfish, thresholds -----------------------
    L2 = Layer(len(L1.rows), "tanh")
    EPS, BQ, BT, BS = 1e-3, 500.0, 50.0, 250.0
    P, Q = {}, {}
    for R in ROWS:
        cells = [cell[R, C] for C in range(W)]
        P["r", R] = L2.unit(f"potential of row {R}", {k: EPS / P1 for k in cells})
        Q["r", R] = L2.unit(f"adjacent jellyfish in row {R}", {k: BQ / P1 for k in cells}, -BQ * 0.45)
    for C in range(W):
        cells = [cell[R, C] for R in ROWS]
        P["c", C] = L2.unit(f"potential of column {C}", {k: EPS / P1 for k in cells})
        Q["c", C] = L2.unit(f"adjacent jellyfish in column {C}", {k: BQ / P1 for k in cells}, -BQ * 0.45)
    tr = {k: L2.unit(f"[r > {k}]", {row: BT}, -BT * (k + 0.5)) for k in range(1, 9)}
    tc = {k: L2.unit(f"[c > {k}]", {col: BT}, -BT * (k + 0.5)) for k in range(0, 9)}
    allowed = {R: L2.unit(f"row {R} allowed", {m: BS, row: BS, v: -BS}, BS * (0.5 - R)) for R in ROWS}
    s = {k: L2.unit(f"[σ ≥ {k}]", {m: BS, v: -BS}, BS * (0.5 - k)) for k in range(3)}

    # ---- layer 3 (relu): potentials per direction, vetoes, tide events ---------------------------------------
    # a tanh threshold t is ±1: [x] = (t + 1)/2. M switches a unit off when its condition is false.
    L3 = Layer(len(L2.rows), "relu")
    M = 4.0
    up, down, left, right, veto = [], [], [], [], {"up": [], "down": [], "left": [], "right": []}
    for R in ROWS:
        up.append(L3.unit(f"up, row {R}", {P["r", R]: 1 / EPS, tr[R]: M / 2}, -M / 2))          # r > R
        down.append(L3.unit(f"down, row {R}", {P["r", R]: 1 / EPS, tr[R - 1]: -M / 2, allowed[R]: M / 2}, -M))
    for C in range(W):
        if C <= 8:
            left.append(L3.unit(f"left, column {C}", {P["c", C]: 1 / EPS, tc[C]: M / 2}, -M / 2))  # c > C
        if C >= 1:
            right.append(L3.unit(f"right, column {C}", {P["c", C]: 1 / EPS, tc[C - 1]: -M / 2}, -M / 2))

    def veto_unit(name, key, gate_pos, gate_neg, const):
        """relu(0.475·Q − potential + 0.475 − M·(1 − [agent next to it])): > 0 only with an adjacent jellyfish."""
        w = {Q[key]: 0.475, P[key]: -1 / EPS}
        for k in gate_pos:
            w[k] = w.get(k, 0) + M / 2
        for k in gate_neg:
            w[k] = w.get(k, 0) - M / 2
        return L3.unit(name, w, 0.475 - M + M * const)

    # [r = R+1] = ([r>R] − [r>R+1])/2; the constant terms of the thresholds that do not exist go into the bias
    for R in ROWS:
        if R + 1 <= 8:
            veto["up"].append(veto_unit(f"jellyfish above (row {R})", ("r", R), [tr[R]], [tr[R + 1]], 0))
        veto["down"].append(veto_unit(f"jellyfish below (row {R})", ("r", R), [tr[R - 2]], [tr[R - 1]], 0))
    for C in range(W):
        if C <= 8:   # agent in column C+1
            if C + 1 <= 8:
                veto["left"].append(veto_unit(f"jellyfish on the left (column {C})", ("c", C), [tc[C]], [tc[C + 1]], 0))
            else:
                veto["left"].append(veto_unit(f"jellyfish on the left (column {C})", ("c", C), [tc[C]], [], 0.5))
        if C >= 1:   # agent in column C−1
            if C - 2 >= 0:
                veto["right"].append(veto_unit(f"jellyfish on the right (column {C})", ("c", C), [tc[C - 2]], [tc[C - 1]], 0))
            else:
                veto["right"].append(veto_unit(f"jellyfish on the right (column {C})", ("c", C), [], [tc[C - 1]], 0.5))
    forced = L3.unit("σ < 0: must move up", {s[0]: -0.5}, 0.5)
    no_down = L3.unit("σ ≤ 0: cannot move down", {s[1]: -0.5}, 0.5)
    wait_here = L3.unit("σ = 1: waits here", {s[1]: 0.5, s[2]: -0.5})
    wait_down = L3.unit("σ ≥ 2: moves down to wait", {s[2]: 0.5}, 0.5)

    # ---- output (linear) -------------------------------------------------------------------------------------
    L4 = Layer(len(L3.rows), "linear")
    L4.unit("noop", {wait_here: PHI})
    L4.unit("up", {**{k: LAMBDA * (1 + 3 * DELTA) for k in up}, **{k: -W_VETO for k in veto["up"]},
                   forced: W_FORCE, no_down: PHI})
    L4.unit("down", {**{k: LAMBDA * (1 + 2 * DELTA) for k in down}, **{k: -W_VETO for k in veto["down"]},
                     no_down: -W_FORCE, wait_down: PHI})
    L4.unit("left", {**{k: LAMBDA * (1 + DELTA) for k in left}, **{k: -W_VETO for k in veto["left"]}})
    L4.unit("right", {**{k: LAMBDA for k in right}, **{k: -W_VETO for k in veto["right"]}})

    layers = [L1, L2, L3, L4]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"source": "experiments/low_tide/bfs_net/build.py: the Low Tide BFS heuristic as a potential",
                   "layers": [L.to_json() for L in layers]}, f, separators=(",", ":"))
    return layers


if __name__ == "__main__":
    from small_games.core import get_game
    from small_games.weights import load_weights

    layers = build()
    load_weights(OUT, get_game("low_tide"))          # the file must pass the checks like any other
    sizes = " → ".join([str(layers[0].n_in)] + [f"{len(L.rows)} ({L.activation})" for L in layers])
    nnz = sum(len(ws) for L in layers for ws, _ in L.rows)
    print(f"{OUT.relative_to(ROOT)}: {sizes}, {nnz} non-zero weights")
