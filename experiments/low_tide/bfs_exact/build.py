"""
The exact Low Tide BFS heuristic, as the weights of an MLP without the game's limits: builds bfs_exact.json.

bfs_net (../bfs_net) stays within the game's limits (3 layers of 64) and computes a potential-based version of the
BFS. This network instead makes the same move as `heuristic_policy` in every state, but it needs many more neurons:
one layer for each step of the BFS and 5 units per cell in each layer (see README.md).

Idea: instead of searching from the agent, a wave starts from the reachable shells and at each layer advances by one
cell, only into the allowed cells (rows up to the last safe one, no jellyfish, no cells next to a jellyfish if the
tide is rising). Each cell counts for how many layers it has been reached: after K layers the count is K + 1 − d,
with d the distance from the nearest shell. The BFS move is the agent's neighbor with the highest count, with ties
broken in the order up, down, left, right (the same order in which the BFS visits the neighbors). The tide rules are
those of the heuristic.

All internal values are small integers, so the computation in float64 is exact. The logits have margins ≥ 1000:
exp(−1000) = 0 in float64, so even when sampling from the softmax the network always chooses the same move.

    uv run python experiments/low_tide/bfs_exact/build.py [--k 98]

K = 98 is enough for every possible state: in a grid of 100 cells the nearest shell is at distance ≤ 99.
"""

import argparse
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "bfs_exact.json"

H = W = 10
CELLS = [(r, c) for r in range(H) for c in range(W)]
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))  # noop, up, down, left, right
DIRS = (1, 2, 3, 4)
K_ALL = H * W - 2                 # the BFS finds a shell at distance ≤ 99: the neighbor needs d ≤ 98
G = 1e4                           # logit scale: the tie preference is worth 0.1 G = 1000
TIE = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
FALLBACK = 2 * G                  # logit of the waiting move when no shell is reachable


def neighbors(r, c):
    return [(r + dr, c + dc) for dr, dc in MOVES[1:] if 0 <= r + dr < H and 0 <= c + dc < W]


def input_names():
    names = []
    for ch in ("agent", "water", "shell", "jellyfish"):
        names += [f"{ch} {r},{c}" for r, c in CELLS]
    return names + ["rising", "countdown", "stun"]


class Layer:
    """A layer: each unit is act(weighted sum of units of the previous layer + bias), by name."""

    def __init__(self, prev_names, activation="relu"):
        self.prev = {n: i for i, n in enumerate(prev_names)}
        self.activation = activation
        self.names, self.rows, self.bias = [], [], []

    def unit(self, name, weights, bias=0.0):
        assert name not in self.names, name
        row = collections.defaultdict(float)
        for k, v in weights:
            row[self.prev[k]] += v
        self.names.append(name)
        self.rows.append({j: v for j, v in row.items() if v != 0})
        self.bias.append(float(bias))

    def carry(self, *names):
        for n in names:
            self.unit(n, [(n, 1.0)])

    def to_json(self):
        W_ = [[i, j, v] for i, row in enumerate(self.rows) for j, v in sorted(row.items())]
        b = [[i, v] for i, v in enumerate(self.bias) if v != 0]
        return {"in": len(self.prev), "out": len(self.names), "activation": self.activation,
                "W": {"sparse": W_}, "b": {"sparse": b}}


def cell(prefix, r, c):
    return f"{prefix} {r},{c}"


def build(K=K_ALL):
    assert K >= 1
    layers = []

    def new(activation="relu"):
        L = Layer(layers[-1].names if layers else input_names(), activation)
        layers.append(L)
        return L

    ag = [cell("agent", r, c) for r, c in CELLS]

    # 1. from the observation: copies of the cells, nearby jellyfish, water level, agent's row
    L = new()
    for r, c in CELLS:
        L.unit(cell("ag", r, c), [(cell("agent", r, c), 1)])
        L.unit(cell("sh", r, c), [(cell("shell", r, c), 1)])
        L.unit(cell("je", r, c), [(cell("jellyfish", r, c), 1)])
        # > 0 only if the tide is rising and a neighboring cell has a jellyfish (its value: how many there are)
        L.unit(cell("nj", r, c), [(cell("jellyfish", *n), 1) for n in neighbors(r, c)] + [("rising", 4)], -4)
    L.unit("level", [(cell("water", r, 0), 1) for r in range(H)])              # flooded rows
    L.unit("rising", [("rising", 1)])
    L.unit("u", [("countdown", 10)], -5)                                        # 0 if countdown ≤ 2, ≥ 1 if ≥ 3
    L.unit("row", [(a, r) for a, (r, c) in zip(ag, CELLS)])                    # agent's row
    L.unit("row 0", [(cell("agent", 0, c), 1) for c in range(W)])              # 1 if the agent is in row 0

    # 2. v = the tide is rising and the change is near (countdown ≤ 2): margin 2 instead of 1
    L = new()
    for r, c in CELLS:
        L.carry(cell("ag", r, c), cell("sh", r, c), cell("je", r, c), cell("nj", r, c))
    L.carry("level", "rising", "row", "row 0")
    L.unit("v", [("rising", 1), ("u", -1)])

    # 3. last safe row: deepest = 9 − level − rising − v.  m_R = deepest − R, e = deepest − agent's row
    L = new()
    for r, c in CELLS:
        L.carry(cell("ag", r, c), cell("sh", r, c), cell("je", r, c), cell("nj", r, c))
    deepest = [("level", -1), ("rising", -1), ("v", -1)]
    for R in range(H):
        L.unit(f"ok1 {R}", deepest, 9 - R + 1)       # [R ≤ deepest] = relu(m_R + 1) − relu(m_R)
        L.unit(f"ok0 {R}", deepest, 9 - R)
    e = deepest + [("row", -1)]
    neg = [(k, -v) for k, v in e]
    L.unit("t1", neg, -9)                            # below the last safe row: [e ≤ −1] = t1 − t2
    L.unit("t2", neg, -9 - 1)
    L.unit("d1", e, 9 - 1)                           # two or more rows above it: [e ≥ 2] = d1 − d2
    L.unit("d2", e, 9 - 2)
    L.unit("fu", neg + [("row 0", -1)], 1 - 9)       # on the last safe row and not in row 0

    # 4. allowed cells and sources of the wave (shells in allowed cells)
    L = new()
    for r, c in CELLS:
        ok = [(f"ok1 {r}", 1), (f"ok0 {r}", -1), (cell("je", r, c), -1), (cell("nj", r, c), -1)]
        L.unit(cell("A", r, c), ok)
        L.unit(cell("R", r, c), ok + [(cell("sh", r, c), 1), (cell("ag", r, c), -1)], -1)
        L.carry(cell("ag", r, c))
    L.unit("tide", [("t1", 1), ("t2", -1)])
    L.unit("waiting down", [("d1", 1), ("d2", -1)])
    L.unit("waiting up", [("fu", 1)])

    # 5. K wave steps: R_k(x) = min(1, R_{k−1} over the neighboring cells and x) if x is allowed.
    #    R_k = p − q with p = relu(s), q = relu(s − 1). C counts the steps in which the cell was already reached.
    for k in range(1, K + 1):
        L = new()
        reached = (lambda r, c: [(cell("R", r, c), 1)]) if k == 1 else \
                  (lambda r, c: [(cell("p", r, c), 1), (cell("q", r, c), -1)])
        for r, c in CELLS:
            s = [t for n in neighbors(r, c) + [(r, c)] for t in reached(*n)] + [(cell("A", r, c), 5)]
            L.unit(cell("p", r, c), s, -5)
            L.unit(cell("q", r, c), s, -5 - 1)
            L.unit(cell("C", r, c), reached(r, c) + ([] if k == 1 else [(cell("C", r, c), 1)]))
            L.carry(cell("ag", r, c))
            if k < K:
                L.carry(cell("A", r, c))
        L.carry("tide", "waiting down", "waiting up")

    # 6. the count C_K = C + p − q in the agent's four neighbors: Z = relu(C_K(y) − M (1 − agent(x)))
    M = K + 2
    L = new()
    for a in DIRS:
        for r, c in CELLS:
            y = (r + MOVES[a][0], c + MOVES[a][1])
            if 0 <= y[0] < H and 0 <= y[1] < W:
                L.unit(f"Z{a} {r},{c}", [(cell("C", *y), 1), (cell("p", *y), 1), (cell("q", *y), -1),
                                         (cell("ag", r, c), M)], -M)
    L.carry("tide", "waiting down", "waiting up")

    # 7. count per direction, and waiting only if no neighbor has been reached
    zs = [n for n in L.names if n.startswith("Z")]
    L = new()
    for a in DIRS:
        L.unit(f"count {a}", [(n, 1) for n in zs if n.startswith(f"Z{a} ")])
    none = [(n, -1) for n in zs]
    L.unit("wait down", none + [("waiting down", 1)])
    L.unit("wait up", none + [("waiting up", 1)])
    L.unit("wait here", none + [("waiting down", -1), ("waiting up", -1)], 1)
    L.carry("tide")

    # output: G (count + preference), waiting, and "move up" if the agent is below the last safe row
    L = new("linear")
    L.unit("noop", [("wait here", FALLBACK)])
    L.unit("up", [("count 1", G), ("wait up", FALLBACK), ("tide", G * (K + 5))], G * TIE[1])
    L.unit("down", [("count 2", G), ("wait down", FALLBACK)], G * TIE[2])
    L.unit("left", [("count 3", G)], G * TIE[3])
    L.unit("right", [("count 4", G)], G * TIE[4])
    return layers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=K_ALL, help=f"wave steps (default {K_ALL}: exact in every state)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    layers = build(args.k)
    with open(args.out, "w") as f:
        json.dump({"layers": [L.to_json() for L in layers]}, f, separators=(",", ":"))
    hidden = [len(L.names) for L in layers[:-1]]
    nz = sum(len(row) for L in layers for row in L.rows) + sum(v != 0 for L in layers for v in L.bias)
    print(f"{args.out.relative_to(Path.cwd()) if args.out.is_relative_to(Path.cwd()) else args.out}: "
          f"{len(hidden)} hidden layers, {sum(hidden)} neurons (max {max(hidden)} per layer), "
          f"{nz} non-zero weights, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
