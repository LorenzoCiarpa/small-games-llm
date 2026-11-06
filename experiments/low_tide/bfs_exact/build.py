"""
L'euristica BFS di Low Tide, esatta, come pesi di un MLP senza i limiti del gioco: costruisce bfs_exact.json.

bfs_net (../bfs_net) sta nei limiti del gioco (3 strati da 64) e calcola una versione a potenziale della BFS.
Questa rete invece fa la stessa mossa di `heuristic_policy` in ogni stato, ma le servono molti più neuroni:
uno strato per ogni passo della BFS e 5 unità per cella in ogni strato (vedi README.md).

Idea: invece di cercare dall'agente, un'onda parte dalle conchiglie raggiungibili e a ogni strato avanza di una
cella, solo nelle celle consentite (righe sopra l'ultima sicura, niente meduse, niente celle vicine a una medusa se
la marea sale). Ogni cella conta per quanti strati è stata raggiunta: dopo K strati il conteggio è K + 1 − d, con d
la distanza dalla conchiglia più vicina. La mossa della BFS è il vicino dell'agente con il conteggio più alto, a
parità nell'ordine su, giù, sinistra, destra (lo stesso ordine con cui la BFS visita i vicini). Le regole della
marea sono quelle dell'euristica.

Tutti i valori interni sono interi piccoli, quindi il calcolo in float64 è esatto. I logit hanno margini ≥ 1000:
exp(−1000) = 0 in float64, quindi anche campionando dalla softmax la rete sceglie sempre la stessa mossa.

    uv run python experiments/low_tide/bfs_exact/build.py [--k 98]

K = 98 basta per ogni stato possibile: in una griglia di 100 celle la conchiglia più vicina è a distanza ≤ 99.
"""

import argparse
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "bfs_exact.json"

H = W = 10
CELLS = [(r, c) for r in range(H) for c in range(W)]
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))  # fermo, su, giù, sinistra, destra
DIRS = (1, 2, 3, 4)
K_ALL = H * W - 2                 # la BFS trova una conchiglia a distanza ≤ 99: al vicino serve d ≤ 98
G = 1e4                           # scala dei logit: la preferenza a parità vale 0.1 G = 1000
TIE = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
FALLBACK = 2 * G                  # logit della mossa di attesa quando nessuna conchiglia è raggiungibile


def neighbors(r, c):
    return [(r + dr, c + dc) for dr, dc in MOVES[1:] if 0 <= r + dr < H and 0 <= c + dc < W]


def input_names():
    names = []
    for ch in ("agente", "acqua", "conchiglia", "medusa"):
        names += [f"{ch} {r},{c}" for r, c in CELLS]
    return names + ["sale", "countdown", "stordito"]


class Layer:
    """Uno strato: ogni unità è act(somma pesata di unità dello strato precedente + bias), per nome."""

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

    ag = [cell("agente", r, c) for r, c in CELLS]

    # 1. dall'osservazione: copie delle celle, meduse vicine, livello dell'acqua, riga dell'agente
    L = new()
    for r, c in CELLS:
        L.unit(cell("ag", r, c), [(cell("agente", r, c), 1)])
        L.unit(cell("sh", r, c), [(cell("conchiglia", r, c), 1)])
        L.unit(cell("je", r, c), [(cell("medusa", r, c), 1)])
        # > 0 solo se la marea sale e una cella vicina ha una medusa (vale il numero di meduse vicine)
        L.unit(cell("nj", r, c), [(cell("medusa", *n), 1) for n in neighbors(r, c)] + [("sale", 4)], -4)
    L.unit("livello", [(cell("acqua", r, 0), 1) for r in range(H)])            # righe allagate
    L.unit("sale", [("sale", 1)])
    L.unit("u", [("countdown", 10)], -5)                                        # 0 se countdown ≤ 2, ≥ 1 se ≥ 3
    L.unit("riga", [(a, r) for a, (r, c) in zip(ag, CELLS)])                   # riga dell'agente
    L.unit("riga 0", [(cell("agente", 0, c), 1) for c in range(W)])            # 1 se l'agente è nella riga 0

    # 2. v = la marea sale e manca poco (countdown ≤ 2): margine 2 invece di 1
    L = new()
    for r, c in CELLS:
        L.carry(cell("ag", r, c), cell("sh", r, c), cell("je", r, c), cell("nj", r, c))
    L.carry("livello", "sale", "riga", "riga 0")
    L.unit("v", [("sale", 1), ("u", -1)])

    # 3. ultima riga sicura: deepest = 9 − livello − sale − v.  m_R = deepest − R, e = deepest − riga dell'agente
    L = new()
    for r, c in CELLS:
        L.carry(cell("ag", r, c), cell("sh", r, c), cell("je", r, c), cell("nj", r, c))
    deepest = [("livello", -1), ("sale", -1), ("v", -1)]
    for R in range(H):
        L.unit(f"ok1 {R}", deepest, 9 - R + 1)       # [R ≤ deepest] = relu(m_R + 1) − relu(m_R)
        L.unit(f"ok0 {R}", deepest, 9 - R)
    e = deepest + [("riga", -1)]
    neg = [(k, -v) for k, v in e]
    L.unit("t1", neg, -9)                            # sotto l'ultima riga sicura: [e ≤ −1] = t1 − t2
    L.unit("t2", neg, -9 - 1)
    L.unit("d1", e, 9 - 1)                           # due righe o più sopra: [e ≥ 2] = d1 − d2
    L.unit("d2", e, 9 - 2)
    L.unit("fu", neg + [("riga 0", -1)], 1 - 9)      # sull'ultima riga sicura e non nella riga 0

    # 4. celle consentite e sorgenti dell'onda (conchiglie in celle consentite)
    L = new()
    for r, c in CELLS:
        ok = [(f"ok1 {r}", 1), (f"ok0 {r}", -1), (cell("je", r, c), -1), (cell("nj", r, c), -1)]
        L.unit(cell("A", r, c), ok)
        L.unit(cell("R", r, c), ok + [(cell("sh", r, c), 1), (cell("ag", r, c), -1)], -1)
        L.carry(cell("ag", r, c))
    L.unit("marea", [("t1", 1), ("t2", -1)])
    L.unit("attesa giù", [("d1", 1), ("d2", -1)])
    L.unit("attesa su", [("fu", 1)])

    # 5. K passi dell'onda: R_k(x) = min(1, R_{k−1} sulle celle vicine e su x) se x è consentita.
    #    R_k = p − q con p = relu(s), q = relu(s − 1). C conta i passi in cui la cella era già raggiunta.
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
        L.carry("marea", "attesa giù", "attesa su")

    # 6. il conteggio C_K = C + p − q nei quattro vicini dell'agente: Z = relu(C_K(y) − M (1 − agente(x)))
    M = K + 2
    L = new()
    for a in DIRS:
        for r, c in CELLS:
            y = (r + MOVES[a][0], c + MOVES[a][1])
            if 0 <= y[0] < H and 0 <= y[1] < W:
                L.unit(f"Z{a} {r},{c}", [(cell("C", *y), 1), (cell("p", *y), 1), (cell("q", *y), -1),
                                         (cell("ag", r, c), M)], -M)
    L.carry("marea", "attesa giù", "attesa su")

    # 7. conteggio per direzione, e l'attesa solo se nessun vicino è stato raggiunto
    zs = [n for n in L.names if n.startswith("Z")]
    L = new()
    for a in DIRS:
        L.unit(f"conteggio {a}", [(n, 1) for n in zs if n.startswith(f"Z{a} ")])
    none = [(n, -1) for n in zs]
    L.unit("aspetta giù", none + [("attesa giù", 1)])
    L.unit("aspetta su", none + [("attesa su", 1)])
    L.unit("aspetta qui", none + [("attesa giù", -1), ("attesa su", -1)], 1)
    L.carry("marea")

    # uscita: G (conteggio + preferenza), attesa, e "sali" se l'agente è sotto l'ultima riga sicura
    L = new("linear")
    L.unit("fermo", [("aspetta qui", FALLBACK)])
    L.unit("su", [("conteggio 1", G), ("aspetta su", FALLBACK), ("marea", G * (K + 5))], G * TIE[1])
    L.unit("giù", [("conteggio 2", G), ("aspetta giù", FALLBACK)], G * TIE[2])
    L.unit("sinistra", [("conteggio 3", G)], G * TIE[3])
    L.unit("destra", [("conteggio 4", G)], G * TIE[4])
    return layers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=K_ALL, help=f"passi dell'onda (default {K_ALL}: esatta in ogni stato)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    layers = build(args.k)
    with open(args.out, "w") as f:
        json.dump({"layers": [L.to_json() for L in layers]}, f, separators=(",", ":"))
    hidden = [len(L.names) for L in layers[:-1]]
    nz = sum(len(row) for L in layers for row in L.rows) + sum(v != 0 for L in layers for v in L.bias)
    print(f"{args.out.relative_to(Path.cwd()) if args.out.is_relative_to(Path.cwd()) else args.out}: "
          f"{len(hidden)} strati nascosti, {sum(hidden)} neuroni (max {max(hidden)} per strato), "
          f"{nz} pesi diversi da zero, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
