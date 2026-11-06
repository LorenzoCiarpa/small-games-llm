"""
L'euristica di Low Tide scritta come pesi di un MLP: costruisce weights/low_tide/bfs_net.json.

L'euristica (`heuristic_policy` in src/small_games/games/low_tide/baselines.py) cerca con una BFS la conchiglia più
vicina, evitando le meduse e le righe che stanno per allagarsi. Una BFS esatta non entra in una rete con 3 strati
da 64 unità (vedi README.md). Questa rete calcola una versione a potenziale dello stesso algoritmo:

- **conchiglie:** ognuna vale 6^(18 − d), con d la distanza di Manhattan dall'agente. Le conchiglie di ogni
  direzione si sommano: sopra (righe più in alto), sotto (righe più in basso ma consentite), a sinistra e a
  destra (colonne, qualsiasi riga). Vince la direzione con la somma più alta: la conchiglia più vicina pesa più di
  tutte quelle più lontane insieme, salvo 6 o più conchiglie a un passo in più. A parità, la preferenza della BFS:
  su, giù, sinistra, destra.
- **meduse:** mai entrare in una cella con una medusa. Tolta quella direzione, vince la successiva.
- **marea:** come l'euristica. Se l'agente è sotto l'ultima riga sicura sale; non scende sotto quella riga; se non
  c'è nessuna conchiglia aspetta una riga sopra.

`policy_logits(obs)` è la specifica, scritta direttamente sull'osservazione. `build()` scrive i pesi che la
calcolano; check.py verifica che rete e specifica scelgano la stessa mossa.

    uv run python experiments/low_tide/bfs_net/build.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "weights" / "low_tide" / "bfs_net.json"

H = W = 10
ROWS = range(3, 9)               # righe con conchiglie e meduse viste dalla rete (la 9 non è mai consentita)
BASE, T = 6, 18                  # una conchiglia a distanza d vale BASE^(T-d)
P1 = BASE ** (T - 1)             # valore di una conchiglia adiacente: l'unità di misura dei potenziali
DELTA = 0.02                     # preferenza a parità: su ×(1+3δ), giù ×(1+2δ), sinistra ×(1+δ)
LAMBDA = 1.4e16                  # scala dei logit: anche a distanza 17 i pareggi si rompono di ~100
W_VETO, W_FORCE = 1e3 * LAMBDA, 1e6 * LAMBDA
PHI = 30.0                       # logit dell'attesa: la softmax la sceglie con probabilità ~1, ma resta sotto ogni potenziale
JELLY = 0.5                      # valore (in unità P1) di una medusa adiacente
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
    """Potenziali (su, giù, sinistra, destra) in unità P1, le meduse adiacenti dirette dell'agente e σ.

    Con jelly_value=JELLY i potenziali sono quelli delle unità di direzione della rete, in cui una medusa adiacente
    entra come un valore 0.5 che poi il veto toglie di mezzo; la specifica li usa con jelly_value=0.
    """
    g, r, c, level, rising, countdown = decode(obs)
    margin = (1 + (countdown <= 2)) if rising else 0
    deepest = 9 - level - margin            # ultima riga consentita
    sigma = deepest - r                     # righe consentite sotto l'agente (< 0: deve salire)
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
    """I 5 logit della specifica (fermo, su, giù, sinistra, destra), dall'osservazione."""
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
        """Aggiunge un neurone: weights = {indice d'ingresso: peso}. Restituisce il suo indice."""
        self.rows.append(({int(k): float(v) for k, v in weights.items() if v != 0}, float(bias)))
        self.names.append(name)
        return len(self.rows) - 1

    def to_json(self):
        return {"in": self.n_in, "out": len(self.rows), "activation": self.activation,
                "W": {"sparse": [[i, j, w] for i, (ws, _) in enumerate(self.rows) for j, w in sorted(ws.items())]},
                "b": {"sparse": [[i, b] for i, (_, b) in enumerate(self.rows) if b != 0]}}


def build():
    # ---- strato 1 (relu): una conchiglia o medusa per cella, più posizione e marea --------------------------
    L1 = Layer(403, "relu")
    cell = {}
    for R in ROWS:
        for C in range(W):
            w = {IDX_SHELL + 10 * R + C: P1, IDX_JELLY + 10 * R + C: P1 * JELLY}
            for ar in range(H):                       # l'agente in (ar, ac): pot(d) − P1, sempre ≤ 0
                for ac in range(W):
                    w[IDX_AGENT + 10 * ar + ac] = pot(abs(R - ar) + abs(C - ac)) - P1
            cell[R, C] = L1.unit(f"cella ({R},{C})", w)
    row = L1.unit("riga dell'agente r", {IDX_AGENT + p: p // 10 for p in range(100)})
    col = L1.unit("colonna dell'agente c", {IDX_AGENT + p: p % 10 for p in range(100)})
    m = L1.unit("m = 9 − r − L", {**{IDX_AGENT + p: -(p // 10) for p in range(100)},
                                  **{IDX_WATER + 10 * R: -1 for R in range(3, 10)}}, 9)
    v = L1.unit("margine ≈ 0/1/2", {IDX_RISING: 2, IDX_COUNTDOWN: -1})

    # ---- strato 2 (tanh): potenziali per riga e colonna, meduse adiacenti, soglie --------------------------
    L2 = Layer(len(L1.rows), "tanh")
    EPS, BQ, BT, BS = 1e-3, 500.0, 50.0, 250.0
    P, Q = {}, {}
    for R in ROWS:
        cells = [cell[R, C] for C in range(W)]
        P["r", R] = L2.unit(f"potenziale riga {R}", {k: EPS / P1 for k in cells})
        Q["r", R] = L2.unit(f"medusa adiacente in riga {R}", {k: BQ / P1 for k in cells}, -BQ * 0.45)
    for C in range(W):
        cells = [cell[R, C] for R in ROWS]
        P["c", C] = L2.unit(f"potenziale colonna {C}", {k: EPS / P1 for k in cells})
        Q["c", C] = L2.unit(f"medusa adiacente in colonna {C}", {k: BQ / P1 for k in cells}, -BQ * 0.45)
    tr = {k: L2.unit(f"[r > {k}]", {row: BT}, -BT * (k + 0.5)) for k in range(1, 9)}
    tc = {k: L2.unit(f"[c > {k}]", {col: BT}, -BT * (k + 0.5)) for k in range(0, 9)}
    allowed = {R: L2.unit(f"riga {R} consentita", {m: BS, row: BS, v: -BS}, BS * (0.5 - R)) for R in ROWS}
    s = {k: L2.unit(f"[σ ≥ {k}]", {m: BS, v: -BS}, BS * (0.5 - k)) for k in range(3)}

    # ---- strato 3 (relu): potenziali per direzione, veti, eventi della marea --------------------------------
    # una soglia tanh t vale ±1: [x] = (t + 1)/2. M spegne un'unità quando la sua condizione è falsa.
    L3 = Layer(len(L2.rows), "relu")
    M = 4.0
    up, down, left, right, veto = [], [], [], [], {"up": [], "down": [], "left": [], "right": []}
    for R in ROWS:
        up.append(L3.unit(f"su, riga {R}", {P["r", R]: 1 / EPS, tr[R]: M / 2}, -M / 2))          # r > R
        down.append(L3.unit(f"giù, riga {R}", {P["r", R]: 1 / EPS, tr[R - 1]: -M / 2, allowed[R]: M / 2}, -M))
    for C in range(W):
        if C <= 8:
            left.append(L3.unit(f"sinistra, colonna {C}", {P["c", C]: 1 / EPS, tc[C]: M / 2}, -M / 2))  # c > C
        if C >= 1:
            right.append(L3.unit(f"destra, colonna {C}", {P["c", C]: 1 / EPS, tc[C - 1]: -M / 2}, -M / 2))

    def veto_unit(name, key, gate_pos, gate_neg, const):
        """relu(0.475·Q − potenziale + 0.475 − M·(1 − [agente accanto])): > 0 solo con una medusa adiacente."""
        w = {Q[key]: 0.475, P[key]: -1 / EPS}
        for k in gate_pos:
            w[k] = w.get(k, 0) + M / 2
        for k in gate_neg:
            w[k] = w.get(k, 0) - M / 2
        return L3.unit(name, w, 0.475 - M + M * const)

    # [r = R+1] = ([r>R] − [r>R+1])/2; i termini costanti delle soglie che non esistono vanno nel bias
    for R in ROWS:
        if R + 1 <= 8:
            veto["up"].append(veto_unit(f"medusa sopra (riga {R})", ("r", R), [tr[R]], [tr[R + 1]], 0))
        veto["down"].append(veto_unit(f"medusa sotto (riga {R})", ("r", R), [tr[R - 2]], [tr[R - 1]], 0))
    for C in range(W):
        if C <= 8:   # agente in colonna C+1
            if C + 1 <= 8:
                veto["left"].append(veto_unit(f"medusa a sinistra (colonna {C})", ("c", C), [tc[C]], [tc[C + 1]], 0))
            else:
                veto["left"].append(veto_unit(f"medusa a sinistra (colonna {C})", ("c", C), [tc[C]], [], 0.5))
        if C >= 1:   # agente in colonna C−1
            if C - 2 >= 0:
                veto["right"].append(veto_unit(f"medusa a destra (colonna {C})", ("c", C), [tc[C - 2]], [tc[C - 1]], 0))
            else:
                veto["right"].append(veto_unit(f"medusa a destra (colonna {C})", ("c", C), [], [tc[C - 1]], 0.5))
    forced = L3.unit("σ < 0: deve salire", {s[0]: -0.5}, 0.5)
    no_down = L3.unit("σ ≤ 0: non può scendere", {s[1]: -0.5}, 0.5)
    wait_here = L3.unit("σ = 1: aspetta qui", {s[1]: 0.5, s[2]: -0.5})
    wait_down = L3.unit("σ ≥ 2: scende ad aspettare", {s[2]: 0.5}, 0.5)

    # ---- uscita (linear) ------------------------------------------------------------------------------------
    L4 = Layer(len(L3.rows), "linear")
    L4.unit("fermo", {wait_here: PHI})
    L4.unit("su", {**{k: LAMBDA * (1 + 3 * DELTA) for k in up}, **{k: -W_VETO for k in veto["up"]},
                   forced: W_FORCE, no_down: PHI})
    L4.unit("giù", {**{k: LAMBDA * (1 + 2 * DELTA) for k in down}, **{k: -W_VETO for k in veto["down"]},
                    no_down: -W_FORCE, wait_down: PHI})
    L4.unit("sinistra", {**{k: LAMBDA * (1 + DELTA) for k in left}, **{k: -W_VETO for k in veto["left"]}})
    L4.unit("destra", {**{k: LAMBDA for k in right}, **{k: -W_VETO for k in veto["right"]}})

    layers = [L1, L2, L3, L4]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"source": "experiments/low_tide/bfs_net/build.py: l'euristica BFS di Low Tide come potenziale",
                   "layers": [L.to_json() for L in layers]}, f, separators=(",", ":"))
    return layers


if __name__ == "__main__":
    from small_games.core import get_game
    from small_games.weights import load_weights

    layers = build()
    load_weights(OUT, get_game("low_tide"))          # il file deve passare i controlli come ogni altro
    sizes = " → ".join([str(layers[0].n_in)] + [f"{len(L.rows)} ({L.activation})" for L in layers])
    nnz = sum(len(ws) for L in layers for ws, _ in L.rows)
    print(f"{OUT.relative_to(ROOT)}: {sizes}, {nnz} pesi diversi da zero")
