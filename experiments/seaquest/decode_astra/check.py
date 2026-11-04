"""
Verifica la lettura a mano di weights/seaquest/paper_astra.json (vedi README.md).

Gioca gli stessi episodi di `uv run evaluate seaquest paper_astra` (seed 10000.., sticky 0.1, azioni campionate
dalla softmax con lo stesso generatore). In ogni stato ricalcola, direttamente dallo stato del gioco (posizioni di
sottomarino, pesci, sottomarini nemici, siluri nemici, sub, ossigeno) e senza usare i pesi:

  * i 32 neuroni del primo strato, con le formule della sezione 4 del README;
  * i 6 logit, con le formule della sezione 6 (valore delle righe);
  * l'azione scelta dalla regola semplificata della sezione 7 (solo condizioni vere o false);

e li confronta con il forward pass vero. Poi riassume il comportamento (righe, morti, punti).

    uv run python experiments/seaquest/decode_astra/check.py [--episodes 200]      # ~1.5 min
    uv run python experiments/seaquest/decode_astra/check.py --ablations          # famiglie di neuroni spente
    uv run python experiments/seaquest/decode_astra/check.py --esempi             # le situazioni della sezione 6
"""

import argparse
import collections
from pathlib import Path

import numpy as np

from small_games.core import get_game
from small_games.policy import policy_rng, sample_action
from small_games.weights import load_weights

ROOT = Path(__file__).resolve().parents[3]
ACTIONS = ("fermo", "sx", "su", "dx", "giu", "fuoco")
NOOP, LEFT, UP, RIGHT, DOWN, FIRE = range(6)
T = np.tanh

# Primo strato, "pericolo vicino" (h16-h23): contributo di un oggetto per colonna 0..9
NEAR_ENEMY = np.array([0, 0, 0.2, 0.4, 6, 7, 6, 2, 0.2, 0])         # pesce o sottomarino nemico
NEAR_BULLET = np.array([0, 0, 2, 4, 7, 7, 7, 5, 2, 0])              # siluro nemico
NEAR_DIVER = np.array([0, 0, -1.8, -3.6, -1, 0, -1, -3, -1.8, 0])   # sub da salvare
# Secondo strato: attrazione per distanza in righe d = |riga dell'oggetto − riga di arrivo|
A = [0.6, 0.45, 0.34, 0.255, 0.19, 0.145, 0.11, 0.08]               # sub             (≈ 0.6·0.75^d)
C = [0.35, 0.12, 0.04, 0.01]                                        # bersagli a sx   (≈ 0.35·3^−d)


def forward(layers, obs):
    x, out = np.asarray(obs, dtype=np.float64), []
    for W, b, act in layers:
        x = W @ x + b
        if act == "tanh":
            x = np.tanh(x)
        out.append(x)
    return out


def grid(g):
    """Le celle occupate da nemici (pesci e sottomarini), siluri nemici e sub, dallo stato del gioco."""
    enemy, bullet, diver = (np.zeros((10, 10), bool) for _ in range(3))
    for x, y, *_ in g.e_fish + g.e_subs:
        enemy[y, x] = True
    for x, y, *_ in g.e_bullets:
        bullet[y, x] = True
    for x, y, *_ in g.divers:
        diver[y, x] = True
    return enemy, bullet, diver


def layer1(g):
    """Il primo strato ricostruito dallo stato del gioco (sezione 4): il vettore dei 32 neuroni."""
    enemy, bullet, diver = grid(g)
    danger, reward = enemy | bullet, enemy | diver          # i due canali dell'osservazione (sono degli OR)
    h = np.empty(32)
    for r in range(9):                                      # h0-h8: il sottomarino è nella riga r
        h[r] = T(6.0) if g.sub_y == r else T(-6.0)
    for j in range(2, 9):                                   # h9-h15: nella riga j ci sono più sub che siluri
        n = (reward[j] & ~danger[j]).sum() - (danger[j] & ~reward[j]).sum()
        h[9 + j - 2] = T(8 * n - 4)
    for j in range(1, 9):                                   # h16-h23: pericolo vicino alle colonne 4-6, riga j
        both, only_b, only_d = danger[j] & reward[j], danger[j] & ~reward[j], reward[j] & ~danger[j]
        s = NEAR_ENEMY[both].sum() + NEAR_BULLET[only_b].sum() + NEAR_DIVER[only_d].sum()
        h[16 + j - 1] = T(s - 3)
    o2_cell2 = g.oxygen * 10 // 200 >= 3                    # h24: ossigeno < 60 e almeno un sub a bordo
    cell8 = g.diver_count >= 1 or g.oxygen * 10 // 200 >= 9
    h[24] = T(6 * cell8 - 6 * o2_cell2 - 3)
    for j in range(2, 9):                                   # h25-h31: qualcosa nelle colonne 0-3, riga j
        h[25 + j - 2] = T(6 * danger[j, :4].sum() - 3)
    return h


def unpack(h):
    """I neuroni del primo strato come frazioni x = (h + 1) / 2 in [0, 1], indicizzate per riga."""
    x = (h + 1) / 2
    div = {j: x[9 + j - 2] for j in range(2, 9)}
    near = {j: x[16 + j - 1] for j in range(1, 9)}
    left = {j: x[25 + j - 2] for j in range(2, 9)}
    return div, near, left, x[24]


def value(k, div, left, near):
    """Valore della riga k (sezione 5): attrazione di sub e bersagli, meno il pericolo vicino in quella riga."""
    v = -0.70
    v += 2 * sum(A[abs(j - k)] * div[j] for j in div)
    v += 2 * sum(C[abs(j - k)] * left[j] for j in left if abs(j - k) < len(C))
    v -= 6 * near.get(k, 0.0)
    return v


def logits(g, h):
    """I 6 logit con le formule della sezione 6, a partire dal primo strato ricostruito."""
    div, near, left, low = unpack(h)
    r = g.sub_y
    lo = 2 * low - 1                                        # [ossigeno basso e sub a bordo] come ±1
    if r == 0:                                              # in superficie
        fire = 1.8
    else:                                                   # resta nella riga r (e spara)
        fire = 31.8 + 30 * T(value(r, div, left, near))
    if r == 0:
        up = 0.0
    elif r == 1:
        up = 28 + 40 * T(6 * lo) + 12 * T(3 * lo)          # 80 se deve emergere, −24 altrimenti
    elif r == 2:                                            # verso la riga 1: vietato se non deve emergere
        up = 18 + 12 * T(3 * lo) + 30 * T(value(1, div, left, near) - 8.0 + 9.5 * low)
    else:                                                   # verso la riga r − 1
        up = 30 + 30 * T(value(r - 1, div, left, near) - 0.6 * near.get(r - 2, 0.0) + 1.5 * low)
    if r == 8:
        down = 0.0
    else:                                                   # verso la riga r + 1
        down = 30 + 30 * T(value(r + 1, div, left, near) - 0.6 * near.get(r + 2, 0.0))
        if r == 1:
            down += 4 - 4 * T(3 * lo)
    return np.array([-100, -100, up, -100, down, fire], dtype=np.float64)


def rule(g):
    """La regola semplificata della sezione 7: solo condizioni vere o false, nessun peso."""
    enemy, bullet, diver = grid(g)
    r = g.sub_y
    low = g.oxygen < 60 and g.diver_count >= 1
    danger = {j for j in range(1, 9) if enemy[j, 4:7].any() or bullet[j, 3:8].any()}
    weak = {j for j in range(1, 9) if enemy[j, 7] or bullet[j, 2] or bullet[j, 8]} - danger
    if r == 0:
        return FIRE if 1 in danger else DOWN               # aspetta in superficie, poi si immerge
    if r == 1 and low:
        return UP                                          # emerge

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
    if r == 1 and cand[DOWN] >= best - 0.3:                # dalla riga 1 la discesa vince i pareggi
        return DOWN
    return FIRE if cand[FIRE] >= best - 0.1 else max(cand, key=cand.get)


def death_cause(g, prev, where, sticky, h_before, h_now, lg_now):
    """Perché è finito l'episodio: (chi ha colpito il sottomarino, come ci si è trovato).

    h_now e lg_now sono primo strato e logit della rete nell'ultimo stato, h_before quelli del passo prima."""
    y0, dc0, _, o0 = prev
    if g.oxygen < 0 or o0 <= 0:                             # anche se emerge proprio all'ultimo passo
        return "", "ossigeno finito"
    if g.sub_y == 0 and dc0 == 0:
        return "", "sticky: emerso senza sub a bordo" if sticky else "emerso senza sub a bordo"
    sub = (g.sub_x, g.sub_y)
    who = "?"
    for kind, objs in (("nemico", g.e_fish + g.e_subs), ("siluro nemico", g.e_bullets)):
        for o in objs:                                      # dopo l'urto un oggetto può fare ancora un passo
            if (o[0], o[1]) == sub or where.get(id(o)) == sub:
                who = f"{kind} da {'sx' if o[2] else 'dx'}"
    near = 16 + g.sub_y - 1                                 # il neurone "pericolo vicino" della riga fatale
    if sticky:
        if g.sub_y == y0:
            return who, "sticky: tenuto fermo"
        seen = h_before is not None and h_before[near] > 0
        return who, "sticky: spostato, pericolo già acceso" if seen else "sticky: spostato, pericolo non acceso"
    if y0 == 0:
        return who, "tuffo dalla superficie (softmax)"
    if h_now[near] <= 0:                                   # c'era anche un sub nella riga (h9-h15)?
        diver = 2 <= g.sub_y <= 8 and h_now[9 + g.sub_y - 2] > 0
        return who, "pericolo non acceso, sub nella riga" if diver else "pericolo non acceso"
    if max(lg_now[UP], lg_now[DOWN], lg_now[FIRE]) < 10:   # una mossa sicura vale almeno 11.7
        return who, "in trappola: nessuna mossa sicura"
    return who, "altro"


def play(layers, episodes, seed0, verify=True):
    env = get_game("seaquest").make_env()
    st, by_row, deaths, surf = (collections.Counter() for _ in range(4))
    max_diff = 0.0
    scores = []
    for ep in range(episodes):
        seed = seed0 + ep
        rng = policy_rng(seed)
        obs, info = env.reset(seed=seed)
        g = env.env.env                                     # il gioco MinAtar, con il suo stato
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
            st["div_spawn"] += g.d_spawn_timer == 0         # in questo passo compare un sub
            st["div_seen"] += sum(1 for d in g.divers if 2 <= d[1] <= 8)
            st["div_away"] += sum(1 for d in g.divers if 2 <= d[1] <= 8 and (d[0] < 5) != d[2] and d[0] != 5)
            prev = (g.sub_y, g.diver_count, g.surface, g.oxygen)
            where = {id(o): (o[0], o[1]) for o in g.e_fish + g.e_subs + g.e_bullets}
            a = sample_action(logit_true, rng)
            st["not_argmax"] += a != a_true
            obs, rwd, term, trunc, info = env.step(a)
            if g.sub_y > 0:
                st["div_taken"] += max(0, g.diver_count - prev[1])
            elif not prev[2]:                               # è appena emerso
                surf[min(prev[1], 6)] += 1
                if prev[1] == 6:
                    st["bonus"] += prev[3] * 10 // 200
            if term:
                cause = death_cause(g, prev, where, env.last_action != a, h_before, h_true, logit_true)
                deaths[cause] += 1
                st["trap_edge"] += cause[1].startswith("in trappola") and prev[0] in (1, 2, 8)  # riga di partenza
            h_before = h_true
            done = term or trunc
        scores.append(info["score"])
        st["steps"] += info["t"]
    return st, by_row, deaths, surf, max_diff, np.array(scores)


def pinned_off(layers, units):
    """Copia della rete con i neuroni del primo strato in `units` fissati a −1 (spenti)."""
    (W1, b1, a1), *rest = layers
    W1, b1 = W1.copy(), b1.copy()
    W1[units] = 0
    b1[units] = -50
    return [(W1, b1, a1), *rest]


def observation(sub_row, fish=(), bullets=(), divers=(), oxygen=200, divers_on_board=0):
    """Un'osservazione costruita a mano: sottomarino in colonna 5 rivolto a sx, oggetti in (riga, colonna)."""
    o = np.zeros(600)

    def put(r, c, ch):
        o[(r * 10 + c) * 6 + ch] = 1

    put(sub_row, 5, 0), put(sub_row, 6, 0), put(sub_row, 6, 4)     # prua, poppa, scia della poppa
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
    ("mare vuoto, riga 5", dict(sub_row=5)),
    ("sub nella mia riga", dict(sub_row=5, divers=[(5, 1)])),
    ("sub una riga sotto", dict(sub_row=5, divers=[(6, 1)])),
    ("sub tre righe sotto", dict(sub_row=5, divers=[(8, 1)])),
    ("bersaglio a sx nella mia riga, (5,2)", dict(sub_row=5, fish=[(5, 2)])),
    ("bersaglio a sx una riga sopra, (4,2)", dict(sub_row=5, fish=[(4, 2)])),
    ("nemico in (5,4), davanti alla prua", dict(sub_row=5, fish=[(5, 4)])),
    ("nemico in (5,6), sulla poppa", dict(sub_row=5, fish=[(5, 6)])),
    ("nemico in (5,7)", dict(sub_row=5, fish=[(5, 7)])),
    ("siluro nemico in (5,3)", dict(sub_row=5, bullets=[(5, 3)])),
    ("siluro nemico in (5,8)", dict(sub_row=5, bullets=[(5, 8)])),
    ("nemico in (6,5), una riga sotto", dict(sub_row=5, fish=[(6, 5)])),
    ("nemico in (7,5), due righe sotto", dict(sub_row=5, fish=[(7, 5)])),
    ("nemico in (6,6) e sub in (6,3) e (6,8)", dict(sub_row=5, fish=[(6, 6)], divers=[(6, 3), (6, 8)])),
    ("sub (7,1), bersaglio (4,2), nemico (6,6)", dict(sub_row=5, fish=[(4, 2), (6, 6)], divers=[(7, 1)])),
    ("ossigeno 50, 2 sub a bordo", dict(sub_row=5, oxygen=50, divers_on_board=2)),
    ("ossigeno 50, nessun sub a bordo", dict(sub_row=5, oxygen=50)),
    ("ossigeno 50, 2 sub, nemico in (4,5)", dict(sub_row=5, fish=[(4, 5)], oxygen=50, divers_on_board=2)),
    ("riga 2", dict(sub_row=2)),
    ("riga 1", dict(sub_row=1)),
    ("riga 1, ossigeno 50, 2 sub", dict(sub_row=1, oxygen=50, divers_on_board=2)),
    ("riga 0 (superficie)", dict(sub_row=0)),
    ("riga 0, nemico in (1,5)", dict(sub_row=0, fish=[(1, 5)])),
    ("riga 8", dict(sub_row=8)),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--ablations", action="store_true", help="punteggio con alcune famiglie di neuroni spente")
    ap.add_argument("--esempi", action="store_true", help="logit della rete vera in situazioni costruite a mano")
    args = ap.parse_args()

    layers = load_weights(ROOT / "weights/seaquest/paper_astra.json", get_game("seaquest"))
    eps = args.episodes

    if args.esempi:
        print(f"{'situazione (riga 5 se non detto)':42s} {'su':>6s} {'giù':>6s} {'fuoco':>6s}   p(su, giù, fuoco)")
        for name, kw in EXAMPLES:
            lg = forward(layers, observation(**kw))[-1]
            p = np.exp(lg - lg.max())
            p /= p.sum()
            print(f"{name:42s} {lg[UP]:6.1f} {lg[DOWN]:6.1f} {lg[FIRE]:6.1f}   {p[UP]:.2f} {p[DOWN]:.2f} {p[FIRE]:.2f}")
        return

    if args.ablations:
        variants = {
            "rete intera": [],
            "senza sub (h9-h15)": list(range(9, 16)),
            "senza pericolo vicino (h16-h23)": list(range(16, 24)),
            "senza emersione (h24)": [24],
            "senza bersagli a sx (h25-h31)": list(range(25, 32)),
            "senza sub, bersagli e pericolo": list(range(9, 24)) + list(range(25, 32)),
        }
        for name, units in variants.items():
            st, _, deaths, surf, _, scores = play(pinned_off(layers, units), eps, args.seed0, verify=False)
            se = scores.std(ddof=1) / np.sqrt(len(scores))
            oxy = deaths["", "ossigeno finito"]
            print(f"{name:34s} {scores.mean():6.1f} ± {se:3.1f}   passi/episodio {st['steps'] / eps:5.0f}"
                  f"   emersioni/episodio {sum(surf.values()) / eps:4.1f}   morti per ossigeno {oxy / eps:4.0%}")
        return

    st, by_row, deaths, surf, max_diff, scores = play(layers, eps, args.seed0)
    n = st["n"]
    se = scores.std(ddof=1) / np.sqrt(len(scores))
    print(f"stati visitati: {n} ({eps} episodi), punteggio {scores.mean():.1f} ± {se:.1f}")
    print(f"sottomarino in colonna 5, rivolto a sinistra: {st['col5'] / n:.2%} degli stati")
    print(f"primo strato, stati in cui la ricostruzione differisce (> 1e-9): {st['bad_h']}"
          f"  (acceso/spento diverso: {st['bad_onoff']})")
    print(f"logit, differenza massima: {max_diff:.3f}")
    print(f"azione (argmax) diversa, esclusi i quasi-pareggi < 0.5: {st['bad_arg']}  (quasi-pareggi: {st['ties']})")
    print(f"regola semplificata, stessa azione della rete: {st['rule_same'] / n:.1%} degli stati; "
          f"{st['rule_same_clear'] / max(st['clear'], 1):.1%} dove la rete ha un margine > 3 ({st['clear'] / n:.0%})")

    print("\nargmax della rete per riga del sottomarino:")
    for r in range(9):
        tot = sum(by_row[r, a] for a in ACTIONS)
        row = " ".join(f"{a}:{by_row[r, a] / tot:4.0%}" for a in ("su", "giu", "fuoco"))
        print(f"  riga {r}: {tot / n:5.1%} degli stati | {row}")
    print(f"azione campionata diversa dall'argmax: {st['not_argmax'] / n:.1%} dei passi")
    print(f"ossigeno < 60 con almeno un sub a bordo: {st['low'] / n:.1%} degli stati, argmax su in "
          f"{st['low_up'] / max(st['low'], 1):.0%} di questi")

    print(f"\nmorti ({eps} episodi):")
    hows = collections.Counter()
    for (who, how), v in deaths.items():
        hows[how] += v
    for how, v in hows.most_common():
        whos = sorted(((w, c) for (w, h), c in deaths.items() if h == how and w), key=lambda t: -t[1])
        print(f"  {how:46s} {v:4d} {v / eps:4.0%}  " + ", ".join(f"{w} {c}" for w, c in whos))
    print(f"  (in trappola nelle righe 1, 2 e 8: {st['trap_edge']})")
    print(f"\nemersioni per episodio: {sum(surf.values()) / eps:.1f}; sub a bordo quando emerge:",
          " ".join(f"{k}:{surf[k]}" for k in sorted(surf)))
    print(f"sub comparsi per episodio {st['div_spawn'] / eps:.1f}, raccolti {st['div_taken'] / eps:.1f}; "
          f"dei sub visibili (righe 2-8), {st['div_away'] / max(st['div_seen'], 1):.0%} ha già passato la colonna 5")
    print(f"punti per episodio {scores.mean():.1f}: nemici colpiti {(scores.sum() - st['bonus']) / eps:.1f}, "
          f"emersioni con 6 sub {st['bonus'] / eps:.1f}; passi per episodio {st['steps'] / eps:.0f}")


if __name__ == "__main__":
    main()
