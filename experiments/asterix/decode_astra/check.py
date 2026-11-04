"""
Verifica la lettura a mano di weights/asterix/paper_astra.json (vedi README.md).

Gioca episodi con la rete come `evaluate` (softmax, sticky 0.1, stessi seed e stesso generatore). In ogni stato
ricalcola, direttamente dallo stato del gioco e senza usare i pesi:

  - le pre-attivazioni dei 32 neuroni del primo strato (sezione 4 del README);
  - lo stato dei 32 neuroni del secondo strato: spento, a metà, acceso (sezione 5);
  - i 6 logit con le formule della sezione 6;
  - i logit della regola semplificata della sezione 7;

e li confronta con il forward pass vero. Con --regola N gioca anche N episodi con la regola al posto della rete,
con --varianti N gioca N episodi con ognuna delle varianti della rete della sezione 9 (un pezzo tolto).

    uv run python experiments/asterix/decode_astra/check.py                  # ~40 s
    uv run python experiments/asterix/decode_astra/check.py --regola 1000    # + ~40 s
    uv run python experiments/asterix/decode_astra/check.py --varianti 1000  # + ~3 min
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
FERMO, SU, GIU = 0, 2, 4
MOSSE = {FERMO: "fermo", SU: "su", GIU: "giu"}


def forward(layers, obs):
    """Le pre-attivazioni di ogni strato (per l'ultimo, i logit)."""
    x, out = np.asarray(obs, dtype=np.float64), []
    for W, b, act in layers:
        z = W @ x + b
        out.append(z)
        x = np.tanh(z) if act == "tanh" else z
    return out


# ---------------------------------------------------------------- lo stato del gioco

def entita(g):
    """riga -> (colonna, va a destra, è un tesoro). Al più un'entità per riga, nelle righe 1-8."""
    return {e[1]: (e[0], e[2], e[3]) for e in g.entities if e is not None}


def arrivo(x, destra):
    """Celle che mancano alla colonna 5 a un'entità che ci sta andando incontro (0: è già nella colonna 5).
    None se si allontana dalla colonna 5."""
    if x == 5:
        return 0
    if (x < 5) == destra:
        return abs(x - 5)
    return None


def scia(x, destra):
    """La colonna della scia: la cella da cui l'entità arriva."""
    c = x - 1 if destra else x + 1
    return c if 0 <= c <= 9 else None


def in_arrivo(g):
    """riga -> d, per i nemici e per i tesori che stanno andando verso la colonna 5."""
    nemici, tesori = {}, {}
    for k, (x, destra, oro) in entita(g).items():
        d = arrivo(x, destra)
        if d is not None:
            (tesori if oro else nemici)[k] = d
    return nemici, tesori


# ---------------------------------------------------------------- sezione 4: primo strato

NEMICO_VICINO = {0: 6, 1: 6, 2: 5}         # h0-h15, righe sopra e sotto: nemico in arrivo a d celle
NEMICO_MIA = {0: 6, 1: 6, 2: 4, 3: 3}      # h16-h23, la mia riga
# effetti collaterali della scia (sezione 4.5): la scia di un tesoro nei neuroni di pericolo,
# quella di un nemico nei neuroni tesoro
SCIA_VICINO = {4: -5, 5: -6, 6: -5}
SCIA_MIA = {3: -3, 4: -4, 5: -9, 6: -4, 7: -3}
SCIA_TESORO = {1: -8, 2: -8, 3: -16, 4: -16, 5: -24, 6: -16, 7: -16, 8: -8, 9: -8}


def guardia_sopra(k, r):
    """Peso del giocatore in riga r nel neurone 'pericolo in riga k, sopra di me'."""
    if k == 1:
        return {1: 8, 2: 6, 3: 3}.get(r, 0)
    return {k + 1: 4, k + 2: 3}.get(r, 0)


def guardia_sotto(k, r):
    if k == 8:
        return {8: 8, 7: 6, 6: 3}.get(r, 0)
    return {k - 1: 4, k - 2: 3}.get(r, 0)


def guardia_mia(k, r):
    return (7 if k in (1, 8) else 4) if r == k else 0


def quota_tesoro(k, r):
    """Il termine del giocatore nel neurone tesoro della riga k: + se il tesoro è sopra, − se è sotto."""
    n = abs(r - k)
    t = 0.0 if n == 0 else 0.8 if n == 1 else 1.2 + 0.1 * n
    return t if r >= k else -t


def primo_strato(g):
    """Le 32 pre-attivazioni del primo strato, dallo stato del gioco."""
    r, ent = g.player_y, entita(g)
    pre = np.full(32, -8.0)
    for k in range(1, 9):
        vicino = mia = tesoro = 0.0
        if k in ent:
            x, destra, oro = ent[k]
            d, c = arrivo(x, destra), scia(x, destra)
            if oro:
                tesoro = 8.0 if d is not None else 0.0
                vicino, mia = SCIA_VICINO.get(c, 0), SCIA_MIA.get(c, 0)
            else:
                if d is not None:
                    vicino, mia = NEMICO_VICINO.get(d, 0), NEMICO_MIA.get(d, 0)
                tesoro = SCIA_TESORO.get(c, 0)
        pre[k - 1] += vicino + guardia_sopra(k, r)      # h0-h7:   pericolo sopra
        pre[7 + k] += vicino + guardia_sotto(k, r)      # h8-h15:  pericolo sotto
        pre[15 + k] += mia + guardia_mia(k, r)          # h16-h23: pericolo nella mia riga
        pre[23 + k] += tesoro + quota_tesoro(k, r)      # h24-h31: tesoro
    return pre


# ---------------------------------------------------------------- sezione 6: le formule

def logit_formule(pre):
    """I logit della sezione 6, a partire dal primo strato."""
    h = np.tanh(pre)
    a_su, a_giu, a_fermo = (1 + h[0:8]).sum(), (1 + h[8:16]).sum(), (1 + h[16:24]).sum()
    t = h[24:32]                                         # tesoro nelle righe 1..8
    A, B = np.tanh(21.6 + 24 * t), np.tanh(8 + 20 * t)
    C, D = np.tanh(-8 + 20 * t), np.tanh(-21.6 + 24 * t)
    lo = np.full(6, -1000.0)
    lo[FERMO] = 17 - 80 * np.tanh(a_fermo - 1) + 16 * (B[:7] - C[:7]).sum() + 16 * B[7]
    lo[SU] = 0.15 - 80 * np.tanh(a_su - 1) + 6 * (C[:7] - D[:7]).sum()
    lo[GIU] = -0.69 - 80 * np.tanh(a_giu - 1) + 6 * (A[1:] - B[1:]).sum()
    return lo


# ---------------------------------------------------------------- sezione 7: la regola

ATTRAZIONE = {1: 12.0, 2: 8.0, 3: 5.3, 4: 3.1, 5: 1.8, 6: 1.1, 7: 0.8}   # tesoro in arrivo a n righe


def colonna_scia_oro(g, r):
    """Colonna della scia del tesoro nella mia riga (None se non c'è)."""
    e = entita(g).get(r)
    return scia(e[0], e[1]) if e is not None and e[2] else None


def allarmi(g):
    """Allarme di fermo, su e giù (tabella della sezione 5)."""
    r = g.player_y
    nemici, _ = in_arrivo(g)
    c = colonna_scia_oro(g, r) if r in (1, 8) else None

    def spostamento(passo):
        t1, t2 = r + passo, r + 2 * passo
        if not 1 <= t1 <= 8:                             # contro il muro: la mossa ti lascia dove sei
            if c is not None and 4 <= c <= 6:            # effetto di bordo della scia di un tesoro
                return 0.0
            d = nemici.get(r)
            return 2.0 if d is not None and d <= 2 else 1.0
        a = 0.0
        d = nemici.get(t1)
        if t1 in (1, 8):                                 # verso una riga di bordo
            a += 2.0 if d is not None and d <= 2 else 0.036
        elif d is not None and d <= 2:
            a += 1.96 if d <= 1 else 1.76
        d = nemici.get(t2) if 1 <= t2 <= 8 else None
        if d is not None and d <= 2:
            a += 1.76 if d <= 1 else 1.0
        return a

    d = nemici.get(r)
    if r in (1, 8):
        if d is not None and d <= 3:
            fermo = 2.0 if d <= 2 else 1.96
        else:
            fermo = 0.0 if c is not None and 3 <= c <= 7 else 0.24
    else:
        fermo = {0: 1.96, 1: 1.96, 2: 1.0, 3: 0.24}.get(d, 0.0)
    return fermo, spostamento(-1), spostamento(+1)


def regola(g):
    """I logit della regola semplificata della sezione 7."""
    r = g.player_y
    _, tesori = in_arrivo(g)
    a_fermo, a_su, a_giu = allarmi(g)
    lo = np.full(6, -1000.0)
    lo[FERMO] = 1 - 80 * np.tanh(a_fermo - 1) + 32 * (r in tesori)
    lo[SU] = 0.15 - 80 * np.tanh(a_su - 1) + sum(ATTRAZIONE[r - k] for k in tesori if k < r)
    lo[GIU] = -80 * np.tanh(a_giu - 1) + sum(ATTRAZIONE[k - r] for k in tesori if k > r)
    return lo


# ---------------------------------------------------------------- sezione 5: il secondo strato

def stato_allarme(a):
    """Lo stato atteso di g0, g1, g2 = tanh(allarme − 1): -1 spento, 0 a metà, +1 acceso."""
    return -1 if a < 0.5 else (0 if a <= 1.5 else 1)


def stato_reale(v):
    return -1 if v < -0.3 else (0 if v <= 0.3 else 1)


def secondo_strato(g):
    """Stato atteso dei 32 neuroni del secondo strato (g3-g31 solo il segno: +1 acceso, -1 spento)."""
    r = g.player_y
    _, tesori = in_arrivo(g)
    a_fermo, a_su, a_giu = allarmi(g)
    st = np.full(32, -1)
    st[0], st[1], st[2] = stato_allarme(a_su), stato_allarme(a_giu), stato_allarme(a_fermo)
    for k in range(1, 9):
        s = r - k if k in tesori else None               # + tesoro sopra, − tesoro sotto, 0 nella mia riga
        on = lambda soglia: 1 if s is not None and s >= soglia else -1
        if k >= 2:
            st[4 * k - 2] = on(-2)                       # A: tesoro sopra, nella mia riga o al più 2 sotto
        st[4 * k - 1] = on(0)                            # B: tesoro sopra o nella mia riga
        if k <= 7:
            st[4 * k] = on(1)                            # C: tesoro sopra
            st[4 * k + 1] = on(3)                        # D: tesoro sopra, ad almeno 3 righe
    return st


def stato_secondo_reale(g_true):
    st = np.where(g_true > 0, 1, -1)
    st[:3] = [stato_reale(v) for v in g_true[:3]]
    return st


# ---------------------------------------------------------------- le situazioni

SITUAZIONI = ("nessuna mossa sicura", "nemico nella mia riga", "tesoro nella mia riga", "riga di bordo",
              "tesoro in un'altra riga", "niente in arrivo")


def situazione(g):
    """La prima che vale, in quest'ordine."""
    r = g.player_y
    nemici, tesori = in_arrivo(g)
    if min(allarmi(g)) >= 0.5:
        return SITUAZIONI[0]
    if r in nemici and nemici[r] <= 3:
        return SITUAZIONI[1]
    if r in tesori:
        return SITUAZIONI[2]
    if r in (1, 8):
        return SITUAZIONI[3]
    if tesori:
        return SITUAZIONI[4]
    return SITUAZIONI[5]


def sopravvive(g, a):
    gg = copy.deepcopy(g)
    gg.act(a)
    return not gg.terminal


def softmax(lo):
    p = np.exp(lo - lo.max())
    return p / p.sum()


def gioca(game, logit, episodes, seed0):
    """Punteggio medio, errore standard e durata media degli episodi di una politica, con il protocollo di
    `evaluate`. logit(env, obs) -> i 6 logit."""
    ret, durata = [], []
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
        durata.append(env.t)
    ret = np.array(ret)
    return ret.mean(), ret.std(ddof=1) / np.sqrt(len(ret)), np.mean(durata)


def varianti(layers):
    """La rete con un pezzo tolto (sezione 9). Gli indici sono quelli delle sezioni 4-6."""
    (W1, b1, f1), (W2, b2, f2), (W3, b3, f3) = layers
    QUIETE = 60.93                       # -80·tanh(-1): quanto vale un allarme spento
    TESORO = slice(3, 32)                # g3-g31

    def rete(W1n=W1, W3n=W3, b3n=b3):
        return [(W1n, b1, f1), (W2, b2, f2), (W3n, b3n, f3)]

    out = {}
    W, b = W3.copy(), b3.copy()
    W[:, TESORO] = 0
    b[FERMO], b[GIU] = 1.0, 0.0          # i valori che le parti tolte davano senza tesori
    out["senza tesori (g3-g31 non arrivano ai logit)"] = rete(W3n=W, b3n=b)
    W, b = W3.copy(), b3.copy()
    W[FERMO, TESORO] = 0
    b[FERMO] = 1.0
    out["senza il +32 a fermo per il tesoro nella mia riga"] = rete(W3n=W, b3n=b)
    W, b = W3.copy(), b3.copy()
    W[[SU, GIU], TESORO] = 0
    b[GIU] = 0.0
    out["senza l'attrazione verso i tesori sopra e sotto"] = rete(W3n=W, b3n=b)
    W = W1.copy()
    player = np.arange(0, 600, 6)        # canale 0, il giocatore
    for i in range(16):                  # h0-h15: tolgo il giocatore a 2 righe dalla riga guardata
        W[i, player[W1[i, player] == 3]] = 0
    out["sopra e sotto guardano solo la riga d'arrivo"] = rete(W1n=W)
    W, b = W3.copy(), b3.copy()
    W[:, 0:3] = 0
    b[[FERMO, SU, GIU]] += QUIETE
    out["senza allarmi (g0-g2 come se fossero spenti)"] = rete(W3n=W, b3n=b)
    W, b = W3.copy(), b3.copy()
    W[FERMO, 2] = 0
    b[FERMO] += QUIETE
    out["senza l'allarme di fermo (g2)"] = rete(W3n=W, b3n=b)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--regola", type=int, default=0, help="episodi da giocare con la regola della sezione 7")
    ap.add_argument("--varianti", type=int, default=0, help="episodi da giocare con ogni variante della sezione 9")
    args = ap.parse_args()

    game = get_game("asterix")
    layers = load_weights(ROOT / "weights/asterix/paper_astra.json", game)
    n = fuori_colonna = bad_h = bad_g = bad_arg = ties = bordo = 0
    max_h = max_f = max_r = tv = 0.0
    bad_g_neuroni = collections.Counter()
    casi = collections.Counter()
    ties_sit = collections.Counter()
    niente_bloccato = 0
    prob = collections.defaultdict(lambda: np.zeros(3))
    righe = collections.Counter()
    morti = collections.Counter()
    addosso = collections.Counter()     # morti entrando in una riga con un nemico già nella colonna 5
    righe_morte = collections.Counter()
    ritorni = []
    tesori_nati = 0

    for ep in range(args.episodes):
        seed = args.seed0 + ep
        env, rng = game.make_env(), policy_rng(seed)
        obs, _ = env.reset(seed=seed)
        g = env.env.env
        visti = list(g.entities)
        done, tot = False, 0.0
        while not done:
            h_pre, g_pre, logit_true = forward(layers, obs)
            g_true = np.tanh(g_pre)
            n += 1
            fuori_colonna += g.player_x != 5
            righe[g.player_y] += 1

            pre = primo_strato(g)
            dh = float(np.abs(pre - h_pre).max())
            max_h = max(max_h, dh)
            bad_h += dh > 1e-9

            diff = secondo_strato(g) != stato_secondo_reale(g_true)
            bad_g += diff.any()
            for i in np.nonzero(diff)[0]:
                bad_g_neuroni[i] += 1

            max_f = max(max_f, float(np.abs(logit_formule(pre) - logit_true).max()))

            lo = regola(g)
            max_r = max(max_r, float(np.abs(lo - logit_true).max()))
            tv += 0.5 * np.abs(softmax(lo) - softmax(logit_true)).sum()
            top2 = np.sort(lo)[-2:]
            sit = situazione(g)
            if top2[1] - top2[0] < 1.0:
                ties += 1
                ties_sit[sit] += 1
            elif np.argmax(lo) != np.argmax(logit_true):
                bad_arg += 1
            c = colonna_scia_oro(g, g.player_y)
            if g.player_y in (1, 8) and c is not None and 3 <= c <= 7:
                bordo += arrivo(*entita(g)[g.player_y][:2]) is None      # tesoro che si allontana

            casi[sit] += 1
            if sit == SITUAZIONI[-1]:
                niente_bloccato += max(allarmi(g)[1:]) >= 0.5
            prob[sit] += softmax(logit_true)[[FERMO, SU, GIU]]
            trappola = sit == SITUAZIONI[0]
            esce = tuple(sopravvive(g, a) for a in (FERMO, SU, GIU)) if trappola else None
            riga = g.player_y
            nemici_prima = in_arrivo(g)[0]

            a = sample_action(logit_true, rng)
            obs, r, term, trunc, _ = env.step(a)
            tot += r
            done = term or trunc
            tesori_nati += sum(1 for i, e in enumerate(g.entities)
                               if e is not None and e[3] and e is not visti[i])
            visti = list(g.entities)
            if term:
                righe_morte[riga] += 1
                if env.last_action != a:
                    causa = "sticky: l'ambiente ripete la mossa precedente"
                elif trappola and any(esce):
                    causa = "nessuna mossa sicura, ma una sopravviveva"
                elif trappola:
                    causa = "nessuna mossa sicura, e nessuna sopravviveva"
                else:
                    causa = "altro"
                morti[causa, riga in (1, 8)] += 1
                eseguita = env.last_action
                dest = riga - 1 if eseguita == SU else riga + 1 if eseguita == GIU else riga
                if dest != riga and 1 <= dest <= 8 and nemici_prima.get(dest) == 0:
                    addosso[causa] += 1
        ritorni.append(tot)

    print(f"stati visitati: {n} ({args.episodes} episodi); punteggio della rete {np.mean(ritorni):.1f}")
    print(f"stati con il giocatore fuori dalla colonna 5: {fuori_colonna}")
    print(f"primo strato, stati con pre-attivazione diversa dalla lettura: {bad_h} (scarto massimo {max_h:.1e})")
    print(f"secondo strato, stati in cui lo stato di un neurone è diverso: {bad_g} {dict(bad_g_neuroni)}")
    print(f"logit con le formule della sezione 6, differenza massima: {max_f:.1e}")
    print(f"regola della sezione 7: differenza massima dei logit {max_r:.2f}, distanza media tra le softmax {tv / n:.4f}")
    print(f"  azione (argmax) diversa, esclusi i quasi-pareggi < 1: {bad_arg}  (quasi-pareggi: {ties})")
    print("  quasi-pareggi per situazione:", dict(ties_sit.most_common()))
    print(f"  stati con l'effetto di bordo della scia di un tesoro: {bordo}")

    print("\nprobabilità media delle mosse della rete, per situazione:")
    for s in SITUAZIONI:
        p = prob[s] / max(casi[s], 1)
        row = " ".join(f"{MOSSE[a]}:{p[i]:4.0%}" for i, a in enumerate(MOSSE))
        print(f"  {s:25s} {casi[s] / n:6.1%} degli stati | {row}")
    print(f"  in '{SITUAZIONI[-1]}', stati in cui su o giù ha un allarme >= 0.5: "
          f"{niente_bloccato / max(casi[SITUAZIONI[-1]], 1):.0%}")

    print("\nriga del giocatore:", " ".join(f"{k}:{righe[k] / n:.1%}" for k in range(1, 9)))
    print(f"tesori: raccolti {sum(ritorni):.0f} su {tesori_nati} comparsi ({sum(ritorni) / tesori_nati:.1%})")
    print(f"morti ({sum(morti.values())}), nelle righe 1-8 e di cui nelle righe di bordo 1 e 8:")
    for causa in sorted({c for c, _ in morti}, key=lambda c: -morti[c, False] - morti[c, True]):
        print(f"  {causa}: {morti[causa, False] + morti[causa, True]} (bordo {morti[causa, True]}; "
              f"entrando in una riga con un nemico già nella colonna 5: {addosso[causa]})")
    print("  riga della morte:", " ".join(f"{k}:{righe_morte[k]}" for k in range(1, 9)))

    if args.regola:
        m, se, t = gioca(game, lambda env, obs: regola(env.env.env), args.regola, args.seed0)
        print(f"\nla regola della sezione 7 al posto della rete, {args.regola} episodi: {m:.1f} ± {se:.1f}"
              f" (durata media {t:.0f} passi)")
    if args.varianti:
        print(f"\nvarianti della rete, {args.varianti} episodi:")
        tutte = {"la rete com'è": layers, **varianti(layers)}
        for nome, var in tutte.items():
            m, se, t = gioca(game, lambda env, obs: forward(var, obs)[-1], args.varianti, args.seed0)
            print(f"  {nome:50s} {m:5.1f} ± {se:.1f}   durata media {t:4.0f} passi")


if __name__ == "__main__":
    main()
