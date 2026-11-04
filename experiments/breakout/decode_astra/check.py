"""
Verifica la lettura a mano di weights/breakout/paper_astra.json (vedi README.md).

Prima riscrive tutti i pesi dalle formule del README e li confronta con il file. Poi gioca gli stessi episodi di
`uv run evaluate breakout paper_astra --episodes N` (seed 10000.., sticky 0.1, azioni campionate dalla softmax): in
ogni stato ricalcola, dallo stato del gioco e senza usare i pesi, i neuroni dei due strati nascosti e i 6 logit come
li descrive il README, e li confronta con il forward pass vero. Infine misura quanto è giusta la previsione della
rete e perché perde la palla.

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
ACTIONS = ("fermo", "sx", "su", "dx", "giu", "fuoco")
# h0..h3: una direzione della palla ciascuno, (dx, dy) con dy = +1 verso il basso
DIRS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
DIR_NAMES = ("giù-dx", "giù-sx", "su-dx", "su-sx")
MINATAR_DIR = {0: (-1, -1), 1: (1, -1), 2: (1, 1), 3: (-1, 1)}   # ball_dir di MinAtar -> (dx, dy)
MOVE = (0, -1, 0, 1, 0, 0)                                        # spostamento della racchetta per azione


def forward(layers, obs):
    x, out = np.asarray(obs, dtype=np.float64), []
    for W, b, act in layers:
        x = W @ x + b
        if act == "tanh":
            x = np.tanh(x)
        out.append(x)
    return out


def pm(b):
    """[x] del README: +1 se vero, -1 se falso (un neurone tanh saturo)."""
    return 1.0 if b else -1.0


def step_x(x, dx):
    """Un passo orizzontale con le pareti di MinAtar: fuori dalla griglia la palla resta sul bordo e rimbalza."""
    nx = x + dx
    if nx < 0:
        return 0, 1
    if nx > 9:
        return 9, -1
    return nx, dx


def predicted_flight(r, c, dx, dy):
    """La colonna di arrivo che la rete assegna alla palla in (r, c) con direzione (dx, dy) (sezione 4 del README),
    e i passi che la palla impiega, secondo lo stesso modello, ad arrivare in riga 8 scendendo.

    Pareti e soffitto come in MinAtar. Se sale, la palla rimbalza appena la riga sopra è una riga di mattoni
    (1-3), come se i mattoni ci fossero tutti; poi scende senza ostacoli. La colonna è quella in cui si trova
    quando arriva in riga 8; se scende ed è già nelle righe 6-8, il punto medio tra le due colonne che la
    prendono (quella in riga 8 e quella dopo). In riga 9 (palla persa) la colonna stessa.
    """
    if r == 9:
        return float(c), 0
    x, y, s, going_down = c, r, 0, dy > 0
    while dy < 0:
        x, dx = step_x(x, dx)
        s += 1
        if y - 1 < 0 or 1 <= y - 1 <= 3:
            dy = 1                      # soffitto o mattone: resta nella riga, inverte la verticale
        else:
            y -= 1
    while y < 8:
        x, dx = step_x(x, dx)
        y += 1
        s += 1
    if going_down and r >= 6:
        return (x + step_x(x, dx)[0]) / 2, s
    return float(x), s


# TARGET[k, r, c]: colonna prevista per la palla in (r, c) con direzione DIRS[k]; STEPS: passi previsti
_TABLE = np.array([[[predicted_flight(r, c, *d) for c in range(10)] for r in range(10)] for d in DIRS])
TARGET, STEPS = _TABLE[..., 0], _TABLE[..., 1].astype(int)


def rebuilt_layers():
    """I pesi della rete riscritti dalle formule del README (sezioni 4-6): servono a controllare che la lettura
    riproduca il file peso per peso, anche nelle celle che la palla non visita mai giocando."""
    cell = lambda r, c, ch: (r * 10 + c) * 6 + ch
    W0, b0 = np.zeros((32, 600)), np.zeros(32)
    for k in range(4):                                  # h0..h3: 0.02·T sulla palla, −0.02·p sulla racchetta
        for r in range(10):
            for c in range(10):
                W0[k, cell(r, c, 3)] = 0.02 * TARGET[k, r, c]
        for p in range(10):
            W0[k, cell(9, p, 0)] = -0.02 * p
    for r in range(10):
        for c in range(10):                             # h4: righello sulle colonne, h5: righello sulle righe
            W0[4, cell(r, c, 3)] = {0: 10, 9: 170}.get(c, 20 * c)
            W0[4, cell(r, c, 4)] = -20 * c
            W0[5, cell(r, c, 3)] = 70 if r == 8 else 10 * r
            W0[5, cell(r, c, 4)] = -10 * r
    b0[5] = 5
    W1, b1 = np.zeros((32, 32)), np.zeros(32)
    for k, (dx, dy) in enumerate(DIRS):                 # g(2k): a sinistra, g(2k+1): a destra
        for side, sign in ((0, -1), (1, 1)):
            W1[2 * k + side, [k, 4, 5]] = 1000 * sign, 200 * dx, 200 * dy
            b1[2 * k + side] = -415
    W2, b2 = np.zeros((6, 32)), np.array([0.0, 90, -100, 90, -100, -100])
    W2[1, [0, 2, 4, 6]] = 30
    W2[3, [1, 3, 5, 7]] = 30
    return [(W0, b0), (W1, b1), (W2, b2)]


def read_direction(e):
    """La direzione come la legge la rete: dalla palla e dalla sua scia (cella precedente)."""
    right = e.ball_x > e.last_x or e.ball_x == e.last_x == 0
    down = e.ball_y > e.last_y or (e.ball_y == e.last_y and e.ball_y != 8)
    return right, down


def decode(e):
    """Neuroni nascosti e logit ricostruiti dalle regole del README, a partire dallo stato del gioco."""
    right, down = read_direction(e)
    k = DIRS.index((1 if right else -1, 1 if down else -1))
    p = e.pos
    T = TARGET[:, e.ball_y, e.ball_x]
    h = np.zeros(32)
    h[:4] = np.tanh(0.02 * (T - p))             # h0..h3: (colonna prevista - racchetta) / 50
    h[4], h[5] = pm(right), pm(down)            # h4: va a destra, h5: scende
    delta = T[k] - p
    go_left, go_right = delta <= -1, delta >= 1
    g = np.zeros(32)
    g[:8] = -1.0
    g[2 * k], g[2 * k + 1] = pm(go_left), pm(go_right)
    logits = np.array([0.0, 30 * pm(go_left), -100.0, 30 * pm(go_right), -100.0, -100.0])
    return h, g, logits, k, T[k]


def true_flight(e):
    """Passi che mancano finché la palla arriva in riga 8 scendendo, e le colonne in cui la racchetta la prende.

    Simula le regole di MinAtar (pareti, soffitto, mattoni veri) senza la racchetta, su una copia dei mattoni.
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
    raise RuntimeError("la palla non torna giù")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--sticky", type=float, default=0.1)
    args = ap.parse_args()

    game = get_game("breakout")
    layers = load_weights(ROOT / "weights/breakout/paper_astra.json", game)

    # 0. controllo statico: i pesi riscritti dalle formule del README contro il file
    n_par = n_diff = n_nonzero = 0
    for (W, b, _), (W_r, b_r) in zip(layers, rebuilt_layers()):
        n_par += W.size + b.size
        n_diff += int(np.sum(np.abs(W - W_r) > 1e-9) + np.sum(np.abs(b - b_r) > 1e-9))
        n_nonzero += int(np.count_nonzero(W_r) + np.count_nonzero(b_r))
    print(f"pesi riscritti dalle formule del README: {n_nonzero} diversi da zero; "
          f"parametri diversi dal file: {n_diff} su {n_par}\n")

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
        e = env.env.env                                   # il Breakout di MinAtar: palla, scia, racchetta, mattoni
        done, score, flight = False, 0.0, None
        while not done:
            # 1. la lettura contro la rete vera
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

            # 2. i voli: dalla partita, o dalla palla appena ribattuta (palla e scia in riga 8), al rimbalzo dopo
            steps, catch = true_flight(e)
            if flight is None or e.ball_y == e.last_y == 8:
                dist = min(abs(e.pos - x) for x in catch)
                flight = {"margin": steps + 1 - dist, "sticky": 0, "dir_err": 0, "wrong": 0}
                flights += 1
                margin_all[flight["margin"]] += 1
                if flight["margin"] == 1:
                    margin1_dist[dist] += 1

            # 3. la direzione letta dalla rete contro quella vera del gioco
            if MINATAR_DIR[e.ball_dir] != DIRS[k]:
                flight["dir_err"] += 1
                if e.ball_y == e.last_y == 8:
                    dir_err["rimbalzo sullo spigolo della racchetta"] += 1
                elif e.ball_y == e.last_y:
                    dir_err["mattone colpito dall'alto"] += 1
                else:
                    dir_err["altro"] += 1

            # 4. la previsione: le colonne in cui la rete sta ferma sono colonne che prendono davvero la palla?
            phase = "sale" if DIRS[k][1] < 0 else ("scende, righe 0-2" if e.ball_y <= 2 else "scende, righe 3-8")
            pred_n[phase] += 1
            pred_ok[phase] += {x for x in range(10) if abs(t - x) < 1} <= catch
            if DIRS[k][1] < 0 and MINATAR_DIR[e.ball_dir] == DIRS[k]:
                shorter += steps < STEPS[k, e.ball_y, e.ball_x]   # il volo vero dura meno di quello previsto?

            a = sample_action(logit_true, rng)
            # 5. la scelta della rete avvicina la racchetta alle colonne di presa vere (o la tiene lì)?
            d_now = min(abs(e.pos - x) for x in catch)
            d_next = min(abs(min(9, max(0, e.pos + MOVE[a])) - x) for x in catch)
            if d_next > max(d_now - 1, 0):
                flights_wrong += flight["wrong"] == 0
                flight["wrong"] += 1
                wrong_phase[phase] += 1
            obs, r, term, trunc, _ = env.step(a)
            if env.last_action != a:
                flight["sticky"] += 1                   # l'ambiente ha ripetuto un'azione diversa da quella scelta
                flights_2sticky += flight["sticky"] == 2
            score += r
            done = term or trunc
            if term:
                margin_miss[flight["margin"]] += 1
                sticky_miss[flight["sticky"]] += 1
                miss_dir_err += flight["dir_err"] > 0
                if flight["wrong"]:
                    miss_kind["la rete ha fatto mosse sbagliate"] += 1
                elif flight["margin"] == 1:
                    miss_kind["margine 1, scelte della rete tutte giuste"] += 1
                else:
                    miss_kind["margine > 1, scelte della rete tutte giuste"] += 1
        scores.append(score)
        lengths.append(env.t)
        cut.append(not term)               # arrivato al limite di passi

    scores = np.array(scores)
    se = scores.std(ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
    print(f"stati visitati: {n} ({args.episodes} episodi, sticky {args.sticky})")
    cut = np.array(cut, dtype=bool)
    print(f"punteggio: {scores.mean():.2f} ± {se:.2f}, lunghezza media {np.mean(lengths):.0f}, "
          f"episodi arrivati al limite di 10000 passi: {cut.sum()}")
    if cut.any() and (~cut).any():
        print(f"  punteggio medio: {scores[cut].mean():.1f} negli episodi arrivati al limite, "
              f"{scores[~cut].mean():.1f} in quelli finiti con la palla persa")
    print("\nla lettura contro la rete")
    print(f"  h0..h3, stati in cui il valore sbaglia (> 1e-9): {bad_h}  (differenza massima {max_dh:.1e})")
    print(f"  h4, h5 (direzione), stati in cui il segno sbaglia: {bad_dir}")
    print(f"  neuroni inutilizzati (h6..h31, g8..g31) diversi da 0: {bad_zero}")
    print(f"  g0..g7, stati in cui acceso/spento sbaglia: {bad_g}")
    print(f"  logit, differenza massima: {max_dl:.4f}")
    print(f"  azione (argmax) diversa, esclusi i quasi-pareggi < 0.5: {bad_arg}  (quasi-pareggi: {ties})")
    print("  argmax della rete:", " ".join(f"{ACTIONS[a]}:{by_action[a] / n:.1%}" for a in range(6)))

    print("\nla rete contro il gioco")
    n_err = sum(dir_err.values())
    print(f"  direzione letta diversa da quella vera: {n_err} stati ({n_err / n:.2%})", dict(dir_err))
    print("  previsione giusta (le colonne in cui sta ferma prendono la palla), per fase:")
    for ph in ("sale", "scende, righe 0-2", "scende, righe 3-8"):
        print(f"    {ph:18s} {pred_ok[ph]:8d} / {pred_n[ph]:8d}  ({pred_ok[ph] / max(pred_n[ph], 1):.1%})")
    print(f"  palla che sale: stati in cui il volo vero è più corto di quello previsto: {shorter}")

    misses = sum(margin_miss.values())
    print(f"\nvoli (dalla racchetta alla racchetta): {flights}, palle perse: {misses}")
    print("  margine al decollo = azioni disponibili - colonne da percorrere:")
    groups = [("1", lambda m: m == 1), ("2-4", lambda m: 2 <= m <= 4), ("5-9", lambda m: 5 <= m <= 9),
              ("≥ 10", lambda m: m >= 10), ("≤ 0", lambda m: m <= 0)]
    for name, f in groups:
        a = sum(v for m, v in margin_all.items() if f(m))
        b = sum(v for m, v in margin_miss.items() if f(m))
        if a:
            print(f"    margine {name:4s}: {a:7d} voli ({a / flights:5.1%}), "
                  f"palle perse {b:4d} ({b / a:.2%} dei voli)")
    print(f"  voli con margine 1, per colonne da percorrere: {dict(sorted(margin1_dist.items()))}")
    print(f"  palle perse per sostituzioni sticky nel volo: {dict(sorted(sticky_miss.items()))}")
    print(f"  voli con almeno 2 sostituzioni sticky: {flights_2sticky} ({flights_2sticky / flights:.1%})")
    print(f"  palle perse in voli con almeno uno stato di direzione letta male: {miss_dir_err}")
    print("  mosse sbagliate della rete (non avvicinano la racchetta alle colonne di presa vere), per fase:",
          dict(wrong_phase))
    print(f"  voli con almeno una mossa sbagliata della rete: {flights_wrong} ({flights_wrong / flights:.1%})")
    print("  palle perse per tipo di volo:", dict(miss_kind))


if __name__ == "__main__":
    main()
