"""
PPO diventa più sparso o più leggibile andando avanti con l'addestramento, o con un'altra entropia? Le misure
principali di src/small_games/analysis.py, più leggere, su tutte le corse PPO e sui loro checkpoint, accanto alle
reti scritte a mano dello stesso gioco (commento in README.md).

    uv run --group analysis python experiments/ppo_over_training/analyze.py --workers 8   # ~5 min
    uv run --group analysis python experiments/ppo_over_training/analyze.py --plot-only   # solo la figura

Il primo strato è guardato solo sugli ingressi vivi del gioco (quelli che non sono sempre zero, presi da
experiments/<gioco>/ppo/results/analysis/analysis.json): sugli altri PPO non riceve gradiente e lascia i pesi
iniziali. Per ogni rete:
  punteggio            media su `--episodes` episodi del protocollo (campionata)
  massa nel 10%        quota della somma dei |w| (tutti gli strati) che sta nel 10% dei pesi più grandi
  ingressi per il 90%  mediana, sui neuroni del primo strato, della frazione di ingressi vivi che fa il 90% del |w|
  pruning 50%          punteggio tenendo il 50% dei pesi più grandi di ogni strato, diviso per il punteggio
  primo strato 25%     punteggio tenendo il 25% dei pesi vivi più grandi del primo strato (gli altri strati interi),
                       diviso per il punteggio
  quantità chiave      media, sulle quantità dell'algoritmo scritto a mano (KEY_QUANTITIES), del miglior |r| di un
                       neurone e dell'R² lineare dello strato migliore
  albero               accordo di un albero di profondità 6 sulle grandezze del gioco con l'argmax della rete

Scrive results/sweep.json (tutti i numeri), results/sweep.md (le tabelle) e results/sweep.png.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from small_games.analysis import (KEY_QUANTITIES, collect, distill, hidden_acts, key_fit, prune, prune_first_layer,
                                  score, weight_stats)
from small_games.core import get_game
from small_games.features import FEATURES
from small_games.weights import load_weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"

MINATAR_STEPS = (5, 10, 25, 50, 100, 150, 200)
GAMES = {
    "low_tide": {"hand": ("claude", "gpt"), "runs": {"ppo_100m": (5, 10, 25, 50, 75, 100)}},
    **{g: {"hand": ("paper_astra",), "runs": {r: MINATAR_STEPS for r in ("ppo_ent01", "ppo_ent05", "ppo_ent10")}}
       for g in ("breakout", "seaquest", "asterix")},
}


def checkpoint(game, run, steps):
    """The weight file of `run` after `steps` million steps (the final one is in weights/)."""
    p = ROOT / "experiments" / game / "ppo" / "results" / run / f"checkpoint_{steps:04d}M.json"
    return p if p.is_file() else ROOT / "weights" / game / f"{run}.json"


def live_inputs(game_name):
    report = json.loads((ROOT / "experiments" / game_name / "ppo" / "results" / "analysis" / "analysis.json").read_text())
    live = np.zeros(get_game(game_name).obs_dim, dtype=bool)
    live[report["live_inputs"]] = True
    return live


def measure(game_name, path, episodes, collect_episodes, workers):
    game = get_game(game_name)
    seed0 = game.protocol.seed0
    names, feat = FEATURES[game_name]
    layers = load_weights(path, game)
    live = live_inputs(game_name)
    W0, b0, act0 = layers[0]
    absw = np.sort(np.concatenate([np.abs(W0[:, live]).ravel()] + [np.abs(W).ravel() for W, _, _ in layers[1:]]))[::-1]
    full = score(game, layers, episodes, seed0, workers)[0]
    half = score(game, prune(layers, 0.5), episodes, seed0, workers)[0]
    first = score(game, prune_first_layer(layers, 0.25, live), episodes, seed0, workers)[0]

    X, ep = collect(game, layers, collect_episodes, seed0 + 5000)
    acts = hidden_acts(layers, X)
    F = np.stack([feat(o) for o in X])
    test = ep >= int(0.7 * collect_episodes)
    best_r, r2 = [], []
    for fn in KEY_QUANTITIES[game_name].values():
        rows = key_fit(acts, fn(F, names), test)
        best_r.append(max(x["best_r"] for x in rows))
        r2.append(max(max(x["linear_r2"], 0.0) for x in rows))
    (_, agree), = distill(F, acts[-1].argmax(axis=1), ep, (6,)).values()
    return {
        "score": full,
        "top10_mass": float(absw[:int(np.ceil(0.1 * len(absw)))].sum() / absw.sum()),
        "inputs_for_90pct_frac": weight_stats([(W0[:, live], b0, act0)])[0]["inputs_for_90pct_frac"],
        "prune50_rel": half / full if full > 0 else None,
        "first25_rel": first / full if full > 0 else None,
        "key_best_r": float(np.mean(best_r)),
        "key_r2": float(np.mean(r2)),
        "tree6_agreement": agree,
        "states": int(len(X)),
    }


def run_all(args):
    res = json.loads((OUT / "sweep.json").read_text()) if (OUT / "sweep.json").is_file() else {}
    for g, spec in GAMES.items():
        if args.games and g not in args.games:
            continue
        res.setdefault(g, {"hand": {}, "runs": {}})
        for h in spec["hand"]:
            t0 = time.time()
            res[g]["hand"][h] = measure(g, ROOT / "weights" / g / f"{h}.json", args.episodes, args.collect, args.workers)
            print(f"[{g}] {h}: {fmt(res[g]['hand'][h])} ({time.time() - t0:.0f}s)", flush=True)
        for run, steps in spec["runs"].items():
            rows = res[g]["runs"][run] = {}
            for s in steps:
                t0 = time.time()
                rows[str(s)] = measure(g, checkpoint(g, run, s), args.episodes, args.collect, args.workers)
                print(f"[{g}] {run} {s}M: {fmt(rows[str(s)])} ({time.time() - t0:.0f}s)", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "sweep.json").write_text(json.dumps(res, indent=1))
    return res


def fmt(m):
    p = lambda x: "—" if x is None else f"{x:.2f}"
    return (f"punti {m['score']:.1f}, massa10 {m['top10_mass']:.2f}, ingr90 {m['inputs_for_90pct_frac']:.2f}, "
            f"prune50 {p(m['prune50_rel'])}, primo25 {p(m['first25_rel'])}, r {m['key_best_r']:.2f}, "
            f"R² {m['key_r2']:.2f}, albero {m['tree6_agreement']:.2f}")


# ------------------------------------------------------------------------------------------------ report
COLS = [("punteggio", "score", "{:.1f}"), ("massa nel 10% dei pesi", "top10_mass", "{:.0%}"),
        ("ingressi vivi per il 90% (primo strato)", "inputs_for_90pct_frac", "{:.0%}"),
        ("pruning 50%", "prune50_rel", "{:.0%}"), ("primo strato al 25%", "first25_rel", "{:.0%}"),
        ("quantità chiave: miglior \\|r\\|", "key_best_r", "{:.2f}"), ("quantità chiave: R²", "key_r2", "{:.2f}"),
        ("albero prof. 6: accordo", "tree6_agreement", "{:.0%}")]


def tables(res):
    out = []
    for g, r in res.items():
        out.append(f"### {g}\n")
        out.append("| rete | " + " | ".join(c[0] for c in COLS) + " |")
        out.append("|---|" + "---|" * len(COLS))
        rows = [(f"`{h}` (a mano)", m) for h, m in r["hand"].items()]
        rows += [(f"`{run}` {s}M", m) for run, ckpts in r["runs"].items() for s, m in ckpts.items()]
        for name, m in rows:
            out.append(f"| {name} | " + " | ".join("—" if m[k] is None else f.format(m[k]) for _, k, f in COLS) + " |")
        out.append("")
    return "\n".join(out)


# reference palette (dataviz skill), light mode: categorical slots 1-3, ink and grid
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PLOTS = [("score", "punteggio"), ("top10_mass", "massa nel 10%\ndei pesi più grandi"),
         ("first25_rel", "punteggio con il 25% dei\npesi del primo strato (rel.)"),
         ("key_best_r", "quantità chiave:\nmiglior |r| di un neurone"),
         ("tree6_agreement", "accordo di un albero\ndi profondità 6")]
TITLES = {"low_tide": "Low Tide", "breakout": "Breakout", "seaquest": "Seaquest", "asterix": "Asterix"}
# colour = entropy coefficient (Low Tide's ppo_100m has the default 0.01)
RUN_STYLE = {"ppo_100m": 0, "ppo_ent01": 0, "ppo_ent05": 1, "ppo_ent10": 2}
ENTROPY_LABELS = ("PPO, entropia 0.01", "PPO, entropia 0.05", "PPO, entropia 0.1")


def plot(res, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    games = [g for g in GAMES if g in res]
    fig, axes = plt.subplots(len(PLOTS), len(games), figsize=(3.1 * len(games), 2.1 * len(PLOTS)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for c, g in enumerate(games):
        for r, (key, label) in enumerate(PLOTS):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            for run, ckpts in res[g]["runs"].items():
                xs = [int(s) for s, m in ckpts.items() if m[key] is not None]
                ys = [m[key] for m in ckpts.values() if m[key] is not None]
                ax.plot(xs, ys, color=SERIES[RUN_STYLE[run]], lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1.2,
                        zorder=3)
            for k, (h, m) in enumerate(res[g]["hand"].items()):
                if m[key] is None:
                    continue
                ax.axhline(m[key], color=MUTED, lw=1.4, ls=("--", ":")[k], zorder=2)
                # the first label at the right end, the second at the left, so that equal values do not collide
                ax.annotate(h, ((1.0, 0.0)[k], m[key]), xycoords=("axes fraction", "data"),
                            xytext=((-2, 2)[k], (3, -3)[k]), textcoords="offset points", ha=("right", "left")[k],
                            va=("bottom", "top")[k], fontsize=7, color=MUTED)
            if key != "score":
                ax.set_ylim(-0.03, 1.12)
            ax.grid(axis="y", color=GRID, lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            if r == 0:
                ax.set_title(TITLES[g], fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, color=INK)
            if r == len(PLOTS) - 1:
                ax.set_xlabel("milioni di passi", fontsize=8, color=MUTED)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=4.5, mec=SURFACE) for c in SERIES]
    handles.append(Line2D([], [], color=MUTED, lw=1.4, ls="--"))
    fig.legend(handles, [*ENTROPY_LABELS, "rete scritta a mano (il nome è sulla linea)"], loc="upper center",
               ncol=4, fontsize=8, frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle("PPO lungo l'addestramento, accanto alle reti scritte a mano", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--collect", type=int, default=10, help="episodi giocati dalla rete per le quantità e l'albero")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--games", nargs="*", help=f"solo questi giochi (default: {' '.join(GAMES)})")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    res = json.loads((OUT / "sweep.json").read_text()) if args.plot_only else run_all(args)
    (OUT / "sweep.md").write_text(tables(res))
    plot(res, OUT / "sweep.png")
