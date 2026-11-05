"""
Si riesce a leggere un algoritmo nei pesi della rete PPO di asterix? Le stesse misure sulla rete PPO
(weights/asterix/ppo.json, da train.py) e sulle reti scritte a mano (paper_astra.json (GPT-6 Astra, dall'articolo)):
pesi (anche solo sugli ingressi che non sono sempre zero), pruning, ablazione di ogni unità, correlazione delle
unità con quantità del gioco, distillazione in alberi di decisione, immagini del primo strato, e se le quantità
usate dall'algoritmo scritto a mano si ritrovano nei neuroni (quantità chiave). Gli strumenti sono in
src/small_games/analysis.py.

    uv run --group analysis python experiments/asterix/ppo/analyze.py              # ~10-30 min
    uv run --group analysis python experiments/asterix/ppo/analyze.py --quick      # pochi episodi, per provare

Scrive results/analysis/: analysis.json (tutti i numeri), trees_<rete>.txt, fields_<rete>.png.
"""

import argparse
from pathlib import Path

from small_games.analysis import analyse, key_quantities, live_input_stats

HERE = Path(__file__).resolve().parent
GAME = "asterix"
W = HERE.parents[2] / "weights" / GAME
NETS = {k: p for k, p in {"paper_astra": W / "paper_astra.json", "ppo_v1": W / "ppo_v1.json",
                          "ppo": W / "ppo.json"}.items() if p.is_file()}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only-key", action="store_true",
                    help="solo quantità chiave e ingressi vivi (aggiunti ad analysis.json), senza rifare il resto")
    ap.add_argument("--nets", help=f"solo queste reti, separate da virgole (default: {', '.join(NETS)})")
    args = ap.parse_args()
    kw = dict(score_episodes=10, ablation_episodes=5, collect_episodes=5) if args.quick else {}
    nets = {k: NETS[k] for k in args.nets.split(",")} if args.nets else NETS
    out = HERE / "results" / "analysis"
    if not args.only_key:
        analyse(GAME, nets, out, workers=args.workers, **kw)
    live_input_stats(GAME, nets, out, live_from=NETS, workers=args.workers,
                     collect_episodes=kw.get("collect_episodes", 40), score_episodes=kw.get("score_episodes", 200))
    key_quantities(GAME, nets, out, collect_episodes=kw.get("collect_episodes", 40))
