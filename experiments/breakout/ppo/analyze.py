"""
Can an algorithm be read in the weights of breakout's PPO network? The same measurements on the PPO network
(weights/breakout/ppo.json, from train.py) and on the hand-written networks (paper_astra.json (GPT-6 Astra, from the
article)): weights (also restricted to the inputs that are not always zero), pruning, ablation of each unit,
correlation of the units with game quantities, distillation into decision trees, images of the first layer, and
whether the quantities used by the hand-written algorithm are found in the neurons (key quantities). The tools are in
src/small_games/analysis.py.

    uv run --group analysis python experiments/breakout/ppo/analyze.py              # ~10-30 min
    uv run --group analysis python experiments/breakout/ppo/analyze.py --quick      # few episodes, to try it out

Writes results/analysis/: analysis.json (all the numbers), trees_<net>.txt, fields_<net>.png.
"""

import argparse
from pathlib import Path

from small_games.analysis import analyse, key_quantities, live_input_stats

HERE = Path(__file__).resolve().parent
GAME = "breakout"
W = HERE.parents[2] / "weights" / GAME
NETS = {k: p for k, p in {"paper_astra": W / "paper_astra.json", "ppo_v1": W / "ppo_v1.json",
                          "ppo": W / "ppo.json"}.items() if p.is_file()}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only-key", action="store_true",
                    help="only key quantities and live inputs (added to analysis.json), without redoing the rest")
    ap.add_argument("--nets", help=f"only these networks, comma-separated (default: {', '.join(NETS)})")
    args = ap.parse_args()
    kw = dict(score_episodes=10, ablation_episodes=5, collect_episodes=5) if args.quick else {}
    nets = {k: NETS[k] for k in args.nets.split(",")} if args.nets else NETS
    out = HERE / "results" / "analysis"
    if not args.only_key:
        analyse(GAME, nets, out, workers=args.workers, **kw)
    live_input_stats(GAME, nets, out, live_from=NETS, workers=args.workers,
                     collect_episodes=kw.get("collect_episodes", 40), score_episodes=kw.get("score_episodes", 200))
    key_quantities(GAME, nets, out, collect_episodes=kw.get("collect_episodes", 40))
