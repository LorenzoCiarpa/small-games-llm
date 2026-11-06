"""
Control: PPO on Breakout with an actor larger than the rules allow (default 256-256 tanh), to see whether the
~6-point plateau depends on the size of the network. The network does not respect Breakout's NetSpec (<= 32 units),
so it is neither exported nor evaluated: only the training curve matters.

    uv run --group ppo python experiments/breakout/ppo/control_big.py                 # 256-256, 6M steps
    uv run --group ppo python experiments/breakout/ppo/control_big.py --hidden 512,512 --steps 10000000

Result (results/control_256x256/train_log.csv): 6.8 points after 6M steps, the same plateau as the 32-32 network.
"""

import argparse
from pathlib import Path

import small_games.ppo as ppo

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--steps", type=int, default=6_000_000)
    args = ap.parse_args()
    hidden = tuple(int(x) for x in args.hidden.split(","))
    ppo._check_spec = lambda cfg, game: None          # outside the rules on purpose
    ppo.export_actor = lambda *a, **k: None           # no weights to export
    cfg = ppo.PPOConfig(game="breakout", hidden=hidden, activation="tanh", actions=(0, 1, 3),
                        total_steps=args.steps)
    out = Path(__file__).resolve().parent / "results" / f"control_{'x'.join(map(str, hidden))}"
    ppo.train(cfg, out / "unused.json", out)
