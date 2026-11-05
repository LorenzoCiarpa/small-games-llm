"""
Controllo: PPO su Breakout con un attore più grande di quanto le regole ammettono (default 256-256 tanh), per
capire se il plateau a ~6 punti dipende dalla taglia della rete. La rete non rispetta il NetSpec di Breakout
(<= 32 unità), quindi non viene esportata né valutata: conta solo la curva di addestramento.

    uv run --group ppo python experiments/breakout/ppo/control_big.py                 # 256-256, 6M passi
    uv run --group ppo python experiments/breakout/ppo/control_big.py --hidden 512,512 --steps 10000000

Risultato (results/control_256x256/train_log.csv): 6.8 punti dopo 6M passi, lo stesso plateau della rete 32-32.
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
    ppo._check_spec = lambda cfg, game: None          # fuori dalle regole di proposito
    ppo.export_actor = lambda *a, **k: None           # niente pesi da esportare
    cfg = ppo.PPOConfig(game="breakout", hidden=hidden, activation="tanh", actions=(0, 1, 3),
                        total_steps=args.steps)
    out = Path(__file__).resolve().parent / "results" / f"control_{'x'.join(map(str, hidden))}"
    ppo.train(cfg, out / "unused.json", out)
