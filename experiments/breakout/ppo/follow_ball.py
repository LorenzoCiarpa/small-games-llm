"""
Termine di paragone per la rete PPO di Breakout: la policy "segui la palla", che sposta la racchetta verso la colonna
in cui la palla è adesso, senza prevedere dove arriverà. Basterebbe un solo neurone lineare per scriverla.

    uv run python experiments/breakout/ppo/follow_ball.py               # 1000 episodi, protocollo del gioco

Fa 8.4 ± 0.3 punti su 1000 episodi: poco più di PPO (5.8-6.8), molto meno della rete di Astra (512), che prevede
la colonna d'arrivo.
"""

import argparse

from small_games.core import get_game
from small_games.evaluate import evaluate
from small_games.features import breakout


class FollowBall:
    """Sinistra se la palla è a sinistra della racchetta, destra se è a destra, altrimenti ferma."""

    def __call__(self, obs, rng):
        paddle, ball_x = breakout(obs)[:2]
        return 1 if ball_x < paddle else 3 if ball_x > paddle else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    s = evaluate(get_game("breakout"), FollowBall(), episodes=args.episodes, workers=args.workers)
    print(f"segui la palla: {s['mean_return']:.2f} ± {s['stderr']:.2f} ({s['protocol']['episodes']} episodi)")
