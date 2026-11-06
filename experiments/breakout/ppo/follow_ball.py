"""
Reference point for Breakout's PPO network: the "follow the ball" policy, which moves the paddle toward the column
where the ball is now, without predicting where it will land. A single linear neuron would be enough to write it.

    uv run python experiments/breakout/ppo/follow_ball.py               # 1000 episodes, the game's protocol

It scores 8.4 ± 0.3 points over 1000 episodes: a little more than PPO (5.8-6.8), much less than Astra's network
(512), which predicts the landing column.
"""

import argparse

from small_games.core import get_game
from small_games.evaluate import evaluate
from small_games.features import breakout


class FollowBall:
    """Left if the ball is to the left of the paddle, right if it is to the right, otherwise no-op."""

    def __call__(self, obs, rng):
        paddle, ball_x = breakout(obs)[:2]
        return 1 if ball_x < paddle else 3 if ball_x > paddle else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    s = evaluate(get_game("breakout"), FollowBall(), episodes=args.episodes, workers=args.workers)
    print(f"follow the ball: {s['mean_return']:.2f} ± {s['stderr']:.2f} ({s['protocol']['episodes']} episodes)")
