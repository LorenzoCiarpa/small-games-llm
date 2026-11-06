"""
PPO on Low Tide: trains a 403-64-64-5 tanh policy (two hidden layers of 64, the maximum number of units allowed),
saves it to weights/low_tide/ppo.json and scores it with the game's protocol.

    uv run --group ppo python experiments/low_tide/ppo/train.py                     # configuration below
    uv run --group ppo python experiments/low_tide/ppo/train.py --total-steps 2000000

The algorithm is in src/small_games/ppo.py; the results (training curve, checkpoints, evaluation) in results/ppo/.
"""

from pathlib import Path

from small_games.ppo import PPOConfig, main

CONFIG = PPOConfig(game="low_tide", hidden=(64, 64), activation="tanh", total_steps=20_000_000)

if __name__ == "__main__":
    main(CONFIG, Path(__file__).parent)
