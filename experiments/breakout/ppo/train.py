"""
PPO on Breakout (MinAtar): trains the 600-32-32-6 tanh policy (the same architecture as the hand-written networks),
saves it to weights/breakout/ppo.json and scores it with the game's protocol.

    uv run --group ppo python experiments/breakout/ppo/train.py                     # configuration below
    uv run --group ppo python experiments/breakout/ppo/train.py --total-steps 2000000

weights/breakout/ppo.json comes from the ppo_ent05 run of experiments/ppo_runs.py, which has this same configuration
(and the same seed): its curve, checkpoints and evaluation are in results/ppo_ent05/. The runs with ent_coef 0.01 and
0.1 are in results/ppo_ent01/ and results/ppo_ent10/, the first attempt (all 6 actions, ent_coef 0.01, 30M steps) in
results/ppo_v1/; see README.md. The algorithm is in src/small_games/ppo.py.
"""

from pathlib import Path

from small_games.ppo import PPOConfig, main

# MinAtar's minimal action set: no-op, left, right (up, down and fire do nothing)
CONFIG = PPOConfig(game="breakout", hidden=(32, 32), activation="tanh", actions=(0, 1, 3),
                   ent_coef=0.05, total_steps=200_000_000)

if __name__ == "__main__":
    main(CONFIG, Path(__file__).parent)
