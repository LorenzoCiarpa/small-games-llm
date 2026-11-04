"""
PPO su Low Tide: addestra una policy 403-64-64-5 tanh (due strati nascosti da 64, il massimo di unità ammesso), la
salva in weights/low_tide/ppo.json e la valuta con il protocollo del gioco.

    uv run --group ppo python experiments/low_tide/ppo/train.py                     # configurazione sotto
    uv run --group ppo python experiments/low_tide/ppo/train.py --total-steps 2000000

L'algoritmo è in src/small_games/ppo.py; i risultati (curva di training, checkpoint, valutazione) in results/ppo/.
"""

from pathlib import Path

from small_games.ppo import PPOConfig, main

CONFIG = PPOConfig(game="low_tide", hidden=(64, 64), activation="tanh", total_steps=20_000_000)

if __name__ == "__main__":
    main(CONFIG, Path(__file__).parent)
