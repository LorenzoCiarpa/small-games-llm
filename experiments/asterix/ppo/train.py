"""
PPO su Asterix (MinAtar): addestra la policy 600-32-32-6 tanh (la stessa architettura delle reti scritte a mano),
la salva in weights/asterix/ppo.json e la valuta con il protocollo del gioco.

    uv run --group ppo python experiments/asterix/ppo/train.py                     # configurazione sotto
    uv run --group ppo python experiments/asterix/ppo/train.py --total-steps 2000000

Un primo tentativo con tutte e 6 le azioni e 30M passi è in weights/asterix/ppo_v1.json e results/ppo_v1/
(vedi README.md). L'algoritmo è in src/small_games/ppo.py; i risultati (curva di training, checkpoint, valutazione) in results/ppo/.
"""

from pathlib import Path

from small_games.ppo import PPOConfig, main

# il minimal action set di MinAtar: tutte tranne fuoco, che in Asterix non fa niente
CONFIG = PPOConfig(game="asterix", hidden=(32, 32), activation="tanh", actions=(0, 1, 2, 3, 4), total_steps=100_000_000)

if __name__ == "__main__":
    main(CONFIG, Path(__file__).parent)
