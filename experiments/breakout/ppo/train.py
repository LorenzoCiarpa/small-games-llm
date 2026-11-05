"""
PPO su Breakout (MinAtar): addestra la policy 600-32-32-6 tanh (la stessa architettura delle reti scritte a mano),
la salva in weights/breakout/ppo.json e la valuta con il protocollo del gioco.

    uv run --group ppo python experiments/breakout/ppo/train.py                     # configurazione sotto
    uv run --group ppo python experiments/breakout/ppo/train.py --total-steps 2000000

Un primo tentativo con tutte e 6 le azioni e 30M passi è in weights/breakout/ppo_v1.json e results/ppo_v1/
(vedi README.md). L'algoritmo è in src/small_games/ppo.py; i risultati (curva di training, checkpoint, valutazione) in results/ppo/.
"""

from pathlib import Path

from small_games.ppo import PPOConfig, main

# il minimal action set di MinAtar: fermo, sinistra, destra (su, giù e fuoco non fanno niente)
CONFIG = PPOConfig(game="breakout", hidden=(32, 32), activation="tanh", actions=(0, 1, 3), total_steps=100_000_000)

if __name__ == "__main__":
    main(CONFIG, Path(__file__).parent)
