"""
PPO su Seaquest (MinAtar): addestra la policy 600-32-32-6 tanh (la stessa architettura delle reti scritte a mano),
la salva in weights/seaquest/ppo.json e la valuta con il protocollo del gioco.

    uv run --group ppo python experiments/seaquest/ppo/train.py                     # configurazione sotto
    uv run --group ppo python experiments/seaquest/ppo/train.py --total-steps 2000000

weights/seaquest/ppo.json viene dalla corsa ppo_ent05 di experiments/ppo_runs.py, che ha questa stessa configurazione
(e lo stesso seed): la sua curva, i checkpoint e la valutazione sono in results/ppo_ent05/. Le corse con ent_coef
0.01 e 0.1 sono in results/ppo_ent01/ e results/ppo_ent10/, il primo tentativo (tutte e 6 le azioni, ent_coef 0.01,
30M passi) in results/ppo_v1/; vedi README.md. L'algoritmo è in src/small_games/ppo.py.
"""

from pathlib import Path

from small_games.ppo import PPOConfig, main

# il minimal action set di MinAtar per Seaquest: tutte e sei le azioni
CONFIG = PPOConfig(game="seaquest", hidden=(32, 32), activation="tanh", actions=None,
                   ent_coef=0.05, total_steps=200_000_000)

if __name__ == "__main__":
    main(CONFIG, Path(__file__).parent)
