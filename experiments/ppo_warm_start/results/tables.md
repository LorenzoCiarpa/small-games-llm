## Scores of all models

Full protocol of each game (1000 MinAtar episodes, 500 Low Tide), mean ± standard error.


### low_tide

| model | type | configuration | score |
|---|---|---|---|
| `ppo_arch_gpt` | PPO from scratch | 43-61-53 relu, ent 0.01, lr 0.0005, 100M steps | 166.9 ± 2.7 |
| `heuristic` | baseline |  | 164.0 ± 1.5 |
| `ppo_100m` | PPO from scratch | 64-64 tanh, ent 0.01, lr 0.0005, 100M steps | 162.7 ± 2.6 |
| `ppo_arch_claude` | PPO from scratch | 46-36 relu/tanh, ent 0.01, lr 0.0005, 100M steps | 158.4 ± 2.9 |
| `ppo_from_gpt` | PPO from the hand-written weights | 43-61-53 relu, ent 0.01, lr 0.0005, 100M steps | 147.3 ± 2.9 |
| `claude` | hand-written |  | 132.3 ± 0.9 |
| `ppo` | PPO from scratch | 64-64 tanh, ent 0.01, lr 0.0005, 20M steps | 125.7 ± 1.9 |
| `gpt` | hand-written |  | 79.9 ± 0.9 |
| `ppo_from_claude_cw` | PPO from the hand-written weights | 46-36 relu/tanh, ent 0.01, lr 0.0005, 100M steps, 2M critic-only steps | 66.6 ± 0.6 |
| `ppo_from_claude` | PPO from the hand-written weights | 46-36 relu/tanh, ent 0.01, lr 0.0005, 100M steps | 31.0 ± 0.3 |
| `random` | baseline |  | 1.9 ± 0.1 |
| `ppo_from_gpt_cw` | PPO from the hand-written weights | 43-61-53 relu, ent 0.01, lr 0.0005, 100M steps, 2M critic-only steps | 0.0 ± 0.0 |

### breakout

| model | type | configuration | score |
|---|---|---|---|
| `paper_astra` | hand-written |  | 512.5 ± 7.6 |
| `ppo_from_astra` | PPO from the hand-written weights | 32-32 tanh, ent 0.05, lr 0.0005, 200M steps | 512.5 ± 7.6 |
| `ppo_ent05` | PPO from scratch | 32-32 tanh, ent 0.05, lr 0.0005, 200M steps | 63.2 ± 1.0 |
| `ppo_ent10` | PPO from scratch | 32-32 tanh, ent 0.1, lr 0.0005, 200M steps | 52.2 ± 1.4 |
| `ppo_ent01` | PPO from scratch | 32-32 tanh, ent 0.01, lr 0.0005, 200M steps | 34.7 ± 0.5 |
| `ppo_v1` | PPO from scratch | 32-32 tanh, ent 0.01, lr 0.0005, 30M steps | 5.8 ± 0.0 |
| `random` | baseline |  | 0.5 ± 0.0 |

### seaquest

| model | type | configuration | score |
|---|---|---|---|
| `ppo_from_astra` | PPO from the hand-written weights | 32-32 tanh, ent 0.05, lr 0.0005, 200M steps | 82.5 ± 2.1 |
| `ppo_ent05` | PPO from scratch | 32-32 tanh, ent 0.05, lr 0.0005, 200M steps | 55.9 ± 1.3 |
| `paper_astra` | hand-written |  | 51.4 ± 1.4 |
| `ppo_ent10` | PPO from scratch | 32-32 tanh, ent 0.1, lr 0.0005, 200M steps | 34.4 ± 1.0 |
| `ppo_v1` | PPO from scratch | 32-32 tanh, ent 0.01, lr 0.0005, 30M steps | 5.4 ± 0.1 |
| `ppo_ent01` | PPO from scratch | 32-32 tanh, ent 0.01, lr 0.0005, 200M steps | 4.4 ± 0.0 |
| `random` | baseline |  | 0.1 ± 0.0 |

### asterix

| model | type | configuration | score |
|---|---|---|---|
| `paper_astra` | hand-written |  | 54.8 ± 1.1 |
| `ppo_from_astra` | PPO from the hand-written weights | 32-32 tanh, ent 0.05, lr 0.0005, 200M steps | 50.8 ± 1.0 |
| `ppo_ent05` | PPO from scratch | 32-32 tanh, ent 0.05, lr 0.0005, 200M steps | 24.6 ± 0.5 |
| `ppo_ent01` | PPO from scratch | 32-32 tanh, ent 0.01, lr 0.0005, 200M steps | 19.4 ± 0.4 |
| `ppo_ent10` | PPO from scratch | 32-32 tanh, ent 0.1, lr 0.0005, 200M steps | 15.5 ± 0.4 |
| `ppo_v1` | PPO from scratch | 32-32 tanh, ent 0.01, lr 0.0005, 30M steps | 9.1 ± 0.2 |
| `random` | baseline |  | 0.5 ± 0.0 |

## Starting from the hand-written weights: how much it improves

Checkpoints every 5M steps evaluated on 200 episodes; start and end with the full protocol. The training curve (mean over the last 1M steps) also sees what happens between two checkpoints, but it is noisier: on Breakout the policy does not change and the curve still oscillates by ±15%.

| game | run | hand-written network | minimum of the training curve in the first 10% (step) | worst checkpoint (step) | back to the hand-written network's level at | +10% at | best checkpoint (step) | end | end − hand-written network | from scratch, same arch. |
|---|---|---|---|---|---|---|---|---|---|---|
| breakout | `ppo_from_astra` | Astra 512.5 | 444.5 (15M) | 512.7 (0) | does not drop | never | 512.7 (0) | 512.5 ± 7.6 | +0.0 (+0%) | 63.2 ± 1.0 |
| seaquest | `ppo_from_astra` | Astra 51.4 | 36.9 (4.9M) | 33.3 (5.0M) | 20M | 25M | 87.6 (190M) | 82.5 ± 2.1 | +31.0 (+60%) | 55.9 ± 1.3 |
| asterix | `ppo_from_astra` | Astra 54.8 | 36.0 (8.9M) | 38.9 (15M) | 140M | never | 55.4 (175M) | 50.8 ± 1.0 | -4.0 (-7%) | 24.6 ± 0.5 |
| low_tide | `ppo_from_gpt` | GPT 79.9 | 1.3 (1.1M) | 46.1 (5.0M) | 15M | 15M | 149.5 (100M) | 147.3 ± 2.9 | +67.3 (+84%) | 166.9 ± 2.7 |
| low_tide | `ppo_from_gpt_cw` | GPT 79.9 | 0.0 (4.2M) | 0.0 (5.0M) | never | never | 79.7 (0) | 0.0 ± 0.0 | -79.9 (-100%) | 166.9 ± 2.7 |
| low_tide | `ppo_from_claude` | Claude 132.3 | 6.3 (5.7M) | 5.4 (5.0M) | never | never | 132.1 (0) | 31.0 ± 0.3 | -101.2 (-77%) | 158.4 ± 2.9 |
| low_tide | `ppo_from_claude_cw` | Claude 132.3 | 38.4 (4.3M) | 39.8 (5.0M) | never | never | 132.1 (0) | 66.6 ± 0.6 | -65.7 (-50%) | 158.4 ± 2.9 |

## Starting from the hand-written weights: how much time is saved

Threshold: 95% of the final score of the run from scratch; steps at the first checkpoint that passes it. The hours are those of the run from scratch on this machine (1 core per run).

| game | run | threshold | from the hand-written network, passes it at | from scratch, passes it at | saving | from scratch, reaches the hand-written network at | from scratch, reaches the final score of this run at |
|---|---|---|---|---|---|---|---|
| breakout | `ppo_from_astra` | 60.1 | 0 | 105M | 105M steps ≈ 0.9 h | never | never |
| seaquest | `ppo_from_astra` | 53.1 | 20M | 170M | 150M steps ≈ 1.4 h | 160M | never |
| asterix | `ppo_from_astra` | 23.4 | 0 | 160M | 160M steps ≈ 1.4 h | never | never |
| low_tide | `ppo_from_gpt` | 158.5 | never | 95M | none: it does not get there | 5.0M | 90M |
| low_tide | `ppo_from_gpt_cw` | 158.5 | never | 95M | none: it does not get there | 5.0M | 0 |
| low_tide | `ppo_from_claude` | 150.4 | never | 95M | none: it does not get there | 30M | 5.0M |
| low_tide | `ppo_from_claude_cw` | 150.4 | never | 95M | none: it does not get there | 30M | 5.0M |

## The gradient during training

Norm of the actor's gradient on the full batch (8192 steps), before clipping; medians. Clipping cuts the joint norm of actor and critic at 0.5.

| game | run | first update (random critic) | first 1M steps | of which policy gradient / entropy | last 10M steps | minibatches cut: first 1M / last 10M |
|---|---|---|---|---|---|---|
| breakout | `ppo_from_astra` | 4.45e-13 | 4.35e-13 | 1.97e-14 / 4.25e-13 | 4.3e-13 | 1% / 0% |
| breakout | `ppo_ent05` | not logged (earlier run: see the fixed-weight table) | | | | |
| seaquest | `ppo_from_astra` | 0.434 | 0.531 | 0.505 / 0.11 | 0.136 | 100% / 15% |
| seaquest | `ppo_ent05` | not logged (earlier run: see the fixed-weight table) | | | | |
| asterix | `ppo_from_astra` | 0.999 | 0.767 | 0.69 / 0.286 | 0.243 | 100% / 73% |
| asterix | `ppo_ent05` | not logged (earlier run: see the fixed-weight table) | | | | |
| low_tide | `ppo_from_gpt` | 18.6 | 3.8 | 3.02 / 0.0845 | 0.917 | 100% / 100% |
| low_tide | `ppo_from_gpt_cw` | 2.18 | 1.57 | 1.56 / 0.0756 | 0.83 | 3% / 100% |
| low_tide | `ppo_arch_gpt` | 0.0756 | 0.184 | 0.185 / 0.0125 | 0.426 | 72% / 100% |
| low_tide | `ppo_from_claude` | 2.87 | 2.29 | 2.29 / 0.03 | 1.76 | 79% / 95% |
| low_tide | `ppo_from_claude_cw` | 14.4 | 25.8 | 25.9 / 1.12 | 1.64 | 14% / 97% |
| low_tide | `ppo_arch_claude` | 0.0374 | 0.0965 | 0.1 / 0.00674 | 0.358 | 24% / 100% |

## How much the weights move

| game | run | ‖θ₀‖ | ‖θ_end − θ₀‖ | relative | per layer | from scratch: ‖θ_end − θ₀‖ (relative) | zeros of the hand-written network now ≠ 0 | same move as the hand-written network (on its states / on its own) |
|---|---|---|---|---|---|---|---|---|
| breakout | `ppo_from_astra` | 3587.8 | 7e-06 | 0.00% | 0.0% · 0.0% · 0.0% | 51.2 (29%) | 0% | 100% / 100% |
| seaquest | `ppo_from_astra` | 498.3 | 43.9 | 8.80% | 34.7% · 8.1% · 0.5% | 55.5 (491%) | 51% | 37% / 56% |
| asterix | `ppo_from_astra` | 1451.3 | 32.6 | 2.25% | 13.8% · 7.1% · 0.1% | 52 (52%) | 42% | 68% / 70% |
| low_tide | `ppo_from_gpt` | 1600.4 | 27.5 | 1.72% | 36.9% · 2.2% · 1.6% · 0.2% | 39.9 (239%) | 80% | 31% / 36% |
| low_tide | `ppo_from_gpt_cw` | 1600.4 | 20.8 | 1.30% | 28.2% · 1.8% · 0.9% · 0.3% | 39.9 (239%) | 68% | 15% / 0% |
| low_tide | `ppo_from_claude` | 45620.7 | 40.8 | 0.09% | 0.2% · 0.0% · 12.9% | 42.3 (330%) | 60% | 17% / 18% |
| low_tide | `ppo_from_claude_cw` | 45620.7 | 32.9 | 0.07% | 0.2% · 0.0% · 8.9% | 42.3 (330%) | 60% | 18% / 24% |

## The gradient at fixed weights

Actor frozen, critic adapted for 2M steps; mean of the last quarter. Norms before clipping (max_grad_norm 0.5 on actor + critic).

| game | network | entropy | ‖θ‖ | ‖g‖ minibatch | ‖g‖ full batch | of which policy gradient | of which entropy | ‖g‖/‖θ‖ | full batch with the random critic |
|---|---|---|---|---|---|---|---|---|---|
| breakout | `astra` | 0.000 | 3587.8 | 4.21e-13 | 4.18e-13 | 1.21e-14 | 4.2e-13 | 1.2e-16 | 4.54e-13 |
| breakout | `random` | 1.099 | 11.3 | 0.0765 | 0.0601 | 0.0602 | 0.000135 | 0.0053 | 0.0568 |
| breakout | `ppo_5M` | 0.528 | 174.3 | 0.454 | 0.164 | 0.185 | 0.0501 | 0.00094 | 0.21 |
| breakout | `ppo_50M` | 0.764 | 177.8 | 0.337 | 0.144 | 0.142 | 0.0221 | 0.00081 | 0.117 |
| breakout | `ppo_final` | 0.738 | 181.2 | 0.24 | 0.085 | 0.0823 | 0.0162 | 0.00047 | 0.0498 |
| breakout | `from_astra_final` | 0.000 | 3587.8 | 4.25e-13 | 4.23e-13 | 1.22e-14 | 4.24e-13 | 1.2e-16 | 4.6e-13 |
| seaquest | `astra` | 0.143 | 498.3 | 0.966 | 0.367 | 0.344 | 0.0957 | 0.00074 | 0.308 |
| seaquest | `random` | 1.792 | 11.3 | 0.0742 | 0.0545 | 0.0545 | 4.19e-05 | 0.0048 | 0.0419 |
| seaquest | `ppo_5M` | 1.479 | 21.1 | 0.225 | 0.0839 | 0.0865 | 0.0262 | 0.004 | 0.0711 |
| seaquest | `ppo_50M` | 1.267 | 45.0 | 0.328 | 0.126 | 0.136 | 0.0288 | 0.0028 | 0.109 |
| seaquest | `ppo_final` | 1.229 | 57.1 | 0.31 | 0.115 | 0.117 | 0.0286 | 0.002 | 0.143 |
| seaquest | `from_astra_final` | 0.685 | 499.8 | 0.35 | 0.131 | 0.133 | 0.023 | 0.00026 | 0.15 |
| asterix | `astra` | 0.286 | 1759.6 | 2.29 | 1.14 | 1.06 | 0.951 | 0.00065 | 1.46 |
| asterix | `random` | 1.609 | 11.3 | 0.0418 | 0.0146 | 0.0146 | 9.28e-06 | 0.0013 | 0.0134 |
| asterix | `ppo_5M` | 1.275 | 101.7 | 0.206 | 0.0762 | 0.0763 | 0.014 | 0.00075 | 0.0691 |
| asterix | `ppo_50M` | 1.211 | 108.9 | 0.267 | 0.0997 | 0.0988 | 0.0136 | 0.00092 | 0.0976 |
| asterix | `ppo_final` | 1.190 | 113.5 | 0.316 | 0.111 | 0.112 | 0.0162 | 0.00098 | 0.106 |
| asterix | `from_astra_final` | 0.693 | 1451.4 | 0.746 | 0.289 | 0.293 | 0.0269 | 0.0002 | 0.282 |
| low_tide | `gpt` | 0.620 | 1600.4 | 4.41 | 1.32 | 1.31 | 0.075 | 0.00082 | 2.1 |
| low_tide | `random_arch_gpt` | 1.609 | 16.7 | 0.05 | 0.0247 | 0.0247 | 1.4e-05 | 0.0015 | 0.0369 |
| low_tide | `from_gpt_final` | 0.291 | 1600.7 | 2.69 | 1.09 | 1.09 | 0.0174 | 0.00068 | 1.54 |
| low_tide | `from_gpt_cw_final` | 0.847 | 1600.6 | 3.18 | 1.45 | 1.44 | 0.0626 | 0.0009 | 0.884 |
| low_tide | `arch_gpt_final` | 0.414 | 43.7 | 1.22 | 0.409 | 0.41 | 0.0119 | 0.0093 | 0.328 |
| low_tide | `claude` | 0.351 | 45620.7 | 77.7 | 28.3 | 28.3 | 1.09 | 0.00062 | 25.8 |
| low_tide | `random_arch_claude` | 1.609 | 12.8 | 0.0757 | 0.0415 | 0.0415 | 1.12e-05 | 0.0032 | 0.0486 |
| low_tide | `from_claude_final` | 0.493 | 45620.9 | 4.38 | 1.88 | 1.88 | 0.0106 | 4.1e-05 | 3.31 |
| low_tide | `from_claude_cw_final` | 0.067 | 45620.9 | 4.64 | 2.4 | 2.42 | 0.0258 | 5.3e-05 | 3.59 |
| low_tide | `arch_claude_final` | 0.363 | 44.4 | 0.997 | 0.352 | 0.352 | 0.00823 | 0.0079 | 0.454 |
| low_tide | `random_64x64` | 1.609 | 16.0 | 0.108 | 0.0524 | 0.0524 | 3.38e-05 | 0.0033 | 0.0751 |
| low_tide | `ppo_100m_5M` | 0.452 | 26.8 | 0.609 | 0.237 | 0.237 | 0.00761 | 0.0088 | 0.276 |
| low_tide | `ppo_100m_50M` | 0.391 | 52.3 | 0.618 | 0.235 | 0.236 | 0.0057 | 0.0045 | 0.213 |
| low_tide | `ppo_100m_final` | 0.377 | 57.0 | 0.7 | 0.252 | 0.253 | 0.00715 | 0.0044 | 0.254 |
