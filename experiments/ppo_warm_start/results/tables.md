## Punteggi di tutti i modelli

Protocollo completo di ogni gioco (1000 episodi MinAtar, 500 Low Tide), media ± errore standard.


### low_tide

| modello | tipo | configurazione | punteggio |
|---|---|---|---|
| `ppo_arch_gpt` | PPO da zero | 43-61-53 relu, ent 0.01, lr 0.0005, 100M passi | 166.9 ± 2.7 |
| `heuristic` | baseline |  | 164.0 ± 1.5 |
| `ppo_100m` | PPO da zero | 64-64 tanh, ent 0.01, lr 0.0005, 100M passi | 162.7 ± 2.6 |
| `ppo_arch_claude` | PPO da zero | 46-36 relu/tanh, ent 0.01, lr 0.0005, 100M passi | 158.4 ± 2.9 |
| `ppo_from_gpt` | PPO dai pesi a mano | 43-61-53 relu, ent 0.01, lr 0.0005, 100M passi | 147.3 ± 2.9 |
| `claude` | scritta a mano |  | 132.3 ± 0.9 |
| `ppo` | PPO da zero | 64-64 tanh, ent 0.01, lr 0.0005, 20M passi | 125.7 ± 1.9 |
| `gpt` | scritta a mano |  | 79.9 ± 0.9 |
| `ppo_from_claude_cw` | PPO dai pesi a mano | 46-36 relu/tanh, ent 0.01, lr 0.0005, 100M passi, 2M passi di solo critico | 66.6 ± 0.6 |
| `ppo_from_claude` | PPO dai pesi a mano | 46-36 relu/tanh, ent 0.01, lr 0.0005, 100M passi | 31.0 ± 0.3 |
| `random` | baseline |  | 1.9 ± 0.1 |
| `safe` | baseline |  | 0.0 ± 0.0 |
| `ppo_from_gpt_cw` | PPO dai pesi a mano | 43-61-53 relu, ent 0.01, lr 0.0005, 100M passi, 2M passi di solo critico | 0.0 ± 0.0 |

### breakout

| modello | tipo | configurazione | punteggio |
|---|---|---|---|
| `paper_astra` | scritta a mano |  | 512.5 ± 7.6 |
| `ppo_from_astra` | PPO dai pesi a mano | 32-32 tanh, ent 0.05, lr 0.0005, 200M passi | 512.5 ± 7.6 |
| `ppo_ent05` | PPO da zero | 32-32 tanh, ent 0.05, lr 0.0005, 200M passi | 63.2 ± 1.0 |
| `ppo_ent10` | PPO da zero | 32-32 tanh, ent 0.1, lr 0.0005, 200M passi | 52.2 ± 1.4 |
| `ppo_ent01` | PPO da zero | 32-32 tanh, ent 0.01, lr 0.0005, 200M passi | 34.7 ± 0.5 |
| `ppo_v1` | PPO da zero | 32-32 tanh, ent 0.01, lr 0.0005, 30M passi | 5.8 ± 0.0 |
| `random` | baseline |  | 0.5 ± 0.0 |

### seaquest

| modello | tipo | configurazione | punteggio |
|---|---|---|---|
| `ppo_from_astra` | PPO dai pesi a mano | 32-32 tanh, ent 0.05, lr 0.0005, 200M passi | 82.5 ± 2.1 |
| `ppo_ent05` | PPO da zero | 32-32 tanh, ent 0.05, lr 0.0005, 200M passi | 55.9 ± 1.3 |
| `paper_astra` | scritta a mano |  | 51.4 ± 1.4 |
| `ppo_ent10` | PPO da zero | 32-32 tanh, ent 0.1, lr 0.0005, 200M passi | 34.4 ± 1.0 |
| `ppo_v1` | PPO da zero | 32-32 tanh, ent 0.01, lr 0.0005, 30M passi | 5.4 ± 0.1 |
| `ppo_ent01` | PPO da zero | 32-32 tanh, ent 0.01, lr 0.0005, 200M passi | 4.4 ± 0.0 |
| `random` | baseline |  | 0.1 ± 0.0 |

### asterix

| modello | tipo | configurazione | punteggio |
|---|---|---|---|
| `paper_astra` | scritta a mano |  | 54.8 ± 1.1 |
| `ppo_from_astra` | PPO dai pesi a mano | 32-32 tanh, ent 0.05, lr 0.0005, 200M passi | 50.8 ± 1.0 |
| `ppo_ent05` | PPO da zero | 32-32 tanh, ent 0.05, lr 0.0005, 200M passi | 24.6 ± 0.5 |
| `ppo_ent01` | PPO da zero | 32-32 tanh, ent 0.01, lr 0.0005, 200M passi | 19.4 ± 0.4 |
| `ppo_ent10` | PPO da zero | 32-32 tanh, ent 0.1, lr 0.0005, 200M passi | 15.5 ± 0.4 |
| `ppo_v1` | PPO da zero | 32-32 tanh, ent 0.01, lr 0.0005, 30M passi | 9.1 ± 0.2 |
| `random` | baseline |  | 0.5 ± 0.0 |

## Partire dai pesi scritti a mano: quanto migliora

Checkpoint ogni 5M passi valutati su 200 episodi; inizio e fine con il protocollo completo. La curva di addestramento (media sugli ultimi 1M passi) vede anche quello che succede fra due checkpoint, ma è più rumorosa: su Breakout la policy non cambia e oscilla comunque di ±15%.

| gioco | corsa | rete a mano | minimo della curva di addestramento nel primo 10% (passo) | checkpoint peggiore (passo) | torna al livello della rete a mano a | +10% a | miglior checkpoint (passo) | fine | fine − rete a mano | da zero, stessa arch. |
|---|---|---|---|---|---|---|---|---|---|---|
| breakout | `ppo_from_astra` | Astra 512.5 | 444.5 (15M) | 512.7 (0) | non scende | mai | 512.7 (0) | 512.5 ± 7.6 | +0.0 (+0%) | 63.2 ± 1.0 |
| seaquest | `ppo_from_astra` | Astra 51.4 | 36.9 (4.9M) | 33.3 (5.0M) | 20M | 25M | 87.6 (190M) | 82.5 ± 2.1 | +31.0 (+60%) | 55.9 ± 1.3 |
| asterix | `ppo_from_astra` | Astra 54.8 | 36.0 (8.9M) | 38.9 (15M) | 140M | mai | 55.4 (175M) | 50.8 ± 1.0 | -4.0 (-7%) | 24.6 ± 0.5 |
| low_tide | `ppo_from_gpt` | GPT 79.9 | 1.3 (1.1M) | 46.1 (5.0M) | 15M | 15M | 149.5 (100M) | 147.3 ± 2.9 | +67.3 (+84%) | 166.9 ± 2.7 |
| low_tide | `ppo_from_gpt_cw` | GPT 79.9 | 0.0 (4.2M) | 0.0 (5.0M) | mai | mai | 79.7 (0) | 0.0 ± 0.0 | -79.9 (-100%) | 166.9 ± 2.7 |
| low_tide | `ppo_from_claude` | Claude 132.3 | 6.3 (5.7M) | 5.4 (5.0M) | mai | mai | 132.1 (0) | 31.0 ± 0.3 | -101.2 (-77%) | 158.4 ± 2.9 |
| low_tide | `ppo_from_claude_cw` | Claude 132.3 | 38.4 (4.3M) | 39.8 (5.0M) | mai | mai | 132.1 (0) | 66.6 ± 0.6 | -65.7 (-50%) | 158.4 ± 2.9 |

## Partire dai pesi scritti a mano: quanto tempo si risparmia

Soglia: 95% del punteggio finale della corsa da zero; passi al primo checkpoint che la supera. Le ore sono quelle della corsa da zero su questa macchina (1 core per corsa).

| gioco | corsa | soglia | da mano la supera a | da zero la supera a | risparmio | da zero raggiunge la rete a mano a | da zero raggiunge il finale di questa corsa a |
|---|---|---|---|---|---|---|---|
| breakout | `ppo_from_astra` | 60.1 | 0 | 105M | 105M passi ≈ 0.9 h | mai | mai |
| seaquest | `ppo_from_astra` | 53.1 | 20M | 170M | 150M passi ≈ 1.4 h | 160M | mai |
| asterix | `ppo_from_astra` | 23.4 | 0 | 160M | 160M passi ≈ 1.4 h | mai | mai |
| low_tide | `ppo_from_gpt` | 158.5 | mai | 95M | nessuno: non ci arriva | 5.0M | 90M |
| low_tide | `ppo_from_gpt_cw` | 158.5 | mai | 95M | nessuno: non ci arriva | 5.0M | 0 |
| low_tide | `ppo_from_claude` | 150.4 | mai | 95M | nessuno: non ci arriva | 30M | 5.0M |
| low_tide | `ppo_from_claude_cw` | 150.4 | mai | 95M | nessuno: non ci arriva | 30M | 5.0M |

## Il gradiente durante l'addestramento

Norma del gradiente dell'attore sul batch intero (8192 passi), prima del clipping; mediane. Il clipping taglia la norma congiunta di attore e critico a 0.5.

| gioco | corsa | primo aggiornamento (critico casuale) | primi 1M passi | di cui policy gradient / entropia | ultimi 10M passi | minibatch tagliati: primi 1M / ultimi 10M |
|---|---|---|---|---|---|---|
| breakout | `ppo_from_astra` | 4.45e-13 | 4.35e-13 | 1.97e-14 / 4.25e-13 | 4.3e-13 | 1% / 0% |
| breakout | `ppo_ent05` | non registrato (corsa precedente: vedi la tabella a pesi fissi) | | | | |
| seaquest | `ppo_from_astra` | 0.434 | 0.531 | 0.505 / 0.11 | 0.136 | 100% / 15% |
| seaquest | `ppo_ent05` | non registrato (corsa precedente: vedi la tabella a pesi fissi) | | | | |
| asterix | `ppo_from_astra` | 0.999 | 0.767 | 0.69 / 0.286 | 0.243 | 100% / 73% |
| asterix | `ppo_ent05` | non registrato (corsa precedente: vedi la tabella a pesi fissi) | | | | |
| low_tide | `ppo_from_gpt` | 18.6 | 3.8 | 3.02 / 0.0845 | 0.917 | 100% / 100% |
| low_tide | `ppo_from_gpt_cw` | 2.18 | 1.57 | 1.56 / 0.0756 | 0.83 | 3% / 100% |
| low_tide | `ppo_arch_gpt` | 0.0756 | 0.184 | 0.185 / 0.0125 | 0.426 | 72% / 100% |
| low_tide | `ppo_from_claude` | 2.87 | 2.29 | 2.29 / 0.03 | 1.76 | 79% / 95% |
| low_tide | `ppo_from_claude_cw` | 14.4 | 25.8 | 25.9 / 1.12 | 1.64 | 14% / 97% |
| low_tide | `ppo_arch_claude` | 0.0374 | 0.0965 | 0.1 / 0.00674 | 0.358 | 24% / 100% |

## Quanto si muovono i pesi

| gioco | corsa | ‖θ₀‖ | ‖θ_fine − θ₀‖ | relativo | per strato | da zero: ‖θ_fine − θ₀‖ (relativo) | zeri della rete a mano ora ≠ 0 | stessa mossa della rete a mano (sui suoi stati / sui propri) |
|---|---|---|---|---|---|---|---|---|
| breakout | `ppo_from_astra` | 3587.8 | 7e-06 | 0.00% | 0.0% · 0.0% · 0.0% | 51.2 (29%) | 0% | 100% / 100% |
| seaquest | `ppo_from_astra` | 498.3 | 43.9 | 8.80% | 34.7% · 8.1% · 0.5% | 55.5 (491%) | 51% | 37% / 56% |
| asterix | `ppo_from_astra` | 1451.3 | 32.6 | 2.25% | 13.8% · 7.1% · 0.1% | 52 (52%) | 42% | 68% / 70% |
| low_tide | `ppo_from_gpt` | 1600.4 | 27.5 | 1.72% | 36.9% · 2.2% · 1.6% · 0.2% | 39.9 (239%) | 80% | 31% / 36% |
| low_tide | `ppo_from_gpt_cw` | 1600.4 | 20.8 | 1.30% | 28.2% · 1.8% · 0.9% · 0.3% | 39.9 (239%) | 68% | 15% / 0% |
| low_tide | `ppo_from_claude` | 45620.7 | 40.8 | 0.09% | 0.2% · 0.0% · 12.9% | 42.3 (330%) | 60% | 17% / 18% |
| low_tide | `ppo_from_claude_cw` | 45620.7 | 32.9 | 0.07% | 0.2% · 0.0% · 8.9% | 42.3 (330%) | 60% | 18% / 24% |

## Il gradiente a pesi fissi

Attore fermo, critico adattato per 2M passi; media dell'ultimo quarto. Norme prima del clipping (max_grad_norm 0.5 su attore + critico).

| gioco | rete | entropia | ‖θ‖ | ‖g‖ minibatch | ‖g‖ batch intero | di cui policy gradient | di cui entropia | ‖g‖/‖θ‖ | batch intero col critico casuale |
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
