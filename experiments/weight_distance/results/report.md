# Distanza tra le reti: risultati

Generato da `analyze.py` (42 s, seed 1234). Il commento ai risultati è nel README.

## 1. Com'è fatta ogni rete

Per strato: parametri (pesi + bias), quota non nulla, quanti valori diversi compaiono tra i non nulli, quota di non nulli con al più 2 decimali, norma L2, rango stabile (‖W‖²_F / ‖W‖²_2: quante direzioni contano davvero), unità nascoste usate (che leggono qualcosa e sono lette dallo strato dopo).

| rete | strato | forma | attivazione | parametri | non nulli | valori distinti | ≤ 2 decimali | norma | max |w| | rango stabile | unità usate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| astra/breakout | 1 | 600->32 | tanh | 19232 | 4.0% | 56 | 100% | 1673.63 | 180.00 | 1.1 | 6/32 |
|  | 2 | 32->32 | tanh | 1056 | 3.0% | 5 | 100% | 3165.09 | 1000.00 | 4.3 | 8/32 |
|  | 3 | 32->6 | linear | 198 | 6.6% | 3 | 100% | 231.08 | 100.00 | 2.0 |  |
| astra/seaquest | 1 | 600->32 | tanh | 19232 | 2.1% | 13 | 100% | 125.19 | 8.00 | 10.6 | 32/32 |
|  | 2 | 32->32 | tanh | 1056 | 35.5% | 35 | 88% | 74.09 | 15.21 | 7.6 | 26/32 |
|  | 3 | 32->6 | linear | 198 | 16.2% | 8 | 100% | 476.56 | 244.00 | 2.8 |  |
| astra/asterix | 1 | 600->32 | tanh | 19232 | 2.6% | 33 | 100% | 223.75 | 24.00 | 9.7 | 32/32 |
|  | 2 | 32->32 | tanh | 1056 | 8.0% | 8 | 100% | 147.40 | 24.00 | 7.2 | 32/32 |
|  | 3 | 32->6 | linear | 198 | 26.3% | 9 | 100% | 1739.06 | 1000.00 | 2.3 |  |
| claude/low_tide | 1 | 403->46 | relu | 18584 | 4.1% | 38 | 100% | 11771.87 | 1024.00 | 8.2 | 46/46 |
|  | 2 | 46->36 | tanh | 1692 | 8.5% | 16 | 100% | 44074.96 | 5122.50 | 2.3 | 36/36 |
|  | 3 | 36->5 | linear | 185 | 21.6% | 13 | 100% | 256.66 | 160.00 | 2.6 |  |
| gpt/low_tide | 1 | 403->43 | relu | 17372 | 2.9% | 9 | 100% | 64.07 | 10.00 | 1.2 | 43/43 |
|  | 2 | 43->61 | relu | 2684 | 11.8% | 22 | 100% | 416.52 | 68.00 | 13.7 | 61/61 |
|  | 3 | 61->53 | relu | 3286 | 9.0% | 3 | 100% | 607.33 | 64.00 | 4.7 | 53/53 |
|  | 4 | 53->5 | linear | 270 | 24.1% | 11 | 100% | 1419.46 | 1000.00 | 2.0 |  |
| ppo/breakout-grid | 1 | 500->32 | tanh | 16032 | 100.0% | 15745 | 0% | 117.05 | 7.88 | 1.4 | 32/32 |
|  | 2 | 32->32 | tanh | 1056 | 100.0% | 1056 | 0% | 30.33 | 3.75 | 2.4 | 32/32 |
|  | 3 | 32->5 | linear | 165 | 100.0% | 165 | 0% | 18.09 | 5.34 | 1.9 |  |
| ppo/pong-grid | 1 | 500->32 | tanh | 16032 | 100.0% | 15750 | 0% | 67.48 | 5.24 | 3.6 | 32/32 |
|  | 2 | 32->32 | tanh | 1056 | 100.0% | 1056 | 0% | 28.00 | 3.18 | 3.7 | 32/32 |
|  | 3 | 32->5 | linear | 165 | 100.0% | 165 | 0% | 9.84 | 1.95 | 1.9 |  |
| ppo/pacman-grid | 1 | 500->32 | tanh | 16032 | 100.0% | 15748 | 0% | 90.46 | 7.28 | 8.2 | 32/32 |
|  | 2 | 32->32 | tanh | 1056 | 100.0% | 1056 | 0% | 23.09 | 2.38 | 6.9 | 32/32 |
|  | 3 | 32->5 | linear | 165 | 100.0% | 165 | 1% | 15.05 | 3.07 | 2.8 |  |
| ppo/freeway-grid | 1 | 500->32 | tanh | 16032 | 100.0% | 15802 | 0% | 82.56 | 8.72 | 3.7 | 32/32 |
|  | 2 | 32->32 | tanh | 1056 | 100.0% | 1056 | 0% | 22.00 | 3.08 | 7.3 | 32/32 |
|  | 3 | 32->5 | linear | 165 | 100.0% | 165 | 1% | 14.53 | 3.14 | 2.3 |  |

Tutta la rete: i valori più usati e quanta parte dei pesi non nulli coprono gli 8 più frequenti.

| rete | autore | non nulli | valori distinti | ≤ 2 decimali | 8 valori più usati | valori più usati (conteggio) |
|---|---|---|---|---|---|---|
| astra/breakout | GPT-6 Astra | 4.0% | 61 | 100% | 34% | 0.18 (×37), 0.16 (×34), 0.14 (×34), 0.12 (×34), 0.1 (×34), 0.08 (×34) |
| astra/seaquest | GPT-6 Astra | 3.9% | 52 | 94% | 56% | 6 (×121), 8 (×70), -8 (×70), -3 (×50), 0.45 (×38), 0.12 (×38) |
| astra/asterix | GPT-6 Astra | 3.1% | 42 | 100% | 53% | -8 (×71), 6 (×48), 16 (×40), -16 (×39), 24 (×38), 8 (×34) |
| claude/low_tide | Claude Opus | 4.7% | 62 | 100% | 44% | 1 (×125), 5 (×52), 64 (×44), 32 (×44), 16 (×44), 8 (×44) |
| gpt/low_tide | GPT-6 Astra | 5.0% | 40 | 100% | 81% | 1 (×502), 2 (×100), -2 (×87), 3 (×70), -64 (×58), 4 (×50) |
| ppo/breakout-grid | PPO | 100.0% | 16960 | 0% | 0% | 0.115927 (×3), 0.028718 (×3), 0.020599 (×3), -0.027989 (×3), 0.174443 (×2), 0.149518 (×2) |
| ppo/pong-grid | PPO | 100.0% | 16955 | 0% | 0% | 0.115927 (×3), 0.040025 (×3), -0.001615 (×3), -0.016648 (×3), 0.604628 (×2), 0.430204 (×2) |
| ppo/pacman-grid | PPO | 100.0% | 16957 | 0% | 0% | 0.117085 (×3), 0.068172 (×3), 0.009367 (×3), -0.006724 (×3), -0.078656 (×3), 0.971892 (×2) |
| ppo/freeway-grid | PPO | 100.0% | 17014 | 0% | 0% | 0.186329 (×2), 0.181882 (×2), 0.159208 (×2), 0.153081 (×2), 0.130031 (×2), 0.129297 (×2) |

## 2. Cosa legge il primo strato

Per ogni canale d'ingresso: quota delle connessioni non nulle di W1 che partono da quel canale e, tra parentesi, quota della massa Σ|W1|. Le due misure possono essere molto diverse: nelle reti scritte a mano la scala dei pesi è arbitraria (la rete di Breakout legge la racchetta con pesi da 0.1 e la palla con pesi da 100), quindi la massa sottostima i canali letti con pesi piccoli. Ultima colonna: quanti ingressi sono letti da almeno un'unità.

| rete | player | danger | reward | projectile | trail | gauge | ingressi letti |
|---|---|---|---|---|---|---|---|
| astra/breakout | 5% (0%) | 0% (0%) | 0% (0%) | 72% (50%) | 23% (50%) | 0% (0%) | 208/600 |
| astra/seaquest | 25% (24%) | 42% (45%) | 32% (30%) | 0% (0%) | 0% (0%) | 1% (1%) | 245/600 |
| astra/asterix | 20% (6%) | 29% (23%) | 17% (34%) | 0% (0%) | 34% (37%) | 0% (0%) | 216/600 |

| rete | agent | water | shell | jelly | rising | countdown | stun | ingressi letti |
|---|---|---|---|---|---|---|---|---|
| claude/low_tide | 51% (81%) | 10% (3%) | 29% (10%) | 10% (6%) | 0% (0%) | 0% (0%) | 0% (0%) | 311/403 |
| gpt/low_tide | 86% (93%) | 3% (1%) | 5% (2%) | 5% (2%) | 1% (1%) | 0% (0%) | 0% (0%) | 150/403 |

| rete | agent | moving | reward | obstacle | previous | ingressi letti |
|---|---|---|---|---|---|---|
| ppo/breakout-grid | 20% (5%) | 20% (43%) | 20% (6%) | 20% (3%) | 20% (42%) | 500/500 |
| ppo/pong-grid | 20% (8%) | 20% (42%) | 20% (5%) | 20% (6%) | 20% (39%) | 500/500 |
| ppo/pacman-grid | 20% (54%) | 20% (26%) | 20% (6%) | 20% (4%) | 20% (11%) | 500/500 |
| ppo/freeway-grid | 20% (49%) | 20% (18%) | 20% (4%) | 20% (7%) | 20% (22%) | 500/500 |

## 3. Distanza nello spazio dei pesi, strato per strato

Solo tra reti con la stessa forma. Per ogni strato (pesi e bias insieme) e per tutta la rete:

- **distanza relativa** ‖A − B‖ / √(‖A‖² + ‖B‖²): 0 = identici, 1 = scorrelati (come due vettori ortogonali), √2 = opposti; il coseno dice la stessa cosa (1 = stessa direzione, 0 = scorrelati);
- **grezza**: così come sono scritte; **allineata**: dopo aver riordinato (e, per tanh, cambiato di segno) le unità nascoste di B nel modo che la avvicina di più ad A, che non cambia la funzione di B;
- **nullo**: la stessa allineata, ma con B sostituita da B coi valori rimescolati a caso dentro ogni strato (stessi valori, stessa sparsità, nessuna struttura), media su 20 rimescolamenti. Se *allineata* ≈ *nullo*, le due reti non hanno in comune più di quanto ne avrebbero due reti a caso coi loro stessi valori.

Attenzione: nelle reti scritte a mano la norma è dominata da pochi pesi enormi (fino a 1000), quindi ‖A − B‖ di uno strato o di tutta la rete misura soprattutto dove stanno quei pochi pesi; per tanh saturata 100 e 1000 fanno quasi la stessa cosa. Il coseno e le sezioni 4–6 non dipendono dalla scala.

Verifica dell'allineamento: una rete contro una sua copia con le unità rimescolate dà distanza 0.0e+00 (deve essere 0).

**Reti scritte da GPT-6 Astra (600-32-32-6, tanh), giochi diversi**

| coppia | strato | ‖A−B‖ grezza | ‖A‖ / ‖B‖ | rel. grezza | ‖A−B‖ allineata | rel. allineata | coseno allineato | rel. nullo |
|---|---|---|---|---|---|---|---|---|
| astra/breakout vs astra/seaquest | strato 1 | 1678.32 | 1673.63 / 125.19 | 1.00 | 1678.29 | 1.00 | 0.00 | 1.00 ± 0.00 |
|  | strato 2 | 3155.87 | 3165.09 / 74.09 | 1.00 | 3138.54 | 0.99 | 0.37 | 0.99 ± 0.00 |
|  | strato 3 | 679.17 | 231.08 / 476.56 | 1.28 | 679.17 | 1.28 | -0.82 | 1.02 ± 0.14 |
|  | tutta | 3638.34 | 3587.79 / 498.26 | 1.00 | 3623.31 | 1.00 | -0.00 | 0.99 ± 0.00 |
| astra/breakout vs astra/asterix | strato 1 | 1687.02 | 1673.63 / 223.75 | 1.00 | 1675.70 | 0.99 | 0.06 | 1.00 ± 0.00 |
|  | strato 2 | 3172.51 | 3165.09 / 147.40 | 1.00 | 3087.30 | 0.97 | 0.54 | 0.98 ± 0.00 |
|  | strato 3 | 1799.34 | 231.08 / 1739.06 | 1.03 | 1799.34 | 1.03 | -0.20 | 0.98 ± 0.04 |
|  | tutta | 4018.52 | 3587.79 / 1759.58 | 1.01 | 3946.77 | 0.99 | 0.03 | 0.98 ± 0.01 |
| astra/seaquest vs astra/asterix | strato 1 | 251.89 | 125.19 / 223.75 | 0.98 | 202.38 | 0.79 | 0.44 | 0.95 ± 0.00 |
|  | strato 2 | 165.72 | 74.09 / 147.40 | 1.00 | 138.49 | 0.84 | 0.37 | 0.75 ± 0.03 |
|  | strato 3 | 1827.25 | 476.56 / 1739.06 | 1.01 | 1822.82 | 1.01 | -0.04 | 1.08 ± 0.06 |
|  | tutta | 1851.96 | 498.26 / 1759.58 | 1.01 | 1839.25 | 1.01 | -0.02 | 1.08 ± 0.06 |

**Reti addestrate con PPO (500-32-32-5, tanh), giochi diversi**

| coppia | strato | ‖A−B‖ grezza | ‖A‖ / ‖B‖ | rel. grezza | ‖A−B‖ allineata | rel. allineata | coseno allineato | rel. nullo |
|---|---|---|---|---|---|---|---|---|
| ppo/breakout-grid vs ppo/pong-grid | strato 1 | 134.82 | 117.05 / 67.48 | 1.00 | 125.74 | 0.93 | 0.15 | 0.97 ± 0.00 |
|  | strato 2 | 38.29 | 30.33 / 28.00 | 0.93 | 33.21 | 0.80 | 0.35 | 0.80 ± 0.02 |
|  | strato 3 | 21.53 | 18.09 / 9.84 | 1.05 | 17.58 | 0.85 | 0.32 | 0.82 ± 0.02 |
|  | tutta | 141.79 | 122.26 / 73.71 | 0.99 | 131.23 | 0.92 | 0.18 | 0.95 ± 0.00 |
| ppo/breakout-grid vs ppo/pacman-grid | strato 1 | 145.78 | 117.05 / 90.46 | 0.99 | 139.36 | 0.94 | 0.12 | 0.96 ± 0.00 |
|  | strato 2 | 38.20 | 30.33 / 23.09 | 1.00 | 30.54 | 0.80 | 0.37 | 0.84 ± 0.01 |
|  | strato 3 | 22.45 | 18.09 / 15.05 | 0.95 | 17.99 | 0.76 | 0.42 | 0.71 ± 0.04 |
|  | tutta | 152.36 | 122.26 / 94.57 | 0.99 | 143.80 | 0.93 | 0.14 | 0.95 ± 0.00 |
| ppo/breakout-grid vs ppo/freeway-grid | strato 1 | 142.98 | 117.05 / 82.56 | 1.00 | 126.80 | 0.89 | 0.23 | 0.97 ± 0.00 |
|  | strato 2 | 38.11 | 30.33 / 22.00 | 1.02 | 31.12 | 0.83 | 0.33 | 0.85 ± 0.01 |
|  | strato 3 | 22.94 | 18.09 / 14.53 | 0.99 | 16.61 | 0.72 | 0.50 | 0.72 ± 0.03 |
|  | tutta | 149.74 | 122.26 / 86.66 | 1.00 | 131.62 | 0.88 | 0.24 | 0.95 ± 0.00 |
| ppo/pong-grid vs ppo/pacman-grid | strato 1 | 113.11 | 67.48 / 90.46 | 1.00 | 107.20 | 0.95 | 0.10 | 0.96 ± 0.00 |
|  | strato 2 | 36.72 | 28.00 / 23.09 | 1.01 | 28.11 | 0.77 | 0.41 | 0.81 ± 0.02 |
|  | strato 3 | 17.23 | 9.84 / 15.05 | 0.96 | 14.85 | 0.83 | 0.35 | 0.79 ± 0.04 |
|  | tutta | 120.16 | 73.71 / 94.57 | 1.00 | 111.82 | 0.93 | 0.13 | 0.94 ± 0.00 |
| ppo/pong-grid vs ppo/freeway-grid | strato 1 | 106.49 | 67.48 / 82.56 | 1.00 | 102.29 | 0.96 | 0.08 | 0.96 ± 0.00 |
|  | strato 2 | 35.26 | 28.00 / 22.00 | 0.99 | 27.94 | 0.78 | 0.40 | 0.82 ± 0.01 |
|  | strato 3 | 17.27 | 9.84 / 14.53 | 0.98 | 15.50 | 0.88 | 0.24 | 0.81 ± 0.03 |
|  | tutta | 113.50 | 73.71 / 86.66 | 1.00 | 107.17 | 0.94 | 0.11 | 0.94 ± 0.00 |
| ppo/pacman-grid vs ppo/freeway-grid | strato 1 | 122.87 | 90.46 / 82.56 | 1.00 | 107.98 | 0.88 | 0.22 | 0.95 ± 0.00 |
|  | strato 2 | 29.58 | 23.09 / 22.00 | 0.93 | 27.10 | 0.85 | 0.28 | 0.84 ± 0.01 |
|  | strato 3 | 19.59 | 15.05 / 14.53 | 0.94 | 13.58 | 0.65 | 0.58 | 0.68 ± 0.03 |
|  | tutta | 127.88 | 94.57 / 86.66 | 1.00 | 112.16 | 0.87 | 0.24 | 0.94 ± 0.00 |

## 4. Unità del primo strato: hanno un gemello nell'altra rete?

Ogni unità del primo strato è il vettore dei suoi pesi in ingresso. Le unità di A e di B sono accoppiate una a una (algoritmo ungherese) massimizzando il coseno (in valore assoluto per tanh). Nullo: lo stesso con gli ingressi di B rimescolati, media su 20 rimescolamenti.

| coppia | unità attive | coseno medio | mediana | > 0.9 | > 0.5 | nullo: medio | nullo: > 0.5 | le 5 coppie migliori |
|---|---|---|---|---|---|---|---|---|
| astra/breakout vs astra/seaquest | 6 / 32 | 0.00 | 0.00 | 0 | 0 | 0.10 | 0.0 | 0.00, 0.00, 0.00, 0.00, 0.00 |
| astra/breakout vs astra/asterix | 6 / 32 | 0.05 | 0.00 | 0 | 0 | 0.09 | 0.0 | 0.19, 0.12, 0.00, 0.00, 0.00 |
| astra/seaquest vs astra/asterix | 32 / 32 | 0.35 | 0.16 | 0 | 15 | 0.10 | 0.0 | 0.75, 0.75, 0.74, 0.74, 0.73 |
| claude/low_tide vs gpt/low_tide | 46 / 43 | 0.20 | 0.08 | 3 | 7 | 0.12 | 0.9 | 1.00, 1.00, 0.91, 0.82, 0.68 |
| ppo/breakout-grid vs ppo/pong-grid | 32 / 32 | 0.17 | 0.18 | 0 | 0 | 0.08 | 0.0 | 0.37, 0.31, 0.23, 0.23, 0.22 |
| ppo/breakout-grid vs ppo/pacman-grid | 32 / 32 | 0.12 | 0.11 | 0 | 0 | 0.09 | 0.0 | 0.31, 0.21, 0.20, 0.19, 0.17 |
| ppo/breakout-grid vs ppo/freeway-grid | 32 / 32 | 0.21 | 0.21 | 0 | 0 | 0.10 | 0.0 | 0.42, 0.41, 0.37, 0.36, 0.35 |
| ppo/pong-grid vs ppo/pacman-grid | 32 / 32 | 0.11 | 0.10 | 0 | 0 | 0.09 | 0.0 | 0.24, 0.22, 0.18, 0.17, 0.15 |
| ppo/pong-grid vs ppo/freeway-grid | 32 / 32 | 0.09 | 0.09 | 0 | 0 | 0.09 | 0.0 | 0.17, 0.16, 0.14, 0.12, 0.12 |
| ppo/pacman-grid vs ppo/freeway-grid | 32 / 32 | 0.22 | 0.21 | 0 | 0 | 0.09 | 0.0 | 0.48, 0.33, 0.32, 0.30, 0.30 |

Le coppie con coseno > 0.9 (unità quasi identiche):

- claude/low_tide unità 1 (1 ingressi: rising) ≈ gpt/low_tide unità 42 (1 ingressi: rising): coseno 1.00
- claude/low_tide unità 35 (10 ingressi: agent) ≈ gpt/low_tide unità 34 (10 ingressi: agent): coseno 1.00
- claude/low_tide unità 18 (19 ingressi: agent, shell) ≈ gpt/low_tide unità 30 (10 ingressi: agent): coseno 0.91

## 5. Rappresentazioni (CKA lineare) sugli stessi stati

CKA tra le uscite di due strati sugli stessi stati: 1 = la stessa rappresentazione a meno di rotazioni e scala, 0 = nulla in comune. *Rete casuale*: CKA tra lo strato di A e lo stesso strato di una rete con la stessa forma e pesi gaussiani (media su 5): quanto viene "gratis" dal fatto di leggere gli stessi ingressi.

**claude/low_tide (righe) vs gpt/low_tide (colonne)**, 6000 stati: Low Tide, giocato da euristica, claude e gpt (in parti uguali)

| strato | gpt/low_tide 1 | gpt/low_tide 2 | gpt/low_tide 3 | gpt/low_tide 4 | A vs rete casuale |
|---|---|---|---|---|---|
| claude/low_tide 1 | 0.04 | 0.11 | 0.11 | 0.00 | 0.07 |
| claude/low_tide 2 | 0.54 | 0.17 | 0.21 | 0.56 | 0.09 |
| claude/low_tide 3 | 0.51 | 0.08 | 0.10 | 0.75 | 0.03 |

**astra/breakout (righe) vs astra/seaquest (colonne)**, 5219 stati: i tre giochi MinAtar, giocati dalla loro rete e a caso

| strato | astra/seaquest 1 | astra/seaquest 2 | astra/seaquest 3 | A vs rete casuale |
|---|---|---|---|---|
| astra/breakout 1 | 0.19 | 0.04 | 0.08 | 0.22 |
| astra/breakout 2 | 0.07 | 0.11 | 0.42 | 0.24 |
| astra/breakout 3 | 0.05 | 0.07 | 0.27 | 0.10 |

**astra/breakout (righe) vs astra/asterix (colonne)**, 5219 stati: i tre giochi MinAtar, giocati dalla loro rete e a caso

| strato | astra/asterix 1 | astra/asterix 2 | astra/asterix 3 | A vs rete casuale |
|---|---|---|---|---|
| astra/breakout 1 | 0.22 | 0.22 | 0.06 | 0.20 |
| astra/breakout 2 | 0.06 | 0.06 | 0.03 | 0.22 |
| astra/breakout 3 | 0.05 | 0.05 | 0.02 | 0.10 |

**astra/seaquest (righe) vs astra/asterix (colonne)**, 5219 stati: i tre giochi MinAtar, giocati dalla loro rete e a caso

| strato | astra/asterix 1 | astra/asterix 2 | astra/asterix 3 | A vs rete casuale |
|---|---|---|---|---|
| astra/seaquest 1 | 0.53 | 0.55 | 0.10 | 0.39 |
| astra/seaquest 2 | 0.05 | 0.05 | 0.01 | 0.16 |
| astra/seaquest 3 | 0.05 | 0.05 | 0.02 | 0.18 |

## 6. Comportamento

**Stessa azione sugli stessi stati.** Probabilità che le due policy scelgano la stessa azione campionando (e il caso: due scelte indipendenti con le stesse frequenze di azioni), e scegliendo l'argmax (pareggi divisi in parti uguali); TV = distanza in variazione totale media tra le due distribuzioni (0 = identiche, 1 = disgiunte).

| stati | coppia | stessa azione (campionata) | caso | stessa azione (argmax) | caso | TV |
|---|---|---|---|---|---|---|
| Low Tide | claude/low_tide vs gpt/low_tide | 0.50 | 0.23 | 0.58 | 0.29 | 0.47 |
| Low Tide | claude/low_tide vs euristica | 0.43 | 0.19 | 0.41 | 0.18 | 0.57 |
| Low Tide | gpt/low_tide vs euristica | 0.35 | 0.18 | 0.34 | 0.18 | 0.65 |
| breakout | astra/breakout vs astra/seaquest | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| breakout | astra/breakout vs astra/asterix | 0.27 | 0.27 | 0.45 | 0.45 | 0.73 |
| seaquest | astra/seaquest vs astra/breakout | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| seaquest | astra/seaquest vs astra/asterix | 0.14 | 0.10 | 0.11 | 0.07 | 0.85 |
| asterix | astra/asterix vs astra/breakout | 0.69 | 0.69 | 0.83 | 0.83 | 0.31 |
| asterix | astra/asterix vs astra/seaquest | 0.05 | 0.04 | 0.01 | 0.01 | 0.89 |

**Cross-play** (200 episodi per casella, protocollo dei giochi: sticky 0.1, azioni campionate). Righe: chi gioca; colonne: il gioco. Se "conoscendone una si risolvono tutte", fuori dalla diagonale i punteggi dovrebbero restare alti.

| rete \ gioco | breakout | seaquest | asterix |
|---|---|---|---|
| astra/breakout | 512.7 ± 16.9 | 0.0 ± 0.0 | 0.6 ± 0.1 |
| astra/seaquest | 0.5 ± 0.0 | 48.3 ± 2.9 | 0.6 ± 0.1 |
| astra/asterix | 0.5 ± 0.0 | 0.0 ± 0.0 | 54.5 ± 2.5 |
| random | 0.5 ± 0.1 | 0.0 ± 0.0 | 0.5 ± 0.1 |

## 7. Le reti di Astra per i quattro giochi, anche con forme diverse

Le tre reti MinAtar sono 600-32-32-6; quella di Low Tide (`gpt/low_tide`) è 403-43-61-53-5, con un strato nascosto in più e le azioni in un altro ordine. Per confrontarle, ogni rete è riscritta, *senza cambiare quello che calcola*, in uno spazio comune:

- ingressi: griglia 10×10 con 7 canali (i 6 di MinAtar più l'acqua di Low Tide) e i 3 scalari di Low Tide; in Low Tide agente → player, meduse → danger, conchiglie → reward. Una rete ha peso 0 sugli ingressi che non ha;
- azioni: fermo, sinistra, su, destra, giù, fuoco (Low Tide non ha fuoco: riga di zeri);
- strati nascosti: allargati con unità a zero quando servono (un'unità senza pesi non cambia niente).

Poi:

- **primo strato** e **uscita**: ogni strato è allineato da solo, riordinando (e cambiando di segno, per tanh) le unità nascoste; è l'allineamento più favorevole possibile per quello strato, calcolato in modo esatto;
- **strato intermedio, reti con lo stesso numero di strati nascosti** (le tre MinAtar e `claude/low_tide`): un unico allineamento per tutta la rete, come nella sezione 3 (esatto);
- **strato intermedio con `gpt/low_tide`**, che ha due strati intermedi: ognuno contro quello dell'altra rete, riordinando righe e colonne insieme. Questo problema non si risolve in modo esatto: il coseno trovato è un limite inferiore, e accanto c'è il **limite superiore** di von Neumann (nessun riordino può superarlo). Si tiene la coppia di strati più simile.

Coseno 1 = stessi pesi a meno di un fattore di scala, 0 = niente in comune; *nullo* = lo stesso allineamento con i valori di B rimescolati a caso (media ± dev. std. su 20). `claude/low_tide` è incluso come riferimento: stesso gioco di `gpt/low_tide`, autore diverso.

Verifiche: ogni rete contro una sua copia con le unità rimescolate dà coseno minimo 1.000000 con i metodi esatti (deve essere 1); il metodo approssimato ritrova gli strati intermedi di `gpt/low_tide` rimescolati con coseno minimo 0.82.

**Coseno allineato per strato** (tra parentesi il nullo; "≤" = limite superiore).

| coppia | primo strato | intermedio | strati intermedi confrontati | uscita | tutta la rete |
|---|---|---|---|---|---|
| astra/breakout vs astra/seaquest | 0.00 (0.02) | 0.37 (0.40) | 2 / 2 | -0.82 (-0.11) | -0.00 (0.06) |
| astra/breakout vs astra/asterix | 0.06 (0.02) | 0.57 (0.46) | 2 / 2 | -0.20 (0.21) | 0.03 (0.04) |
| gpt/low_tide vs astra/breakout | 0.00 (0.06) | 0.40 (0.43), ≤ 0.94 | 3 / 2 | 0.02 (0.14) | - |
| claude/low_tide vs astra/breakout | 0.00 (0.09) | 0.33 (0.45) | 2 / 2 | -0.10 (0.12) | 0.28 (0.39) |
| astra/seaquest vs astra/asterix | 0.44 (0.12) | 0.37 (0.50) | 2 / 2 | -0.04 (-0.22) | -0.02 (-0.22) |
| gpt/low_tide vs astra/seaquest | 0.14 (0.06) | 0.80 (0.78), ≤ 0.92 | 2 / 2 | 0.12 (0.10) | - |
| claude/low_tide vs astra/seaquest | 0.30 (0.14) | 0.72 (0.68) | 2 / 2 | -0.03 (-0.15) | 0.11 (0.10) |
| gpt/low_tide vs astra/asterix | 0.03 (0.06) | 0.70 (0.55), ≤ 0.97 | 3 / 2 | 0.07 (0.04) | - |
| claude/low_tide vs astra/asterix | 0.11 (0.13) | 0.62 (0.60) | 2 / 2 | 0.58 (0.37) | 0.06 (0.05) |
| gpt/low_tide vs claude/low_tide | 0.26 (0.05) | 0.59 (0.70), ≤ 0.89 | 2 / 2 | 0.12 (0.11) | - |

**Primo strato**

| coppia | strati | ‖A‖ / ‖B‖ | ‖A−B‖ grezza | ‖A−B‖ allineata | rel. allineata | coseno grezzo | coseno allineato | coseno nullo |
|---|---|---|---|---|---|---|---|---|
| astra/breakout vs astra/seaquest | 1 / 1 | 1673.63 / 125.19 | 1678.32 | 1678.29 | 1.00 | -0.00 | 0.00 | 0.02 ± 0.00 |
| astra/breakout vs astra/asterix | 1 / 1 | 1673.63 / 223.75 | 1687.02 | 1675.70 | 0.99 | 0.01 | 0.06 | 0.02 ± 0.00 |
| gpt/low_tide vs astra/breakout | 1 / 1 | 64.07 / 1673.63 | 1674.85 | 1674.84 | 1.00 | 0.00 | 0.00 | 0.06 ± 0.01 |
| claude/low_tide vs astra/breakout | 1 / 1 | 11771.87 / 1673.63 | 11890.36 | 11889.69 | 1.00 | -0.00 | 0.00 | 0.09 ± 0.01 |
| astra/seaquest vs astra/asterix | 1 / 1 | 125.19 / 223.75 | 251.89 | 202.31 | 0.79 | 0.04 | 0.44 | 0.12 ± 0.01 |
| gpt/low_tide vs astra/seaquest | 1 / 1 | 64.07 / 125.19 | 141.27 | 132.29 | 0.94 | -0.01 | 0.14 | 0.06 ± 0.00 |
| claude/low_tide vs astra/seaquest | 1 / 1 | 11771.87 / 125.19 | 11766.71 | 11734.44 | 1.00 | 0.05 | 0.30 | 0.14 ± 0.01 |
| gpt/low_tide vs astra/asterix | 1 / 1 | 64.07 / 223.75 | 233.19 | 230.68 | 0.99 | -0.01 | 0.03 | 0.06 ± 0.01 |
| claude/low_tide vs astra/asterix | 1 / 1 | 11771.87 / 223.75 | 11761.36 | 11749.48 | 1.00 | 0.06 | 0.11 | 0.13 ± 0.01 |
| gpt/low_tide vs claude/low_tide | 1 / 1 | 64.07 / 11771.87 | 11764.72 | 11755.42 | 1.00 | 0.11 | 0.26 | 0.05 ± 0.01 |

**Strato intermedio**

| coppia | strati | ‖A‖ / ‖B‖ | ‖A−B‖ grezza | ‖A−B‖ allineata | rel. allineata | coseno grezzo | coseno allineato | coseno nullo |
|---|---|---|---|---|---|---|---|---|
| astra/breakout vs astra/seaquest | 2 / 2 | 3165.09 / 74.09 | 3155.87 | 3138.53 | 0.99 | 0.14 | 0.37 | 0.40 ± 0.01 |
| astra/breakout vs astra/asterix | 2 / 2 | 3165.09 / 147.40 | 3172.51 | 3084.17 | 0.97 | -0.03 | 0.57 | 0.46 ± 0.03 |
| gpt/low_tide vs astra/breakout | 3 / 2 | 607.33 / 3165.09 | 3173.63 | 2972.42 | 0.92 | 0.08 | 0.40 | 0.43 ± 0.01 |
| claude/low_tide vs astra/breakout | 2 / 2 | 44074.96 / 3165.09 | 44200.60 | 43125.83 | 0.98 | -0.00 | 0.33 | 0.45 ± 0.01 |
| astra/seaquest vs astra/asterix | 2 / 2 | 74.09 / 147.40 | 165.72 | 138.49 | 0.84 | -0.01 | 0.37 | 0.50 ± 0.05 |
| gpt/low_tide vs astra/seaquest | 2 / 2 | 416.52 / 74.09 | 384.17 | 360.36 | 0.85 | 0.51 | 0.80 | 0.78 ± 0.01 |
| claude/low_tide vs astra/seaquest | 2 / 2 | 44074.96 / 74.09 | 44055.66 | 44021.88 | 1.00 | 0.26 | 0.72 | 0.68 ± 0.02 |
| gpt/low_tide vs astra/asterix | 3 / 2 | 607.33 / 147.40 | 621.78 | 514.20 | 0.82 | 0.02 | 0.70 | 0.55 ± 0.02 |
| claude/low_tide vs astra/asterix | 2 / 2 | 44074.96 / 147.40 | 44070.73 | 43983.08 | 1.00 | 0.03 | 0.62 | 0.60 ± 0.02 |
| gpt/low_tide vs claude/low_tide | 2 / 2 | 416.52 / 44074.96 | 43947.96 | 43830.58 | 0.99 | 0.31 | 0.59 | 0.70 ± 0.01 |

**Strato di uscita**

| coppia | strati | ‖A‖ / ‖B‖ | ‖A−B‖ grezza | ‖A−B‖ allineata | rel. allineata | coseno grezzo | coseno allineato | coseno nullo |
|---|---|---|---|---|---|---|---|---|
| astra/breakout vs astra/seaquest | 3 / 3 | 231.08 / 476.56 | 679.17 | 679.17 | 1.28 | -0.82 | -0.82 | -0.11 ± 0.31 |
| astra/breakout vs astra/asterix | 3 / 3 | 231.08 / 1739.06 | 1799.34 | 1799.34 | 1.03 | -0.20 | -0.20 | 0.21 ± 0.27 |
| gpt/low_tide vs astra/breakout | 4 / 3 | 1419.46 / 231.08 | 1438.15 | 1433.97 | 1.00 | -0.00 | 0.02 | 0.14 ± 0.05 |
| claude/low_tide vs astra/breakout | 3 / 3 | 256.66 / 231.08 | 370.10 | 361.37 | 1.05 | -0.15 | -0.10 | 0.12 ± 0.24 |
| astra/seaquest vs astra/asterix | 3 / 3 | 476.56 / 1739.06 | 1827.25 | 1822.71 | 1.01 | -0.05 | -0.04 | -0.22 ± 0.20 |
| gpt/low_tide vs astra/seaquest | 4 / 3 | 1419.46 / 476.56 | 1496.73 | 1442.56 | 0.96 | 0.00 | 0.12 | 0.10 ± 0.01 |
| claude/low_tide vs astra/seaquest | 3 / 3 | 256.66 / 476.56 | 555.86 | 547.25 | 1.01 | -0.07 | -0.03 | -0.15 ± 0.30 |
| gpt/low_tide vs astra/asterix | 4 / 3 | 1419.46 / 1739.06 | 2245.22 | 2171.73 | 0.97 | -0.00 | 0.07 | 0.04 ± 0.02 |
| claude/low_tide vs astra/asterix | 3 / 3 | 256.66 / 1739.06 | 1606.57 | 1603.91 | 0.91 | 0.57 | 0.58 | 0.37 ± 0.18 |
| gpt/low_tide vs claude/low_tide | 4 / 3 | 1419.46 / 256.66 | 1442.63 | 1411.29 | 0.98 | -0.00 | 0.12 | 0.11 ± 0.01 |

