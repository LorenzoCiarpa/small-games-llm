# PPO lungo l'addestramento: i pesi diventano più sparsi o più leggibili?

Le analisi di `experiments/<gioco>/ppo/` guardano una sola rete PPO per gioco, alla fine dell'addestramento, e la
trovano densa e illeggibile. Potrebbe dipendere da quando ci si ferma (forse con più passi la rete si "pulisce") o
dal coefficiente di entropia scelto. Qui le misure principali, in versione leggera, girano su tutte le corse PPO e
sui loro checkpoint:

- **giochi MinAtar:** le tre corse da 200M passi con entropia 0.01, 0.05 e 0.1, a 5, 10, 25, 50, 100, 150 e 200M
  passi;
- **Low Tide:** la corsa da 100M passi, a 5, 10, 25, 50, 75 e 100M;
- accanto, le reti scritte a mano dello stesso gioco.

**In breve: no.** In tutte le corse e a ogni checkpoint i pesi PPO restano lontani da quelli scritti a mano, e la
distanza non si accorcia mentre la rete impara.

- **Concentrazione:** il 10% dei pesi più grandi porta il 30-40% della somma dei |w|, dal primo all'ultimo
  checkpoint. Nelle reti scritte a mano il 100%; in pesi gaussiani a caso il 26%.
- **Ingressi per neurone:** ogni neurone del primo strato usa il 53-63% degli ingressi vivi per fare il 90% del suo
  peso, con un lento calo durante l'addestramento. Le reti scritte a mano l'1-3% (35% per Breakout, dove ogni
  neurone è una tabella su tutta la griglia).
- **Pruning:** più la rete impara, meno lo regge. Con il 25% dei pesi vivi del primo strato i primi checkpoint
  tengono gran parte del loro punteggio (che è basso); gli ultimi ne tengono l'1-36%. Le reti scritte a mano il 100%.
- **Quantità dell'algoritmo scritto a mano:** il miglior |r| di un neurone, in media sulle quantità, resta piatto
  (0.10-0.53) e sempre sotto quello delle reti a mano dello stesso gioco (0.47-0.97).
- **Regola:** l'accordo di un albero di profondità 6 resta tra il 29 e il 79%, senza tendenza; le reti a mano
  87-98%.
- **Entropia:** cambia quanto e quando PPO impara, non la forma dei pesi. Le tre corse di ogni gioco hanno curve
  vicine in tutte le misure tranne il punteggio e il pruning, che dipende da quanto la rete ha imparato.

L'unica eccezione apparente è Seaquest con entropia 0.01: regge il pruning al 100% per tutto l'addestramento, ma
solo perché fa 4.4 punti dall'inizio alla fine e non ha niente da perdere.

```bash
uv run --group analysis python experiments/ppo_over_training/analyze.py               # ~5 min con 8 processi
uv run --group analysis python experiments/ppo_over_training/analyze.py --plot-only   # solo figura e tabelle
```

Prima va eseguito `analyze.py` di ogni gioco: da lì vengono gli ingressi vivi.

## Le misure

Il primo strato è guardato solo sugli **ingressi vivi** del gioco, quelli diversi da zero in almeno uno stato
giocato dalle reti analizzate in `experiments/<gioco>/ppo/` (220 su 600 in Breakout, 425 in Seaquest, 315 in
Asterix, 312 su 403 in Low Tide). Sugli altri PPO non riceve gradiente e lascia i pesi dell'inizializzazione, che
gonfierebbero le misure di densità senza dire niente su cosa ha imparato.

| misura | cosa dice | per una rete scritta a mano |
|---|---|---|
| punteggio | media su 100 episodi del protocollo, campionata | |
| massa nel 10% dei pesi | quota della somma dei \|w\| (tutti gli strati) che sta nel 10% dei pesi più grandi | 100%: i pesi diversi da zero sono meno del 10% |
| ingressi vivi per il 90% | mediana, sui neuroni del primo strato, della frazione di ingressi vivi che fa il 90% del loro \|w\| | pochi |
| primo strato al 25% | punteggio tenendo il 25% dei pesi vivi più grandi del primo strato, diviso per il punteggio | 100% |
| quantità chiave: miglior \|r\| | media, sulle quantità dell'algoritmo scritto a mano (`KEY_QUANTITIES` in `analysis.py`), della correlazione più alta con un singolo neurone | alto |
| albero di profondità 6 | accordo con la mossa più probabile della rete, sul 30% degli episodi non usati per costruirlo | alto |

Le quantità e gli alberi usano 10 episodi giocati dalla rete; i valori sono un po' più rumorosi di quelli delle
analisi complete (40 episodi), ma coincidono con quelli entro pochi punti.

## Risultati

![le misure lungo l'addestramento](results/sweep.png)

Ogni colonna è un gioco, ogni riga una misura. Le linee colorate sono le corse PPO (il colore è il coefficiente di
entropia), le linee grigie tratteggiate le reti scritte a mano.

Il primo e l'ultimo checkpoint di ogni corsa (tutti i checkpoint in `results/sweep.md`):

| gioco | rete | punteggio | massa nel 10% | ingressi vivi per il 90% | primo strato al 25% | quantità chiave: \|r\| | albero prof. 6 |
|---|---|---|---|---|---|---|---|
| low_tide | `claude` (a mano) | 133.0 | 100% | 3% | 100% | 0.97 | 98% |
| low_tide | `gpt` (a mano) | 81.9 | 100% | 1% | 100% | 0.66 | 91% |
| low_tide | `ppo_100m` 5M | 77.4 | 32% | 61% | 75% | 0.45 | 66% |
| low_tide | `ppo_100m` 100M | 151.8 | 36% | 55% | 3% | 0.47 | 62% |
| breakout | `paper_astra` (a mano) | 518.5 | 100% | 35% | 100% | 0.73 | 89% |
| breakout | `ppo_ent01` 5M | 6.4 | 41% | 61% | 43% | 0.30 | 65% |
| breakout | `ppo_ent01` 200M | 35.3 | 29% | 62% | 4% | 0.32 | 66% |
| breakout | `ppo_ent05` 5M | 11.1 | 39% | 61% | 49% | 0.38 | 61% |
| breakout | `ppo_ent05` 200M | 61.8 | 30% | 62% | 1% | 0.34 | 71% |
| breakout | `ppo_ent10` 5M | 12.4 | 35% | 63% | 20% | 0.50 | 78% |
| breakout | `ppo_ent10` 200M | 51.3 | 31% | 60% | 2% | 0.43 | 73% |
| seaquest | `paper_astra` (a mano) | 45.7 | 100% | 2% | 100% | 0.47 | 87% |
| seaquest | `ppo_ent01` 5M | 4.3 | 36% | 62% | 99% | 0.22 | 61% |
| seaquest | `ppo_ent01` 200M | 4.5 | 37% | 59% | 98% | 0.18 | 42% |
| seaquest | `ppo_ent05` 5M | 3.6 | 35% | 62% | 108% | 0.25 | 59% |
| seaquest | `ppo_ent05` 200M | 59.6 | 36% | 56% | 2% | 0.22 | 65% |
| seaquest | `ppo_ent10` 5M | 2.7 | 33% | 62% | 89% | 0.20 | 45% |
| seaquest | `ppo_ent10` 200M | 32.6 | 37% | 56% | 32% | 0.26 | 57% |
| asterix | `paper_astra` (a mano) | 55.0 | 100% | 3% | 100% | 0.56 | 94% |
| asterix | `ppo_ent01` 5M | 3.9 | 36% | 61% | 78% | 0.18 | 45% |
| asterix | `ppo_ent01` 200M | 21.1 | 38% | 53% | 32% | 0.12 | 44% |
| asterix | `ppo_ent05` 5M | 2.2 | 34% | 62% | 88% | 0.17 | 35% |
| asterix | `ppo_ent05` 200M | 26.7 | 40% | 53% | 29% | 0.11 | 49% |
| asterix | `ppo_ent10` 5M | 1.6 | 34% | 62% | 83% | 0.18 | 40% |
| asterix | `ppo_ent10` 200M | 14.6 | 41% | 54% | 36% | 0.10 | 45% |

(Un valore sopra il 100% nel pruning è rumore: con 100 episodi la rete potata può fare un po' meglio dell'intera.)

## Cosa se ne ricava

PPO, con questa architettura e senza nessuna pressione verso pesi piccoli o pochi, trova sempre soluzioni dense:
la discesa del gradiente muove ogni peso un po', e niente spinge quelli inutili a zero. Durante l'addestramento la
rete non converge verso una forma sparsa; al contrario, man mano che migliora usa più pesi, e il pruning le costa
di più. Le reti scritte a mano sono sparse perché chi le scrive parte da un algoritmo e lo traduce in poche
connessioni; PPO non parte da un algoritmo, e non ne trova uno lungo la strada.

Una rete PPO sparsa si potrebbe cercare esplicitamente, per esempio con una penalità L1 sui pesi o con pruning e
riaddestramento. Sarebbe un altro esperimento: qui la domanda era che forma ha la soluzione che PPO trova da solo.

## File

- `analyze.py`: le misure su tutti i checkpoint, la figura e le tabelle.
- `results/sweep.json`: tutti i numeri.
- `results/sweep.md`: le tabelle, tutti i checkpoint.
- `results/sweep.png`: la figura.
