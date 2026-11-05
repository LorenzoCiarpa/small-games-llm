# PPO su Asterix: che pesi trova, e si leggono?

`train.py` addestra con PPO una rete 600-32-32-6 tanh, la stessa architettura della rete che GPT-6 Astra ha scritto
a mano nell'articolo (`paper_astra.json`), e la salva in `weights/asterix/ppo.json`. `analyze.py` fa le stesse
misure sulle due reti. La rete di Astra è decodificata neurone per neurone in
[`decode_astra`](../decode_astra/README.md) e fa da termine di paragone. Le misure sono quelle di
[PPO su Low Tide](../../low_tide/ppo/README.md) e di [PPO su Breakout](../../breakout/ppo/README.md), spiegate lì.

**In breve.**

- **Come gioca:** 24.6 punti, meno della metà dei 54.8 di Astra (giocando sempre la mossa più probabile, 46 contro
  65). La curva sale ancora dopo 200M passi.
- **Gioca in un altro modo:** Astra resta fermo nella colonna 5 e sceglie la riga; PPO si muove tra le colonne 3 e 6
  e usa tutte le mosse, ognuna circa un quinto delle volte.
- **Come sono i pesi:** densi e distribuiti.
  - Usa tutti i 315 ingressi vivi, con 10.029 valori diversi; Astra il 4.6%, con 33 valori.
  - Tenendo un quarto dei pesi vivi del primo strato perde tre quarti del punteggio; Astra regge con il 5%.
  - Nessun neurone calcola "c'è un nemico in arrivo nella riga sopra / sotto / mia", gli allarmi su cui si regge
    Astra: |r| 0.08-0.14 contro 0.64-0.67.
  - Un albero di profondità 6 lo imita solo nel 49% degli stati (Astra: 92%).
- **Cosa si legge:** metà del peso del primo strato è sulla **posizione del giocatore**, con mappe lisce; un po' sui
  tesori vicino alle colonne centrali; quasi niente sui nemici e sulla loro direzione. La rete sa bene dove è, e
  poco di cosa sta arrivando.

```bash
uv run --group ppo python experiments/asterix/ppo/train.py          # ~1h50 su un core (200M passi)
uv run --group analysis python experiments/asterix/ppo/analyze.py   # ~10 min con 4 processi
```

## 1. L'addestramento

Come per Breakout (`src/small_games/ppo.py`, PPO con clipping e GAE): attore 600-32-32-6 tanh, critico 600-64-64-1,
64 ambienti, rollout da 128 passi, Adam lr 5·10⁻⁴ che scende a 0, sticky actions 0.1 come nella valutazione, seed
di addestramento lontani da quelli della valutazione. Il minimal action set di MinAtar per Asterix: tutte le azioni
tranne fuoco, che qui non fa niente. **Coefficiente di entropia 0.05**, 200 milioni di passi.

Il primo tentativo (`ppo_v1`: 6 azioni, entropia 0.01, 30M passi) faceva 9.1 punti ed era ancora in salita. Le tre
corse di `experiments/ppo_runs.py` differiscono solo per il coefficiente di entropia:

| corsa | supera 12 punti | ritorno in addestramento a 50M / 100M / 200M | valutazione (1000 episodi) |
|---|---|---|---|
| `ppo_ent01` (0.01) | a 68M passi | 8.6 / 14 / 20 | 19.4 ± 0.4 |
| **`ppo_ent05` (0.05)** | **a 52M passi** | 11 / 17 / 25 | **24.6 ± 0.5** |
| `ppo_ent10` (0.1) | a 65M passi | 8.8 / 12 / 16 | 15.5 ± 0.4 |
| rete di Astra | | | 54.8 ± 1.1 |

`weights/asterix/ppo.json` è `ppo_ent05`. Qui non c'è un plateau da cui uscire, ma una salita lenta e regolare: con
più passi la rete migliorerebbe ancora.

## 2. Come gioca

| | `paper_astra` | `ppo_v1` | `ppo` |
|---|---|---|---|
| punteggio (200 episodi, campionata) | 54.5 ± 2.5 | 9.0 ± 0.5 | 26.8 ± 1.2 |
| punteggio greedy | 65.1 | 10.2 | 45.8 |

Sugli stati giocati dalle due reti (20 episodi, mosse più probabili):

| | `paper_astra` | `ppo` |
|---|---|---|
| colonna del giocatore | sempre la 5 | 3: 13%, 4: 30%, 5: 32%, 6: 16%, altre 9% |
| mosse: fermo, sinistra, su, destra, giù | 79, 0, 11, 0, 10% | 20, 18, 21, 18, 22% |

Astra trasforma Asterix in un gioco a una dimensione: tutti i nemici e i tesori passano dalla colonna 5, quindi
basta stare lì e scegliere la riga. PPO resta vicino al centro ma non ha trovato questa semplificazione.

## 3. Com'è fatto un peso PPO

| | `paper_astra` | `ppo_v1` | `ppo` |
|---|---|---|---|
| pesi diversi da zero | 2.8% | 100.0% | 99.8% |
| valori diversi nel primo strato | 33 | 18.609 | 18.757 |
| ingressi che fanno il 90% del peso di un neurone (mediana, su 600) | 10 | 345 | 282 |
| peso più grande in valore assoluto | 80.00 | 1.75 | 3.95 |

(Lo 0.2% di zeri di `ppo` è la riga di uscita di fuoco, esclusa.)

285 ingressi su 600 sono sempre zero: i canali projectile e gauge, la riga 0 e la riga 9 degli altri, e qualche
cella del giocatore. I pesi PPO su questi restano quelli dell'inizializzazione casuale (vedi Breakout). Ristretti agli
ingressi vivi:

| primo strato, solo i 315 ingressi vivi su 600 | `paper_astra` | `ppo_v1` | `ppo` |
|---|---|---|---|
| pesi diversi da zero | 4.6% | 100.0% | 100.0% |
| valori diversi | 33 | 9.961 | 10.029 |
| ingressi che fanno il 90% del peso di un neurone (mediana, su 315) | 10 | 183 | 167 |
| \|w\| medio sugli ingressi morti / vivi | 0.000 / 0.374 | 0.050 / 0.132 | 0.054 / 0.293 |

Quanto pesa ogni canale nel primo strato (quota della somma dei |w| sugli ingressi vivi):

| | giocatore | nemici (danger) | tesori (reward) | scia (trail) |
|---|---|---|---|---|
| `paper_astra` | 6% | 23% | 34% | 37% |
| `ppo` | 49% | 14% | 28% | 9% |

Astra usa molto la scia, perché è da lì che capisce se un nemico sta arrivando o si allontana (sezione 4.1 di
`decode_astra`). PPO mette metà del peso sulla propria posizione.

**Rete di Astra:** ogni neurone guarda una riga: "c'è un nemico in arrivo nella riga k, sopra / sotto / nella mia?",
"c'è un tesoro in arrivo nella riga k?".

![primo strato di paper_astra.json](results/analysis/fields_paper_astra.png)

**Rete PPO:**

![primo strato di ppo.json](results/analysis/fields_ppo.png)

Nel canale del giocatore ci sono mappe lisce e larghe: u17 è positivo in basso al centro, u10 negativo a destra, u31
positivo in basso a sinistra e negativo in alto a destra. Sono una codifica della posizione per zone. Nel canale
reward diversi neuroni hanno una striscia verticale sulle colonne 4-5, cioè "un tesoro vicino al centro". I canali
danger e trail sono quasi piatti.

## 4. Quanti pesi servono (pruning)

Tutti gli strati (200 episodi):

| pesi tenuti | 100% | 50% | 25% | 10% | 5% | 2% |
|---|---|---|---|---|---|---|
| `paper_astra` | 54.5 | 54.5 | 54.5 | 54.5 | 30.2 | 0.5 |
| `ppo_v1` | 9.0 | 9.1 | 5.6 | 1.6 | 0.7 | 0.6 |
| `ppo` | 26.8 | 17.4 | 0.6 | 0.4 | 0.5 | 0.5 |

Solo il primo strato, sugli ingressi vivi:

| pesi vivi del primo strato tenuti | 100% | 50% | 25% | 10% | 5% |
|---|---|---|---|---|---|
| `paper_astra` | 54.5 | 54.5 | 54.5 | 54.5 | 54.5 |
| `ppo_v1` | 9.0 | 8.7 | 6.0 | 1.8 | 1.2 |
| `ppo` | 26.8 | 19.6 | 7.3 | 1.6 | 0.9 |

## 5. Quanto conta ogni neurone (ablazione)

| | neuroni (costanti) | che fanno perdere più del 10% | più del 50% | perdita massima | riferimento |
|---|---|---|---|---|---|
| `paper_astra` | 64 (0) | 41 | 14 | 50.7 | 55.0 ± 3.5 |
| `ppo_v1` | 64 (0) | 8 | 0 | 1.7 | 9.0 ± 0.7 |
| `ppo` | 64 (0) | 62 | 2 | 15.7 | 26.7 ± 1.7 |

- **Astra:** 14 neuroni da soli tolgono più di metà del punteggio. I due peggiori sono gli allarmi del secondo
  strato per "su" e "giù" (|r| 0.82 con l'arrivo di un nemico due righe sopra o sotto).
- **PPO:** quasi tutti i neuroni contano un po' (62 su 64 più del 10%), quasi nessuno tanto. I più importanti sono
  correlati con la posizione del giocatore (|r| fino a 0.68 con riga o colonna).

## 6. Cosa calcolano i neuroni

### Le quantità dell'algoritmo di Astra

| quantità | `paper_astra` r / R² | `ppo_v1` r / R² | `ppo` r / R² |
|---|---|---|---|
| nemico in arrivo nella mia riga (≤ 3) | **0.67** / 0.45 | 0.10 / 0.04 | 0.08 / 0.06 |
| nemico in arrivo nella riga sotto (≤ 2) | **0.64** / 0.43 | 0.11 / 0.05 | 0.14 / 0.16 |
| nemico in arrivo nella riga sopra (≤ 2) | **0.65** / 0.48 | 0.12 / 0.06 | 0.13 / 0.11 |
| tesoro in arrivo nella mia riga | 0.30 / **1.00** | 0.15 / 0.08 | 0.10 / 0.16 |

Anche in Astra r non arriva a 1, perché ogni allarme è diviso tra otto neuroni, uno per riga (sezione 4 di
`decode_astra`); ma lo strato li contiene in modo leggibile. In PPO queste quantità non ci sono, nemmeno come
combinazione lineare: è coerente con i pesi quasi piatti sui canali dei nemici.

### Neuroni booleani o graduati

| | `paper_astra` strato 0 | `paper_astra` strato 1 | `ppo` strato 0 | `ppo` strato 1 |
|---|---|---|---|---|
| neuroni tanh saturi (\|a\| > 0.95) | 95% | 90% | 12% | 58% |
| neuroni con \|r\| ≥ 0.8 con una grandezza del gioco | 0/32 | 2/32 | 0/32 | 0/32 |
| R² medio di un neurone sulle grandezze del gioco | 0.15 | 0.19 | 0.28 | 0.37 |

In Astra 39 neuroni su 64 sono interruttori (saturi in più del 90% degli stati, e a volte da una parte a volte
dall'altra); in PPO nessuno. Come in Seaquest, le grandezze del vocabolario spiegano i neuroni PPO meglio di quelli
di Astra: quasi tutti i neuroni PPO seguono la posizione del giocatore, che è nel vocabolario, mentre Astra calcola
condizioni su singole righe.

## 7. Si può scrivere come regola? (distillazione)

| profondità | `paper_astra` accordo / foglie / punti | `ppo_v1` | `ppo` |
|---|---|---|---|
| 1 | 78.7% / 2 | 34.2% / 2 | 31.0% / 2 |
| 2 | 82.6% / 4 / 2.2 | 40.3% / 4 / 0.7 | 37.2% / 4 / 1.3 |
| 4 | 88.4% / 16 / 3.3 | 49.9% / 16 / 2.8 | 43.8% / 16 / 2.3 |
| 6 | 92.1% / 40 | 52.3% / 48 | 49.0% / 54 |
| 8 | 93.8% / 68 / 36.3 | 56.7% / 98 / 3.2 | 50.9% / 127 / 4.7 |
| 12 | 95.9% / 112 / 42.5 | 57.7% / 182 / 3.5 | 53.8% / 337 / 6.7 |
| la rete, greedy | 65.1 | 10.2 | 45.8 |

Astra diventa un albero di 68 foglie che gioca a 36 punti. Per PPO nessun albero va oltre il 54% di accordo, e
nessuno gioca: la sua politica è la meno comprimibile dei tre giochi MinAtar. L'albero di profondità 2 dice "torna
verso il centro, e sali se il tesoro più vicino è sopra", e indovina una mossa su tre.

## 8. Conclusione

- **Astra:** i pesi sono il programma. Allarmi per riga, attrazione verso i tesori, pesi tondi su pochi ingressi,
  una strategia a una dimensione.
- **PPO:** gioca a metà di Astra, e i suoi pesi non hanno questa forma:
  - usa tutti gli ingressi vivi, con valori continui, e non regge il pruning;
  - il calcolo è distribuito su tutti i neuroni, nessuno dei quali è un allarme o una condizione;
  - metà del peso è sulla propria posizione, quasi niente sulla direzione dei nemici;
  - il comportamento non si comprime in un albero.

Lungo l'addestramento e con le altre entropie il quadro è lo stesso: vedi
[`experiments/ppo_over_training`](../../ppo_over_training/README.md).

## File

- `train.py`: configurazione e lancio dell'addestramento (quella di `ppo.json`).
- `analyze.py`: tutte le misure; con `--only-key` solo le quantità chiave e le misure sugli ingressi vivi.
- `results/ppo_ent01/`, `results/ppo_ent05/`, `results/ppo_ent10/`: le tre corse da 200M passi (curva
  `train_log.csv`, `checkpoint_*M.json` ogni 5M passi, `eval.json`, `final.pt`, `console.log`).
- `results/ppo_v1/`: il primo tentativo (6 azioni, entropia 0.01, 30M passi).
- `results/analysis/`: `analysis.json` (tutti i numeri), `trees_*.txt` (alberi di profondità 1-4), `fields_*.png`
  (immagini del primo strato).
