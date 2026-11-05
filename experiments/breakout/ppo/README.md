# PPO su Breakout: che pesi trova, e si leggono?

`train.py` addestra con PPO una rete 600-32-32-6 tanh, la stessa architettura della rete che GPT-6 Astra ha scritto
a mano nell'articolo (`paper_astra.json`). La salva in `weights/breakout/ppo.json`, nello stesso formato. `analyze.py`
fa le stesse misure sulle due reti. La rete di Astra è già decodificata neurone per neurone
([`decode_astra`](../decode_astra/README.md)), quindi fa da termine di paragone: mostra che aspetto hanno, con queste
misure, dei pesi che *sono* un algoritmo. Le misure sono quelle di [PPO su Low Tide](../../low_tide/ppo/README.md),
spiegate lì in dettaglio.

**In breve.**

- **Come gioca:** PPO arriva a 63 punti, contro i 512 di Astra. Esce dal plateau a ~6 punti dei primi tentativi
  solo con più entropia (sezione 1), ma resta lontano da Astra.
- **Come sono i pesi:** densi e distribuiti, come su Low Tide.
  - Usa tutti i 220 ingressi vivi (quelli che non sono sempre zero), con 7.027 valori diversi; Astra il 10%, con 53
    valori.
  - Non sopporta il pruning: tenendo metà dei pesi vivi del primo strato non gioca più. Astra regge con un decimo.
  - Nessun neurone calcola le quantità dell'algoritmo di Astra ("dove arriverebbe la palla se scendesse a destra"):
    |r| 0.31-0.34 contro 0.94.
  - Un albero di profondità 6 lo imita nel 72% degli stati (Astra: 89%).
- **Cosa si legge:** i neuroni più importanti del primo strato leggono la **posizione della racchetta** con un
  profilo per colonna, e la palla con macchie larghe, senza una forma riconoscibile. A differenza di Low Tide,
  però, molti neuroni contano: 10 neuroni su 64 da soli tolgono più di metà del punteggio.

```bash
uv run --group ppo python experiments/breakout/ppo/train.py          # ~1h50 su un core (200M passi)
uv run --group analysis python experiments/breakout/ppo/analyze.py   # ~10 min con 4 processi
```

## 1. L'addestramento

L'algoritmo è in `src/small_games/ppo.py`: PPO standard con clipping e GAE, come `ppo.py` di CleanRL.

| | |
|---|---|
| rete (attore) | 600-32-32-6, tanh; esportata così com'è |
| azioni | il minimal action set di MinAtar: fermo, sinistra, destra. Le altre tre (su, giù, fuoco) non fanno niente e hanno logit −100 |
| critico | 600-64-64-1, tanh; non esportato |
| ambienti | 64 in parallelo, rollout da 128 passi (8192 passi per aggiornamento) |
| ottimizzazione | 4 epoche, 8 minibatch, Adam lr 5·10⁻⁴ che scende a 0, clip 0.2 |
| esplorazione | **coefficiente di entropia 0.05** |
| durata | 200 milioni di passi, ~1h50 |
| ambiente | quello della valutazione: sticky actions 0.1, 10.000 passi; seed da 1.000.000 in su, mai quelli della valutazione |

### Il plateau

I primi tentativi si fermavano a ~6 punti:

- **`ppo_v1`** (tutte e 6 le azioni, entropia 0.01, 30M passi): 5.8 punti. Quattro azioni su sei non spostano la
  racchetta, e la policy finisce per non muoversi quasi mai.
- **Una rete più grande** (256-256, fuori regole, `control_big.py`): 6.8 dopo 6M passi. La taglia non c'entra.
- **"Segui la palla"** (`follow_ball.py`, la racchetta va verso la colonna della palla): 8.4. Basterebbe a superare
  il plateau, ma non porta lontano.

Con il minimal action set (3 azioni) e più entropia PPO esce dal plateau. Le tre corse di
`experiments/ppo_runs.py`, 200M passi ciascuna, differiscono solo per il coefficiente di entropia:

| corsa | supera 10 punti | ritorno in addestramento a 50M / 100M / 200M | valutazione (1000 episodi) |
|---|---|---|---|
| `ppo_ent01` (0.01) | a 15M passi | 27 / 31 / 35 | 34.7 ± 0.5 |
| **`ppo_ent05` (0.05)** | **a 3.5M passi** | 37 / 60 / 64 | **63.2 ± 1.0** |
| `ppo_ent10` (0.1) | a 2.2M passi | 26 / 33 / 54 | 52.2 ± 1.4 |
| rete di Astra | | | 512.5 ± 7.6 |

`weights/breakout/ppo.json` è `ppo_ent05`. Dopo 100M passi la sua curva oscilla tra 55 e 68 senza salire più.
Astra fa 512 prevedendo in che colonna arriverà la palla dopo i rimbalzi; PPO, con questa rete e questi passi, non
ci arriva.

## 2. Il punteggio

| | `paper_astra` | `ppo_v1` | `ppo` |
|---|---|---|---|
| punteggio (200 episodi, campionata) | 512.7 ± 16.9 | 5.8 ± 0.1 | 62.4 ± 2.5 |
| punteggio greedy | 512.7 | 5.7 | 69.4 |

## 3. Com'è fatto un peso PPO

Su tutti i 600 ingressi:

| | `paper_astra` | `ppo_v1` | `ppo` |
|---|---|---|---|
| pesi diversi da zero | 3.9% | 100.0% | 99.5% |
| valori diversi nel primo strato | 55 | 18.500 | 18.732 |
| neuroni del primo strato con tutti i pesi a zero | 26 | 0 | 0 |
| ingressi che fanno il 90% del peso di un neurone (mediana, su 600) | 84 | 359 | 270 |
| peso più grande in valore assoluto | 1000.00 | 1.47 | 3.29 |

(Lo 0.5% di zeri di `ppo` sono le righe di uscita delle tre azioni escluse.)

**Gli ingressi vivi.** In Breakout 380 ingressi su 600 sono sempre zero: i canali danger e gauge, la racchetta fuori
dalla riga 9, la palla e la sua scia nella riga 9, i mattoni fuori dalle righe 1-3. PPO non riceve gradiente sui
loro pesi, che restano quelli dell'inizializzazione casuale (verificato: identici tra il checkpoint a 5M passi e la
fine). Sono piccoli, ma diversi da zero e tutti diversi, e gonfiano i numeri sopra senza dire niente su cosa ha
imparato la rete. Ristretti agli ingressi vivi (quelli diversi da zero in almeno uno stato giocato da una delle tre
reti):

| primo strato, solo i 220 ingressi vivi su 600 | `paper_astra` | `ppo_v1` | `ppo` |
|---|---|---|---|
| pesi diversi da zero | 9.9% | 100.0% | 100.0% |
| valori diversi | 53 | 6.983 | 7.027 |
| ingressi che fanno il 90% del peso di un neurone (mediana, su 220) | 77 | 127 | 136 |
| \|w\| medio sugli ingressi morti / vivi | 0.296 / 3.315 | 0.046 / 0.116 | 0.046 / 0.406 |

Il quadro non cambia: la rete PPO usa ogni ingresso vivo, con valori tutti diversi.

Le immagini mostrano i pesi del primo strato dei neuroni più importanti (quelli la cui ablazione fa perdere più
punti, sezione 5): una riga per neurone, un pannello per canale (rosso positivo, blu negativo).

**Rete di Astra:** sei neuroni, e si legge cosa fanno. Due codificano la direzione della palla (gradienti
orizzontale e verticale tra la palla e la sua scia), quattro sono tabelle "cella della palla → colonna d'arrivo",
confrontate con la posizione della racchetta (il gradiente nella riga 9 del canale player).

![primo strato di paper_astra.json](results/analysis/fields_paper_astra.png)

**Rete PPO:**

![primo strato di ppo.json](results/analysis/fields_ppo.png)

- **Racchetta:** nel canale player i pesi vivi sono solo quelli della riga 9, l'unica in cui la racchetta può
  stare, e cambiano con la colonna: alcuni neuroni rispondono alla racchetta a destra (u17, u5), altri a sinistra
  (u1, u23). È la parte più leggibile della rete.
- **Palla:** nei canali projectile e trail ci sono macchie larghe, con qualche preferenza (u5 cresce quando la palla
  è nelle righe alte), ma senza la struttura a tabella di Astra.

## 4. Quanti pesi servono (pruning)

Tenendo la stessa frazione dei pesi più grandi in ogni strato (200 episodi):

| pesi tenuti | 100% | 50% | 25% | 10% | 5% | 2% |
|---|---|---|---|---|---|---|
| `paper_astra` | 512.7 | 512.7 | 512.7 | 512.7 | 512.7 | 0.5 |
| `ppo_v1` | 5.8 | 5.8 | 3.6 | 2.0 | 1.9 | 0.7 |
| `ppo` | 62.4 | 40.1 | 2.9 | 0.7 | 0.3 | 0.3 |

Solo il primo strato, sugli ingressi vivi (i pesi degli ingressi morti a zero, poi si tiene la frazione indicata dei
pesi vivi più grandi; gli altri strati restano interi):

| pesi vivi del primo strato tenuti | 100% | 50% | 25% | 10% | 5% |
|---|---|---|---|---|---|
| `paper_astra` | 512.7 | 512.7 | 512.7 | 512.7 | 0.5 |
| `ppo_v1` | 5.8 | 4.1 | 3.0 | 2.1 | 2.4 |
| `ppo` | 62.4 | 0.6 | 0.4 | 0.5 | 0.6 |

La rete di Astra perde solo zeri fino al 10%. La rete PPO non regge nemmeno la metà dei pesi vivi del primo strato:
servono quasi tutti.

## 5. Quanto conta ogni neurone (ablazione)

Si sostituisce un neurone, a ogni passo, con il valore che aveva in un altro stato preso a caso, e si misura quanti
punti si perdono (100 episodi per neurone).

| | neuroni (costanti) | che fanno perdere più del 10% | più del 50% | perdita massima | riferimento |
|---|---|---|---|---|---|
| `paper_astra` | 64 (50) | 14 | 14 | 517.7 | 518.5 ± 23.2 |
| `ppo_v1` | 64 (0) | 0 | 0 | 0.2 | 5.8 ± 0.1 |
| `ppo` | 64 (0) | 36 | 10 | 49.4 | 61.8 ± 3.2 |

- **Astra:** usa 14 neuroni su 64 (gli altri 50 hanno pesi a zero) e ognuno è indispensabile.
- **PPO:** usa tutti i 64 neuroni. Più della metà contano, e 10 da soli tolgono più di metà del punteggio. È diverso
  da PPO su Low Tide, dove nessun neurone contava: in una rete da 32 unità per strato ogni neurone porta una parte
  del calcolo, e toglierne uno lo rompe. Contare, però, non vuol dire avere un significato: il più importante (u17
  del primo strato, perdita 49 punti su 62) ha |r| 0.45 con la posizione della racchetta, e nessuna grandezza del
  gioco lo spiega meglio.

## 6. Cosa calcolano i neuroni

### Le quantità dell'algoritmo di Astra

Per ogni quantità: la correlazione più alta con un singolo neurone (|r|) e quanto bene la ricostruisce una
combinazione lineare di tutti i neuroni dello strato (R², su episodi non usati per la combinazione). Lo strato
migliore:

| quantità | `paper_astra` r / R² | `ppo_v1` r / R² | `ppo` r / R² |
|---|---|---|---|
| colonna d'arrivo − racchetta | 0.42 / 0.18 | 0.17 / 0.22 | 0.24 / 0.26 |
| arrivo se scendesse a destra − racchetta | **0.94 / 0.92** | 0.30 / 0.35 | 0.31 / 0.48 |
| arrivo se scendesse a sinistra − racchetta | **0.94 / 0.92** | 0.51 / 0.47 | 0.34 / 0.51 |
| palla − racchetta (colonne) | 0.53 / 0.48 | 0.33 / 0.63 | 0.27 / 0.43 |
| la palla scende | **0.84** / 0.72 | 0.30 / 0.26 | 0.56 / 0.44 |

Astra calcola "dove arriverebbe se scendesse a destra/sinistra" in un neurone ciascuna (sezione 4 di
`decode_astra`). Nella rete PPO nessuna di queste quantità c'è in modo pulito, nemmeno come combinazione di neuroni.

### Neuroni booleani o graduati

| | `paper_astra` strato 0 | `paper_astra` strato 1 | `ppo` strato 0 | `ppo` strato 1 |
|---|---|---|---|---|
| neuroni tanh saturi (\|a\| > 0.95) | 6% | 25% | 16% | 83% |
| neuroni con \|r\| ≥ 0.8 con una grandezza del gioco | 4/6 | 0/8 | 0/32 | 0/32 |
| R² medio di un neurone sulle grandezze del gioco | 0.74 | 0.20 | 0.22 | 0.18 |

Il secondo strato PPO è quasi sempre saturo: 12 neuroni su 32 sono a ±1 in più del 90% degli stati. Ma restano
quasi sempre dalla stessa parte e cambiano di rado, e non quando cambia una grandezza del gioco: la correlazione
migliore è 0.39 (con "la palla scende"). Sono interruttori senza un nome.

## 7. Si può scrivere come regola? (distillazione)

Un albero di decisione sulle 8 grandezze del gioco (racchetta, palla, direzione, colonna d'arrivo, mattoni) impara
a imitare la mossa più probabile della rete; accordo sul 30% degli episodi non usati, poi l'albero gioca da solo.

| profondità | `paper_astra` accordo / foglie / punti | `ppo_v1` | `ppo` |
|---|---|---|---|
| 2 | 71.7% / 4 / 1.1 | 43.6% / 4 / 0.5 | 58.3% / 4 / 0.5 |
| 4 | 82.4% / 15 / 5.6 | 60.0% / 16 / 1.5 | 66.0% / 16 / 2.2 |
| 6 | 89.0% / 49 | 71.4% / 43 | 72.1% / 60 |
| 8 | 95.0% / 143 / 12.1 | 74.8% / 55 / 2.2 | 78.0% / 194 / 4.4 |
| 12 | 98.9% / 354 / 75.6 | 76.2% / 59 / 2.2 | 83.0% / 479 / 8.7 |
| la rete, greedy | 512.7 | 5.7 | 69.4 |

Nemmeno Astra diventa un albero corto: la sua tabella "cella → colonna d'arrivo" non si scrive bene con soglie su
queste grandezze, e l'albero di profondità 12 gioca a 76. Ma con 49 foglie ne imita già l'89% delle mosse. Per PPO,
con 479 foglie l'albero sbaglia ancora una mossa su sei e gioca a 9.

## 8. Conclusione

- **Astra:** i pesi sono il programma. 14 neuroni, ognuno con un nome (direzione della palla, colonna d'arrivo se
  scende a destra o a sinistra, confronto con la racchetta), pesi tondi su un decimo degli ingressi.
- **PPO:** gioca un ottavo di Astra, e i suoi pesi non hanno questa forma:
  - usa tutti gli ingressi vivi, con valori continui, e non regge il pruning;
  - il calcolo è distribuito: molti neuroni contano, nessuno corrisponde a una grandezza del gioco;
  - non calcola le quantità dell'algoritmo di Astra, nemmeno come combinazione lineare;
  - il comportamento non si riassume in una regola corta.

L'unica parte leggibile è la codifica della posizione della racchetta, un profilo per colonna nella riga 9.

Lungo l'addestramento e con le altre entropie il quadro è lo stesso: vedi
[`experiments/ppo_over_training`](../../ppo_over_training/README.md).

## File

- `train.py`: configurazione e lancio dell'addestramento (quella di `ppo.json`).
- `analyze.py`: tutte le misure; con `--only-key` solo le quantità chiave e le misure sugli ingressi vivi.
- `follow_ball.py`: la policy "segui la palla" (8.4 punti).
- `control_big.py`: PPO con un attore 256-256, fuori regole (6.8 punti a 6M passi).
- `results/ppo_ent01/`, `results/ppo_ent05/`, `results/ppo_ent10/`: le tre corse da 200M passi.
  - `train_log.csv`: la curva di addestramento;
  - `checkpoint_*M.json`: l'attore ogni 5M passi;
  - `eval.json`: la valutazione (1000 episodi);
  - `final.pt`: attore e critico in formato PyTorch;
  - `console.log`: l'output.
- `results/ppo_v1/`: il primo tentativo (6 azioni, 30M passi).
- `results/control_256x256/train_log.csv`: la curva della rete 256-256.
- `results/analysis/`:
  - `analysis.json`: tutti i numeri;
  - `trees_*.txt`: gli alberi di profondità 1-4;
  - `fields_*.png`: le immagini del primo strato.
