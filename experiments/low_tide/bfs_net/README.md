# L'euristica BFS di Low Tide scritta nei pesi di un MLP

Le cartelle `decode_*` partono da una rete scritta da un LLM e ne ricavano l'algoritmo. Qui si fa il contrario: si
parte dall'algoritmo dell'euristica di Low Tide (`heuristic_policy`, che usa una BFS) e lo si scrive nei pesi.
La rete deve stare nei limiti del gioco:

- al massimo 3 strati nascosti da 64 unità;
- attivazioni relu, tanh o linear;
- nessun addestramento.

`build.py` scrive `weights/low_tide/bfs_net.json`, una rete 403 → 64 (relu) → 58 (tanh) → 63 (relu) → 5, con 6.806
parametri diversi da zero su 33.663. Ogni peso viene da una formula.

**Risultato:** 166.7 ± 2.0 punti sui 500 episodi del protocollo.

| rete o politica | punteggio |
|---|---|
| euristica BFS | 164.0 ± 1.5 |
| `claude` | 132.3 ± 0.9 |
| `gpt` | 79.9 ± 0.9 |
| migliore PPO (`ppo_arch_gpt`, 100M passi) | 166.9 ± 2.7 |
| `bfs_net` | 166.7 ± 2.0 |

**La rete non è la BFS esatta, che non ci sta (sezione 1).** È una versione a potenziale dello stesso algoritmo:
- va verso la conchiglia più vicina;
- segue le stesse regole sulla marea;
- non entra mai in una medusa.

Fa la stessa mossa della BFS nel 78.8% degli stati. In quasi tutti gli altri, o fa un altro passo ugualmente corto, o
va verso una conchiglia che la BFS esclude (sezione 5).

```bash
uv run python experiments/low_tide/bfs_net/build.py               # scrive i pesi
uv run python experiments/low_tide/bfs_net/check.py               # verifica, ~2 min
uv run evaluate low_tide bfs_net --baseline heuristic
uv run play low_tide --weights bfs_net                            # per guardarla giocare
```

## 1. Perché la BFS esatta non ci sta

**Il primo strato vede una cella "rispetto all'agente" solo con un neurone per cella.** Un neurone del primo
strato fa una somma pesata fissa degli ingressi.
- Il peso di una cella non può cambiare con la posizione dell'agente. Una domanda come "c'è una conchiglia due celle
  alla mia sinistra?" non si può scrivere con un solo neurone per tutte le posizioni dell'agente.
- Si può invece dare a ogni cella il suo neurone: "c'è una conchiglia qui, e quanto è lontana dall'agente?". La
  posizione dell'agente è un one-hot, quindi la distanza è un peso per ogni cella dell'agente.
- Le celle che contano sono le 60 delle righe 3-8. La riga 9 non è mai consentita, perché la marea arriva al
  livello 0 solo per poi risalire.
- Restano 4 neuroni per la posizione dell'agente e la marea, e nessuno per una mappa separata delle meduse.

**Aggirare le meduse richiede un passo di propagazione per ogni passo di cammino.** Una BFS che gira intorno a un
ostacolo propaga la distanza cella per cella. Negli episodi del protocollo il cammino della BFS arriva a 26
passi. Con 3 strati se ne fanno al più 3.

**"La più vicina" è un minimo su 60 celle.** Con due strati relu dopo il primo non si calcola esattamente.

Senza i limiti la BFS esatta si scrive, ma servono 104 strati e 51.103 neuroni: vedi
[`../bfs_exact`](../bfs_exact).

La rete quindi tiene della BFS quello che si può scrivere in 3 strati:
- la conchiglia più vicina, approssimata con un potenziale;
- le regole della marea, esatte;
- le meduse, ma solo come "non entrare nella cella", senza aggirarle lungo un cammino.

## 2. L'algoritmo che la rete calcola

`policy_logits` in `build.py` è la specifica, scritta direttamente sull'osservazione.

- **Conchiglie.** Una conchiglia a distanza di Manhattan d vale 6^(18−d). Le conchiglie si sommano per direzione:
  - **su:** le righe sopra l'agente;
  - **giù:** le righe sotto, ma solo quelle consentite;
  - **sinistra e destra:** le colonne a sinistra e a destra, in qualsiasi riga.

  Vince la direzione con la somma più alta.
  - Una conchiglia pesa 6 volte una conchiglia a un passo in più, quindi la più vicina vince: servono 6 conchiglie
    più lontane di un passo per batterla.
  - Le preferenze a parità sono quelle della BFS (su, giù, sinistra, destra), come fattori 1.06, 1.04, 1.02 e 1.
- **Meduse.** La direzione che porta in una medusa è vietata, e vince la successiva.
- **Marea:**
  - l'ultima riga consentita è `9 − L − margine`, con margine 0 se la marea scende, 1 se sale e 2 se sale e
    cambia entro 2 passi. Sono le regole di `heuristic_policy`;
  - sotto quella riga l'agente sale, e non scende sotto di essa;
  - senza conchiglie aspetta una riga sopra.

Perché sinistra e destra contano le colonne in qualsiasi riga:
- Una conchiglia in diagonale entra sia nella sua direzione verticale sia in quella orizzontale. A parità vince la
  verticale, come nella BFS, che prima cambia riga e poi colonna.
- Se ci sono più conchiglie da una parte, vince quella parte. È la differenza principale dalla BFS: a parità di
  distanza la BFS sceglie con un ordine fisso, il potenziale va dove una mossa avvicina a più conchiglie.

Quanto vale ogni pezzo. Le versioni sono in Python, con la mossa più probabile, sui 500 episodi del protocollo:

| versione | punteggio | stessa mossa della BFS | episodi finiti annegando |
|---|---|---|---|
| euristica BFS | 164.0 ± 1.5 | 100% | 11.0% |
| la conchiglia più vicina con le preferenze della BFS, senza meduse | 148.8 ± 1.6 | 90.6% | 14.0% |
| potenziale, senza il divieto sulle meduse | 146.2 ± 1.7 | 79.6% | 12.6% |
| **potenziale (la specifica)** | **166.2 ± 1.9** | 78.5% | 13.8% |
| la rete `bfs_net`, con le azioni campionate come nel protocollo | 166.7 ± 2.0 | 78.8% | 15.2% |

- **Le meduse:** valgono 15-20 punti. Una puntura stordisce per 3 passi, e vicino all'acqua che sale fa annegare.
- **I pareggi:** risolti dal potenziale invece che dall'ordine fisso, valgono qualche punto in più della BFS.

## 3. Struttura della rete

```
403 → 64 (relu) → 58 (tanh) → 63 (relu) → 5 (linear)
```

| strato | cosa fa |
|---|---|
| 1 (relu, 64) | **legge**: una conchiglia o medusa per cella (60), riga e colonna dell'agente, margine dall'acqua, margine di sicurezza |
| 2 (tanh, 64 → 58) | **misura**: potenziale di ogni riga e colonna, medusa adiacente in ogni riga e colonna, soglie sulla posizione e sulla marea |
| 3 (relu, 63) | **smista**: potenziale di ogni direzione, veti per le meduse adiacenti, eventi della marea |
| 4 (linear, 5) | **decide**: un logit per mossa |

I potenziali sono misurati in unità di P1 = 6^17 ≈ 1.7·10¹³, il valore di una conchiglia adiacente. Una soglia tanh
satura vale ±1. La scrivo [x] = (t + 1)/2, che vale 1 se la condizione è vera e 0 se è falsa.

### Strato 1: una conchiglia per cella, con la distanza dentro

Per la cella (R, C), con R tra 3 e 8:

```
pre = P1·conchiglia(R,C) + 0.5·P1·medusa(R,C) + Σ_celle dell'agente (pot(d) − P1)·agente
pot(d) = 6^(18−d) per 1 ≤ d ≤ 17, altrimenti 0          d = |R − r| + |C − c|
```

L'agente è un one-hot, quindi la somma sulle sue celle vale un solo termine, `pot(d) − P1`, che è sempre ≤ 0. Il
neurone vale:

| cosa c'è nella cella | valore |
|---|---|
| conchiglia | 6^(18−d): P1 se adiacente, P1/6 a distanza 2, e così via |
| medusa adiacente all'agente | 0.5·P1 |
| medusa più lontana, o cella vuota | 0 |

È il "cancello" della rete di GPT: un numero passa solo se la cella contiene qualcosa. Qui però la distanza non va
costruita con delle rampe, perché viene direttamente dal peso della cella in cui sta l'agente.

Gli altri 4 neuroni:

| neurone | valore |
|---|---|
| r | riga dell'agente (pesi 0..9 sulle sue celle) |
| c | colonna dell'agente |
| m | `9 − r − L`, le righe asciutte sotto l'agente (come h0 di `claude.json`) |
| v | `relu(2·sale − countdown/5)` |

v vale 1.8 o 1.6 quando il margine è 2, da 1.4 a 1.0 quando è 1, 0 quando la marea scende. Approssima il margine
entro 0.4, e alle soglie del secondo strato basta.

### Strato 2: potenziali, meduse adiacenti, soglie (tanh)

| neuroni | pre-attivazione | valore |
|---|---|---|
| potenziale riga R (6) e colonna C (10) | 10⁻³ · (somma delle celle)/P1 | tanh nella zona lineare: 10⁻³ volte il potenziale, errore relativo ≤ 2.5·10⁻⁶ |
| medusa adiacente in riga R (6) e colonna C (10) | 500 · ((somma delle celle)/P1 − 0.45) | ±1 |
| [r > k], k = 1..8, e [c > k], k = 0..8 (17) | 50 · (r − k − 0.5) | ±1 |
| riga R consentita (6) | 250 · (m + r − v − R + 0.5) | ±1: R ≤ 9 − L − margine |
| [σ ≥ k], k = 0, 1, 2 (3) | 250 · (m − v − k + 0.5) | ±1, con σ = (ultima riga consentita) − r |

Come si riconosce una medusa adiacente. La riga subito sopra l'agente ha una sola cella adiacente, quella nella sua
colonna; lo stesso vale per la riga sotto e per le colonne accanto. Il potenziale di quella riga o colonna, in
unità di P1, cade quindi in tre fasce separate:

| cella adiacente | potenziale |
|---|---|
| vuota | 0 - 0.4 (le altre conchiglie della riga sono ad almeno 2 passi, e insieme valgono al più 0.4) |
| medusa | 0.5 - 0.9 |
| conchiglia | 1.0 - 1.4 |

La soglia a 0.45 separa la prima fascia dalle altre due. La conchiglia si distingue dalla medusa nel terzo strato.

### Strato 3: direzioni, veti, marea (relu)

Le unità di direzione lasciano passare il potenziale di una riga o colonna solo se è dalla parte giusta
dell'agente (M = 4 spegne l'unità):

```
su, riga R           relu(pot_R − M·(1 − [r > R]))
giù, riga R          relu(pot_R − M·(1 − [r < R]) − M·(1 − [R consentita]))
sinistra, colonna C  relu(pot_C − M·(1 − [c > C]))
destra, colonna C    relu(pot_C − M·(1 − [c < C]))
```

Il veto per una medusa adiacente, per esempio sopra:

```
medusa sopra (riga R)   relu(0.475·Q_R − pot_R + 0.475 − M·(1 − [r = R + 1]))
```

Il valore dipende dalla fascia del potenziale di quella riga:

| fascia | Q | valore |
|---|---|---|
| vuota | −1 | ≤ 0 |
| medusa | +1 | 0.95 − pot, tra 0.05 e 0.45: acceso |
| conchiglia | +1 | 0.95 − pot ≤ −0.05: spento |

Una sola relu fa così una domanda non monotona ("medusa sì, conchiglia no"), combinando una soglia e un valore
lineare. Il cancello `[r = R + 1] = [r > R] − [r > R + 1]` lo accende solo per la riga subito sopra l'agente.

Gli ultimi 4 neuroni sono gli eventi della marea, 0 o 1:
- σ < 0, deve salire;
- σ ≤ 0, non può scendere;
- σ = 1, aspetta qui;
- σ ≥ 2, scende ad aspettare.

### Uscita

```
su       = Λ·1.06·Σ su       − W_veto·(medusa sopra)    + W_forza·[σ < 0] + 30·[σ ≤ 0]
giù      = Λ·1.04·Σ giù      − W_veto·(medusa sotto)    − W_forza·[σ ≤ 0] + 30·[σ ≥ 2]
sinistra = Λ·1.02·Σ sinistra − W_veto·(medusa a sinistra)
destra   = Λ·Σ destra        − W_veto·(medusa a destra)
fermo    = 30·[σ = 1]
```

con Λ = 1.4·10¹⁶, W_veto = 10³·Λ e W_forza = 10⁶·Λ.
- **Λ:** è scelto perché anche a distanza 17 un pareggio si rompa di ~100 logit. La softmax campiona quindi la mossa
  della specifica con probabilità ~1.
- **L'attesa:** vale 30, sotto il potenziale di qualsiasi conchiglia (≥ ~5000 logit, a distanza 17), quindi conta
  solo quando non se ne vede nessuna.
- **Primo tentativo:** l'attesa valeva 2. La softmax la sceglieva solo il 65% delle volte, e la rete faceva 159
  invece di 166.

**I pesi sono enormi:** 6^17 nel primo strato, logit fino a ~10²². Tutti i valori del primo strato sono interi
esatti in float64 (< 2^53), quindi conchiglie a distanza 17 (valore 6) e adiacenti (valore 1.7·10¹³) convivono
senza errori di arrotondamento. Il forward pass di riferimento lavora in float64.

## 4. Verifica

`check.py` riscrive i pesi dalle formule e gioca i 500 episodi del protocollo con la rete:

```
pesi riscritti dalle formule uguali al file: True
stati: 456230 (500 episodi); punteggio 166.66 ± 1.97
rete ≠ specifica (argmax): 0
potenziali per direzione, errore relativo massimo rispetto alla formula: 2.5e-06; stati con un potenziale che dovrebbe essere 0 e non lo è: 0
stati con un veto (medusa adiacente) diverso dalla formula: 0
stessa mossa dell'euristica BFS: 78.8%
quando la mossa è diversa:
    9.7% degli stati  nessuna conchiglia raggiungibile per la BFS
    6.2% degli stati  un altro passo di un cammino minimo
    4.8% degli stati  un passo più lungo (altra conchiglia, o la medusa aggirata in altro modo)
    0.5% degli stati  resta fermo
```

## 5. Dove si discosta dalla BFS

| quota degli stati | cosa succede |
|---|---|
| 78.8% | la stessa mossa |
| 9.7% | **la BFS non vede conchiglie raggiungibili e aspetta, la rete invece si muove** |
| 6.2% | **un altro passo ugualmente corto** |
| 4.8% | **un passo più lungo** |
| 0.5% | **resta ferma**: la BFS ha una conchiglia raggiungibile, la rete non ne vede una utile |

- **La BFS aspetta, la rete si muove (9.7%).** Le conchiglie ci sono, ma la BFS le esclude:
  - stanno in una riga che la marea sta per coprire: per la rete contano ancora per sinistra e destra;
  - oppure, con la marea che sale, sono accanto a una medusa: la BFS blocca anche le celle vicine alle meduse, la
    rete solo la medusa.
- **Un altro passo ugualmente corto (6.2%).** È il pareggio risolto dal potenziale invece che dall'ordine fisso.
- **Un passo più lungo (4.8%).**
  - La rete va verso un gruppo di conchiglie invece che verso la singola più vicina.
  - Oppure aggira una medusa scegliendo la seconda direzione per potenziale, mentre la BFS prende il primo cammino
    libero.

La rete annega nel 15.2% degli episodi, contro l'11.0% della BFS. Il motivo è che non tiene le distanze dalle
meduse quando la marea sale. Fa comunque qualche punto in più, perché raccoglie meglio.

## 6. Cosa resta fuori

- **Aggirare le meduse lungo un cammino, e le celle accanto alle meduse quando la marea sale.** Servirebbero più
  strati, o neuroni per una mappa delle meduse che il primo strato non ha.
- **La riga 9.** Non è mai consentita, quindi non costa niente.
- **Conchiglie a più di 17 passi.** Su questa spiaggia non esistono.
