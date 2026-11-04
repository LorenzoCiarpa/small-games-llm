# Come si legge la rete `gpt.json` di Low Tide

`weights/low_tide/gpt.json` è la rete scritta a mano da GPT-6 Astra per Low Tide, senza training e senza eseguire
codice. Fa 79.9 ± 1.0 punti su 500 episodi. Per confronto, `claude.json` ne fa 132.3, l'euristica (BFS) 164.0 e
una policy casuale 2.0.

Questa analisi la decodifica neurone per neurone, come
[`decode_claude_low_tide`](../decode_claude_low_tide/README.md) fa per la rete di Claude. Anche questa è un
algoritmo scritto in forma di pesi, ma è un algoritmo diverso, costruito con trucchi diversi:

- **vista:** guarda le celle una per una, ma solo in una finestra di 6×4 celle;
- **conchiglie:** per ognuna calcola un'attrazione che dipende dalla distanza;
- **mossa:** somma le attrazioni per direzione e aggiunge una spinta che tiene l'agente dentro la finestra.

La lettura è verificata contro la rete vera su 186.768 stati di gioco. La coincidenza è esatta, a meno degli
arrotondamenti (vedi [Verifica](#7-verifica)):

```bash
uv run python experiments/decode_gpt_low_tide/check.py            # ~1 min
uv run python experiments/decode_gpt_low_tide/check.py --episodes 500
```

## 1. Gioco e posizioni

Gioco e osservazione sono descritti nelle sezioni 1 e 2 di
[`decode_claude_low_tide`](../decode_claude_low_tide/README.md#2-come-si-leggono-le-posizioni). In breve:

```
indice = canale*100 + riga*10 + colonna       agente 0..99, acqua 100..199, conchiglie 200..299, meduse 300..399
400 marea che sale, 401 countdown, 402 stordimento
```

- **Righe:** la riga 0 è in alto (terra), la riga 9 in basso (mare).
- **Acqua:** con livello L copre le righe da `10−L` a 9.
- **Azioni:** 0 fermo, 1 su (riga−1), 2 giù (riga+1), 3 sinistra, 4 destra.
- **m (margine):** le righe asciutte tra l'agente e l'acqua, `m = 9 − riga − L`.
- **Battigia:** l'ultima riga asciutta, `9 − L`.

## 2. Struttura della rete

```
403 → 43 (relu) → 61 (relu) → 53 (relu) → 5 (linear)       23.612 parametri, 1.174 diversi da zero
```

| strato | cosa fa |
|---|---|
| 1 (43) | **legge**: 24 celle della finestra, posizione dell'agente come "rampe", livello e direzione della marea |
| 2 (61) | **misura**: attrazione di ogni conchiglia (6 − distanza), vicinanza di ogni medusa, pericolo della marea |
| 3 (53) | **smista per direzione**: somma le attrazioni delle conchiglie sopra, sotto, a sinistra, a destra |
| 4 (5) | **decide**: spinta verso la finestra + attrazioni − meduse adiacenti ± regole di emergenza |

Tutti gli strati nascosti sono relu. La rete non usa tanh né condizioni ±1 come quella di Claude. Usa due trucchi
lineari a tratti, che conviene vedere prima.

### Trucco 1: le rampe

Una **rampa** è `relu(x − a)`: vale 0 fino ad `a`, poi cresce di 1 per ogni passo. Con le rampe di una coordinata
intera (la riga o la colonna dell'agente) si costruiscono due cose.

- **Un gradino:** `relu(x − a) − relu(x − a − 1)` vale 1 se `x ≥ a + 1`, altrimenti 0.
- **Una tenda, cioè meno il valore assoluto:** `relu(x − a) − 2·relu(x − R)` vale `(R − a) − |x − R|` per `x ≥ a`.
  Così la rete calcola la **distanza** tra l'agente e una cella.

### Trucco 2: il cancello

Si scrive `relu(v + 64·g − 64)`, con g ∈ {0, 1} e 0 ≤ v ≤ 64:

- con `g = 1` vale `v`;
- con `g = 0` vale 0.

È un "lascia passare v solo se g è vero", cioè un AND tra un numero e una condizione. Il cancello g è di solito un
gradino costruito con le rampe.

## 3. Primo strato: cosa legge (43 neuroni)

| neuroni | ingressi | valore |
|---|---|---|
| h0-h23 | conchiglia (+1) e medusa (−1) di una cella, bias 1 | 2 conchiglia, 1 vuota, 0 medusa |
| h24-h30 | agente in riga r, pesi `r − 2 − k` | rampe sulla riga: `relu(r − 2 − k)`, k = 0..6 |
| h31-h34 | agente in colonna c, pesi `c − 5 − k` | rampe sulla colonna: `relu(c − 5 − k)`, k = 0..3 |
| h35 | acqua, colonna 0, righe 3..9 (peso 1) + 10·marea che sale | `L + 10·[sale]` |
| h36 | agente in riga 0 (peso 2) o 1 (peso 1) | `relu(2 − r)` |
| h37-h41 | acqua in riga 5, 6, 7, 8, 9 della colonna 0 + marea che sale, bias −1 | sale **e** L ≥ 5, 4, 3, 2, 1 |
| h42 | marea che sale | 0 o 1 |

**La finestra.** h0-h23 guardano le 24 celle delle righe 3-8 e delle colonne 6-9, una per neurone: la cella in riga
R e colonna C è il neurone `4·(R−3) + (C−6)`. Per esempio, h0 ha pesi `[236: +1, 336: −1]`, cioè conchiglia e
medusa in riga 3, colonna 6.

**Il resto della spiaggia è invisibile:**

- le colonne 0-5;
- la riga 9;
- in tutto, 46 delle 70 celle della zona di marea.

**Cosa non legge:** countdown (401) e stordimento (402), come la rete di Claude.

## 4. Secondo strato: cosa misura (61 neuroni)

### n0-n23: l'attrazione di ogni conchiglia

Per la cella (R, C) della finestra:

```
pre = 32·h_cella + h24 − 2·h(25+R−3) + h31 − 2·h(32+C−6) − 32·h(37+R−3) − 60 − (R−3) − (C−6)
```

- **Righe:** `h24 − 2·h(25+R−3)` è una tenda, cioè `(R − 2) − |r − R|`.
- **Colonne:** `h31 − 2·h(32+C−6)` è un'altra tenda, `(C − 5) − |c − C|`.
- **Bias:** cancella le costanti.

Con una conchiglia nella cella (`h_cella = 2`) resta

```
n = relu(6 − d)        d = |r − R| + |c − C|   (distanza di Manhattan)
```

Il valore è quindi 5 per una conchiglia adiacente, 1 per una a distanza 5, 0 da distanza 6 in poi.

- **Cella vuota** (`h_cella = 1`): pre = 32 − 58 − d, negativo.
- **Medusa** (`h_cella = 0`): pre ancora più negativo.

Due dettagli:

- **Le rampe partono dalla riga 2 e dalla colonna 5.** Un agente nelle righe 0-1 viene visto come se fosse in riga
  2, uno nelle colonne 0-4 come se fosse in colonna 5. Succede in meno del 5% degli stati.
- **Il termine `−32·h(37+R−3)` spegne la conchiglia quando la marea la minaccia.** Per la riga R il neurone usato è
  "sale e L ≥ 8 − R", cioè: la marea sale e la riga della conchiglia ha margine ≤ 1. La rete ignora le conchiglie
  che l'acqua sta per raggiungere.

### n24-n47: le meduse vicine

È lo stesso calcolo con `−32·h_cella` al posto di `+32` e senza il termine della marea. Passano solo le meduse
(`h_cella = 0`):

```
n = relu(2 − d)         1 se la medusa è adiacente all'agente, 0 se è più lontana
```

### n48-n58: le rampe, copiate

n48-n54 copiano h24-h30, n55-n58 copiano h31-h34. In un MLP ogni strato vede solo il precedente, quindi la
posizione dell'agente va "passata avanti" se serve più tardi. La relu non cambia un valore già ≥ 0.

### n59, n60: il pericolo della marea

```
n59 = relu(h24 − h36 + h35 − 15)
n60 = relu(h24 − h36 + h35 − 14)
```

- **La riga:** `h24 − h36 = relu(r − 2) − relu(2 − r) = r − 2` per ogni riga. Due rampe opposte ricostruiscono r.
- **La marea:** `h35 = L + 10·[sale]`. Con la marea che scende la somma resta sotto zero. Con la marea che sale
  vale `r + L − 7 = 2 − m`, perché `r + L = 9 − m`.

Quindi:

```
n59 = relu(2 − m) se la marea sale, altrimenti 0      (2 a m = 0, 1 a m = 1)
n60 = relu(3 − m) se la marea sale, altrimenti 0      (3, 2, 1 a m = 0, 1, 2)
```

## 5. Terzo strato: le direzioni (53 neuroni)

Ogni neurone somma le attrazioni di una riga o di una colonna della finestra e le fa passare con un cancello
(trucco 2) solo se quella riga o colonna è nella direzione giusta. Il cancello è un gradino costruito con le rampe
copiate.

| neuroni | somma | cancello | direzione |
|---|---|---|---|
| p0-p5 | attrazioni della riga R = 3..8 | `[r ≥ R + 1]` | conchiglie **sopra** |
| p6-p11 | attrazioni della riga R | `[r ≤ R − 1]` | conchiglie **sotto** |
| p12-p15 | attrazioni della colonna C = 6..9 | `[c ≥ C + 1]` | conchiglie **a sinistra** (p15 sempre 0) |
| p16-p19 | attrazioni della colonna C | `[c ≤ C − 1]` | conchiglie **a destra** |
| p20-p39 | lo stesso per le meduse vicine | | medusa adiacente sopra, sotto, a sx, a dx |
| p40-p50 | copie delle rampe (n48-n58) | | |
| p51, p52 | copie di n59, n60 | | |

Per esempio, p0 ha pesi `n0..n3: +1, n49: +64, n50: −64` e bias −64. Il gradino `n49 − n50 = relu(r−3) − relu(r−4)`
vale `[r ≥ 4]`, quindi p0 lascia passare le conchiglie della riga 3 solo se l'agente sta più in basso.

Una conchiglia in diagonale conta in **due** direzioni. Per esempio, una conchiglia in alto a sinistra spinge sia
verso su sia verso sinistra.

## 6. Ultimo strato: la regola (5 logit)

Riscritto in funzione della posizione dell'agente (r, c), con `[x]` = 1 se vero e 0 se falso:

```
fermo = −4
su    = −12   + 12·[r≥4] + 0.75·[r≥7] + 12·[r=9]           + 0.9·(conchiglie sopra) − 25·(medusa sopra) + 1000·n59
giù   =  12.75 − 12·[r≥3] − 0.75·[r≥6] − 12·[r≥8]          + 0.9·(conchiglie sotto) − 25·(medusa sotto) − 1000·n60
sx    = −12   + 12·[c≥7] + 0.75·[c=9]                      + 0.9·(conchiglie a sx)  − 25·(medusa a sx)
dx    =  12.75 − 12·[c≥6] − 0.75·relu(c−7) − 11.25·relu(c−8) + 0.9·(conchiglie a dx) − 25·(medusa a dx)
```

"Conchiglie sopra" è la somma di `6 − d` sulle conchiglie visibili sopra l'agente (sezione 5). "Medusa sopra" vale 1
se c'è una medusa nella cella subito sopra.

### La spinta verso la finestra

Senza conchiglie e senza pericolo, i logit dipendono solo dalla posizione:

| riga r | ≤ 2 | 3 | 4-5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| su | −12 | −12 | 0 | 0 | 0.75 | 0.75 | 12.75 |
| giù | **12.75** | 0.75 | 0.75 | 0 | 0 | −12 | −12 |

| colonna c | ≤ 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|
| sx | −12 | −12 | 0 | 0 | 0.75 |
| dx | **12.75** | 0.75 | 0.75 | 0 | −12 |

È un campo di potenziale:

- **Fuori dalla finestra** (righe 3-8, colonne 6-9) la spinta è forte, circa 12.75 contro −12, e riporta l'agente
  dentro.
- **Dentro la finestra** le mosse che restano nella finestra valgono 0 o 0.75, quelle che ne uscirebbero −12 (su
  dalla riga 3, giù dalla riga 8, sx dalla colonna 6, dx dalla colonna 9). L'agente vaga, con "fermo" sempre a −4.

La spinta tiene l'agente nella zona che la rete riesce a vedere.

### Le conchiglie: un gradiente

Ogni conchiglia visibile a distanza d aggiunge `0.9·(6 − d)` alle mosse che la avvicinano. Una conchiglia adiacente
vale 4.5, una a distanza 5 vale 0.9. Più conchiglie nella stessa direzione si sommano. È come seguire il gradiente
di un campo di attrazione: si va dove l'attrazione totale è più forte, non verso la conchiglia più vicina.

### Meduse ed emergenze

- **Medusa adiacente:** −25 alla mossa che ci porta sopra. È un divieto quasi assoluto.
- **Marea che sale e m ≤ 1:** +1000 (m = 1) o +2000 (m = 0) a "su". Scappa, e batte tutto.
- **Marea che sale e m ≤ 2:** −1000·(3 − m) a "giù". Scendere è vietato.
- **Conchiglie con margine ≤ 1 mentre la marea sale:** ignorate (sezione 4).

### L'algoritmo

```
se la marea sale e m ≤ 1:   su
altrimenti, per ogni mossa (su, giù, sx, dx):
    punteggio = spinta verso la finestra (righe 3-8, colonne 6-9)
              + 0.9 × Σ (6 − distanza) sulle conchiglie visibili in quella direzione
                (escluse quelle che la marea che sale sta per raggiungere)
              − 25 se nella cella di destinazione c'è una medusa
    giù vietato se la marea sale e m ≤ 2
fermo = −4; si campiona la mossa dalla softmax dei punteggi
```

Quando la marea scende non c'è nessuna regola: la rete non segue la battigia e non sa che le conchiglie nuove
compaiono lì.

## 7. Verifica

`check.py` gioca 200 episodi con la rete (seed 10000-10199, sticky 0.1, campionando dalla softmax come
`evaluate`). In ogni stato ricalcola, **direttamente dallo stato del gioco e senza usare i pesi**, i 43 neuroni del
primo strato e i 5 logit con le formule della sezione 6, e li confronta con il forward pass vero.

| controllo | risultato |
|---|---|
| stati visitati | 186.768 |
| primo strato: stati in cui la lettura sbaglia | **0** |
| logit: differenza massima | 1.1·10⁻¹³ (la rete è tutta relu e lineare: le formule sono esatte) |
| mossa con logit più alto diversa, esclusi i quasi-pareggi (< 0.5) | **0** |

I quasi-pareggi sono 61.158 stati. Sono per lo più l'agente dentro la finestra vuota, dove i logit sono 0 o 0.75 e la
rete vaga.

Dove sta l'agente:

| colonna | 0-4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|
| % degli stati | 0.4% | 1.1% | 14.3% | 27.7% | 36.1% | 20.5% |

| riga | 0-1 | 2 | 3 | 4 | 5 | 6 | 7 | 8-9 |
|---|---|---|---|---|---|---|---|---|
| % degli stati | 3.7% | 10.4% | 25.7% | 21.6% | 18.7% | 13.5% | 5.1% | 1.2% |

L'agente passa il 98.6% del tempo nelle colonne 6-9. In media solo il **9.4%** delle conchiglie presenti sulla
griglia è dentro la finestra: quelle che vede le raccoglie, le altre si accumulano dove non guarda.

## 8. Limiti visibili nei pesi

- **Vede solo 24 delle 70 celle della zona di marea** (34%): righe 3-8, colonne 6-9. Le colonne 0-5 e la riga 9
  non esistono per la rete, e la spinta verso la finestra la tiene lì anche quando la finestra è vuota. Per questo
  raccoglie 86.6 conchiglie per episodio, contro le 154 di Claude.
  - La causa più probabile è il limite di 64 unità per strato. Con un neurone per cella per le conchiglie e uno
    per le meduse, 24 celle occupano già 48 neuroni; tutta la zona di marea ne richiederebbe 140.
  - Claude ha evitato il problema con neuroni che riassumono un'intera riga o colonna (i confronti in binario).
- **Non sfrutta il ciclo della marea.** Non ha regole per la marea che scende, quindi non segue la battigia dove
  compaiono le conchiglie nuove.
- **Non legge countdown (401) e stordimento (402).**
- **Annega nel 15.6% degli episodi, ma solo per colpa delle sticky actions.** Senza sticky actions
  (`uv run evaluate low_tide gpt --sticky 0`) annega in 0 episodi su 500 e fa 90.5 punti.
  - **Meccanismi:** sono gli stessi della rete di Claude. Le sticky actions ripetono un "giù" mentre la marea sale.
    Nei 65 annegamenti su 500 episodi, 34 casi sono solo sticky e 31 sono sticky + medusa che stordisce sulla
    battigia.
  - **Perché più spesso:** con la marea che sale, la rete sceglie "giù" a m = 3 nel 3.4% dei passi, contro l'1.3%
    di Claude, cioè 2.6 volte di più. È la spinta verso la finestra: con la marea alta la riga 3 è vicina
    all'acqua, ma la rete vuole stare nelle righe 3-8 (dalla riga 2 "giù" vale 12.75). Ogni discesa a m = 3 è
    un'occasione per due sticky consecutive.
- **Meduse: va meglio di Claude** (6.7 punture per episodio contro 21.7). Vede le meduse adiacenti in tutta la
  finestra, non solo sulla battigia, e passa poco tempo sulla battigia, dove compaiono quelle nuove.

## 9. Confronto con `claude.json`

| | `claude.json` | `gpt.json` |
|---|---|---|
| architettura | 403-46-36-5, relu e tanh | 403-43-61-53-5, solo relu |
| pesi diversi da zero | 953 | 1.174 |
| conchiglie | tutta la zona di marea, ma solo in croce (riga e colonna) e senza distanza | solo 24 celle, con distanza di Manhattan |
| meduse | solo sulla battigia, solo adiacenti | adiacenti, dentro la finestra |
| marea | strategia completa: segue la battigia quando scende, 2 righe di margine quando sale | solo emergenze: su a m ≤ 1, mai giù a m ≤ 2 |
| trucco principale | confronti in binario (potenze di 2), tanh saturo come booleano | rampe relu (gradini, valore assoluto), cancello `relu(v + 64g − 64)` |
| idea di fondo | regole di priorità ("se… allora…") | campo di potenziale + gradiente di attrazione |
| punteggio (sticky 0.1) | 132.3 ± 0.9 | 79.9 ± 1.0 |
| conchiglie / punture / annegamenti | 154 / 21.7 / 4.2% | 86.6 / 6.7 / 15.6% |
| senza sticky actions | 140.7, nessun annegamento | 90.5, nessun annegamento |

Le due reti sono due programmi diversi per lo stesso gioco:

- **Claude** vede poco di ogni cosa, ma ovunque, e ha una strategia legata al ciclo della marea.
- **GPT** vede bene una zona piccola, con distanze vere, ma resta chiuso in quella zona.
