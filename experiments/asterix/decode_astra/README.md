# Come si legge la rete `paper_astra.json` di Asterix

`weights/asterix/paper_astra.json` è la rete che GPT-6 Astra ha scritto a mano, senza training, per Asterix di
MinAtar nell'articolo "Weight-space programming" (Esperimento 5). Fa 54.8 ± 1.1 punti su 1000 episodi; una policy
casuale ne fa 0.5.

Questa analisi la decodifica neurone per neurone, come [`decode_claude`](../../low_tide/decode_claude/README.md) e
[`decode_gpt`](../../low_tide/decode_gpt/README.md) fanno per Low Tide. Anche questa rete è un algoritmo scritto in
forma di pesi, e piuttosto semplice:

- **non si muove mai in orizzontale:** sinistra, destra e fuoco hanno bias −1000. Il giocatore resta nella colonna
  5 e sceglie solo la riga. Tutti i nemici e i tesori attraversano la colonna 5, quindi il gioco diventa a una
  dimensione;
- **per ogni mossa (fermo, su, giù) calcola un allarme:** i nemici che stanno arrivando verso la colonna 5 nella
  riga in cui la mossa porta il giocatore, e in quella dopo;
- **tra le mosse senza allarme sceglie quella che va verso i tesori:** fermo se un tesoro sta arrivando nella sua
  riga, altrimenti verso le righe con tesori in arrivo, con un'attrazione che cala con la distanza in righe.

La lettura è verificata contro la rete vera su 171.311 stati di gioco. Il primo strato coincide esattamente; la
regola finale, fuori dai quasi-pareggi, sceglie la stessa mossa della rete in tutti gli stati tranne 13, e giocata al
posto della rete fa lo stesso punteggio (vedi [Verifica](#8-verifica)):

```bash
uv run python experiments/asterix/decode_astra/check.py                  # ~40 s
uv run python experiments/asterix/decode_astra/check.py --regola 1000    # + la regola al posto della rete
uv run python experiments/asterix/decode_astra/check.py --varianti 1000  # + la rete con un pezzo tolto
```

## 1. Il gioco in breve

La griglia è 10×10. Le regole sono in `.venv/lib/python3.12/site-packages/minatar/environments/asterix.py`.

- **Giocatore:** parte in riga 5, colonna 5. Si muove di una cella per passo nelle quattro direzioni, ma resta
  nelle righe 1-8: "su" dalla riga 1 e "giù" dalla riga 8 non lo spostano.
- **Nemici e tesori:** compaiono sul bordo sinistro (colonna 0, e vanno verso destra) o destro (colonna 9, e vanno
  verso sinistra), in una riga libera tra 1 e 8. **In ogni riga c'è al più un'entità.** Un'entità su tre è un
  tesoro. Attraversano tutta la riga in linea retta, poi escono.
- **Tesoro toccato:** +1. **Nemico toccato:** l'episodio finisce. Non ci sono vite né altri modi di morire.
- **Ordine dentro un passo:** prima si muove il giocatore e si controllano le collisioni, poi (se è il loro turno)
  si muovono le entità e si controllano di nuovo.
- **Velocità:** all'inizio le entità avanzano di una cella ogni 5 passi e ne compare una ogni 10. Ogni ~100 passi
  la difficoltà sale. Dal passo ~816 le entità si muovono a ogni passo, dal passo ~918 ne compare una a ogni passo.
  Il timer non è nell'osservazione: guardando un fotogramma non si sa quando le entità si muoveranno.
- **Fuoco:** in Asterix non fa niente.
- **Sticky actions:** con probabilità 0.1 l'ambiente ripete l'azione precedente al posto di quella scelta.

## 2. Come si leggono le posizioni

L'osservazione ha 600 numeri in {0, 1}: la griglia 10×10 con 6 canali per cella.

```
indice = (riga*10 + colonna)*6 + canale

canale 0 giocatore    1 pericolo (i nemici)    2 premio (i tesori)
       3 proiettile   4 scia                   5 indicatore          (3 e 5 sono sempre vuoti in Asterix)
```

**La riga 0 è in alto, la colonna 0 a sinistra.** Per esempio il giocatore nella casella di partenza (5, 5) è
l'indice `55*6 + 0 = 330`; un nemico in (4, 3) è `43*6 + 1 = 259`; la scia in (4, 5) è `45*6 + 4 = 274`.

```
           col 0 1 2 3 4 5 6 7 8 9
 riga 0       . . . . . . . . . .      sempre vuota
 riga 1  ┐    ← entità che vanno a sinistra compaiono in colonna 9
 ...     │    righe 1-8: una entità al più per riga; il giocatore sta qui
 riga 8  ┘    → entità che vanno a destra compaiono in colonna 0
 riga 9       . . . . . . . . . .      sempre vuota
                        ↑
                    colonna 5: il giocatore non la lascia mai
```

- **Scia:** è la cella da cui l'entità arriva (quella dietro di lei), quindi dice in che direzione va. Un nemico in
  (4, 3) con la scia in (4, 2) va verso destra, cioè verso la colonna 5.
- **Azioni:** 0 fermo, 1 sinistra, 2 su (riga − 1), 3 destra, 4 giù (riga + 1), 5 fuoco.

Tre nomi che uso in tutto il documento:

- **r:** la riga del giocatore;
- **in arrivo:** un'entità che si muove verso la colonna 5, o che ci sta sopra. Una che si è già lasciata la
  colonna 5 alle spalle si sta **allontanando**;
- **d:** per un'entità in arrivo, le celle che le mancano per raggiungere la colonna 5 (0 se è già lì). Un nemico
  in (4, 3) che va a destra ha d = 2.

## 3. Struttura della rete

```
600 → 32 (tanh) → 32 (tanh) → 6 (linear)        20.486 parametri, 637 diversi da zero
```

| strato | cosa fa |
|---|---|
| 1 (tanh, 32) | **misura**, riga per riga: c'è un nemico in arrivo vicino alla colonna 5, sopra, sotto o nella riga del giocatore? C'è un tesoro in arrivo, e quante righe sopra o sotto? |
| 2 (tanh, 32) | **riassume**: tre allarmi, uno per mossa (somma dei pericoli), e quattro soglie sul tesoro di ogni riga |
| 3 (linear, 6) | **decide**: ogni mossa vale 61 senza allarme e 0 o meno con allarme, più l'attrazione dei tesori; sinistra, destra e fuoco valgono −1000 |

I pesi diversi da zero sono 468 nel primo strato, 53 nel secondo e 46 nell'ultimo, più 70 bias. Sinistra, destra e
fuoco non hanno pesi, solo il bias −1000: non vengono mai campionate, e le sticky actions possono solo ripetere
un'azione già eseguita. Nel primo strato **tutti i pesi sul giocatore stanno nella colonna 5**: la rete guarda il
giocatore solo lì, e infatti non ce lo porta mai via. Le 32 unità del primo strato hanno tutte bias −8.

## 4. Primo strato: cosa misura (32 neuroni, tanh)

| neuroni | domanda (k = riga guardata) | serve a |
|---|---|---|
| h0-h7 | c'è un nemico in arrivo nella riga k = 1..8, una o due righe **sopra** di me? | allarme di "su" |
| h8-h15 | c'è un nemico in arrivo nella riga k, una o due righe **sotto** di me? | allarme di "giù" |
| h16-h23 | c'è un nemico in arrivo nella riga k, che è **la mia**? | allarme di "fermo" |
| h24-h31 | c'è un **tesoro** in arrivo nella riga k? Quante righe sopra o sotto di me? | attrazione |

Ogni neurone guarda una sola riga. Somma tre cose: quello che c'è nella riga (nemico o tesoro, e se è in arrivo),
dove sta il giocatore rispetto a quella riga, e il bias −8.

### 4.1 Il nemico: in arrivo o no, con la scia

Prendiamo h3, che guarda la riga 4 dal punto di vista "sopra di me". I suoi pesi sulla riga 4:

```
colonna            3     4     5     6     7
pericolo (nemico)  +5    +6    +11   +6    +5        indici 259, 265, 271, 277, 283
scia                     −5    −6    −5              indici 268, 274, 280
```

Il trucco è nella scia. **La scia di un nemico che si allontana sta sulla cella dal lato della colonna 5, e il suo
peso cancella quello del nemico:**

- nemico in (4, 4) che va a destra (in arrivo, d = 1): la scia è in (4, 3), peso 0 → **+6**;
- nemico in (4, 4) che va a sinistra (si allontana): la scia è in (4, 5), peso −6 → 6 − 6 = **0**;
- nemico in (4, 3) che va a destra (in arrivo, d = 2): scia in (4, 2), peso 0 → **+5**;
- nemico in (4, 5), nella colonna 5: 11 − 5 (la scia è in (4, 4) o (4, 6)) → **+6**.

Lo stesso schema vale per tutti i neuroni di pericolo. Il contributo del nemico, E, è:

| nemico nella riga k | E in h0-h15 (righe sopra e sotto) | E in h16-h23 (la mia riga) |
|---|---|---|
| in arrivo, d ≤ 1 (anche d = 0) | 6 | 6 |
| in arrivo, d = 2 | 5 | 4 |
| in arrivo, d = 3 | 0 | 3 |
| più lontano, o che si allontana, o nessuno | 0 | 0 |

Nella propria riga la rete guarda una cella più in là (colonne 2-8 invece di 3-7, pesi in `h19`: 3, 4, 9, 10, 9,
4, 3 sul nemico e −3, −4, −9, −4, −3 sulla scia).

### 4.2 Quale riga conta: il peso del giocatore

Ogni neurone ha un peso sul giocatore solo nelle celle (r, 5) da cui quella riga è rilevante:

- **h3 (riga 4, sopra):** giocatore in (5, 5) **+4** (indice 330), in (6, 5) **+3** (indice 390). Cioè: la riga 4 è
  la riga d'arrivo di "su" se sono in riga 5, la riga dopo se sono in riga 6.
- **h11 (riga 4, sotto):** giocatore in (3, 5) +4, in (2, 5) +3.
- **h19 (riga 4, mia):** giocatore in (4, 5) +4.

La pre-attivazione è quindi `−8 + E + P`:

| nemico in arrivo | riga d'arrivo (P = 4) | riga dopo (P = 3) | mia riga (P = 4) | riga che non mi riguarda (P = 0) |
|---|---|---|---|---|
| d ≤ 1 | 2 | 1 | 2 | −2 |
| d = 2 | 1 | 0 | 0 | −3 (sopra, sotto) / −4 (mia) |
| d = 3 | −4 | −5 | −1 | −8 (sopra, sotto) / −5 (mia) |
| nessuno | −4 | −5 | −4 | −8 |

Nel resto del documento conviene usare `1 + h`, cioè quanto il neurone è sopra il suo valore "spento" (−1). È il
suo **contributo all'allarme**:

| pre-attivazione | ≤ −3 | −2 | −1 | 0 | 1 | 2 | ≥ 3 |
|---|---|---|---|---|---|---|---|
| 1 + tanh(pre) | ≈ 0 | 0.04 | 0.24 | 1.00 | 1.76 | 1.96 | 2.00 |

Esempio, giocatore in riga 5, nemico in (4, 3) che va a destra: h3 = tanh(−8 + 5 + 4) = tanh(1) = 0.76, contributo
1.76 all'allarme di "su".

### 4.3 I bordi

Le righe 1 e 8 hanno pesi diversi, perché lì una delle due mosse verticali non sposta il giocatore.

- **h0 (riga 1, sopra):** giocatore in (1, 5) **+8**, in (2, 5) +6, in (3, 5) +3 (indici 90, 150, 210). Con il
  giocatore in riga 1, "su" lo lascia dov'è: h0 guarda la riga 1 stessa, e anche **senza nemici** vale tanh(0) = 0,
  cioè contributo 1. È un **muro**. Con un nemico in arrivo a d ≤ 2 nella riga 1 sale a 2.
- **Giocatore in riga 2:** il peso 6 invece di 4 rende la riga 1 più pericolosa, e senza nemici h0 vale tanh(−2):
  contributo 0.04, cioè "su" verso il bordo costa 1.2 punti.
- **h16 e h23 (la mia riga, se è la 1 o la 8):** giocatore +7 invece di +4. Un nemico a d ≤ 3 dà almeno 1.96, e
  **senza nemici** il contributo è 0.24: stare fermi sul bordo costa un po'.
- **h15 (riga 8, sotto)** fa lo stesso per la riga 8 e "giù": giocatore in (8, 5) +8, (7, 5) +6, (6, 5) +3.
- **h7 (riga 8, sopra) e h8 (riga 1, sotto)** non hanno pesi sul giocatore: non esiste una riga sotto la 8 da cui
  guardare la 8 "sopra di me". Sono neuroni morti, con pre-attivazione al massimo −2.

### 4.4 h24-h31: il tesoro, con la distanza in righe

Prendiamo h27, la riga 4. Sui tesori e sulla scia della riga 4 ha una **scala**:

```
colonna   0    1    2    3    4    5    6    7    8    9
tesoro    +8   +8   +16  +16  +24  +24  +24  +16  +16  +8
scia           −8   −8   −16  −16  −24  −16  −16  −8   −8
```

Un tesoro **in arrivo** ha la scia dal lato del bordo, dove il peso della scia è 8 in meno del suo; uno che **si
allontana** ha la scia dal lato della colonna 5, dove il peso della scia è uguale al suo:

- tesoro in (4, 2) che va a destra: 16 − 8 (scia in (4, 1)) = **8**;
- tesoro in (4, 2) che va a sinistra: 16 − 16 (scia in (4, 3)) = **0**.

Fa 8 per ogni tesoro in arrivo e 0 per ogni tesoro che si allontana, in tutte le colonne.

Il giocatore ha pesi in tutta la colonna 5: **+0.8, +1.4, +1.5, +1.6** se è 1, 2, 3, 4 righe sotto la riga 4
(cioè il tesoro è sopra di lui), **−0.8, −1.4, −1.5** se è sopra, 0 se è nella riga 4. In generale, a n righe di
distanza, `t = 0.8` per n = 1 e `t = 1.2 + 0.1·n` per n ≥ 2, con il segno + se il tesoro è sopra.

Quindi, con un tesoro in arrivo, `h = tanh(−8 + 8 ± t) = tanh(±t)`:

| tesoro in arrivo nella riga k | h |
|---|---|
| nella mia riga | 0 |
| 1, 2, 3, ..., 7 righe sopra | 0.66, 0.89, 0.91, 0.92, 0.94, 0.95, 0.96 |
| 1, 2, 3, ... righe sotto | −0.66, −0.89, −0.91, ... |
| nessun tesoro in arrivo | ≈ −1 (pre-attivazione ≤ −6) |

Il neurone codifica tre cose insieme: se c'è un tesoro in arrivo, se è sopra o sotto, e quanto è lontano. Il
secondo strato le separa con quattro soglie.

### 4.5 Effetti collaterali della scia

Il canale scia è unico per nemici e tesori, quindi la scia di un tesoro entra nei neuroni di pericolo (pesi da −3 a
−9) e quella di un nemico nei neuroni tesoro (da −8 a −24). Quasi sempre spinge più in basso un neurone già spento,
e non cambia niente. Fa eccezione il bordo:

- con il giocatore in riga 1 e un tesoro nella riga 1 la cui scia è nelle colonne 4-6, il **muro** di h0 scende da
  tanh(0) a tanh(−5) o tanh(−6): sparisce;
- se la scia è nelle colonne 3-7, sparisce anche il costo di 0.24 dello stare fermi sul bordo (h16).

Lo stesso in riga 8. Sono tesori che si sono appena lasciati alle spalle la colonna 5 (o tesori in arrivo a d = 1,
ma lì "fermo" vince comunque): 284 stati su 171.311 nei 200 episodi di `check.py`. In quei momenti il giocatore non è
più spinto via dal bordo: resta, sale o scende come al centro.

### Cosa il primo strato non guarda

- **Il giocatore fuori dalla colonna 5.** Non ci va mai.
- **Nemici lontani dalla colonna 5:** più di 2 celle nelle righe sopra e sotto, più di 3 nella propria riga. Nemici
  a più di 2 righe di distanza.
- **I tesori che si allontanano.**
- **I canali proiettile e indicatore:** sono sempre vuoti.

## 5. Secondo strato: allarmi e soglie (32 neuroni, tanh)

| neurone | pre-attivazione | significato |
|---|---|---|
| g0 | 7 + Σ h0..h7 | allarme di "su" |
| g1 | 7 + Σ h8..h15 | allarme di "giù" |
| g2 | 7 + Σ h16..h23 | allarme di "fermo" |
| A_k = g(4k−2), k = 2..8 | 21.6 + 24·h(23+k) | tesoro in arrivo in riga k: sopra, nella mia riga o al più 2 righe sotto |
| B_k = g(4k−1), k = 1..8 | 8 + 20·h(23+k) | tesoro in arrivo in riga k: sopra o nella mia riga |
| C_k = g(4k), k = 1..7 | −8 + 20·h(23+k) | tesoro in arrivo in riga k: sopra |
| D_k = g(4k+1), k = 1..7 | −21.6 + 24·h(23+k) | tesoro in arrivo in riga k: sopra di almeno 3 righe (graduale) |

Per esempio, per la riga 4 i quattro neuroni sono g14 (A), g15 (B), g16 (C), g17 (D). Mancano A_1, C_8 e D_8: un
tesoro in riga 1 non può stare sotto il giocatore, uno in riga 8 non può stargli sopra.

### Gli allarmi: il bias 7 compensa sette neuroni spenti

Un neurone di pericolo spento vale circa −1, quindi otto spenti sommano −8. Il bias 7 ne compensa sette:

`g = tanh(7 + Σ h) = tanh(−1 + Σ (1 + h)) = tanh(A − 1)`

dove **A = Σ (1 + h)** è l'allarme della mossa: la somma dei contributi della sezione 4.2. Con A = 0, g = −0.76;
con A = 1, g = 0; con A = 2, g = 0.76; con A ≥ 3, g ≥ 0.96. In pratica, giocando, A cade sempre in uno di questi
gruppi: 0-0.15, 0.24-0.35 (solo per "fermo"), 1.00-1.11, 1.76 o più.

Con le tabelle della sezione 4, l'allarme di ogni mossa è:

| mossa | da dove viene l'allarme | contributo |
|---|---|---|
| su | nemico in arrivo nella riga sopra, d ≤ 1 / d = 2 | 1.96 / 1.76 |
| | nemico in arrivo due righe sopra, d ≤ 1 / d = 2 | 1.76 / 1.00 |
| | se la riga sopra è la 1: nemico a d ≤ 2 / nessuno | 2.00 / 0.04 |
| su dalla riga 1 (muro) | nessun nemico in riga 1 / nemico a d ≤ 2 | 1.00 / 2.00 |
| giù | lo stesso, verso il basso | |
| fermo (righe 2-7) | nemico in arrivo nella mia riga, d ≤ 1 / d = 2 / d = 3 | 1.96 / 1.00 / 0.24 |
| fermo (righe 1 e 8) | nemico in arrivo nella mia riga, d ≤ 3 / nessuno | 1.96-2.00 / 0.24 |

A questi si aggiungono i **residui**. Un nemico in arrivo a d ≤ 1 porta a −2 (contributo 0.04) i suoi neuroni che
non hanno il peso del giocatore. Se il nemico è lontano dal giocatore succede in tutti e tre i gruppi: i tre allarmi
salgono insieme di 0.04, i tre logit scendono insieme di circa 1.2 punti e la scelta non cambia. Se il nemico è
vicino, il residuo va solo alle mosse che non lo riguardano (un caso in cui pesa è nella sezione 9).

### Le soglie sul tesoro

Con h = tanh(±t) della sezione 4.4 le quattro soglie cadono così:

| tesoro in arrivo in riga k | A | B | C | D |
|---|---|---|---|---|
| nessuno | −0.98 | −1 | −1 | −1 |
| 3 o più righe sotto | ≤ −0.12 | −1 | −1 | −1 |
| 2 righe sotto | 0.34 | −1 | −1 | −1 |
| 1 riga sotto | 1 | −1 | −1 | −1 |
| nella mia riga | 1 | 1 | −1 | −1 |
| 1 riga sopra | 1 | 1 | 1 | −1 |
| 2 righe sopra | 1 | 1 | 1 | −0.34 |
| 3, 4, 5, 6, 7 righe sopra | 1 | 1 | 1 | 0.12, 0.48, 0.69, 0.81, 0.87 |

A e D sono graduali vicino alla soglia, B e C sono booleani. Lo strato successivo le usa a coppie.

## 6. Ultimo strato: la regola (6 logit)

```
fermo = 17    − 80·g2 + 16·Σ(k=1..7) (B_k − C_k) + 16·B_8
su    = 0.15  − 80·g0 +  6·Σ(k=1..7) (C_k − D_k)
giù   = −0.69 − 80·g1 +  6·Σ(k=2..8) (A_k − B_k)
sinistra = destra = fuoco = −1000
```

Le coppie si leggono con la tabella delle soglie:

- **B_k − C_k** vale 2 solo con il tesoro in arrivo **nella mia riga**, 0 in tutti gli altri casi. Fermo prende
  +32. B_8 da solo vale −1 senza tesoro: 16·B_8 = −16, che con il 17 del bias lascia 1.
- **C_k − D_k** vale 2 con il tesoro 1 riga sopra, 1.34 a 2 righe, poi scende man mano che D si accende.
- **A_k − B_k** fa lo stesso per i tesori sotto. Da spenti, gli A_k valgono −0.984 e non −1: sette di loro danno
  6 × 7 × 0.016 = 0.69, che il bias −0.69 di giù cancella.

Riscritti con l'allarme A di ogni mossa:

```
fermo = 1    − 80·tanh(A_fermo − 1) + 32·[tesoro in arrivo nella mia riga]
su    = 0.15 − 80·tanh(A_su − 1)    + Σ a(n)   sui tesori in arrivo n righe sopra
giù   = 0    − 80·tanh(A_giù − 1)   + Σ a(n)   sui tesori in arrivo n righe sotto

a(n) = 12, 8.0, 5.3, 3.1, 1.8, 1.1, 0.8   per n = 1, 2, ..., 7
```

Il termine dell'allarme, al variare di A:

| A | 0 | 0.04 | 0.24 | 1 | 1.76 | 1.96 | 2 | 3.52 |
|---|---|---|---|---|---|---|---|---|
| −80·tanh(A − 1) | **60.9** | 59.7 | 51.4 | **0** | −51.4 | −59.7 | **−60.9** | −79.0 |

Ecco i logit veri della rete in alcuni casi, con il giocatore in riga 5 (in riga 1 e 8 dove indicato):

| situazione | fermo | su | giù | mossa |
|---|---|---|---|---|
| niente in arrivo | 61.9 | 61.1 | 60.9 | vaga: fermo 56%, su 24%, giù 20% |
| tesoro in arrivo nella mia riga | **93.9** | 61.1 | 60.8 | fermo |
| tesoro in arrivo 1 riga sopra | 61.9 | **73.1** | 60.8 | su |
| tesoro in arrivo 2 / 3 / 4 righe sopra | 61.9 | 69.1 / 66.3 / 64.2 | 60.8 | su (100% / 98% / 88%) |
| tesoro 1 riga sopra e uno 2 righe sotto | 61.9 | **73.1** | 68.7 | su (99%) |
| tesoro 2 righe sopra e uno 1 riga sotto | 61.9 | 69.1 | **72.7** | giù (97%) |
| nemico in arrivo nella mia riga, d = 1 | −58.7 | 59.8 | 59.7 | su o giù |
| nemico in arrivo nella mia riga, d = 2 | 1.0 | 60.9 | 60.7 | su o giù |
| nemico in arrivo nella mia riga, d = 3 | 52.4 | 61.1 | 60.9 | su o giù |
| nemico in arrivo nella mia riga, d = 4 | 61.9 | 61.1 | 60.9 | come "niente": non lo vede |
| nemico in arrivo nella riga sopra, d ≤ 1 | 60.7 | −59.5 | 59.7 | fermo (73%) o giù |
| nemico in arrivo due righe sopra, d ≤ 1 | 60.7 | −51.2 | 59.7 | fermo o giù |
| nemico in arrivo due righe sopra, d = 2 | 61.9 | 0.1 | 60.7 | fermo o giù |
| nemico nella mia riga d = 1, tesoro 1 riga sopra | −58.7 | **71.8** | 59.6 | su |
| nemico a d = 1 sopra, nella mia riga e sotto | −61.1 | −62.0 | −62.1 | nessuna salvezza: vaga |
| nemico a d = 2 sopra, a d = 1 nella mia riga e sotto | −60.0 | **−54.4** | −61.1 | su: il male minore |
| riga 1, niente in arrivo | 52.4 | 0.1 | **60.9** | giù: si allontana dal bordo |
| riga 8, niente in arrivo | 52.4 | **61.1** | 0.0 | su |

La gerarchia viene dalle grandezze:

- **Sicurezza prima di tutto.** Una mossa con allarme 1 perde 61 punti, una con allarme 1.76 o più ne perde più di
  110, e i tesori danno al massimo 32. Finché c'è una mossa senza allarme, la rete non ne sceglie una con allarme 1
  o più. L'allarme 0.24 di "fermo" (nemico a d = 3, o il bordo) costa invece solo 9.6 punti.
- **Poi il tesoro nella mia riga:** fermo arriva a 93.9, più di quanto "su" o "giù" possano avere anche con tesori
  in tutte le righe (61 + 32.1).
- **Poi i tesori nelle altre righe:** vince il lato con l'attrazione totale più grande. Più tesori nella stessa
  direzione si sommano, come il gradiente di un campo di attrazione.
- **Niente in vista:** i tre logit differiscono di meno di 1 punto e la rete campiona: resta ferma più spesso.
- **Ai bordi:** stare fermi costa 9.6 punti e la mossa contro il muro vale 0: la rete torna verso il centro, a meno
  che la strada sia bloccata o arrivi un tesoro nella riga.

## 7. L'algoritmo

```
resta nella colonna 5 (sinistra, destra, fuoco: −1000)

per ogni nemico in arrivo verso la colonna 5:  d = celle che gli mancano (0 se è già nella colonna 5)

allarme(fermo) = nemico nella mia riga:          d ≤ 1 → 1.96,  d = 2 → 1,  d = 3 → 0.24
                 (nelle righe 1 e 8:              d ≤ 2 → 2,     d = 3 → 1.96,  altrimenti 0.24)
allarme(su)    = nemico nella riga sopra:        d ≤ 1 → 1.96,  d = 2 → 1.76
               + nemico due righe sopra:         d ≤ 1 → 1.76,  d = 2 → 1
                 (dalla riga 1 "su" non sposta:   1, o 2 con un nemico in riga 1 a d ≤ 2;
                  verso la riga 1:                2 con un nemico a d ≤ 2, altrimenti 0.04)
allarme(giù)   = lo stesso, verso il basso

punteggio(m)     = base(m) − 80·tanh(allarme(m) − 1)          base: fermo 1, su 0.15, giù 0
                   (61 senza allarme, 0 con allarme 1, da −51 a −80 con allarme 1.76 o più)
punteggio(fermo) += 32                  se un tesoro è in arrivo nella mia riga
punteggio(su)    += a(n)                per ogni tesoro in arrivo n righe sopra     a = 12, 8, 5.3, 3.1, 1.8, 1.1, 0.8
punteggio(giù)   += a(n)                per ogni tesoro in arrivo n righe sotto

campiona la mossa dalla softmax dei punteggi
```

In parole: il giocatore fa il **portiere della colonna 5**. Tutto passa di lì, quindi gli basta scegliere la riga:
aspetta i tesori che arrivano nella sua riga, si sposta verso le righe dove ne arrivano altri, e scappa in verticale
quando un nemico si avvicina alla colonna 5 nella sua riga o in quella dove sta andando.

Questa regola non riporta due dettagli: gli effetti della scia ai bordi (sezione 4.5), che la regola di `check.py`
include come eccezione, e i residui di 0.04 (sezione 5), che ignora.

## 8. Verifica

`check.py` gioca 200 episodi con la rete come `evaluate` (seed 10000-10199, stesso generatore per il
campionamento, sticky 0.1): il punteggio, 54.5, è lo stesso di `uv run evaluate asterix paper_astra --episodes 200`.
In ogni stato ricalcola, **direttamente dallo stato del gioco e senza usare i pesi**, le pre-attivazioni del primo
strato (sezione 4, con gli effetti della scia), lo stato del secondo strato, i logit con le formule della sezione 6
e quelli della regola della sezione 7. Poi li confronta con il forward pass vero.

| controllo | risultato |
|---|---|
| stati visitati | 171.311 |
| stati con il giocatore fuori dalla colonna 5 | **0** |
| primo strato: stati in cui una pre-attivazione è diversa dalla lettura | **0** (scarto massimo 1.8·10⁻¹⁵) |
| secondo strato: stati in cui lo stato di un neurone è diverso (g0-g2: spento < −0.3, a metà, acceso > 0.3; g3-g31: segno) | **0** |
| logit con le formule della sezione 6: differenza massima | 1.0·10⁻¹³ |
| regola della sezione 7: differenza massima dei logit | 9.1 (i residui di 0.04 sommati, sezione 5) |
| regola: distanza media tra la sua softmax e quella della rete | 0.007 |
| regola: mossa con logit più alto diversa, esclusi i quasi-pareggi (< 1) | **13** su 133.412 |

I quasi-pareggi sono 37.899 stati: 29.820 sono "niente in arrivo" (61.9 contro 61.1 contro 60.9), 6.528 hanno
tesori in arrivo in altre righe, ma con attrazioni piccole o simili dalle due parti. Lì la rete campiona e vaga.

Giocata al posto della rete sugli stessi 1000 seed di `evaluate` (`--regola 1000`), la regola della sezione 7 fa
**55.7 ± 1.1** punti, contro **54.8 ± 1.1** della rete.

Probabilità media di ogni mossa secondo la rete, per situazione (vale la prima che si applica, in quest'ordine):

| situazione | % degli stati | fermo | su | giù |
|---|---|---|---|---|
| nessuna mossa senza allarme (tutti ≥ 0.5) | 0.8% | 22% | 42% | 36% |
| nemico in arrivo nella mia riga, d ≤ 3 | 4.8% | 7% | 46% | 46% |
| tesoro in arrivo nella mia riga | 31.4% | 100% | | |
| riga 1 o 8 | 7.5% | 64% | 17% | 20% |
| tesoro in arrivo in un'altra riga | 21.2% | 33% | 33% | 34% |
| niente in arrivo | 34.3% | 73% | 14% | 13% |

- **Tesoro nella mia riga:** un terzo del tempo la rete aspetta un tesoro che le sta arrivando addosso.
- **Riga 1 o 8:** "fermo" al 64% è il caso in cui la strada verso il centro ha un allarme. Il bordo costa 0.24,
  meno di qualunque allarme, e la rete aspetta lì.
- **Niente in arrivo:** il 73% di "fermo" è più del 56% del caso vuoto perché nel 58% di questi stati "su" o "giù"
  ha un allarme di almeno 0.5, e la scelta resta tra fermo e l'altra mossa.

Il giocatore sta nelle righe 2-7 tra il 13% e il 15% del tempo ciascuna, nelle righe 1 e 8 il 7.5% e il 7.0%.

## 9. Limiti visibili nei pesi

### Quanto conta ogni pezzo

`check.py --varianti 1000` gioca la rete con un pezzo tolto alla volta (1000 episodi, protocollo di `evaluate`):

| variante | punteggio | durata media |
|---|---|---|
| la rete com'è | 54.8 ± 1.1 | 870 |
| senza allarmi (g0-g2 fissati al valore "spento") | 2.7 ± 0.1 | 94 |
| senza l'allarme di fermo (g2) | 12.9 ± 0.3 | 354 |
| sopra e sotto guardano solo la riga d'arrivo (tolti i pesi 3 sul giocatore in h0-h15) | 17.4 ± 0.6 | 398 |
| senza tesori (g3-g31 tolti dall'ultimo strato) | 22.9 ± 0.6 | 873 |
| senza l'attrazione verso i tesori sopra e sotto | 34.3 ± 0.7 | 900 |
| senza il +32 a fermo per il tesoro nella mia riga | 46.3 ± 1.0 | 821 |

- **Sopravvivere conta più che cercare i tesori.** Senza nessun segnale sui tesori la rete fa ancora 22.9: resta viva
  quanto prima (873 passi contro 870) e prende i tesori che le arrivano addosso per caso. Senza allarmi muore in un
  centinaio di passi.
- **Senza l'allarme di fermo (g2)** la rete non scappa dai nemici che le arrivano nella riga: la durata scende da
  870 a 354 passi e il punteggio a 12.9.
- **Guardare due righe avanti è essenziale:** con la sola riga d'arrivo il punteggio scende da 54.8 a 17.4.

### Il bordo è dove muore

Le righe 1 e 8 sono il 14.5% degli stati ma il 63.5% delle morti: 127 su 200 nei 200 episodi di `check.py` (riga 1:
71, riga 8: 56; le altre righe tra 5 e 16 ciascuna).

| causa della morte (200 episodi) | morti | di cui sul bordo | di cui entrando in una riga con un nemico già nella colonna 5 |
|---|---|---|---|
| nessuna mossa senza allarme, ma una delle tre sopravviveva al passo | 90 | 89 | 68 |
| sticky: l'ambiente ha ripetuto la mossa precedente | 68 | 8 | 31 |
| nessuna mossa senza allarme, e nessuna sopravviveva | 42 | 30 | 17 |

Un caso tipico: il giocatore è in riga 1, un nemico gli arriva incontro nella riga 1 a d = 2, e nella riga 2 c'è un
nemico **già nella colonna 5**. Gli allarmi sono fermo 2.0, su 2.0 (il muro), giù 1.96, e i logit veri della rete
fermo −61.0, su −62.0, giù −59.9: con il 68% di probabilità il giocatore scende dentro il nemico. Restare fermo, o
"su", lo avrebbe tenuto vivo almeno per quel passo, perché un nemico a 2 celle non arriva alla colonna 5 in un passo
solo. I pesi spiegano perché:

- **un nemico nella colonna 5 vale quanto uno a 1 cella.** Il peso del nemico in colonna 5 è 11 (10 nella propria
  riga), più alto degli altri, ma la sua scia sta sempre in colonna 4 o 6, con peso −5 (−4), e lo riporta a 6: lo
  stesso valore di un nemico a d = 1. Eppure entrare in una cella occupata uccide subito, mentre un nemico a d = 1
  colpisce solo se in quel passo le entità si muovono;
- **l'allarme satura a 2.** Sul bordo i pesi del giocatore sono più grandi (7 e 8 invece di 4), quindi un nemico a 2
  celle nella propria riga dà già 2.0, più di un nemico nella colonna 5 della riga accanto (1.96);
- **i residui vanno nella direzione sbagliata.** Il nemico nella colonna 5 della riga 2 porta a −2 anche i neuroni
  "riga 2, sopra di me" e "riga 2, la mia", e aggiunge 0.04 agli allarmi di su e di fermo: la mossa che entra nel
  nemico guadagna un altro punto;
- **la rete non sa quando le entità si muovono.** Il timer non è nell'osservazione, e la rete tratta quasi allo
  stesso modo un nemico che può arrivare in questo passo e uno che arriverà fra 2 o 3.

Sul bordo il giocatore ha una sola via di fuga. Stare fermo costa 9.6 punti, abbastanza per farlo ripartire quando
la riga accanto è libera, ma se la riga accanto ha un allarme resta lì (sezione 8: sul bordo, senza nemici né tesori
nella sua riga, resta fermo il 64% delle volte), e lì la prossima trappola è più probabile.

### Sticky actions

68 morti su 200 (34%) arrivano quando l'ambiente ripete la mossa precedente al posto di quella scelta, 31 di queste
dentro un nemico già nella colonna 5. Senza sticky actions (`uv run evaluate asterix paper_astra --sticky 0`) la
rete fa **68.9 ± 1.2** punti su 1000 episodi, con episodi di 998 passi in media invece di 870.

### Tesori

Nei 200 episodi di `check.py` la rete raccoglie il 68.8% dei tesori che compaiono (10.906 su 15.851).
L'attrazione somma i tesori in arrivo riga per riga, ma non guarda quanto sono lontani dalla colonna 5 né se il
giocatore farà in tempo ad arrivare: un tesoro appena comparso in colonna 0 pesa quanto uno a una cella. Oltre le 4
righe di distanza vale meno di 2 punti: la mossa verso il tesoro diventa la più probabile, non quella sicura.

### Altro

- **Nessun movimento laterale.** È una scelta di progetto: tutto passa dalla colonna 5, quindi nessun tesoro è
  irraggiungibile. Ma l'unica schivata possibile è in verticale.
- **Due neuroni morti:** h7 (riga 8, sopra) e h8 (riga 1, sotto) non hanno pesi sul giocatore e non superano mai −2.
  Sono il prezzo di aver scritto le famiglie in modo simmetrico.

### Cosa resta non spiegato

Niente di sostanziale. Ogni neurone del primo strato ha una formula esatta in termini di gioco (scarto 10⁻¹⁵), e
così il secondo strato e i logit. Due cose sporcano la regola semplice:

- **gli effetti della scia ai bordi** (sezione 4.5): 284 stati su 171.311 nei 200 episodi di `check.py`. La regola
  li include come eccezione;
- **i residui di 0.04 per nemico** (sezione 5), che la regola ignora: spostano i logit fino a 9.1 punti, ma quasi
  sempre di quantità uguali per le tre mosse. La distanza media tra le due softmax è 0.007 e la regola gioca come la
  rete (55.7 contro 54.8).
