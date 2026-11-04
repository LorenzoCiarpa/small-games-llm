# Come si legge la rete `paper_astra.json` di Breakout

`weights/breakout/paper_astra.json` è la rete che GPT-6 Astra ha scritto a mano per Breakout (MinAtar)
nell'articolo "Weight-space programming" (Esperimento 5), senza training. Fa 512.5 ± 7.6 punti su 1000 episodi
(l'articolo riporta 517.4 ± 7.4); una policy casuale ne fa 0.5.

Questa analisi la decodifica neurone per neurone, come [`decode_claude`](../../low_tide/decode_claude/README.md) e
[`decode_gpt`](../../low_tide/decode_gpt/README.md) fanno per le due reti di Low Tide. Il risultato è che la rete è
un algoritmo classico scritto in forma di pesi: **prevedi in che colonna arriverà la palla e porta lì la racchetta**.

- **Il primo strato non calcola la traiettoria, la cerca in una tabella.** La palla occupa una sola cella, quindi i
  suoi 100 pesi verso un neurone sono una tabella "cella → colonna di arrivo". Astra ha calcolato i rimbalzi su
  pareti, soffitto e mattoni per ogni cella e per ogni direzione, e ha scritto i 400 risultati nei pesi.
- **Usa 6 neuroni su 32 nel primo strato e 8 su 32 nel secondo.** Gli altri 50 hanno tutti i pesi a zero.

La lettura è verificata in due modi (vedi [Verifica](#8-verifica)). Riscrivendo i pesi dalle formule di questo
documento si riottiene il file, parametro per parametro. Giocando, la lettura coincide con la rete vera in tutti i
1.456.882 stati visitati:

```bash
uv run python experiments/breakout/decode_astra/check.py                            # ~2 min
uv run python experiments/breakout/decode_astra/check.py --sticky 0 --episodes 10   # senza sticky actions
```

## 1. Il gioco in breve

Le regole sono quelle di MinAtar (`minatar/environments/breakout.py`), che la repo usa senza modifiche.

- **Griglia:** 10×10. La racchetta è una cella in riga 9 e si sposta di una colonna per passo. Parte in colonna 4.
- **Mattoni:** 30, nelle righe 1-3. Ognuno vale +1. Quando sono finiti tutti, al passaggio successivo della palla in
  fondo ricompaiono le tre righe.
- **Palla:** si muove solo in diagonale, una riga e una colonna per passo, quindi ha 4 direzioni. Parte in riga 3,
  dalla colonna 0 verso giù-destra o dalla colonna 9 verso giù-sinistra (a caso).
- **Pareti e soffitto:** se la palla dovrebbe uscire dalla griglia, resta sul bordo e quella componente della
  direzione si inverte. Per questo passa **due passi di fila** nella colonna 0 (o 9, o nella riga 0).
- **Mattone:** se la palla dovrebbe entrare in una cella con un mattone, il mattone sparisce, la palla resta nella
  sua riga (si sposta solo di colonna) e la direzione verticale si inverte. Eccezione: subito dopo un mattone, un
  secondo mattone al passo seguente viene attraversato.
- **Fondo:** a ogni passo prima si muove la racchetta, poi la palla. Quando la palla in riga 8 dovrebbe scendere in
  riga 9:
  - se la racchetta è sotto la colonna della palla, la palla rimbalza e tiene la direzione orizzontale;
  - se la racchetta è sotto la colonna verso cui la palla va, la palla rimbalza "di spigolo" e inverte anche la
    direzione orizzontale;
  - altrimenti la palla è persa e l'episodio finisce.
- **Azioni:** 0 fermo, 1 sinistra, 3 destra. Le azioni 2 (su), 4 (giù) e 5 (fuoco) in Breakout non fanno niente,
  come "fermo".
- **Sticky actions:** con probabilità 0.1 l'ambiente ripete l'azione eseguita al passo prima, al posto di quella
  scelta.
- **Limite:** l'episodio è tagliato a 10.000 passi.

La palla e la racchetta hanno la stessa velocità orizzontale, una colonna per passo.

## 2. Come si leggono le posizioni

L'osservazione ha 600 numeri in {0, 1}: 10×10 celle per 6 canali.

```
indice = (riga*10 + colonna)*6 + canale          riga 0 in alto, colonna 0 a sinistra

canale 0  racchetta      sempre in riga 9: indici 540, 546, ..., 594
canale 1  -              sempre vuoto in Breakout
canale 2  mattoni
canale 3  palla
canale 4  scia           la cella in cui la palla era al passo prima
canale 5  -              sempre vuoto in Breakout
```

Per esempio la palla in riga 5, colonna 6 è l'indice (56)·6 + 3 = 339, la scia in riga 6, colonna 7 è
(67)·6 + 4 = 406, la racchetta in colonna 9 è (99)·6 + 0 = 594. Scrivo (r, c) per la cella in riga r e colonna c.

```
           col 0 1 2 3 4 5 6 7 8 9
 riga 0                                soffitto
 riga 1  ┐
 riga 2  │ mattoni
 riga 3  ┘
 riga 4  ┐
 ...     │ la palla vola
 riga 8  ┘ l'ultima riga prima del fondo
 riga 9                                la racchetta
```

**La direzione della palla** non è nell'osservazione, ma la scia la dà: è palla − scia. Chiamo le 4 direzioni
giù-dx, giù-sx, su-dx, su-sx ("giù" vuol dire verso la racchetta).

Tre nomi che uso in tutto il documento:

- **colonne di presa:** quando la palla è in riga 8 nella colonna x e va verso x + dx, la racchetta la prende se
  dopo la sua mossa sta in x oppure in x + dx. Contro una parete la colonna di presa è una sola.
- **T:** la colonna di arrivo che la rete prevede; **p:** la colonna della racchetta; **Δ = T − p**.
- **volo:** il tragitto della palla da un rimbalzo sulla racchetta al successivo.

## 3. Struttura della rete

```
600 → 32 (tanh) → 32 (tanh) → 6 (linear)       20.486 parametri, 815 diversi da zero
```

| strato | neuroni usati | pesi ≠ 0 | cosa fa |
|---|---|---|---|
| 1 (tanh, 32) | 6: h0-h5 | 769 + 1 bias | **misura**: per ognuna delle 4 direzioni, Δ = dove arriverebbe la palla − racchetta (h0-h3); la direzione della palla (h4, h5) |
| 2 (tanh, 32) | 8: g0-g7 | 24 + 8 bias | **condizioni**: "la palla va nella direzione k e arriva a sinistra (o a destra) della racchetta", come booleani ±1 |
| 3 (linear, 6) | 2 logit su 6 | 8 + 5 bias | **regola**: sinistra se una condizione "a sinistra" è vera, destra se una "a destra" è vera, altrimenti fermo |

- **Neuroni vuoti:** h6-h31 e g8-g31 non hanno nessun peso, né in entrata né in uscita. Valgono tanh(0) = 0.
- **Cosa legge il primo strato:** la palla (553 pesi, 72%), la scia (180, 23%), la racchetta (36, 5%).
  **Nessun peso sui mattoni.**

## 4. Primo strato (6 neuroni, tanh)

| neurone | domanda | valore |
|---|---|---|
| h0 | se la palla andasse giù-dx, dove arriverebbe rispetto alla racchetta? | tanh(0.02·Δ) ≈ Δ/50 |
| h1 | lo stesso, giù-sx | tanh(0.02·Δ) |
| h2 | lo stesso, su-dx | tanh(0.02·Δ) |
| h3 | lo stesso, su-sx | tanh(0.02·Δ) |
| h4 | la palla va a destra? | +1 sì, −1 no |
| h5 | la palla scende? | +1 sì, −1 no |

### h0-h3: dove arriverà la palla, in una tabella

Ogni neurone h0-h3 è dedicato a una direzione, ma **non sa** in che direzione va la palla: calcola dove arriverebbe
se andasse nella sua. Sceglie il secondo strato quale dei quattro ascoltare.

```
palla in (r, c)            peso 0.02 · T_k(r, c)       una tabella di 100 valori per neurone
racchetta in colonna p     peso −0.02 · p              (la colonna 0 pesa 0)
bias 0
h_k = tanh(0.02 · (T_k − p))
```

- **La tabella:** la palla è in una sola cella, quindi dei 100 pesi sul canale palla ne conta uno solo, quello
  della sua cella. È una tabella "cella → colonna di arrivo", con la colonna moltiplicata per 0.02.
- **La racchetta:** anche lei è in una sola cella, quindi i suoi pesi sottraggono la sua colonna.
- **La scala 0.02:** con |Δ| ≤ 9 la pre-attivazione resta tra −0.18 e 0.18, dove tanh è quasi una retta
  (tanh 0.18 = 0.178). Qui tanh non fa da interruttore come negli altri strati: lascia passare Δ/50 col suo segno.

**Cosa c'è nella tabella.** T_k(r, c) è la colonna in cui la palla, partita da (r, c) nella direzione k, si trova
quando arriva in riga 8 scendendo. Astra l'ha calcolata simulando le regole del gioco:

- **pareti e soffitto** esattamente come in MinAtar, compreso il passo in più sul bordo;
- **se la palla sale,** rimbalza appena la riga sopra di lei è una riga di mattoni (1-3), **come se i mattoni ci
  fossero tutti**: dalle righe 4-8 rimbalza in riga 4, dalle righe 2 e 3 subito, dalle righe 0 e 1 sul soffitto;
  poi scende senza ostacoli;
- **se la palla scende ed è già nelle righe 6-8,** T è il punto medio delle due colonne di presa (per esempio 7.5
  se la prendono le colonne 7 e 8), così la racchetta si ferma in una qualunque delle due. Dalle righe 0-5 T è la
  colonna in riga 8.

Le tabelle sono i pesi del canale palla moltiplicati per 50. Le due delle direzioni a destra:

```
h0 (giù-dx)                                   h2 (su-dx)
        col 0   1   2   3   4   5   6   7   8   9         col 0  1  2  3  4  5  6  7  8  9
riga 0      8   9   9   8   7   6   5   4   3   2   riga 0    9  9  8  7  6  5  4  3  2  1
riga 1      7   8   9   9   8   7   6   5   4   3   riga 1    9  8  7  6  5  4  3  2  1  0
riga 2      6   7   8   9   9   8   7   6   5   4   riga 2    7  8  9  9  8  7  6  5  4  3
riga 3      5   6   7   8   9   9   8   7   6   5   riga 3    6  7  8  9  9  8  7  6  5  4
riga 4      4   5   6   7   8   9   9   8   7   6   riga 4    5  6  7  8  9  9  8  7  6  5
riga 5      3   4   5   6   7   8   9   9   8   7   riga 5    6  7  8  9  9  8  7  6  5  4
riga 6    2.5 3.5 4.5 5.5 6.5 7.5 8.5   9 8.5 7.5   riga 6    7  8  9  9  8  7  6  5  4  3
riga 7    1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5   9 8.5   riga 7    8  9  9  8  7  6  5  4  3  2
riga 8    0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5   9   riga 8    9  9  8  7  6  5  4  3  2  1
riga 9      0   1   2   3   4   5   6   7   8   9   riga 9    0  1  2  3  4  5  6  7  8  9
```

h1 e h3 sono le stesse tabelle specchiate: T_sx(r, c) = 9 − T_dx(r, 9 − c). La riga 9 (T = c) non serve: la palla
ci arriva solo quando è persa. Gli zeri della tabella sono i pesi che mancano nel file: h0 ha 99 pesi sulla palla,
h1 84, h2 98, h3 82, più 9 sulla racchetta ciascuno.

Tre caselle, con gli indici veri:

- **h0, palla in (6, 7), indice 405:** peso 0.18, T = 9. La palla va in (7, 8), poi in (8, 9); lì la colonna dopo
  sarebbe la 10, quindi la palla resta nella 9 e la prende solo la colonna 9. Punto medio (9 + 9)/2 = 9.
- **h3, palla in (5, 6), indice 339:** peso 0 (manca), T = 0. La palla va in (4, 5), prova a entrare in riga 3,
  rimbalza e resta in riga 4 spostandosi in (4, 4), poi scende in (5, 3), (6, 2), (7, 1), (8, 0).
- **h2, palla in (8, 0), indice 483:** peso 0.18, T = 9. È la palla appena ribattuta nell'angolo sinistro: sale
  fino a (4, 4), rimbalza in (4, 5) e scende fino a (8, 9), all'angolo opposto.

### h4, h5: la direzione, da palla e scia

```
h4: palla in colonna c  peso 20·c   (ma colonna 0: 10, colonna 9: 170)     scia in colonna c  peso −20·c    bias 0
h5: palla in riga r     peso 10·r   (ma riga 8: 70)                         scia in riga r     peso −10·r    bias 5
```

Sono due righelli. Palla − scia è lo spostamento dell'ultimo passo, ±1 in colonna e ±1 in riga: h4 vale ±20 e h5
±10 + 5, abbastanza per saturare tanh. Le eccezioni nei pesi servono per i casi in cui palla e scia hanno la stessa
colonna o la stessa riga, cioè per i rimbalzi:

| h4 | pre-attivazione | lettura | caso |
|---|---|---|---|
| palla a destra della scia | +20 (+10 se la palla è in colonna 9) | destra | |
| palla a sinistra della scia | −20 (−10 se la palla è in colonna 0) | sinistra | |
| palla e scia in colonna 0 | 10 − 0 = **+10** | destra | rimbalzo sulla parete sinistra: ora va a destra |
| palla e scia in colonna 9 | 170 − 180 = **−10** | sinistra | rimbalzo sulla parete destra |

| h5 | pre-attivazione | lettura | caso |
|---|---|---|---|
| palla sotto la scia | +15 (+5 se la palla è in riga 8) | giù | |
| palla sopra la scia | −5 | su | |
| stessa riga r ≠ 8 | 10r − 10r + 5 = **+5** | giù | soffitto, o mattone colpito da sotto: ora scende |
| stessa riga 8 | 70 − 80 + 5 = **−5** | su | rimbalzo sulla racchetta: ora sale |

La lettura sbaglia in due casi, e ogni volta per un solo passo:

- **rimbalzo di spigolo sulla racchetta:** la palla inverte la direzione orizzontale ma si è appena spostata nella
  vecchia, e h4 legge la vecchia;
- **mattone colpito dall'alto** (la palla scende nelle righe 0-2, sopra un mattone): la palla resta nella riga e ora
  sale, ma h5 legge "giù".

Succede nello 0.56% degli stati, e in nessuno dei voli in cui la rete perde la palla (sezioni 8 e 9).

### Esempio: palla in (5, 6) che sale verso sinistra, racchetta in colonna 9

Ingressi accesi: palla 339, scia 406 (riga 6, colonna 7), racchetta 594, più i mattoni, che nessuno legge.

```
h0 = tanh(0.18 − 0.18) =  0             giù-dx: T = 9
h1 = tanh(0.06 − 0.18) = −0.119         giù-sx: T = 3
h2 = tanh(0.14 − 0.18) = −0.040         su-dx:  T = 7
h3 = tanh(0    − 0.18) = −0.178         su-sx:  T = 0, Δ = −9
h4 = tanh(120 − 140)   = −1             va a sinistra
h5 = tanh(50 − 60 + 5) = −0.9999        sale
```

### Cosa il primo strato non guarda

- **I mattoni** (canale 2): nessun peso. La rete non sa dove sono; per le palle che salgono li immagina tutti al
  loro posto.
- **I canali 1 e 5**, che in Breakout sono sempre vuoti.
- **Quando arriverà la palla:** la rete sa dove arriverà, non quando. Se la racchetta non è sull'arrivo previsto
  si muove subito, una colonna per passo, anche quando avrebbe tempo.

## 5. Secondo strato: le condizioni (8 neuroni, tanh)

Due neuroni per direzione, uno per lato:

```
g(2k)   = tanh(−415 − 1000·h_k + 200·sx·h4 + 200·sy·h5)       la palla va nella direzione k e arriva a SINISTRA
g(2k+1) = tanh(−415 + 1000·h_k + 200·sx·h4 + 200·sy·h5)       la palla va nella direzione k e arriva a DESTRA
```

| neuroni | k | h_k | sx (segno su h4) | sy (segno su h5) |
|---|---|---|---|---|
| g0, g1 | giù-dx | h0 | + | + |
| g2, g3 | giù-sx | h1 | − | + |
| g4, g5 | su-dx | h2 | + | − |
| g6, g7 | su-sx | h3 | − | − |

Il bias −415 è −400 − 15, ed è un AND di tre condizioni:

- **la direzione è giusta:** h4 e h5 sono ±1, quindi i due termini da 200 danno +400 solo se la direzione letta
  è la k. Se anche uno solo dei due segni è sbagliato la somma è al più −415 + 0 + 178 = −237, e il neurone è
  spento;
- **Δ supera la soglia:** con la direzione giusta resta −15 ∓ 1000·h_k, e 1000·h_k = 1000·tanh(0.02·Δ) vale
  ±10 per Δ = ±0.5 e ±19.997 per Δ = ±1. La soglia cade a |Δ| = 0.75, e Δ è sempre un multiplo di 0.5:

| Δ = T − p | ≤ −1 | −0.5, 0, +0.5 | ≥ +1 |
|---|---|---|---|
| g(2k), "a sinistra" | **da +5 in su, acceso** | da −25 a −5, spento | spento |
| g(2k+1), "a destra" | spento | da −25 a −5, spento | **da +5 in su, acceso** |

La pre-attivazione è sempre ad almeno 4.99 dallo zero, quindi ogni neurone vale ±1 entro 10⁻⁴: è un booleano. In
ogni stato al più uno degli 8 è acceso.

Nell'esempio di prima (palla su-sx, Δ = −9): g6 = tanh(−415 + 178 + 200 + 200) = tanh(163) = +1. Gli altri sette
sono spenti, tutti sotto −190.

## 6. Ultimo strato: la regola (6 logit)

```
fermo = 0
sx    = 90 + 30·(g0 + g2 + g4 + g6)
dx    = 90 + 30·(g1 + g3 + g5 + g7)
su = giù = fuoco = −100
```

Il bias 90 sembra strano perché **i neuroni spenti valgono −1, non 0**. Con i quattro neuroni "a sinistra"
spenti, sx = 90 − 120 = −30; con uno acceso, sx = 90 + 30·(1 − 3) = +30. Riscritti con `[x]` = +1 se x è vero,
−1 se è falso, e T la colonna prevista per la direzione letta:

```
fermo = 0
sx    = 30·[T ≤ p − 1]
dx    = 30·[T ≥ p + 1]
su = giù = fuoco = −100
```

| dove arriva la palla | sx | dx | fermo | mossa |
|---|---|---|---|---|
| a sinistra della racchetta (T ≤ p − 1) | **30** | −30 | 0 | sinistra |
| sopra la racchetta (\|T − p\| ≤ 0.5) | −30 | −30 | **0** | fermo |
| a destra della racchetta (T ≥ p + 1) | −30 | **30** | 0 | destra |

Il distacco tra il primo e il secondo logit è sempre 30, quindi la softmax sceglie la mossa con probabilità
almeno 1 − 2·10⁻¹³: la rete campionata gioca come la sua argmax. Infatti `evaluate --greedy` dà lo stesso
punteggio di `evaluate` sugli stessi 200 episodi (512.67). Su, giù e fuoco non vengono mai scelti; in Breakout
farebbero come "fermo".

## 7. L'algoritmo

```
direzione = palla − scia                     (pareti, soffitto, mattoni e racchetta sistemati nei pesi)
T = colonna in cui la palla arriverà in riga 8, presa da una tabella precalcolata:
    se sale:   rimbalzo sulla prima riga di mattoni sopra di lei (come se ci fossero tutti) o sul soffitto,
               poi giù fino alla riga 8, rimbalzando sulle pareti
    se scende: giù fino alla riga 8, rimbalzando sulle pareti
               (dalle righe 6-8: a metà tra le due colonne di presa)
se T è a sinistra della racchetta di almeno 1 colonna:  sinistra
se T è a destra della racchetta di almeno 1 colonna:    destra
altrimenti:                                             fermo
```

La rete rifà la previsione a ogni passo dalla cella in cui la palla si trova, quindi si corregge da sola quando la
palla devia (un buco tra i mattoni, un mattone colpito dall'alto).

**Perché immaginare tutti i mattoni.** Per una palla che sale, "i mattoni ci sono tutti" è il caso in cui torna
giù **prima**. Se i mattoni mancano, la palla sale di più e torna più tardi, quindi la rete ha più tempo per
correggere, non meno. Nei dati il volo vero non è mai più corto di quello previsto (sezione 8). Non sappiamo se
Astra l'abbia scelto per questo.

Il prezzo sono mosse sprecate: mentre la palla sale, la racchetta va verso un arrivo che non è quello vero. Senza
sticky actions il tempo in più basta sempre e la rete non perde nessuna palla. Con le sticky actions queste mosse
sprecate pesano in 29 delle 102 palle perse (sezione 9).

## 8. Verifica

`check.py` fa due controlli.

- **Statico:** riscrive tutti i pesi dalle formule delle sezioni 4-6, con la tabella T calcolata simulando le
  regole della sezione 4, e li confronta con il file.
- **In gioco:** gioca 200 episodi con la rete, gli stessi di `uv run evaluate breakout paper_astra --episodes 200`
  (seed 10000-10199, sticky 0.1, azioni campionate dalla softmax). In ogni stato ricalcola, **dallo stato del
  gioco e senza usare i pesi**, i neuroni dei due strati nascosti e i 6 logit con le formule delle sezioni 4-6, e
  li confronta con il forward pass vero.

| controllo | risultato |
|---|---|
| pesi riscritti dalle formule delle sezioni 4-6, contro il file | **0** parametri diversi su 20.486 (815 diversi da zero in entrambi) |
| stati visitati | 1.456.882 |
| h0-h3: stati in cui il valore sbaglia (di più di 10⁻⁹) | **0** (differenza massima 3·10⁻¹⁷) |
| h4, h5 (direzione): stati in cui il segno sbaglia | **0** |
| h6-h31, g8-g31 diversi da 0 | **0** |
| g0-g7: stati in cui acceso/spento sbaglia | **0** |
| logit: differenza massima | 0.003 (tanh non arriva mai esattamente a ±1) |
| mossa con logit più alto diversa | **0** (quasi-pareggi sotto 0.5: nessuno) |

La rete sta ferma nel 50.7% degli stati e va a sinistra o a destra nel 24.7% ciascuna. Il punteggio dei 200
episodi è 512.66 ± 16.87, lo stesso di `evaluate`.

**La rete contro il gioco.** Lo script simula anche il volo vero, con i mattoni veri, e controlla la previsione.
"Giusta" vuol dire che le colonne in cui la rete starebbe ferma sono colonne di presa.

| fase (direzione letta) | stati | previsione giusta |
|---|---|---|
| la palla sale | 722.118 | 42.6% |
| la palla scende, righe 0-2 | 89.868 | 88.4% |
| la palla scende, righe 3-8 | 644.896 | **100%** |

- **Palla che sale:** la previsione è giusta solo quando i mattoni sono davvero dove la rete li immagina. Ma il
  volo vero non è mai più corto di quello previsto: 0 stati.
- **Palla che scende:** dalla riga 3 in giù la previsione è sempre esatta. Nelle righe 0-2 sbaglia quando sotto la
  palla c'è ancora un mattone che la rimanda su.
- **Direzione letta diversa da quella vera:** 8.115 stati (0.56%): 2.232 dopo un rimbalzo di spigolo e 5.883 dopo
  un mattone colpito dall'alto. Nessuno di questi stati cade in un volo finito con la palla persa.

Senza sticky actions (`check.py --sticky 0 --episodes 10`, 100.000 stati) la lettura coincide ancora in tutti gli
stati. Ogni episodio arriva al limite di 10.000 passi con 834 punti e la rete non perde mai la palla. Lo stesso dà
`uv run evaluate breakout paper_astra --sticky 0 --episodes 200`: 834 in tutti i 200 episodi.

## 9. Limiti visibili nei pesi

Senza sticky actions la rete non perde mai la palla, e il punteggio lo limita solo il tetto di 10.000 passi.

- **Il limite di 10.000 passi.** Senza sticky actions ogni episodio arriva al limite con 834 punti: in 10.000 passi
  ci sono 835 voli da 12 passi, e ognuno rompe circa un mattone. Con sticky 0.1 arrivano al limite 98 episodi su
  200, con 702.5 punti in media. Gli altri 102 finiscono prima con la palla persa, con 330.3 punti in media.
- **Le palle perse.** Sono 102 in 113.332 voli, e in ognuna l'ambiente ha sostituito almeno 2 azioni nel volo
  (2 sostituzioni in 42 casi, 3 in 39, 4 in 18, 5 in 3). Per capire quali voli sono a rischio serve il **margine**:
  le azioni che la racchetta ha prima che la palla arrivi in fondo, meno le colonne che deve percorrere.

  | margine al decollo | voli | palle perse | % dei voli |
  |---|---|---|---|
  | 1 | 5.319 (4.7%) | 46 | **0.86%** |
  | 2-4 | 8.237 (7.3%) | 8 | 0.10% |
  | 5-9 | 62.127 (54.8%) | 27 | 0.04% |
  | ≥ 10 | 37.649 (33.2%) | 21 | 0.06% |

  - **Margine 1 vuol dire da un angolo all'altro.** Tutti i 5.319 voli con margine 1 hanno 9 colonne da
    percorrere: la palla ribattuta in un angolo sale fino alla riga 4, rimbalza sui mattoni della riga 3 e scende
    nell'angolo opposto. Palla e racchetta vanno alla stessa velocità, quindi la racchetta ha 10 azioni per 9
    colonne. Questi voli sono il 4.7% del totale ma danno il 45% delle palle perse.
  - **In questi voli una sostituzione costa solo all'inizio.** La racchetta era ferma nell'angolo (o ci era
    appena arrivata), e l'azione ripetuta non la sposta: ogni sostituzione prima della prima mossa costa una
    colonna. Quando la racchetta corre già verso l'arrivo, ripetere la mossa precedente non cambia niente.
    Esempio, l'episodio con seed 10004 (37 punti, `uv run play breakout --weights paper_astra --seed 10004`):

    ```
    passo  palla   direzione  T  racchetta  scelta  eseguita
     429   (8, 0)  giù-sx     0      0      fermo   fermo     la palla arriva nell'angolo, la racchetta c'è
     430   (8, 0)  su-dx      9      0      dx      fermo     sticky
     431   (7, 1)  su-dx      9      0      dx      fermo     sticky
     432   (6, 2)  su-dx      9      0      dx      dx
     ...                                                      la racchetta insegue, una colonna per passo
     439   (8, 9)  giù-dx     9      7      dx      dx        la racchetta arriva in 8, non in 9: palla persa
    ```

  - **Tre tipi di palla persa:**

    | volo | palle perse |
    |---|---|
    | da un angolo all'altro (margine 1), le scelte della rete sempre giuste | 46 |
    | margine maggiore, le scelte della rete sempre giuste: solo sticky | 27 |
    | la rete ha fatto almeno una mossa che non avvicina la racchetta all'arrivo vero, poi le sticky | 29 |

    Le mosse sbagliate della rete cadono tutte nelle due fasi in cui la previsione può sbagliare: 201.115 mentre
    la palla sale, 3.835 mentre scende nelle righe 0-2, nessuna dalla riga 3 in giù. Le fa nel 43.5% dei voli, ma
    da sole non bastano mai a perdere la palla (senza sticky actions, sezione 8). Nei 29 voli persi la previsione
    era sbagliata per qualche passo, perché la palla è passata da un buco o ha colpito un mattone dall'alto. La
    racchetta ha inseguito l'arrivo sbagliato, poi le sticky actions hanno tolto il resto del margine.
- **La rete non sceglie dove mandare la palla.** Quale delle due colonne di presa la racchetta usa, e quindi se la
  palla tiene o inverte la direzione orizzontale, è un effetto collaterale: i logit dipendono solo da Δ. Non c'è
  niente nei pesi che eviti i voli da un angolo all'altro, e senza mattoni nell'osservazione la rete non potrebbe
  nemmeno mirare.
- **Non vede i mattoni.** Mentre la palla sale, la previsione è giusta solo nel 42.6% degli stati. Come visto in
  sezione 7 sbaglia sempre nel verso del tempo in più: senza sticky actions non costa nessuna palla, con le sticky
  actions entra in 29 palle perse su 102.
- **Quasi tutta la rete è vuota.** 50 neuroni nascosti su 64 non hanno pesi. L'algoritmo intero sta in 14
  neuroni e 815 numeri, 400 dei quali sono la tabella delle traiettorie.
