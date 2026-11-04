# Come si legge la rete `paper_astra.json` di Seaquest

`weights/seaquest/paper_astra.json` è la rete che GPT-6 Astra ha scritto a mano per Seaquest (MinAtar)
nell'articolo [Weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/), senza training.
Qui fa 51.4 ± 1.4 punti su 1000 episodi (53.1 ± 1.4 nell'articolo); una policy casuale ne fa 0.04.

Questa analisi la decodifica neurone per neurone. Il risultato è un algoritmo molto semplice:

- **il sottomarino non si sposta mai in orizzontale.** Resta nella colonna 5, rivolto a sinistra, dove comincia la
  partita, e spara verso sinistra;
- **a ogni passo sceglie la riga.** Confronta tre righe, la sua, quella sopra e quella sotto, e va in quella con il
  valore più alto. Il valore premia i sub da salvare e i bersagli nella linea di tiro, e vieta le righe con un
  pericolo vicino alla colonna 5;
- **emerge** quando l'ossigeno scende sotto 60 e ha almeno un sub a bordo.

Funziona perché in Seaquest tutto si muove in orizzontale, lungo le righe: chi sta fermo in una colonna e sceglie
bene la riga vede passare davanti alla prua sia i nemici da colpire sia i sub da raccogliere.

La lettura è verificata contro la rete vera su 263.355 stati di gioco (vedi [Verifica](#8-verifica)):

```bash
uv run python experiments/seaquest/decode_astra/check.py               # ~1.5 min
uv run python experiments/seaquest/decode_astra/check.py --esempi      # le situazioni della sezione 6
uv run python experiments/seaquest/decode_astra/check.py --ablations   # famiglie di neuroni spente (sezione 9)
```

## 1. Il gioco in breve

È il Seaquest ufficiale di MinAtar (`minatar/environments/seaquest.py`, versione 1.0.15). La griglia è 10×10: la riga
0 è la superficie, le righe 1-8 sono il mare, la riga 9 contiene solo due indicatori.

- **Sottomarino:** occupa due celle della stessa riga, prua e poppa. Parte in riga 0, colonna 5, rivolto a sinistra
  (prua in colonna 5, poppa in colonna 6). Sinistra e destra lo spostano di una colonna e lo girano da quella parte;
  su e giù lo spostano di una riga (righe 0-8). Solo la prua conta per gli urti.
- **Fuoco:** un siluro parte dalla prua nella direzione in cui guarda il sottomarino e avanza di una cella per passo.
  Dopo ogni colpo servono 5 passi di ricarica; nel frattempo "fuoco" non fa niente, cioè il sottomarino resta fermo.
- **Nemici:** pesci e sottomarini compaiono al bordo sinistro o destro, in una riga 1-8 a caso, e attraversano lo
  schermo in orizzontale. All'inizio fanno un passo ogni 6, poi sempre più spesso (fino a uno ogni 3). I sottomarini
  nemici sparano ogni 11 passi siluri che avanzano di una cella per passo.
- **Punti:** +1 per ogni nemico colpito da un siluro.
- **Morte:** toccare un pesce, un sottomarino o un siluro nemico.
- **Sub da salvare:** ne compare uno ogni 31 passi, in una riga 1-8, e attraversa lo schermo (un passo ogni 6).
  Passandoci sopra lo si carica a bordo, fino a 6.
- **Ossigeno:** 200 passi. Cala di 1 a ogni passo sott'acqua; a 0 l'episodio finisce.
- **Emersione (riga 0):**
  - senza sub a bordo l'episodio finisce;
  - con 1-5 sub se ne perde uno e l'ossigeno torna pieno;
  - con 6 sub si perdono tutti, si guadagnano ⌊ossigeno/20⌋ punti (da 0 a 10) e l'ossigeno torna pieno.

  Ogni emersione alza la difficoltà: i nemici compaiono più spesso e si muovono più in fretta.
- **Sticky actions:** con probabilità 0.1 l'ambiente ripete l'azione precedente al posto di quella scelta.

Un dettaglio del codice di MinAtar che conta più avanti: dopo un'emersione con 6 sub il contatore scende a −1, non a
0, quindi il primo sub raccolto dopo lo riporta solo a 0.

## 2. Come si leggono le posizioni

L'osservazione ha 600 numeri in {0, 1}: la griglia 10×10 con 6 canali per cella.

```
indice = (riga*10 + colonna)*6 + canale          riga 0 in alto, colonna 0 a sinistra

canale 0  player      prua e poppa del sottomarino
canale 1  danger      pesci, sottomarini nemici, siluri nemici
canale 2  reward      pesci, sottomarini nemici, sub da salvare
canale 3  projectile  i tuoi siluri
canale 4  trail       la cella dietro a nemici e sub (dice in che direzione vanno), e la poppa
canale 5  gauge       riga 9: ossigeno e sub a bordo
```

Per esempio l'indice 330 = (5·10 + 5)·6 + 0 è il canale player della cella (riga 5, colonna 5), e 557 = (9·10 + 2)·6
+ 5 è il canale gauge della cella (9, 2). Scrivo le celle come (riga, colonna).

**I canali sono degli OR.** Un nemico accende sia danger sia reward; un sub da salvare solo reward; un siluro nemico
solo danger. La rete usa questa sovrapposizione per distinguere gli oggetti (sezione 4).

**La riga 9** contiene due barre nello stesso canale gauge:

- l'ossigeno, da sinistra: le colonne da 0 a ⌊ossigeno/20⌋ − 1;
- i sub a bordo, da destra: con n sub le colonne da 9 − n a 8.

La cella (9, 2) è accesa se e solo se l'ossigeno è almeno 60, perché la barra dei sub non arriva mai alla colonna 2.
La cella (9, 8) è accesa se c'è almeno un sub a bordo, oppure se l'ossigeno è almeno 180.

**Azioni:** 0 fermo, 1 sinistra, 2 su, 3 destra, 4 giù, 5 fuoco.

Due nomi che uso in tutto il documento:

- **r**, la riga del sottomarino;
- **[emergi]**, la condizione "ossigeno < 60 e almeno un sub a bordo", cioè è ora di tornare in superficie.

## 3. Struttura della rete

```
600 → 32 (tanh) → 32 (tanh) → 6 (linear)        20.486 parametri, 803 diversi da zero
```

| strato | cosa fa |
|---|---|
| 1 (tanh, 32) | **legge**: la riga del sottomarino, le righe con un sub da salvare, le righe con un pericolo vicino alla colonna 5, le righe con un bersaglio nella linea di tiro, se è ora di emergere |
| 2 (tanh, 32) | **valuta**: per la riga del sottomarino, il valore di restare, salire o scendere; 6 neuroni sono vuoti |
| 3 (linear, 6) | **sceglie**: fuoco = resta, su, giù; fermo, sinistra e destra valgono sempre −100 |

**La rete non si muove mai in orizzontale.** I logit di fermo, sinistra e destra hanno solo il bias −100, mentre su,
giù e fuoco valgono almeno −24. La softmax non sceglie mai sinistra o destra, e le sticky actions ripetono solo azioni
già eseguite (al primo passo, "fermo"). Quindi il sottomarino resta per tutto l'episodio nella colonna 5, rivolto a sinistra, come alla
partenza: in tutti i 263.355 stati della verifica. "Fuoco" fa da "resta qui": spara se il siluro è carico,
altrimenti non fa niente.

Un'idea da tenere a mente: con tanh un neurone saturo vale **+1 (vero)** o **−1 (falso)**, non 0. Nello strato dopo
i neuroni spenti contano −1 volte il loro peso, e i bias servono anche a compensare questo. Tutto il primo strato è
fatto di condizioni vere o false (con una eccezione, il pericolo vicino). Il secondo strato invece **non** è saturo:
lavora nella parte centrale di tanh, dove l'uscita cresce con l'ingresso, e confronta dei punteggi.

## 4. Primo strato: cosa legge (32 neuroni)

| neuroni | ingressi | domanda | valore |
|---|---|---|---|
| h0-h8 | player, riga r, peso 6, bias −6 | il sottomarino è nella riga r = 0..8? | ±1 |
| h9-h15 | reward +8 e danger −8 su tutta la riga j = 2..8, bias −4 | nella riga j c'è un sub da salvare? | ±1 |
| h16-h23 | danger e reward nelle colonne 2-8 della riga j = 1..8, pesi a campana, bias −3 | nella riga j c'è un pericolo vicino alla colonna 5? | ±1, a volte intermedio |
| h24 | gauge (9, 2) peso −6, (9, 8) peso +6, bias −3 | [emergi]: ossigeno < 60 e almeno un sub a bordo? | ±1 |
| h25-h31 | danger nelle colonne 0-3 della riga j = 2..8, peso 6, bias −3 | nella riga j c'è qualcosa nella linea di tiro? | ±1 |

### h0-h8: la riga del sottomarino

h5 ha peso 6 sugli indici 300, 306, ..., 354, cioè il canale player di tutta la riga 5, e bias −6.

- **Sottomarino in riga 5:** prua e poppa accendono due celle, 6 + 6 − 6 = 6, tanh = +1.
- **Altrove:** −6, tanh = −1.

È la riga del sottomarino in codice "one-hot" (±1). La riga 9 manca: il sottomarino non ci va mai.

### h9-h15: sub da salvare, con una sottrazione tra canali

h12 guarda la riga 5: peso +8 sul canale reward (indici 302, 308, ..., 356) e −8 sul canale danger (301, 307,
..., 355) di tutte le 10 celle, bias −4. Per ogni cella conta reward − danger:

| oggetto nella cella | danger | reward | contributo |
|---|---|---|---|
| pesce o sottomarino nemico | 1 | 1 | 0 |
| sub da salvare | 0 | 1 | **+8** |
| siluro nemico | 1 | 0 | −8 |

Il canale reward mescola nemici e sub, ma sottraendo danger i nemici si cancellano e restano solo i sub. Con un sub
nella riga la pre-attivazione è 8 − 4 = 4 (tanh = +1), senza è −4 (−1). Un effetto collaterale: un siluro nemico
nella stessa riga cancella il sub. La riga 1 non ha un neurone di questo tipo, quindi i sub in riga 1 sono invisibili.

### h16-h23: pericolo vicino, con una campana di pesi

h20 guarda la riga 5, solo le colonne 2-8. Bias −3.

| colonna | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| peso danger (indice) | 2 (313) | 4 (319) | 7 (325) | 7 (331) | 7 (337) | 5 (343) | 2 (349) |
| peso reward (indice) | −1.8 (314) | −3.6 (320) | −1 (326) | 0 | −1 (338) | −3 (344) | −1.8 (350) |
| **nemico** (danger + reward) | 0.2 | 0.4 | **6** | **7** | **6** | 2 | 0.2 |
| **siluro nemico** (solo danger) | 2 | **4** | **7** | **7** | **7** | **5** | 2 |
| **sub** (solo reward) | −1.8 | −3.6 | −1 | 0 | −1 | −3 | −1.8 |

Il neurone vale tanh(somma − 3). Con un solo oggetto nella riga:

- **acceso (+1):** un nemico nelle colonne 4-6, cioè davanti alla prua, sulla prua o sulla poppa, oppure un siluro
  nemico nelle colonne 3-7 (in colonna 3 tanh 1 = 0.76, in colonna 7 tanh 2 = 0.96);
- **mezzo acceso:** un nemico in colonna 7, o un siluro in colonna 2 o 8: tanh(−1) = −0.76, cioè il 12% della
  strada da −1 a +1. Nel secondo strato basta a cambiare la scelta (sezione 6);
- **spento:** un nemico nelle colonne 2, 3 e 8 (0.2-0.4) e tutto quello che sta nelle colonne 0, 1 e 9.

La campana si legge come una regola sensata per il gioco. Un nemico fa un passo ogni 3-6, quindi è pericoloso solo
quando è a una cella dalla prua. Un siluro nemico fa un passo per turno, quindi è pericoloso anche a due celle. Un
nemico in colonna 3 è nella linea di tiro e può essere colpito, quindi conta poco (0.4); uno in colonna 7 è dietro
la poppa, dove i siluri non arrivano, e conta di più (2).

I pesi negativi sul canale reward servono a distinguere nemici e siluri, ma hanno un effetto collaterale: **un sub
nella riga abbassa il pericolo.** Per esempio, un sub in colonna 3 (−3.6) e un nemico in colonna 6 (+6) danno
6 − 3.6 − 3 = −0.6: il neurone resta quasi spento, e la riga sembra sicura.

### h24: è ora di emergere

Peso −6 sull'indice 557, la cella (9, 2), e +6 sul 593, la cella (9, 8). Bias −3.

| ossigeno | sub a bordo | (9, 2) | (9, 8) | pre-attivazione | h24 |
|---|---|---|---|---|---|
| < 60 | ≥ 1 | 0 | 1 | +3 | **+1** |
| < 60 | 0 (o −1) | 0 | 0 | −3 | −1 |
| 60-179 | ≥ 1 | 1 | 1 | −3 | −1 |
| 60-179 | 0 | 1 | 0 | −9 | −1 |
| ≥ 180 | qualsiasi | 1 | 1 | −3 | −1 |

Due celle della barra bastano: la (9, 2) dice "ossigeno ≥ 60", la (9, 8) "almeno un sub a bordo". Con l'ossigeno
sotto 60 la seconda non può essere accesa dall'ossigeno, quindi l'ambiguità della barra condivisa sparisce. h24 è
[emergi].

### h25-h31: bersagli nella linea di tiro

h28 ha peso 6 sul canale danger delle celle (5, 0), (5, 1), (5, 2), (5, 3), cioè gli indici 301, 307, 313, 319,
e bias −3. Vale +1 se in quelle celle c'è almeno un nemico o un siluro nemico. Sono le colonne che un siluro sparato
dalla prua in colonna 5 attraversa: un nemico lì è un bersaglio.

### Cosa il primo strato non guarda

- **I canali projectile e trail:** nessun peso. La rete non sa in che direzione vanno nemici e sub (MinAtar lo
  mostra solo con la scia) e non sa se un suo siluro è già in volo.
- **La riga 1** per sub e bersagli, **la riga 0** del tutto (lì non ci sono mai nemici né sub).
- **La barra dell'ossigeno:** solo 2 celle su 10.
- **Le colonne 0, 1 e 9** per il pericolo vicino: un nemico che arriva da destra diventa visibile in colonna 7.

## 5. Secondo strato: il valore delle righe (32 neuroni, 26 usati)

23 dei 32 neuroni seguono **un solo schema**. Ognuno appartiene a una riga ρ del sottomarino e a una mossa, e guarda
la riga in cui la mossa porta il sottomarino, la **riga di arrivo** k:

| neuroni | mossa | riga del sottomarino ρ | riga di arrivo k |
|---|---|---|---|
| g0-g7 | fuoco (resta) | 1..8 | ρ |
| g8-g14 | su | 2..8 | ρ − 1 |
| g16-g23 | giù | 0..7 | ρ + 1 |

```
pre = 10·h(riga ρ)                                  il cancello: il sottomarino è nella riga ρ?
    + Σ_j a(|j − k|) · h(sub nella riga j)          j = 2..8
    + Σ_j c(|j − k|) · h(bersaglio nella riga j)    j = 2..8
    − 3 · h(pericolo nella riga k)
    − 0.3 · h(pericolo nella riga dopo k)           solo su e giù: k − 1 per su, k + 1 per giù
    + bias

a(d) = 0.6, 0.45, 0.34, 0.255, 0.19, 0.145, 0.11, 0.08     (≈ 0.6 · 0.75^d)
c(d) = 0.35, 0.12, 0.04, 0.01                               (≈ 0.35 · 3^−d)
```

Per esempio g4 (fuoco con il sottomarino in riga 5) ha peso 10 su h5, 0.6 su h12 (sub in riga 5), 0.45 su h11 e
h13 (righe 4 e 6), e così via fino a 0.255 su h9 e h15 (righe 2 e 8); 0.35 su h28 (bersaglio in riga 5), 0.12 su
h27 e h29; −3 su h20 (pericolo in riga 5); bias −10.32.

**Il cancello.** Se il sottomarino non è nella riga ρ, il termine 10·h(riga ρ) vale −10 invece di +10 e la
pre-attivazione scende sotto −12: il neurone è fermo a −1, qualsiasi cosa ci sia sullo schermo. In ogni stato, dei
23 neuroni ne sono vivi al massimo tre, quelli della riga del sottomarino. È una tabella indicizzata dalla riga, costruita con
un AND.

**Il bias.** In tutti e 23 i neuroni il bias è scelto in modo che, con il cancello aperto e tutto il resto spento
(−1), la pre-attivazione valga esattamente **−0.70**. Per g4: 10 − (somma dei pesi a = 2.69) − (somma dei pesi
c = 0.69) + 3 (il pericolo spento conta −3 · −1) − 10.32 = −0.70. I bias diversi da neurone a neurone (−11.96,
−11.09, −10.63, ...) compensano solo il fatto che, vicino ai bordi, la somma dei pesi a e c è più piccola.

**Il valore di una riga.** Riscrivendo con [x] = 1 se vero e 0 se falso (h = 2[x] − 1), il neurone vivo vale
tanh(V), con

```
V(k) = −0.70 + Σ_sub 1.2·0.75^d + Σ_bersagli 0.7·3^−d − 6·[pericolo nella riga k]
```

dove d è la distanza in righe tra il sub (o il bersaglio) e la riga di arrivo k.

- **Sub:** un sub nella riga di arrivo vale +1.2, a una riga +0.9, a due +0.68, e così via fino a 0.16 a sette
  righe. È un'attrazione che cala con la distanza: tutti i sub dello schermo tirano, i vicini di più.
- **Bersagli:** +0.7 nella riga, +0.24 a una riga, +0.08 a due. Attirano solo da vicino.
- **Pericolo:** −6 manda tanh(V) a −1 qualunque siano le attrazioni. È un veto.
- **Pericolo nella riga dopo** (solo su e giù): −0.6. Una piccola cautela: muoversi verso una riga accanto a un
  pericolo costa un po'.

Questi neuroni **non sono saturi**: tanh(−0.70) = −0.60, tanh(0.5) = 0.46. Tanh qui fa solo da compressore:
appiattisce i valori molto alti (molti sub) e quelli molto bassi (pericolo).

### Il ritorno in superficie

- **g8-g14 (su):** hanno anche un peso su h24. Per g9-g14 è 0.75, cioè +1.5 a V quando [emergi]. Per g8 (su dalla
  riga 2 alla riga 1) è 4.75 e il bias è più basso: V vale −8.7 senza [emergi] e +0.8 con. Senza bisogno di
  emergere, la rete non sale mai in riga 1.
- **g15** = tanh(6·h1 + 6·h24 − 6): sottomarino in riga 1 **e** [emergi].
- **g24** = tanh(3·h1 + 3·h2 − 3·h24): sottomarino in riga 1 o 2 **e non** [emergi].
- **g25** = tanh(3·h1 − 3·h24 − 3): sottomarino in riga 1 **e non** [emergi].
- **g26-g31:** nessun peso e nessun bias. Valgono 0 e non contano.

## 6. Ultimo strato: la regola (6 logit)

```
fermo = sx = dx = −100
fuoco = 241.8 + 30·Σ g0..g7
su    = 238   + 30·Σ g8..g14 + 40·g15 − 12·g24
giù   = 244   + 30·Σ g16..g23 + 4·g25
```

I bias sono grandi perché **i neuroni spenti valgono −1**. Con il sottomarino in riga r, 7 degli 8 neuroni "fuoco"
sono spenti e danno −210, che porta 241.8 a 31.8. Lo stesso vale per su (6 spenti su 7: 238 − 180 = 58) e giù (7 su
8: 244 − 210 = 34). Restano:

| riga r | fuoco (resta) | su | giù |
|---|---|---|---|
| 0 | 1.8 | 0 | 30 + 30·tanh(D) |
| 1 | 31.8 + 30·tanh(F) | **80** se [emergi], altrimenti −24 | 38 + 30·tanh(D) (30 + … se [emergi]) |
| 2 | 31.8 + 30·tanh(F) | 30 + 30·tanh(U) se [emergi], altrimenti **−24** | 30 + 30·tanh(D) |
| 3-7 | 31.8 + 30·tanh(F) | 30 + 30·tanh(U) | 30 + 30·tanh(D) |
| 8 | 31.8 + 30·tanh(F) | 30 + 30·tanh(U) | 0 |

con

```
F = V(r)                                              restare
U = V(r − 1) − 0.6·[pericolo in r − 2] + 1.5·[emergi]   salire
D = V(r + 1) − 0.6·[pericolo in r + 2]                  scendere
```

Quindi le tre mosse hanno la stessa forma, 30 + 30·tanh(valore della riga di arrivo), e **vince la riga con il
valore più alto**. Restare ha 1.8 in più e vince i pareggi. Le eccezioni sono ai bordi:

- **riga 0:** fuoco e su valgono 1.8 e 0, giù fino a 60. Si immerge, a meno che la riga 1 non abbia un pericolo;
- **riga 1:** su vale 80 se [emergi] e batte tutto; altrimenti −24, perché emergere senza sub è la morte. Giù ha 6.2
  in più per riportare il sottomarino in basso;
- **riga 2:** salire è vietato se non c'è bisogno di emergere;
- **riga 8:** non può scendere.

La scala 30 rende la scelta quasi deterministica quando i valori sono diversi, e una lotteria quando sono uguali.
Ecco i logit della rete vera in alcune situazioni costruite a mano (`check.py --esempi`; sottomarino in riga 5 se non
detto, oggetti come (riga, colonna)):

| situazione | su | giù | fuoco | p(su, giù, fuoco) |
|---|---|---|---|---|
| mare vuoto | 11.7 | 11.7 | 13.5 | 0.12, 0.12, **0.75** |
| sub nella mia riga, (5, 1) | 35.6 | 35.6 | 45.4 | 0, 0, **1** |
| sub una riga sotto, (6, 1) | 29.1 | 43.6 | 37.4 | 0, **1**, 0 |
| sub tre righe sotto, (8, 1) | 20.4 | 29.1 | 25.9 | 0, **0.96**, 0.04 |
| bersaglio nella mia riga, (5, 2) | 16.8 | 16.8 | 31.2 | 0, 0, **1** |
| bersaglio una riga sopra, (4, 2) | 29.4 | 13.2 | 18.6 | **1**, 0, 0 |
| nemico sulla poppa, (5, 6) | 11.7 | 11.7 | 1.8 | **0.5, 0.5**, 0 |
| nemico in (5, 7), mezzo acceso | 11.7 | 11.7 | 5.2 | **0.5, 0.5**, 0 |
| siluro nemico in (5, 3) | 16.8 | 16.8 | 1.8 | **0.5, 0.5**, 0 |
| nemico una riga sotto, (6, 5) | 11.7 | 0.0 | 13.5 | 0.14, 0, **0.86** |
| nemico due righe sotto, (7, 5) | 11.7 | 4.1 | 13.5 | 0.14, 0, **0.86** |
| nemico (6, 6) "coperto" da sub in (6, 3) e (6, 8) | 29.1 | **42.8** | 37.4 | 0, **1**, 0 |
| sub (7, 1), bersaglio (4, 2), nemico (6, 6) | **43.6** | 0.0 | 38.0 | **1**, 0, 0 |
| ossigeno 50, 2 sub a bordo | **49.7** | 11.7 | 13.5 | **1**, 0, 0 |
| ossigeno 50, nessun sub a bordo | 11.7 | 11.7 | 13.5 | 0.12, 0.12, **0.75** |
| ossigeno 50, 2 sub, nemico in (4, 5) | 0.0 | 11.7 | 13.5 | 0, 0.14, **0.86** |
| riga 2 | −23.9 | 11.7 | 13.5 | 0, 0.14, **0.86** |
| riga 1 | −23.9 | **19.6** | 13.4 | 0, **1**, 0 |
| riga 1, ossigeno 50, 2 sub | **79.9** | 11.7 | 13.4 | **1**, 0, 0 |
| riga 0 | −0.1 | **11.6** | 1.8 | 0, **1**, 0 |
| riga 0, nemico in (1, 5) | −0.1 | 0.0 | 1.8 | **0.12, 0.13**, 0.75 |

Le formule, con i valori veri dei neuroni del primo strato, riproducono i logit della rete entro 0.063 (sezione 8).
I numeri qui sopra sono un po' più bassi di quelli delle formule con ingressi esattamente ±1 (13.5 invece di 13.7
nel mare vuoto), perché un neurone "spento" con pre-attivazione −3 vale tanh(−3) = −0.995 e non −1.

## 7. L'algoritmo

```
il sottomarino resta sempre in colonna 5, rivolto a sinistra; "fuoco" vuol dire "resto qui e sparo"

valore(riga k) = Σ sub nelle righe 2-8                           1.2 · 0.75^(distanza in righe da k)
               + Σ bersagli nelle colonne 0-3 delle righe 2-8    0.7 · 3^−(distanza da k), fino a 3 righe
               − 6   se nella riga k c'è un pericolo vicino alla prua
                     (nemico nelle colonne 4-6, siluro nemico nelle colonne 3-7)
               − 0.7 se c'è un pericolo "mezzo" (nemico in colonna 7, siluro in colonna 2 o 8)

in superficie (riga 0):   giù, a meno che la riga 1 non abbia un pericolo
                          (allora aspetta, ma con fuoco 1.8, su 0, giù 0: è una lotteria, sezione 9)
se ossigeno < 60 e c'è almeno un sub a bordo:
    in riga 1: su (emerge)
    altrove:   "su" ha +1.5 di valore
scegli tra:
    resta e spara    valore(r)
    su               valore(r − 1) − 0.6 se c'è un pericolo in r − 2      (mai sopra la riga 2,
                                                                          se non per emergere)
    giù              valore(r + 1) − 0.6 se c'è un pericolo in r + 2      (non dalla riga 8)
la mossa con il valore più alto; a parità (entro 0.1) resta e spara
```

La strategia segue le regole del gioco:

- **Tutto scorre in orizzontale.** Un sub nella riga del sottomarino prima o poi arriva in colonna 5 e viene
  raccolto, a meno che non stia già andando via. Un nemico nella riga del sottomarino, a sinistra, prima o poi
  incontra un siluro. Basta scegliere la riga.
- **I pericoli sono un veto,** le attrazioni un punteggio graduato.
- **L'ossigeno** detta il ritorno in superficie, e la rete ci torna solo se ha un sub, cioè solo se emergere non è
  fatale.

## 8. Verifica

`check.py` gioca gli stessi 200 episodi di `uv run evaluate seaquest paper_astra --episodes 200` (seed 10000-10199,
sticky 0.1, azioni campionate dalla softmax con lo stesso generatore). In ogni stato ricalcola, **direttamente dallo
stato del gioco e senza usare i pesi**:

- i 32 neuroni del primo strato, con le formule della sezione 4;
- i 6 logit, con le formule della sezione 6;
- l'azione della regola semplificata della sezione 7, fatta solo di condizioni vere o false;

e li confronta con il forward pass vero.

| controllo | risultato |
|---|---|
| stati visitati | 263.355 (punteggio 48.3 ± 2.9, lo stesso di `evaluate`) |
| sottomarino in colonna 5, rivolto a sinistra | **100%** degli stati |
| primo strato: stati in cui la ricostruzione differisce di più di 10⁻⁹ | **0** |
| logit: differenza massima | 0.063 (alcuni neuroni del secondo strato sono presi come ±1 esatti) |
| mossa con logit più alto diversa, esclusi i quasi-pareggi (< 0.5) | **0** (quasi-pareggi: 4.238) |
| regola semplificata della sezione 7: stessa mossa della rete | **97.4%** degli stati; **99.1%** dove la rete ha un margine > 3 (76% degli stati) |

Le differenze della regola semplificata vengono da due semplificazioni: il pericolo vicino trattato come acceso,
mezzo acceso o spento (nella rete i contributi si sommano, sub compresi), e il confronto tra valori al posto di
quello tra tanh, che appiattisce i valori alti.

Quale mossa ha il logit più alto, per riga del sottomarino:

| riga | % degli stati | su | giù | fuoco |
|---|---|---|---|---|
| 0 | 0.9% | | 86% | 14% |
| 1 | 1.9% | 32% | 40% | 28% |
| 2 | 13.6% | 5% | 10% | 85% |
| 3 | 15.8% | 9% | 10% | 81% |
| 4 | 16.2% | 10% | 9% | 80% |
| 5 | 15.9% | 10% | 9% | 81% |
| 6 | 13.6% | 10% | 8% | 82% |
| 7 | 11.8% | 10% | 6% | 84% |
| 8 | 10.4% | 8% | | 92% |

Il sottomarino passa il 97% del tempo nelle righe 2-8, distribuito in modo quasi uniforme. In riga 1 ci arriva solo
per emergere, o per una sticky action; in riga 2 "su" vince solo con [emergi].

## 9. Limiti visibili nei pesi

Nei 200 episodi della verifica la rete fa 48.3 punti a episodio: 42.9 dai nemici colpiti e 5.4 dalle emersioni con
6 sub. Ogni episodio finisce con una morte (nessuno arriva al limite di 10.000 passi), in media dopo 1.317 passi.
`check.py` classifica le 200 morti così:

| come muore | morti | |
|---|---|---|
| **sticky:** una ripetizione di su o giù l'ha portata in una riga il cui pericolo era già acceso un passo prima | 115 | 57% |
| **tuffo dalla superficie:** in riga 0, con un pericolo in riga 1, la softmax sceglie "giù" | 23 | 12% |
| **in trappola:** pericolo nella riga in cui finisce e nessuna mossa sicura (nessun logit sopra 10) | 16 | 8% |
| **sticky:** spostato in una riga il cui pericolo non era ancora acceso (12 sono siluri nemici da sinistra) | 15 | 8% |
| ossigeno finito | 11 | 6% |
| **sticky:** tenuto fermo mentre la rete voleva scappare | 8 | 4% |
| **pericolo non acceso perché c'era un sub nella stessa riga** | 6 | 3% |
| **sticky:** ripetuto "su" fino alla superficie, senza sub a bordo | 6 | 3% |

- **Il pericolo è un veto solo per la riga di arrivo.** Muoversi verso una riga *accanto* a un pericolo costa
  appena 0.6 di valore, meno di un sub a due righe (0.68). Quindi la rete non evita di mettersi a una riga da un
  nemico. Con le sticky actions, la mossa successiva ripete con probabilità 0.1 quella precedente, e la porta addosso
  al nemico. Le sticky actions sono coinvolte nel 72% delle morti (144 su 200). Senza sticky actions
  (`uv run evaluate seaquest paper_astra --sticky 0`, gli stessi 200 seed) la rete fa **110.1 ± 2.9** punti invece
  di 48.3 ± 2.9.
- **In superficie tira a sorte.** In riga 0 i logit sono fuoco 1.8, su 0 e giù fino a 60. Se la riga 1 ha un
  pericolo, giù scende a 0 e le tre mosse finiscono quasi alla pari: a ogni passo di attesa la rete si tuffa sul
  pericolo con probabilità 0.13 (tabella della sezione 6, ultima riga). Nelle altre righe questo non succede, perché
  lì una mossa sicura vale almeno 11.7. Scegliendo sempre la mossa con il logit più alto invece di campionare
  (`uv run evaluate seaquest paper_astra --greedy`, 200 episodi) la rete fa 53.9 ± 3.1.
- **Il suo mondo sono le righe 2-8 della colonna 5.** In riga 2 non può salire (la riga 1 è vietata se non deve
  emergere), in riga 8 non può scendere, e non si sposta mai di lato. Se il pericolo arriva nella sua riga e
  nell'unica riga accanto, non ha vie di fuga: delle 16 morti "in trappola", 14 sono nelle righe 1, 2 e 8.
- **Spara solo a sinistra.** I nemici che arrivano da destra non si possono colpire, solo evitare, e sono quelli
  che la uccidono più spesso: 86 morti, contro 60 per i nemici da sinistra (il resto sono siluri nemici). Il
  rilevatore di pericolo li vede solo dalla colonna 7.
- **I sub coprono i nemici.** I pesi negativi sul canale reward in h16-h23 abbassano il pericolo quando nella riga
  c'è anche un sub. Una riga con un sub e un nemico in colonna 6 sembra sicura e attira (tabella della sezione 6:
  con due sub nella riga la rete scende sul nemico con probabilità 1). 6 morti.
- **Non legge la scia, quindi non sa dove vanno le cose.** Un sub che ha già passato la colonna 5 e si allontana
  attira quanto uno che si avvicina: in media il 20% dei sub visibili. Raccoglie 27.7 dei 43.4 sub che compaiono in
  un episodio.
- **Emerge tardi.** Torna in superficie solo con l'ossigeno sotto 60, quindi un'emersione con 6 sub vale al massimo
  ⌊59/20⌋ = 2 punti invece di 10. Su 1.645 emersioni, 562 sono con 6 sub, e danno 5.4 punti a episodio.
- **Senza sub a bordo non c'è un piano B.** Con l'ossigeno sotto 60 e nessun sub, h24 resta spento e la rete non
  emerge, il che è giusto perché emergere sarebbe fatale. Però non cerca i sub con più urgenza: l'attrazione resta la
  stessa. 11 morti per ossigeno.
- **Capacità inutilizzata:** 6 neuroni del secondo strato (g26-g31) sono vuoti, 3 uscite su 6 sono costanti, i
  canali projectile e trail non hanno nessun peso, e la riga 1 è invisibile per sub e bersagli.

### Quanto conta ogni famiglia di neuroni

`check.py --ablations` rigioca gli stessi 200 episodi con una famiglia del primo strato fissata a −1 (spenta), cioè
come se quella cosa non fosse mai sullo schermo:

| variante | punteggio | passi per episodio | morti per ossigeno |
|---|---|---|---|
| rete intera | 48.3 ± 2.9 | 1.317 | 6% |
| senza sub da salvare (h9-h15) | 39.3 ± 2.7 | 1.011 | 49% |
| senza bersagli nella linea di tiro (h25-h31) | 24.8 ± 1.6 | 1.432 | 4% |
| senza pericolo vicino (h16-h23) | 3.2 ± 0.3 | 147 | 0% |
| senza [emergi] (h24) | 4.3 ± 0.1 | 199 | 96% |
| senza sub, bersagli e pericolo | 1.3 ± 0.1 | 123 | 2% |

- **Pericolo ed emersione sono indispensabili.** Senza il primo la rete muore in media dopo 147 passi; senza il
  secondo resta sott'acqua fino all'ultimo ossigeno.
- **Senza i bersagli il punteggio si dimezza.** La rete vive anche un po' di più, ma colpisce solo quello che le
  passa davanti per caso.
- **I sub servono soprattutto per l'ossigeno.** Senza attrazione verso i sub, quando l'ossigeno finisce spesso non
  ne ha a bordo, e metà degli episodi finisce così.

Tutto il comportamento della rete è spiegato dalle formule delle sezioni 4-6, che coincidono con la rete vera in
ogni stato visitato. La regola semplificata della sezione 7 sbaglia la mossa nel 2.6% degli stati. Gli scarti
vengono dai casi in cui contano i dettagli che la regola arrotonda: più oggetti nella stessa riga che si sommano nel
rilevatore di pericolo, e valori alti che tanh appiattisce.
