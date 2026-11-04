# Come si legge la rete `claude.json` di Low Tide

`weights/low_tide/claude.json` è la rete scritta a mano da Claude Opus per Low Tide, senza training e senza
eseguire codice. Fa 132.3 ± 0.9 punti su 500 episodi; l'euristica (BFS) ne fa 164.0, una policy casuale 2.0.

Questa analisi la decodifica neurone per neurone. Il risultato è che la rete **non è una scatola nera**: è un
algoritmo a regole scritto in forma di pesi. Il primo strato misura poche grandezze del gioco, il secondo le
trasforma in condizioni vere o false, l'ultimo le combina in una regola di priorità tra le mosse.

La lettura è verificata contro la rete vera su 197.920 stati di gioco (vedi [Verifica](#7-verifica)):

```bash
uv run python experiments/decode_claude_low_tide/check.py            # ~1 min
uv run python experiments/decode_claude_low_tide/check.py --episodes 500
```

## 1. Il gioco in breve

La griglia è 10×10 ed è una spiaggia. La marea sale e scende di una riga ogni 3-5 passi, tra un livello basso
(0-2) e uno alto (5-7).

- **Marea che scende:** ogni riga scoperta si riempie di conchiglie (25% per cella) e meduse (8% per cella).
- **Marea che sale:** quello che c'è sulla riga sommersa viene cancellato.
- **Conchiglia:** +1.
- **Medusa:** −1 e 3 passi di stordimento, durante i quali le azioni sono ignorate.
- **Acqua:** l'agente non può entrarci. Se la marea sommerge la riga su cui si trova, annega e l'episodio
  finisce.
- **Sticky actions:** con probabilità 0.1 l'ambiente ripete l'azione precedente al posto di quella scelta.

Le regole complete sono in `src/small_games/games/low_tide/env.py`.

## 2. Come si leggono le posizioni

L'osservazione ha 403 numeri: quattro piani 10×10 binari, poi tre scalari.

```
indice = canale*100 + riga*10 + colonna

agente     0..99      (esattamente una cella vale 1)
acqua      100..199   (le righe da 10-L a 9 valgono 1, L = livello della marea)
conchiglie 200..299
meduse     300..399
400  la marea sta salendo (1) o scendendo (0)
401  countdown: passi al prossimo cambio di livello / 5
402  stordimento: passi rimasti / 3
```

**La riga 0 è in alto (terra), la riga 9 in basso (mare).** Dentro ogni piano gli indici 0-9 sono la riga 0, gli
indici 90-99 la riga 9. Vale uguale per i quattro piani: l'indice 245 è la conchiglia in riga 4, colonna 5.

```
           col 0 1 2 3 4 5 6 7 8 9
 riga 0  ┐                              terra: sempre asciutta
 riga 1  │ righe 0-2: mai acqua,        l'agente parte in riga 1
 riga 2  ┘ mai conchiglie né meduse
 riga 3  ┐
 ...     │ zona di marea: conchiglie e meduse compaiono qui
 riga 9  ┘                              mare: la prima riga che si bagna
```

- **Acqua:** si riempie dal basso. Con livello L sono bagnate le righe da `10-L` a 9. Il livello non supera mai
  7, quindi le righe 0-2 restano asciutte.
- **Azioni:** 0 fermo, 1 su (riga−1, verso terra), 2 giù (riga+1, verso il mare), 3 sinistra, 4 destra.

Due nomi che uso in tutto il documento:

- **battigia:** l'ultima riga asciutta, `9 − L`, quella subito sopra l'acqua;
- **m (margine):** le righe asciutte tra l'agente e l'acqua, `m = 9 − riga − L`. Con m = 0 l'agente è sulla
  battigia; se in quel momento la marea sale di una riga, annega.

## 3. Struttura della rete

```
403 → 46 (relu) → 36 (tanh) → 5 (linear)        20.461 parametri, 953 diversi da zero
```

| strato | cosa fa |
|---|---|
| 1 (relu, 46) | **misura**: margine, direzione della marea, conchiglie in riga e colonna, colonna dell'agente, meduse sulla battigia |
| 2 (tanh, 36) | **condizioni**: ogni neurone è saturo, +1 vuol dire vero e −1 falso; gli AND si fanno con pesi enormi |
| 3 (linear, 5) | **regola**: ogni logit è una somma pesata di condizioni; vince la mossa con la priorità più alta |

Un'idea da tenere a mente: nel primo strato un peso non "premia" o "punisce" una cella. Serve a **misurare** o a
**fare una domanda**. Il peso dice quanto conta l'ingresso nella somma, il bias fa da soglia, la relu risponde
"sì, e quanto" se la somma supera la soglia, altrimenti 0. Il segno negativo serve solo a sottrarre. Chi decide se
una cosa è buona o cattiva è l'ultimo strato.

## 4. Primo strato: cosa misura (46 neuroni, relu)

| neuroni | domanda | valore |
|---|---|---|
| h0 | quante righe asciutte ho tra me e l'acqua? | m, da 0 a 9 |
| h1 | la marea sale? | 0 o 1 |
| h2-h11 | c'è una conchiglia **sopra** di me, nella mia colonna? (uno per colonna 0-9) | ≥ 1 se sì, 0 se no |
| h12-h18 | c'è una conchiglia **a sinistra**, nella mia riga? (uno per riga 3-9) | ≥ 1 se sì, 0 se no |
| h19-h25 | c'è una conchiglia **a destra**, nella mia riga? (uno per riga 3-9) | ≥ 1 se sì, 0 se no |
| h26-h35 | sono nella colonna c? (one-hot) | 0 o 1 |
| h36-h45 | c'è una medusa **sulla battigia** nella colonna k? | ≥ 1 se sì, 0 se no |

### h0: il margine, con un righello

```
acqua, colonna 0, righe 3..9          peso −1    → somma = −L  (conta le righe bagnate)
agente in riga r, qualsiasi colonna   peso −r    → somma = −r  (tutta la riga 4 pesa −4, tutta la riga 7 −7)
bias 9
h0 = 9 − r − L = m
```

- **Pesi sull'agente:** sono un righello. Non importa in che cella sia, importa solo la riga.
- **Pesi sull'acqua:** basta la colonna 0, perché l'acqua è uguale su tutte le colonne. Bastano le righe 3..9,
  perché il livello non supera 7.

Esempio: agente in riga 4, livello 3 (acqua nelle righe 7, 8, 9). h0 = 9 − 4 − 3 = 2, e infatti le righe 5 e 6
sono asciutte. L'agente non entra mai in acqua, quindi h0 non è mai negativo e la relu qui non interviene.

### h1: la direzione della marea

Ha peso 1 sull'ingresso 400 e bias 0: lo copia.

### h2-h11: conchiglia sopra, con un confronto in binario

Prendiamo h2, che guarda la colonna 0 (h3 la colonna 1, e così via):

```
conchiglie, righe 3, 4, 5, ..., 9      pesi 64, 32, 16, 8, 4, 2, 1          (2^(9−riga))
agente,     righe 4, 5, 6, 7, 8, 9     pesi 193, 225, 241, 249, 253, 255    (256 − (2^(10−riga) − 1))
bias −256
```

Con l'agente in riga r della stessa colonna, peso dell'agente e bias insieme valgono `−(2^(10−r) − 1)`, cioè meno
la somma dei pesi di tutte le righe dalla r alla 9. Quindi

`pre-attivazione = (pesi delle conchiglie presenti) − (somma dei pesi delle righe da r in giù)`.

Una potenza di 2 è più grande di tutte le potenze più piccole messe insieme. Così basta **una** conchiglia sopra
l'agente (righe 3..r−1) per battere tutte quelle sotto messe insieme: il neurone si accende (≥ 1) se e solo se c'è
una conchiglia sopra l'agente, nella sua colonna.

Il neurone resta a 0 in due casi:

- **l'agente è in un'altra colonna, o in una riga ≤ 3:** manca il suo peso, e il bias −256 supera qualsiasi somma
  di conchiglie (al massimo 127);
- **non ci sono conchiglie sopra l'agente:** la somma non arriva alla soglia.

Un solo neurone fa quindi un AND tra "sono in questa colonna" e "c'è una conchiglia più in alto di me".

### h12-h18 e h19-h25: conchiglia a sinistra o a destra

È lo stesso trucco, ruotato sulle colonne, con un neurone per ogni riga da 3 a 9 (h12 e h19 la riga 3, h18 e h25
la riga 9). Il bias è −1024.

- **h12-h18 (a sinistra):** conchiglia in colonna k con peso `2^(9−k)` (512 a sinistra, 1 a destra); agente in
  colonna j (1..9) con peso `1024 − (2^(10−j) − 1)`. Si accende se c'è una conchiglia in una colonna < j.
- **h19-h25 (a destra):** conchiglia in colonna k con peso `2^k` (1 a sinistra, 512 a destra); agente in colonna
  j (0..8) con peso `1024 − (2^(j+1) − 1)`. Si accende se c'è una conchiglia in una colonna > j.

### h26-h35: la colonna dell'agente

Peso +1 su tutte le 10 celle della colonna c del piano agente, bias 0. Vale 1 se l'agente è in quella colonna, 0
altrimenti.

### h36-h45: medusa sulla battigia, con la soglia costruita dall'acqua

Per la colonna k (h36 è la colonna 0):

```
meduse, righe 3..9   pesi 8, 16, 32, ..., 512    (2^riga)
acqua,  righe 3..9   pesi 4, 8, 16, ..., 256     (2^(riga−1))
bias −511
```

Con livello L le righe bagnate sono `10−L..9` e la loro somma vale `512 − 2^(9−L)`. Quindi

`pre-attivazione = (meduse) + 1 − 2^(9−L)`.

- **Medusa sulla battigia:** è in riga `9−L`, pesa esattamente `2^(9−L)` e accende il neurone.
- **Meduse più in alto:** sommate insieme valgono al massimo `2^(9−L) − 8` e non bastano.

L'acqua sposta la soglia insieme alla battigia, e il neurone vede **solo** le meduse sulla battigia. Con L = 0 la
soglia è 511 e basta una medusa in riga 9 (peso 512). Con L = 7 la battigia è la riga 2, dove non ci sono mai
meduse, e il neurone resta spento.

### Cosa il primo strato non guarda

- **Countdown (401) e stordimento (402):** nessun peso.
- **Righe 0-2 di acqua, conchiglie e meduse:** nessun peso, ma sono sempre vuote.

L'unica informazione davvero ignorata è il tempo: la rete non sa quando si muoverà la marea, né per quanto resterà
stordita.

## 5. Secondo strato: le condizioni (36 neuroni, tanh)

Ogni neurone ha pesi tali che la pre-attivazione è almeno ±2.5, quindi tanh vale circa **+1 (vero)** o
**−1 (falso)** (tanh 2.5 = 0.987). Gli AND si fanno con pesi molto grandi (1000, 5120): se manca un pezzo, la somma
resta molto sotto zero.

| neurone | pre-attivazione | condizione |
|---|---|---|
| g0 | 5·Σ h12..18 − 2.5 | conchiglia a sinistra nella mia riga |
| g1 | 5·Σ h19..25 − 2.5 | conchiglia a destra nella mia riga |
| g2 | 5·Σ h2..11 + 1000·h1 − 1002.5 | conchiglia sopra **e** marea che sale |
| g3 | 12·h1 − 6·h0 − 3 | sale **e** m ≤ 1 |
| g4 | 18·h1 − 6·h0 − 3 | sale **e** m ≤ 2 |
| g5 | 6·h0 − 60·h1 − 3 | scende **e** m ≥ 1 |
| g6 | −6·h0 + 3 | m = 0 (sono sulla battigia) |
| g7 | 6·h0 + 60·h1 − 75 | sale **e** m ≥ 3 |
| g8-g16 | 5·h(36+i) + 5120·h(27+i) − 5120·h0 − 5122.5 | medusa **subito a sinistra**: sono in colonna i+1, medusa in colonna i, m = 0 |
| g17-g25 | 5·h(37+i) + 5120·h(26+i) − 5120·h0 − 5122.5 | medusa **subito a destra**: sono in colonna i, medusa in colonna i+1, m = 0 |
| g26-g35 | 5·h(36+i) + 5120·h(26+i) − 5122.5 | medusa **sotto di me**, sulla battigia della mia colonna |

Le soglie su m (g3, g4, g5, g7) sono fatte dando a h0 un peso −6 o +6 e scegliendo il bias in modo che il cambio di
segno cada tra due valori interi. Per esempio, con la marea che sale g4 vale 15 − 6·m, positivo per m ≤ 2 e
negativo per m ≥ 3.

## 6. Ultimo strato: la regola (5 logit)

```
fermo = 0
su    =   23.5 + 4.5·g2 + 20·g3
giù   = −160   + 10.5·g5 − 20·g4 − 2.5·g6 + 3·g7 − 15·Σ g26..35
sx    = −128   + 6·g0  − 15·Σ g8..16
dx    = −126.5 + 7.5·g1 − 15·Σ g17..25
```

I bias sembrano strani perché **i neuroni spenti valgono −1, non 0**. Per esempio, i dieci neuroni "medusa sotto"
spenti danno −15 × (−10) = +150, che compensa quasi tutto il −160. Al massimo uno di loro è acceso (quello della
colonna dell'agente), e accenderlo toglie 30.

Riscritti con `[x]` = +1 se x è vero, −1 se è falso:

```
fermo = 0
su    = 23.5 + 4.5·[sale e conchiglia sopra] + 20·[sale e m ≤ 1]
giù   = −10 + 10.5·[scende e m ≥ 1] − 20·[sale e m ≤ 2] − 2.5·[m = 0] + 3·[sale e m ≥ 3]   (−30 se medusa sotto)
sx    = 7   + 6·[conchiglia a sx]                                                        (−30 se medusa subito a sx)
dx    = 8.5 + 7.5·[conchiglia a dx]                                                      (−30 se medusa subito a dx)
```

Ecco i valori nei vari casi. "Fermo" vale sempre 0. Sx vale 13 con una conchiglia a sinistra e 1 senza; dx vale
16 con una conchiglia a destra e 1 senza.

| marea | m | su | giù | mossa scelta |
|---|---|---|---|---|
| scende | 0 | −1 | −6 | dx se c'è una conchiglia a destra (16), poi sx (13), altrimenti vaga |
| scende | ≥ 1 | −1 | **20** (−10 con medusa sotto) | **giù**: segue l'acqua che si ritira |
| sale | 0 | **39** (48 con conchiglia sopra) | −46 | **su**: batte tutto |
| sale | 1 | **39** (48) | −41 | **su** |
| sale | 2 | −1 (8 con conchiglia sopra) | −41 | dx (16) > sx (13) > su verso una conchiglia (8) > vaga |
| sale | ≥ 3 | −1 (8) | 5 (−25 con medusa sotto) | dx (16) > sx (13) > su (8) > **giù** (5) |

**Vaga** vuol dire che sx ≈ 1.08, dx ≈ 1.10, fermo 0 e su −1. La policy campiona dalla softmax dei logit, quindi va
a sinistra o a destra con circa il 40% di probabilità ciascuna, sta ferma il 14% delle volte e sale il 5%.

### L'algoritmo

```
m = righe asciutte tra me e l'acqua

se la marea SCENDE:
    m ≥ 1 → giù                         segui l'acqua che si ritira
                                        (se c'è una medusa sotto, sulla battigia: spostati di lato)
    m = 0 → percorri la battigia:
            conchiglia a destra → dx, altrimenti conchiglia a sinistra → sx, altrimenti vaga
            mai verso una medusa subito a fianco

se la marea SALE:
    m ≤ 1 → su                          scappa: ha la priorità su tutto
    m = 2 → conchiglie nella riga (prima dx, poi sx), poi conchiglia sopra, altrimenti vaga;
            mai giù
    m ≥ 3 → conchiglie nella riga, poi conchiglia sopra, altrimenti giù (per tornare a m = 2)
```

La strategia segue le regole del gioco:

- **Marea che scende:** le conchiglie nuove compaiono solo sulla riga appena scoperta, cioè sulla battigia. La rete
  ci sta attaccata e la spazza in orizzontale.
- **Marea che sale:** l'acqua avanza di una riga ogni 3-5 passi. La rete tiene 2 righe di margine e risale appena
  il margine scende a 1. Intanto raccoglie quello che trova nella sua riga.

## 7. Verifica

`check.py` gioca 200 episodi con la rete (seed 10000-10199, sticky 0.1, campionando dalla softmax come
`evaluate`). In ogni stato ricalcola, **direttamente dallo stato del gioco e senza usare i pesi**, i 46 neuroni del
primo strato e i 5 logit con le formule della sezione 6, e li confronta con il forward pass vero.

| controllo | risultato |
|---|---|
| stati visitati | 197.920 |
| primo strato: stati in cui la lettura (acceso/spento, valore di h0) sbaglia | **0** |
| logit: differenza massima | 0.315 (tanh non arriva mai esattamente a ±1) |
| mossa con logit più alto diversa, esclusi i quasi-pareggi (< 0.5) | **0** |

I quasi-pareggi sono 49.550 stati. Quasi tutti sono sx contro dx quando non ci sono conchiglie (1.08 contro 1.10);
lì la rete campiona e quindi vaga.

Quale mossa ha il logit più alto, per situazione:

| marea | m | % degli stati | fermo | su | giù | sx | dx |
|---|---|---|---|---|---|---|---|
| sale | 0 | 0.3% | | 100% | | | |
| sale | 1 | 14.0% | | 100% | | | |
| sale | 2 | 32.1% | | 2% | | 26% | 72% |
| sale | ≥ 3 | 3.5% | | 3% | 44% | 27% | 26% |
| scende | 0 | 32.9% | 1% | | | 36% | 64% |
| scende | 1 | 15.8% | | | 93% | 2% | 5% |
| scende | 2 | 1.0% | | | 92% | 1% | 7% |
| scende | ≥ 3 | 0.3% | | | 90% | 1% | 9% |

La rete passa due terzi del tempo in due stati: sulla battigia mentre la marea scende (33%) e a m = 2 mentre sale
(32%). Il resto sono le transizioni tra i due.

## 8. Limiti visibili nei pesi

I pesi spiegano anche perché la rete fa 132 e non 164 come l'euristica.

- **Non legge countdown (401) e stordimento (402).** Non sa quando si muoverà la marea, quindi tiene sempre 2 righe
  di margine anche quando avrebbe il tempo di scendere a raccogliere.
- **Vede le conchiglie solo in croce**, cioè nella sua riga e nella sua colonna, e **senza distanza**:
  - una conchiglia in diagonale non esiste per lei;
  - una conchiglia a 1 cella vale quanto una a 8;
  - con conchiglie da entrambi i lati va a destra il 95% delle volte (16 contro 13), anche se quella a sinistra è
    più vicina.
- **Le conchiglie sopra le considera solo quando la marea sale.**
- **Vede le meduse solo sulla battigia.** Quando la marea sale e si muove a m = 2, o sale verso una conchiglia,
  cammina sulle meduse senza vederle: 21.7 punture per episodio in media.
- **Annega nel 4.2% degli episodi, ma solo per colpa delle sticky actions.** Senza sticky actions
  (`uv run evaluate low_tide claude --sticky 0`) annega in 0 episodi su 500 e fa 140.7 punti. Nei 17 annegamenti su
  500 episodi del protocollo ci sono due meccanismi, e in entrambi l'ambiente ripete un "giù" al posto di un "su":
  - **10 casi:** la marea sale e la rete, a m = 3, scende per tornare a m = 2. Le sticky actions ripetono "giù" due
    volte di fila, contro le scelte della rete (all'ultimo passo, a m = 1, la rete sceglie su), e l'agente arriva a
    m = 0 proprio mentre la marea sale. Un margine di 2 righe regge a una ripetizione, non a due.
  - **7 casi:** con la marea che sale, un "giù" ripetuto porta l'agente su una medusa sulla battigia. In 5 casi su 7
    la marea aveva appena invertito e la rete aveva scelto su. L'agente resta stordito 3 passi a m = 0 e la marea sale
    prima che possa muoversi.

La rete di GPT per lo stesso gioco è decodificata in [`decode_gpt_low_tide`](../decode_gpt_low_tide/README.md), con
un confronto tra le due.
