# Low Tide: cosa vede la rete

A ogni passo la rete riceve un vettore di **403 numeri** (float32) e restituisce **5 logit**, uno per azione.
Questo documento spiega cosa rappresenta ciascun numero, quali valori può assumere e quali regole del gioco li
determinano. La fonte è `src/small_games/games/low_tide/env.py`. I vincoli elencati qui sono stati verificati su
600 episodi (circa 310 000 osservazioni, giocati a caso e dall'euristica), senza nessuna eccezione.

## Il gioco in breve

- La spiaggia è una griglia 10×10. La **riga 0** è in alto, lato terra; la **riga 9** è in basso, lato mare. Le
  colonne vanno da 0 (sinistra) a 9 (destra).
- **L'acqua** copre le ultime `livello` righe in basso. Il livello va da 0 a 7, quindi le righe 0–2 sono sempre
  asciutte. Le righe 3–9 sono la zona di marea.
- **La marea** cambia di una riga ogni 3, 4 o 5 passi (estratti a caso).
  - Sale fino a un livello alto, estratto tra 5, 6 e 7, poi scende fino a un livello basso, estratto tra 0, 1 e 2,
    e così via.
  - Quando **scende**, la riga che si scopre si riempie a caso di conchiglie (25% per cella) e meduse (8% per
    cella).
  - Quando **sale**, la riga che viene coperta perde tutto quello che c'era sopra.
- Effetti delle celle:
  - pestare una **conchiglia** dà +1 e la toglie;
  - pestare una **medusa** dà −1, la toglie e paralizza l'agente per 3 passi;
  - se la marea che sale copre la riga dell'agente, l'agente **annega** e l'episodio finisce.
- **Muoversi** verso l'acqua o fuori dalla griglia non ha effetto: l'agente resta dov'è.
- **Azioni:** 0 fermo, 1 su (riga − 1), 2 giù (riga + 1), 3 sinistra (colonna − 1), 4 destra (colonna + 1).
- **Sticky actions:** con probabilità 0.1 viene ripetuta l'azione precedente al posto di quella scelta.
- Un episodio dura al massimo 1000 passi.

## Com'è fatto il vettore

| indici | cosa | valori possibili |
|---|---|---|
| 0–99 | piano **agente** | 0 o 1; esattamente un 1 |
| 100–199 | piano **acqua** | 0 o 1; righe intere a partire dal basso |
| 200–299 | piano **conchiglie** | 0 o 1 |
| 300–399 | piano **meduse** | 0 o 1 |
| 400 | **la marea sale** | 0 o 1 |
| 401 | **conto alla rovescia** della marea | 0.2, 0.4, 0.6, 0.8, 1.0 |
| 402 | **stordimento** | 0, 1/3, 2/3, 1 |

I primi 400 numeri sono quattro griglie 10×10, scritte una dopo l'altra. Dentro ogni griglia l'ordine è per righe:

```
indice = 100 · piano + 10 · riga + colonna          piano: 0 agente, 1 acqua, 2 conchiglie, 3 meduse

piano   = indice // 100
riga    = (indice % 100) // 10
colonna = indice % 10
```

Esempi per la cella in riga 3 e colonna 6: agente → 36, acqua → 136, conchiglia → 236, medusa → 336.

> **Attenzione, l'ordine è diverso da quello dei giochi MinAtar.** In Breakout, Seaquest e Asterix l'indice è
> `(riga · 10 + colonna) · 6 + canale`: i canali di una cella sono vicini. In Low Tide sono vicine le celle di uno
> stesso piano. Chi scrive i pesi deve usare la formula giusta per ciascun gioco.

## I quattro piani

### Agente (indici 0–99)

- C'è sempre **esattamente un 1**, nella cella dell'agente: indice `10 · riga + colonna`.
- All'inizio di ogni episodio l'agente è in **riga 1**, in una colonna a caso.
- Può stare su qualsiasi riga asciutta, cioè dalla 0 alla `9 − livello`. Con la marea a 0 può arrivare fino
  alla riga 9.
- Non sta mai in acqua, tranne nell'**ultima osservazione** di un episodio finito per annegamento. È l'unico caso
  in cui agente e acqua hanno un 1 nella stessa cella.
- Non sta mai sopra una conchiglia o una medusa, perché le raccoglie o ne viene punto nello stesso passo in cui ci
  arriva.

### Acqua (indici 100–199)

- L'acqua occupa sempre **righe intere a partire dal basso**. Con livello L, le righe da `10 − L` a 9 sono tutte 1
  e le altre tutte 0. Non esistono righe allagate a metà.
- Quindi **livello = (somma del piano) / 10**, un intero tra 0 e 7. La riga asciutta più bassa è `9 − livello`.
- Le righe 0–2 non sono mai acqua. All'inizio dell'episodio il livello è tra 2 e 5.
- Frequenza dei livelli nelle osservazioni misurate:

  | livello | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | quota | 3% | 10% | 17% | 20% | 20% | 17% | 10% | 3% |

### Conchiglie (indici 200–299)

- Un 1 per ogni conchiglia.
- Dove possono stare:
  - solo su righe **asciutte della zona di marea** (3–9), mai nelle righe 0–2;
  - mai in acqua;
  - mai nella stessa cella di una medusa;
  - mai sotto l'agente.
- Quando **compaiono**:
  - all'inizio, in tutte le righe asciutte della zona di marea, con probabilità 25% per cella;
  - ogni volta che la marea scende, nella riga appena scoperta, con la stessa probabilità.
- Quando **spariscono**: quando l'agente le raccoglie (+1), o quando la marea che sale copre la loro riga.
- **Non si muovono.** In media se ne vedono circa 3.5 per osservazione, con un massimo misurato di 27.

### Meduse (indici 300–399)

- Le regole sono le stesse delle conchiglie, con probabilità **8% per cella**: compaiono nelle righe scoperte, non
  si muovono e spariscono quando vengono coperte dall'acqua.
- Spariscono anche quando l'agente ci passa sopra. Quel passo dà −1 e fa partire lo stordimento.
- In media se ne vedono circa 2.6 per osservazione, con un massimo misurato di 13.

## I tre scalari

### 400: la marea sale

- Vale **1** se il prossimo cambio di livello alzerà l'acqua, **0** se la abbasserà. È a 1 circa metà del tempo.
- Cambia valore solo nel passo in cui il livello raggiunge il suo obiettivo. Per esempio, se la marea sta salendo
  verso 6, appena l'acqua arriva a 6 il valore passa a 0.
- **L'obiettivo non è nell'osservazione.** La rete non sa se la marea salirà fino a 5, 6 o 7, né se scenderà fino
  a 0, 1 o 2. Lo scopre solo quando questo valore cambia.

### 401: conto alla rovescia

- Vale **(passi che mancano al prossimo cambio di livello) / 5**:

  | valore | 1.0 | 0.8 | 0.6 | 0.4 | 0.2 |
  |---|---|---|---|---|---|
  | passi al cambio | 5 | 4 | 3 | 2 | 1 |

- **0.2 vuol dire che il livello cambia alla fine del prossimo passo.**
- Non vale mai 0. Quando arriva a zero il livello cambia e viene estratto subito un nuovo intervallo (3, 4 o 5
  passi). Dopo un cambio quindi si legge 0.6, 0.8 o 1.0, e si scende di 0.2 a ogni passo.
- All'inizio dell'episodio vale 0.6, 0.8 o 1.0.
- Frequenze misurate: 0.2, 0.4 e 0.6 circa il 25% ciascuno, 0.8 il 17%, 1.0 l'8%.

### 402: stordimento

- Vale **(passi di paralisi rimasti) / 3**, quindi 0, 1/3, 2/3 o 1.
- Nel passo in cui l'agente pesta una medusa, l'osservazione mostra **1**. Le 3 azioni successive vengono
  ignorate e il valore scende così:

  | stordimento letto | 1 | 2/3 | 1/3 | 0 |
  |---|---|---|---|---|
  | la prossima azione | ignorata | ignorata | ignorata | eseguita |

- Quindi, se lo stordimento è maggiore di 0, l'agente resta fermo nel passo successivo qualunque azione scelga la
  rete. È diverso da 0 in circa il 2% delle osservazioni misurate.

## In che ordine succedono le cose in un passo

`step(azione)` esegue, nell'ordine:

1. **Sticky action:** con probabilità 0.1 l'azione scelta viene sostituita dalla precedente.
2. **Movimento.** Se l'agente è stordito, non si muove e lo stordimento cala di 1. Altrimenti si sposta, salvo che
   la cella di arrivo sia fuori dalla griglia o in acqua.
3. **Conchiglia o medusa** nella cella di arrivo: +1, oppure −1 e stordimento a 3.
4. **Marea:** il conto alla rovescia cala di 1. Se arriva a 0, il livello cambia di una riga: la riga coperta
   viene svuotata, quella scoperta viene popolata, e si estrae un nuovo intervallo.
5. **Annegamento:** se l'acqua ora copre la riga dell'agente, l'episodio finisce.
6. **Troncamento:** dopo 1000 passi l'episodio si ferma.

L'osservazione restituita da `step` mostra lo stato *dopo* tutti questi eventi, compreso l'eventuale cambio di
marea.

Da queste regole segue la situazione più pericolosa: **marea che sale (400 = 1) con conto alla rovescia a 0.2
(401).** Alla fine del prossimo passo l'acqua coprirà la riga `9 − livello`, cioè la riga asciutta più bassa. Se
l'agente è lì a fine passo, annega. Per salvarsi deve salire in quel passo, ma non ci riesce se è stordito (402 > 0)
o se la sticky action ripete un'azione diversa (10% dei passi).

## Cosa la rete non vede

L'osservazione contiene quasi tutto lo stato del gioco, ma non:

- **l'obiettivo della marea**: fin dove salirà o scenderà nel ciclo in corso;
- **l'azione precedente**, che la sticky action potrebbe ripetere;
- **il passo corrente** e quanti ne mancano ai 1000;
- **il punteggio**;
- **il futuro**: dove compariranno conchiglie e meduse.

## Esempio: l'inizio dell'episodio con seed 10000

È l'episodio 0 di `evaluate` e quello che si vede lanciando `uv run play low_tide`. Legenda: A agente,
o conchiglia, j medusa, ~ acqua, . sabbia.

```
riga 0   . . . . . . . . . .
riga 1   . . . . . . A . . .
riga 2   . . . . . . . . . .
riga 3   . o o . . . . o j .
riga 4   j o j . . o . o o .
riga 5   j o . . . . o . . o
riga 6   . . . j . . . . . .
riga 7   o o j o j . . . o .
riga 8   ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
riga 9   ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
```

Gli ingressi diversi da 0 sono:

| ingressi | quanti | dettaglio |
|---|---|---|
| agente | 1 | indice 16 = riga 1, colonna 6 |
| acqua | 20 | indici 180–199 = righe 8 e 9, quindi livello 2 |
| conchiglie | 14 | 231, 232, 237 (riga 3), 241, 245, 247, 248 (riga 4), ... |
| meduse | 7 | 338 (riga 3, col. 8), 340, 342 (riga 4), 350 (riga 5), 363, 372, 374 |
| scalari | | 400 = **1** (la marea sale), 401 = **0.8** (cambia tra 4 passi), 402 = **0** (non stordito) |

Lo stato nascosto in questo esempio è che la marea sta salendo verso il livello 6. La rete lo scoprirà solo quando
il valore 400 tornerà a 0.

## Leggere l'osservazione in Python

```python
import numpy as np

grid = obs[:400].reshape(4, 10, 10)          # [piano, riga, colonna]
agente, acqua, conchiglie, meduse = grid
riga, colonna = np.argwhere(agente)[0]
livello = int(acqua.sum()) // 10               # 0..7; riga asciutta più bassa = 9 - livello
sale = obs[400] == 1.0
passi_al_cambio = round(obs[401] * 5)          # 1..5
passi_di_paralisi = round(obs[402] * 3)        # 0..3
```

## Come la legge una rete: due esempi

Una rete non riceve variabili come `livello` o `riga`: le deve ricavare con somme pesate di questi 403 numeri.
Ogni unità calcola `attivazione(Σ peso · ingresso + bias)`.

- In `weights/low_tide/gpt.json` l'unità 0 del primo strato ha peso +1 sull'ingresso 236 (conchiglia in riga 3,
  colonna 6), −1 sul 336 (medusa nella stessa cella) e bias 1. Calcola quindi `relu(1 + conchiglia − medusa)`: vale
  2 se nella cella c'è una conchiglia, 1 se è vuota, 0 se c'è una medusa. In pratica è un giudizio su quella cella.
- Un'unità con peso 1 sugli ingressi 0–9, cioè la riga 0 del piano agente, vale 1 se e solo se l'agente è in riga 0.
  Con 10 unità così, una per riga, la rete sa in che riga si trova l'agente. Per il livello dell'acqua basta una
  cella per riga del piano acqua, perché le righe si allagano intere: per esempio l'ingresso 170 (acqua in riga 7,
  colonna 0) vale 1 se e solo se il livello è almeno 3.
