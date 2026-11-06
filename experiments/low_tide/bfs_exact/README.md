# La BFS di Low Tide esatta in un MLP, senza i limiti del gioco

[`bfs_net`](../bfs_net) sta nei limiti del gioco, 3 strati da 64 unità, e per questo calcola solo una versione a
potenziale della BFS (vedi la sua sezione 1). Qui i limiti si tolgono, per vedere **quanti neuroni servono per fare
esattamente la mossa di `heuristic_policy` in ogni stato.**

`build.py` scrive `bfs_exact.json`:
- 104 strati nascosti relu, fino a 503 unità per strato, **51.103 neuroni** in tutto;
- 273.944 pesi diversi da zero, su 25,5 milioni;
- nessun addestramento: ogni peso viene da una formula.

I limiti del gioco ammettono al massimo 3 × 64 = 192 neuroni.

**Risultato:** la rete fa sempre la stessa mossa dell'euristica:
- in tutti i 470.889 stati dei 500 episodi del protocollo;
- in tutti i 20.000 stati costruiti apposta, con cammini fino a 54 passi.

Gioca quindi gli stessi episodi, con lo stesso punteggio episodio per episodio: **164.0 ± 1.5**, come l'euristica.

La rete supera i limiti del gioco, quindi `uv run evaluate` e `uv run play` non la caricano. `check.py` la legge con
lo stesso `load_weights` del repo, a cui toglie solo i limiti di strati e unità, e la valuta con il forward di
riferimento (`ReferenceMLP`, float64).

```bash
uv run python experiments/low_tide/bfs_exact/build.py     # scrive bfs_exact.json (3.8 MB, non versionato)
uv run python experiments/low_tide/bfs_exact/check.py     # verifica, ~9 min con 8 processi
```

## 1. L'algoritmo

La BFS dell'euristica parte dall'agente e visita i vicini nell'ordine su, giù, sinistra, destra. Si ferma alla prima
conchiglia che toglie dalla coda e fa il primo passo del cammino che ve l'ha portata.

**Una regola equivalente, più comoda per una rete:**

> vai nel vicino consentito più vicino a una conchiglia; a parità, nell'ordine su, giù, sinistra, destra.

Le due regole scelgono la stessa mossa:
- la BFS toglie le celle dalla coda in ordine di distanza;
- a parità di distanza, le toglie in ordine dei cammini: prima quelli che cominciano con su, poi con giù, e così via.

Quindi la prima conchiglia che trova è la più vicina. Il suo cammino comincia con la prima direzione da cui parte un
cammino minimo, cioè la prima direzione il cui vicino è a distanza d − 1 da una conchiglia.

**La rete calcola la distanza dalle conchiglie con un'onda:**
- **Partenza.** L'onda parte dalle conchiglie nelle celle consentite. Sono le celle che la BFS può visitare:
  - righe fino all'ultima sicura;
  - niente meduse;
  - se la marea sale, niente celle vicine a una medusa.
- **Un passo per strato.** A ogni strato l'onda raggiunge le celle consentite vicine a una cella già raggiunta. Dopo k
  strati copre esattamente le celle a distanza ≤ k da una conchiglia.
- **Conteggio.** Ogni cella conta per quanti strati è stata raggiunta: dopo K strati il conteggio è K + 1 − d, con d
  la distanza dalla conchiglia più vicina (0 se d > K).
- **Lettura.** Alla fine la rete legge il conteggio nei quattro vicini dell'agente:
  - logit = 10⁴ · (conteggio + 0.4, 0.3, 0.2 o 0.1) per su, giù, sinistra e destra;
  - vince il vicino più vicino a una conchiglia, e a parità l'ordine della BFS.
- **Marea.** Sono le regole dell'euristica:
  - se l'agente è sotto l'ultima riga sicura, sale;
  - se nessun vicino è stato raggiunto dall'onda, aspetta una riga sopra l'ultima sicura.

## 2. Struttura

| strati | unità | cosa calcolano |
|---|---|---|
| 1 | 405 | copie di agente, conchiglie e meduse; numero di meduse vicine se la marea sale; livello dell'acqua; riga dell'agente; countdown ≥ 3 |
| 2 | 405 | marea che sale e countdown ≤ 2, cioè margine 2 invece di 1 |
| 3 | 425 | per ogni riga, se è entro l'ultima riga sicura; dov'è l'agente rispetto a quella riga |
| 4 | 303 | celle consentite; partenza dell'onda; "agente sotto l'ultima riga"; attesa su o giù |
| 5-102 | 503 (l'ultimo 403) | un passo dell'onda ciascuno, 98 in tutto |
| 103 | 363 | il conteggio nei vicini dell'agente, una unità per cella e direzione |
| 104 | 8 | conteggio per direzione; attesa solo se nessun vicino è stato raggiunto |
| uscita | 5 | logit |

**Per cella e per passo servono 5 unità:**
- **2 per l'"o" dei vicini.** Con s il numero di celle raggiunte attorno, relu(s) − relu(s − 1) = min(1, s). Una sola
  relu non si ferma a 1.
- **1 per il conteggio.**
- **1 per la maschera delle celle consentite.** Serve a ogni passo, quindi va portata avanti di strato in strato.
- **1 per la posizione dell'agente.** Serve solo alla fine, ma anche lei va portata avanti.

**Il calcolo è esatto.** Tutti i valori interni sono interi piccoli, tra 0 e 100, che il float64 rappresenta senza
errori.
- L'unico ingresso non intero è il countdown, k/5 in float32.
- `relu(10 · countdown − 5)` vale 0 per k ≤ 2 e almeno 1 per k ≥ 3, quindi `relu(sale − ·)` dà un 0/1 esatto.

**Anche campionando dalla softmax la mossa è sempre la stessa.** I logit hanno margini ≥ 1000, ed exp(−1000) è 0 in
float64. La rete gioca quindi esattamente gli stessi episodi dell'euristica.

## 3. Quanti neuroni servono

**Profondità: uno strato per passo di cammino.**
- **K = 98 basta per ogni stato possibile.** In 100 celle la conchiglia più vicina è a distanza ≤ 99, e al vicino
  dell'agente ne servono d − 1 ≤ 98.
- **È un limite sicuro, non il minimo.** Con pareti di meduse il cammino minimo più lungo è più corto: negli stati
  costruiti si arriva a 54 passi.
- **Negli episodi del protocollo la conchiglia trovata è al massimo a 26 passi.** Per quegli stati basterebbero
  K = 25 passi, cioè 31 strati e 14.384 neuroni. La rete però sbaglierebbe in uno stato con un cammino più lungo.

**Larghezza: 5 unità per cella**, circa 500 per strato.

| passi dell'onda K | strati nascosti | neuroni | pesi ≠ 0 | esatta se la conchiglia è entro |
|---|---|---|---|---|
| 98 (`bfs_exact.json`) | 104 | 51.103 | 273.944 | sempre |
| 30 | 36 | 16.899 | 87.420 | 31 passi |
| 25 | 31 | 14.384 | 73.705 | 26 passi: tutti gli stati del protocollo |
| limiti del gioco (`bfs_net`, a potenziale) | 3 | 185 | 6.806 | non esatta (78.8% delle mosse) |

`build.py --k N` costruisce le varianti più corte.

**Si può scambiare profondità con larghezza, ma costa molto.** Si possono calcolare le distanze tra tutte le coppie di
celle e raddoppiare il raggio a ogni strato, come nel prodotto di matrici min-plus.
- Bastano log₂ 99 ≈ 7 raddoppi.
- Ogni raddoppio però ha bisogno di un'unità per ogni terna di celle, circa un milione.

Non l'ho costruita.

## 4. Verifica

`check.py`:
1. riscrive i pesi dalle formule e controlla che coincidano con il file;
2. gioca i 500 episodi del protocollo con la rete, campionando dalla softmax come `evaluate`. In ogni stato confronta
   la mossa con quella dell'euristica. Gioca gli stessi episodi con l'euristica e confronta i punteggi;
3. controlla 20.000 stati costruiti apposta, anche fuori dalle regole del gioco:
   - metà casuali, con fino al 45% di meduse e 1-3 conchiglie;
   - un quarto labirinti di meduse;
   - un quarto serpentine.

   Servono a provare cammini lunghi, che negli episodi non capitano.

```
pesi riscritti dalle formule uguali al file: True
rete: 104 strati nascosti, 51103 neuroni, 273944 pesi diversi da zero su 25472463

episodi del protocollo: 500, stati 470889
  mosse diverse dall'euristica: 0
  azione campionata ≠ argmax: 0; margine minimo dei logit: 1000
  punteggio rete 163.97 ± 1.47, euristica 163.97 ± 1.47, episodi con punteggio diverso: 0
  distanza massima della conchiglia trovata dalla BFS: 26 (per questi stati basterebbero K = 25 passi, 31 strati)

stati costruiti (casuali, labirinti, serpentine): 20000
  mosse diverse dall'euristica: 0; margine minimo: 1000
  distanza della conchiglia: mediana 8, massima 54, stati con distanza > 26: 1879

(540 s)
```
