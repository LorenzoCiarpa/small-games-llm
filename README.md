# small-games-llm

Banco di prova per l'idea del post [Weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/):
un LLM scrive a mano, senza training, senza tool e senza eseguire codice, i pesi di un MLP che gioca a un gioco.
Qui ci sono il gioco nuovo **Low Tide** (mai visto prima, mai usato per il training di nessun modello) e i tre giochi
dell'articolo (**Breakout**, **Seaquest**, **Asterix** di MinAtar), tutti dietro la stessa interfaccia, così
`evaluate` e `play` funzionano su qualsiasi gioco registrato.

```bash
uv sync                    # crea .venv da uv.lock (Python 3.12)
uv run evaluate --help     # in fondo: per ogni gioco, le baseline e i pesi disponibili
uv run evaluate --list     # (anche `play --list`) protocolli, reti ammesse, cosa fa ogni baseline
```

**Baseline**: policy di riferimento scritte a mano, non reti, che fanno da scala per i punteggi.
`random` (azione uniforme a caso) c'è per tutti i giochi; Low Tide ha anche `heuristic` (BFS verso la conchiglia
più vicina evitando meduse e righe che si allagano: il riferimento alto) e `safe` (va alla riga 0 e resta lì:
non annega mai, non segna mai). **Greedy** invece riguarda solo le reti: argmax dei logits invece di campionare
dalla softmax (il protocollo dell'articolo è campionato).

## Valutare

```bash
uv run evaluate low_tide claude gpt                     # nomi dei file in weights/low_tide/
uv run evaluate low_tide weights/low_tide/claude.json   # oppure un percorso qualsiasi
uv run evaluate low_tide --baseline heuristic --baseline safe
uv run evaluate breakout paper_astra --baseline random
```

Ogni gioco ha il suo protocollo (episodi, seed, sticky actions, limite di passi), che si può cambiare con
`--episodes`, `--seed0`, `--sticky` e `--max-steps`. `--greedy` usa l'argmax invece di campionare dalla softmax.
L'episodio k usa il seed `seed0 + k` (vedi *Riproducibilità*); `--workers` (default: tutti i core) non cambia i
risultati. Il punteggio usa sempre il forward pass di riferimento (`small_games/weights.py`), mai codice scritto
dal modello.

## Giocare o guardare

```bash
uv run play low_tide                          # giochi tu
uv run play low_tide --weights claude         # guardi una rete
uv run play low_tide --baseline heuristic     # guardi una baseline
uv run play seaquest --weights paper_astra --speed 20
```

Tasti: quelli del gioco (frecce, spazio per stare fermo, `F` per sparare nei giochi MinAtar);
`1 2 3 ...` cambiano chi gioca (tu, rete, baseline) senza ricominciare; `P` pausa, `N` un passo, `+ -` velocità,
`T` a turni o in tempo reale, `R` ricomincia con lo stesso seed, `E` passa al seed + 1, `S` sticky on/off,
`G` greedy/campionata (solo la rete), `C` coordinate.

## Riproducibilità

Un episodio dipende solo dal suo seed: da quel seed ripartono il gioco, le sticky actions e il generatore con
cui la policy campiona (`policy_rng`, lo stesso in `evaluate` e in `play`). Quindi:

- lo stesso comando `evaluate` dà sempre gli stessi numeri, con qualsiasi `--workers`;
- in `play`, `R` (o rilanciare con lo stesso `--seed`) ripete l'episodio identico;
- `play` parte dal seed0 del gioco (10000 per tutti), cioè dall'episodio 0 di `evaluate`, ed `E` passa agli
  episodi 1, 2, ...; in generale `play <gioco> --seed s` mostra esattamente l'episodio `s - seed0`, per esempio
  `uv run play breakout --weights paper_astra --seed 10002` è l'episodio 2.

Vale finché la stessa policy gioca tutto l'episodio con le stesse impostazioni: cambiare chi gioca, `G` o `S`
a metà partita cambia il seguito (e da quel momento non coincide più con `evaluate`).

## I giochi

| gioco | osservazione | azioni | rete ammessa | protocollo |
|---|---|---|---|---|
| `low_tide` | 403 = 4 piani 10×10 + 3 scalari | 5 | ≤ 3 strati nascosti da ≤ 64 unità, relu/tanh/linear | 500 episodi, seed 10000.., sticky 0.1, 1000 passi |
| `breakout`, `seaquest`, `asterix` | 600 = griglia 10×10 × 6 canali | 6 | ≤ 2 strati nascosti da ≤ 32 unità, tanh (l'articolo: 600-32-32-6) | 1000 episodi, seed 10000.., sticky 0.1, 10000 passi |

Cosa rappresentano i 403 ingressi di Low Tide, e quali valori possono avere, è spiegato in
[`docs/low_tide_observation.md`](docs/low_tide_observation.md).

I giochi MinAtar sono quelli ufficiali del pacchetto `minatar==1.0.15` (la versione usata dall'articolo), non una
reimplementazione. L'osservazione è quella dell'articolo: indice `(riga * 10 + colonna) * 6 + canale`, canali
player, danger, reward, projectile, trail, gauge, ciascuno l'OR di alcuni canali del gioco, con la stessa mappatura
scritta nei prompt dell'articolo (vedi `games/minatar_games.py`). Azioni: 0 fermo, 1 sinistra, 2 su, 3 destra,
4 giù, 5 fuoco.

**Verifica dell'integrazione.** Le reti scritte da GPT-6 Astra nell'articolo (`weights/<gioco>/paper_astra.json`)
danno qui gli stessi punteggi dell'articolo, entro circa un errore standard:

| | qui (1000 episodi) | articolo |
|---|---|---|
| Breakout | 512.5 ± 7.6 | 517.4 ± 7.4 |
| Seaquest | 51.4 ± 1.4 | 53.1 ± 1.4 |
| Asterix | 54.8 ± 1.1 | 53.2 ± 1.0 |

**Low Tide** (500 episodi): euristica 164.0 ± 1.5, `claude` 132.3 ± 0.9, `gpt` 79.9 ± 1.0, casuale 2.0, `safe` 0.

## Esperimenti

Gli esperimenti su un solo gioco stanno in `experiments/<gioco>/`, quelli che confrontano più giochi direttamente in
`experiments/`.

- [`experiments/weight_distance/`](experiments/weight_distance): quanto sono lontane tra loro le reti scritte dagli
  LLM? Distanze strato per strato (grezze, dopo aver allineato i neuroni e contro un modello nullo), unità gemelle,
  CKA, azioni sugli stessi stati e cross-play tra giochi, con quattro reti addestrate con PPO come termine di
  paragone. `uv run --group analysis python experiments/weight_distance/analyze.py`
- [`experiments/ppo_over_training/`](experiments/ppo_over_training): le reti PPO diventano più sparse o più
  leggibili andando avanti con l'addestramento, o con un'altra entropia? No: le misure principali su tutte le corse
  e su tutti i checkpoint, accanto alle reti scritte a mano.
  `uv run --group analysis python experiments/ppo_over_training/analyze.py`
- [`experiments/ppo_runs.py`](experiments/ppo_runs.py): lancia, controlla e ferma in background tutti gli
  addestramenti PPO (10 corse, ~2 ore su 8 core).

**Low Tide**

- [`experiments/low_tide/decode_claude/`](experiments/low_tide/decode_claude): come si legge la rete `claude.json`,
  neurone per neurone, e quale algoritmo di gioco rappresenta. La lettura è verificata contro la rete vera su
  ~200.000 stati. `uv run python experiments/low_tide/decode_claude/check.py`
- [`experiments/low_tide/decode_gpt/`](experiments/low_tide/decode_gpt): la stessa analisi per `gpt.json`, con un
  confronto tra le due reti. `uv run python experiments/low_tide/decode_gpt/check.py`
- [`experiments/low_tide/ppo/`](experiments/low_tide/ppo): una rete addestrata con PPO (403-64-64-5) gioca bene
  quanto quella di Claude, e con 100M passi (`ppo_100m.json`) quanto l'euristica, ma i suoi pesi non si leggono come
  un algoritmo. Le stesse misure sulle reti: pesi, pruning, ablazione, quantità dell'algoritmo di Claude nei
  neuroni, distillazione in alberi di decisione.
  `uv run --group ppo python experiments/low_tide/ppo/train.py`, poi `uv run --group analysis python
  experiments/low_tide/ppo/analyze.py`

**Breakout, Seaquest, Asterix**: le reti di GPT-6 Astra dell'articolo (`paper_astra.json`), lette neurone per
neurone come quelle di Low Tide, ognuna con un `check.py` che verifica la lettura contro la rete vera.

- [`experiments/breakout/decode_astra/`](experiments/breakout/decode_astra): prevede la colonna in cui arriverà la
  palla, da una tabella "cella → colonna d'arrivo" scritta nei pesi, e ci porta la racchetta.
- [`experiments/seaquest/decode_astra/`](experiments/seaquest/decode_astra): il sottomarino resta nella colonna 5 e
  sceglie solo la riga, in base a sub da salvare, bersagli nella linea di tiro e pericoli; emerge con poco ossigeno.
- [`experiments/asterix/decode_astra/`](experiments/asterix/decode_astra): un "portiere" nella colonna 5 che
  sceglie solo la riga, evitando i nemici in arrivo e andando verso i tesori.

E le reti PPO con la stessa architettura (600-32-32-6), confrontate con quelle di Astra con le misure di Low Tide.
`uv run --group ppo python experiments/<gioco>/ppo/train.py`, poi `uv run --group analysis python
experiments/<gioco>/ppo/analyze.py`.

- [`experiments/breakout/ppo/`](experiments/breakout/ppo): PPO esce dal plateau a ~6 punti solo con più entropia
  e arriva a 63, contro i 512 di Astra.
- [`experiments/seaquest/ppo/`](experiments/seaquest/ppo): PPO gioca meglio di Astra (55.9 contro 51.4), ma in un
  altro modo: si muove in orizzontale invece di restare nella colonna 5.
- [`experiments/asterix/ppo/`](experiments/asterix/ppo): PPO arriva a 24.6, metà di Astra, ancora in salita.

### PPO trova reti sparse e leggibili come quelle scritte a mano?

No, in nessuno dei quattro giochi. Anche dove gioca bene quanto o meglio delle reti scritte a mano (Low Tide,
Seaquest), PPO trova pesi densi, che servono quasi tutti, e neuroni senza un nome. Le misure sul primo strato sono
ristrette agli ingressi vivi, quelli che non sono sempre zero (sugli altri PPO lascia i pesi iniziali):

| gioco | rete | punteggio | pesi vivi ≠ 0 | valori diversi | ingressi per il 90% del peso di un neurone | punteggio con il 25% dei pesi vivi | neuroni che da soli tolgono > 50% | quantità chiave: miglior \|r\| | albero di prof. 6: accordo |
|---|---|---|---|---|---|---|---|---|---|
| Low Tide | `claude` (a mano) | 132.1 | 5% | 34 | 9 su 312 | 131.9 (100%) | 3 | 0.97 | 100% |
| | `gpt` (a mano) | 79.7 | 3% | 9 | 2 su 312 | 79.7 (100%) | 12 | 0.67 | 92% |
| | `ppo` (20M passi) | 125.5 | 100% | 19.582 | 186 su 312 | 14.0 (11%) | 0 | 0.47 | 64% |
| | `ppo_100m` | 155.5 | 100% | 19.731 | 172 su 312 | 4.3 (3%) | 0 | 0.47 | 64% |
| Breakout | `paper_astra` (a mano) | 512.7 | 10% | 53 | 77 su 220 | 512.7 (100%) | 14 | 0.73 | 89% |
| | `ppo` | 62.4 | 100% | 7.027 | 136 su 220 | 0.4 (1%) | 10 | 0.34 | 72% |
| Seaquest | `paper_astra` (a mano) | 48.3 | 3% | 12 | 9 su 425 | 48.3 (100%) | 29 | 0.46 | 87% |
| | `ppo` | 56.0 | 100% | 13.526 | 240 su 425 | 1.7 (3%) | 14 | 0.22 | 65% |
| Asterix | `paper_astra` (a mano) | 54.5 | 5% | 33 | 10 su 315 | 54.5 (100%) | 14 | 0.57 | 92% |
| | `ppo` | 26.8 | 100% | 10.029 | 167 su 315 | 7.3 (27%) | 2 | 0.11 | 49% |

Punteggi su 200 episodi; "valori diversi" e "ingressi" sono del primo strato; il pruning tiene il 25% dei pesi vivi
più grandi del primo strato; le quantità chiave sono quelle dell'algoritmo scritto a mano (media sulle quantità del
gioco); l'albero imita la mossa più probabile della rete usando le grandezze del gioco.

- **Le reti scritte a mano** sono sparse (3-10% dei pesi), con pochi valori tondi, e il pruning toglie solo zeri. I
  neuroni sono le quantità e le condizioni dell'algoritmo, e il comportamento si scrive come un albero corto.
- **Le reti PPO** usano ogni ingresso vivo con un peso diverso, non reggono il pruning, e nessun neurone calcola le
  quantità dell'algoritmo scritto a mano, nemmeno come combinazione lineare. Su Low Tide nessun neurone conta da
  solo; nei giochi MinAtar, con 32 unità per strato, molti contano, ma nessuno ha un significato riconoscibile.
- **Le tracce leggibili** in PPO sono codifiche di posizione: la riga dell'agente insieme alla marea su Low Tide, la
  colonna della racchetta su Breakout, macchie di posizione del sottomarino e del giocatore su Seaquest e Asterix.
- **Con più addestramento o un'altra entropia** il quadro non cambia: la rete migliora usando più pesi, non meno
  ([`experiments/ppo_over_training/`](experiments/ppo_over_training)).
- **Anche il comportamento** è diverso: in Seaquest e Asterix Astra riduce il gioco a una dimensione (fermo nella
  colonna 5, sceglie la riga), PPO no.

## Struttura

```
src/small_games/
  core.py            Game, Protocol, NetSpec, GridSpec e il registro dei giochi
  weights.py         formato dei pesi (JSON), validazione per gioco, MLP di riferimento
  policy.py          campionamento, policy MLP, baseline random, ricerca dei file in weights/
  evaluate.py        CLI `evaluate`
  play.py            CLI `play` (pygame)
  render.py          disegno comune: pannello, renderer generico per griglie a canali
  ppo.py             PPO per qualsiasi gioco registrato, con salvataggi e ripresa esatta (gruppo `ppo`)
  analysis.py        misure sui pesi: sparsità, pruning, ablazione, probe, alberi (gruppo `analysis`)
  features.py        le grandezze di ogni gioco (il vocabolario per probe e alberi)
  games/
    low_tide/        env.py (regole), baselines.py (euristica: NON darla al modello), render.py
    minatar_games.py Breakout, Seaquest, Asterix
weights/<gioco>/<nome>.json
experiments/<gioco>/<nome>/   esperimenti sui pesi di un gioco (script, dati, risultati)
experiments/<nome>/           esperimenti che confrontano più giochi
```

## Aggiungere un gioco

1. Scrivi un modulo (o un pacchetto) in `src/small_games/games/`: viene importato da solo.
2. L'ambiente deve avere `reset(seed) -> (obs, info)`, `step(a) -> (obs, r, terminated, truncated, info)` e gli
   attributi `sticky_prob`, `last_action`, `max_steps`, `t` (dettagli in `core.py`).
3. Chiama `register(Game(...))` con nome, `make_env`, `obs_dim`, nomi delle azioni, `Protocol`, `NetSpec`, ed
   eventualmente baseline, tasti, `episode_stats` (chiavi di `info` mediate da `evaluate`).
4. Per `play` basta un `GridSpec` (renderer generico, una forma per canale); per un aspetto su misura indica in
   `renderer` una sottoclasse di `small_games.render.Renderer`, come `games/low_tide/render.py`.
5. Metti i pesi in `weights/<nome>/`.

## Formato dei pesi

Lo stesso per tutti i giochi; ogni voce non elencata in un blocco sparse vale 0, e `y = act(W @ x + b)`:

```json
{"layers": [
  {"in": 600, "out": 32, "activation": "tanh",
   "W": {"sparse": [[out, in, valore], ...]},  "b": {"dense": [...]}},
  ...
  {"in": 32, "out": 6, "activation": "linear", "W": {"dense": [[...], ...]}, "b": {"sparse": [[out, valore], ...]}}
]}
```

## Crediti e licenze

- MinAtar (Kenny Young e Tian Tian, GPL-3.0) è una dipendenza installata da PyPI, non è incluso nel repository.
- `weights/{breakout,seaquest,asterix}/paper_astra.json` sono le reti `hand.npz` dell'articolo
  ([RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), `exp5_write_from_scratch/networks/`,
  CC BY 4.0), convertite in questo formato (W = kernel trasposto).
