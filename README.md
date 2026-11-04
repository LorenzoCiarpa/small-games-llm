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

- [`experiments/decode_claude_low_tide/`](experiments/decode_claude_low_tide): come si legge la rete `claude.json`
  di Low Tide, neurone per neurone, e quale algoritmo di gioco rappresenta. La lettura è verificata contro la rete
  vera su ~200.000 stati. `uv run python experiments/decode_claude_low_tide/check.py`
- [`experiments/decode_gpt_low_tide/`](experiments/decode_gpt_low_tide): la stessa analisi per `gpt.json`, con un
  confronto tra le due reti. `uv run python experiments/decode_gpt_low_tide/check.py`

- [`experiments/weight_distance/`](experiments/weight_distance): quanto sono lontane tra loro le reti scritte dagli
  LLM? Distanze strato per strato (grezze, dopo aver allineato i neuroni e contro un modello nullo), unità gemelle,
  CKA, azioni sugli stessi stati e cross-play tra giochi, con quattro reti addestrate con PPO come termine di
  paragone. `uv run --group analysis python experiments/weight_distance/analyze.py`

## Struttura

```
src/small_games/
  core.py            Game, Protocol, NetSpec, GridSpec e il registro dei giochi
  weights.py         formato dei pesi (JSON), validazione per gioco, MLP di riferimento
  policy.py          campionamento, policy MLP, baseline random, ricerca dei file in weights/
  evaluate.py        CLI `evaluate`
  play.py            CLI `play` (pygame)
  render.py          disegno comune: pannello, renderer generico per griglie a canali
  games/
    low_tide/        env.py (regole), baselines.py (euristica: NON darla al modello), render.py
    minatar_games.py Breakout, Seaquest, Asterix
weights/<gioco>/<nome>.json
experiments/<nome>/   esperimenti sui pesi (script, dati di riferimento, risultati)
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
