# Da dove riprendere (stato al 5 novembre 2026, notte)

Questa cartella esiste solo per passare il lavoro da un PC all'altro: quando gli addestramenti PPO della sezione 3
saranno finiti e analizzati, va cancellata.

## 1. Cosa è fatto (tutto committato)

| cosa | dove | stato |
|---|---|---|
| esperimenti divisi per gioco | `experiments/<gioco>/<nome>/` (`weight_distance` resta in `experiments/`) | fatto |
| decodifica delle reti di Claude e GPT per Low Tide | `experiments/low_tide/decode_{claude,gpt}/` | fatto |
| decodifica delle reti di Astra (articolo) | `experiments/{breakout,seaquest,asterix}/decode_astra/` | fatto, verificate con `check.py` |
| trainer PPO | `src/small_games/ppo.py`, gruppo di dipendenze `ppo` (PyTorch CPU) | fatto |
| strumenti di analisi dei pesi | `src/small_games/analysis.py`, `features.py` | fatto |
| PPO su Low Tide, addestramento e analisi | `experiments/low_tide/ppo/` (con README) | **fatto** |
| PPO sui giochi MinAtar, primo tentativo (`ppo_v1`) | `weights/<gioco>/ppo_v1.json`, `experiments/<gioco>/ppo/results/ppo_v1/` | fatto, ma la rete gioca male |
| analisi di Astra e `ppo_v1` sui giochi MinAtar | `experiments/<gioco>/ppo/results/analysis/` | fatto |
| PPO sui giochi MinAtar, versione finale (`ppo`) | | **da rifare: interrotto, vedi sezione 3** |
| README delle cartelle `experiments/<gioco>/ppo/` per MinAtar | | **da scrivere** |

## 2. Risultati finora

### Le reti scritte a mano si leggono come algoritmi

| rete | algoritmo | punteggio |
|---|---|---|
| Low Tide, `claude.json` | regole: segue la battigia quando la marea scende, 2 righe di margine quando sale | 132.3 |
| Low Tide, `gpt.json` | campo di potenziale + attrazione 6 − distanza, ma vede solo 24 celle | 79.9 |
| Breakout, `paper_astra.json` | tabella "cella → colonna d'arrivo" nei pesi, porta lì la racchetta | 512.5 |
| Seaquest, `paper_astra.json` | fermo in colonna 5, sceglie la riga (sub, bersagli, pericoli), emerge con ossigeno < 60 | 51.4 |
| Asterix, `paper_astra.json` | "portiere" in colonna 5, sceglie la riga evitando i nemici in arrivo | 54.8 |

### PPO: punteggi

| gioco | PPO | rete scritta a mano | note |
|---|---|---|---|
| Low Tide | 125.7 (greedy 139.2), 20M passi | Claude 132.3 | PPO gioca bene quanto la rete di Claude |
| Breakout | `ppo_v1` 5.8, 30M passi | Astra 512.5 | plateau: "segui la palla" fa 8.4 |
| Seaquest | `ppo_v1` 5.4, 30M passi | Astra 51.4 | plateau |
| Asterix | `ppo_v1` 9.1, 30M passi | Astra 54.8 | lento, ancora in salita |

### PPO: si leggono i pesi? (Low Tide, l'unico confronto alla pari finora)

Nei dettagli in `experiments/low_tide/ppo/README.md`:

- **Pesi:** PPO ha il 100% dei pesi diversi da zero e 25.004 valori diversi nel primo strato; le reti scritte a
  mano hanno il 4% e 9-34 valori.
- **Pruning:** con il 25% dei pesi PPO fa 11.5, le reti scritte a mano non perdono niente.
- **Ablazione:** nessun neurone PPO conta da solo; nella rete di Claude 7 neuroni costano più del 10%.
- **Quantità dell'algoritmo:** il margine dall'acqua e "conchiglia a destra" non compaiono in nessun neurone PPO,
  nemmeno come combinazione lineare (R² 0.48 e 0.27, contro 1.00 per Claude).
- **Distillazione:** un albero di profondità 6 imita Claude al 99.6%; per PPO servono 442 foglie per arrivare al
  77%.
- **Unica traccia leggibile:** i neuroni PPO più importanti pesano la riga dell'agente insieme alla direzione della
  marea, cioè un "pericolo" sfumato.

Sui giochi MinAtar `ppo_v1` dà lo stesso quadro: pesi densi, nessun neurone importante, nessuna quantità di Astra.
Però gioca troppo male perché il confronto sia alla pari.

## 3. Cosa è stato interrotto e cosa c'è qui

### `ppo_v2_interrotto/<gioco>/`: la versione finale MinAtar

È la configurazione di `experiments/<gioco>/ppo/train.py`: minimal action set di MinAtar (Breakout fermo/sinistra/
destra, Asterix tutte tranne fuoco, Seaquest tutte), 100M passi. È stata fermata a 22-24M passi.

- `train_log.csv`: la curva fino a lì.
- `checkpoint_10.json`: l'attore a 10M passi.
- `console.log`: l'output.

Il trainer non sa riprendere un addestramento a metà: critico e ottimizzatore erano solo in memoria. Si riparte da
zero, ma si perde poco: tra 10M e 24M passi i punteggi non si sono mossi (Breakout 6.0, Asterix 5.7, Seaquest 5.5),
uguali a `ppo_v1`.

### `varianti_breakout/`: prove brevi su Breakout (6M passi previsti, fermate prima)

| variante | punteggio | quando |
|---|---|---|
| base (minimal action set) | 5.9-6.0 | per tutti i 24M passi |
| `sticky_0` (niente sticky actions in addestramento) | 6.5 | a 3.5M passi |
| **`ent_coef_0.05` (entropia ×5)** | **8.1, ancora in salita** | a 2.6M passi: esce dal plateau tra 1.6M e 1.8M |

Un attore 256-256 fuori regole fa 6.8 dopo 6M passi (`experiments/breakout/ppo/control_big.py`): la taglia della
rete non c'entra. Il problema è l'esplorazione, e l'entropia più alta è la prima cosa che aiuta.

## 4. Piano per domani

### 4.1 Preparare l'altro PC

```bash
git pull
uv sync --group ppo --group analysis
uv run --group ppo python -c "import torch; print(torch.__version__)"
```

**Sulla GPU.** Il `pyproject.toml` installa PyTorch **solo CPU** (indice `pytorch-cpu` in `[tool.uv.sources]`).
Una RTX 5080 qui aiuta poco:

- le reti sono minuscole (600-32-32-6);
- il tempo va quasi tutto nei giochi MinAtar, che sono Python e girano su CPU: ~100k passi/s per core il solo gioco,
  ~20k passi/s il trainer con 64 ambienti su un core.

Quello che accelera davvero è **usare più core per gli ambienti**:

- **Ambienti su più processi.** Il trainer oggi fa girare i 64 ambienti in un solo processo. Da fare all'inizio
  della sessione: un vettore di ambienti su N processi (stile `SubprocVecEnv`), opzione `n_workers` in
  `PPOConfig`. Con 8 processi per gioco ci si aspetta 3-5 volte più passi al secondo.
- **Riprendere a metà.** Da aggiungere insieme: un checkpoint completo ogni N aggiornamenti (attore, critico,
  ottimizzatore, numero di aggiornamento, stato dei generatori) e un'opzione `--resume`, così un addestramento lungo
  si può interrompere.
- **Solo se serve.** Passare a giochi su GPU (pgx o gymnax hanno MinAtar in JAX) sarebbe molto più veloce, ma non
  sono il pacchetto ufficiale `minatar==1.0.15` usato dall'articolo e dalla valutazione. Si addestrerebbe su un gioco
  leggermente diverso da quello su cui si valuta.

### 4.2 Addestrare

Prima una prova breve dell'entropia alta su tutti e tre i giochi, poi la corsa lunga con la configurazione migliore:

```bash
# prove da 20M passi, nome diverso per non sovrascrivere
for g in breakout seaquest asterix; do
  uv run --group ppo python experiments/$g/ppo/train.py --total-steps 20000000 --set ent_coef=0.05 --name prova_ent05 &
done; wait
# poi, con la configurazione migliore (se ent_coef 0.05 vince, metterlo nel CONFIG di train.py):
uv run --group ppo python experiments/<gioco>/ppo/train.py            # 100M passi, scrive weights/<gioco>/ppo.json
```

`--set campo=valore` cambia qualsiasi campo di `PPOConfig` (`ent_coef`, `sticky`, `n_envs`, `lr`, ...). Le prove con
nome diverso finiscono in `weights/<gioco>/<nome>.json` e `results/<nome>/`: cancellarle (o non committarle) se non
servono.

### 4.3 Analizzare e scrivere

```bash
uv run --group analysis python experiments/<gioco>/ppo/analyze.py --nets ppo --workers 4   # si aggiunge ad analysis.json
```

- **README di ogni gioco MinAtar:** scrivere `experiments/<gioco>/ppo/README.md` come
  `experiments/low_tide/ppo/README.md`. Le tabelle si generano con `small_games.analysis.markdown_tables(report)`.
  - Per Breakout includere la storia del plateau: `ppo_v1` che non si muove quasi mai, `follow_ball.py` 8.4, il
    controllo 256-256 e le varianti di questa cartella.
- **README principale:** aggiungere una voce per ogni `experiments/<gioco>/ppo/` e una tabella riassuntiva tra i
  giochi.
- **Chiusura:** commit, poi cancellare `tmp/`.

## 5. Cose da sapere

- **Nomi dei pesi:** `weights/<gioco>/ppo_v1.json` è il primo tentativo MinAtar (tutte e 6 le azioni, 30M passi);
  `ppo.json` sarà la versione finale. `analyze.py` include `ppo` da solo quando il file esiste.
- **Ablazione:** è a ricampionamento (un neurone prende il valore che aveva in un altro stato a caso), non a media.
  Con la media, i neuroni "rari ma grandi" delle reti scritte a mano sembravano tutti indispensabili.
- **Vocabolario:** le grandezze dei giochi sono in `features.py`. Per Asterix e Seaquest sono state arricchite (righe
  da −2 a +2, tempi d'arrivo) e le analisi rifatte: tutte le reti di un gioco usano lo stesso vocabolario.
- **Quantità chiave:** sono le grandezze calcolate dall'algoritmo scritto a mano, in `KEY_QUANTITIES` in
  `analysis.py`. Vanno definite come le calcola la rete. Esempi:
  - in Breakout i neuroni di Astra calcolano "dove arriverebbe se scendesse a destra/sinistra", non "dove arriverà";
  - in Seaquest "ora di emergere" è un solo neurone, "almeno un sub a bordo" da solo no.
  Una correlazione bassa può voler dire che la quantità è calcolata condizionata ad altro, non che manca.
- **Attese tra processi:** non usare cicli `while pgrep -f "..."` nei comandi di shell, perché il pattern trova la
  shell stessa. Concatenare i comandi con `;` o `&&`.
