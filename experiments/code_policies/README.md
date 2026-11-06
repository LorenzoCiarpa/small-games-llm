# Un algoritmo scritto come codice invece che come pesi

Le cartelle `decode_*` mostrano che le reti scritte a mano dagli LLM sono algoritmi messi nei pesi. Qui si chiede
allo stesso LLM di scrivere l'algoritmo direttamente come codice Python, con le stesse condizioni:
- una conversazione per gioco e una sola risposta;
- nessun tool: non si esegue codice e non c'è calcolatrice;
- nessun feedback.

Il confronto è tra lo stesso modello che scrive i pesi e lo stesso modello che scrive il codice, sullo stesso gioco.

## I prompt

```bash
uv run python experiments/code_policies/make_prompts.py    # scrive prompts/<gioco>.txt
```

Per Breakout, Seaquest e Asterix il prompt è quello dell'Esperimento 5 dell'articolo
(`exp5_write_from_scratch/runs/<gioco>/message.txt` in
[lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), CC BY 4.0).

**Uguali all'originale, parola per parola:**
- la descrizione del gioco;
- il sorgente esatto di MinAtar 1.0.15;
- l'input (600 numeri, indice `(riga * 10 + colonna) * 6 + canale`);
- la mappatura dei canali.

**Cambiano solo le parti sulla rete:**

| sezione | prompt dei pesi | prompt del codice |
|---|---|---|
| compito | i pesi di una rete | una funzione Python, l'algoritmo che vuoi |
| output | 6 logit, azione campionata dalla softmax | l'azione, un intero |
| rete / funzione | 600-32-32-6 tanh, ogni peso parte da 0 | `policy(obs, rng)`, regole sotto |
| come si valuta | 1000 episodi, sticky 0.1, 10.000 passi | uguale, con l'azione restituita dalla funzione |
| come si scrive | un blocco `edits` | un blocco `python` con il file completo |

Per Low Tide il prompt ha la stessa struttura:
- la descrizione del gioco;
- il sorgente di `env.py` (non `baselines.py`, che contiene l'euristica);
- l'input (403 numeri), l'output (5 azioni) e il protocollo (500 episodi, 1000 passi).

Se il prompt usato per far scrivere i pesi di Low Tide a Claude e GPT era diverso, conviene metterne qui la parte
sul gioco, per confrontare le due cose alla pari.

**Regole della funzione**, scritte nel prompt:
- **Solo l'osservazione corrente.** Nessuna memoria tra una chiamata e l'altra, come le reti con cui si confronta.
  Le tabelle costanti calcolate al caricamento vanno bene.
- **Solo standard library e numpy.** Non si importa il gioco.
- **Al massimo circa 1 ms per chiamata.**
- **Errori.** Se la funzione dà errore o restituisce un valore non valido, l'azione è 0.

La cartella `prompts/` non è versionata, perché contiene il sorgente di MinAtar (GPL-3.0).

## Come usarli

**Astra, con Codex senza tool:**

```bash
uv run python experiments/code_policies/run_codex.py low_tide --dry-run   # controlla cosa vedrebbe il modello
uv run python experiments/code_policies/run_codex.py low_tide             # gpt-6-astra, xhigh, come l'articolo
uv run python experiments/code_policies/run_codex.py breakout             # ... un gioco per volta
```

Ogni chiamata è una conversazione nuova. La risposta va in `answers/<modello>/<gioco>/`:
- `answer.md`, la risposta;
- `policy.py`, l'ultimo blocco `python`;
- `events.jsonl`, gli eventi grezzi;
- `run.json`, tempi, token e controlli.

**Come si tolgono i tool.** In Codex 0.164 la configurazione dell'articolo (`codex_no_tools.toml`) non basta.
Gli strumenti di gpt-6-astra vengono dal catalogo dei modelli di Codex:
- `tool_mode = "code_mode_only"` dà il code mode, con dentro apply_patch, MCP e un orologio;
- `multi_agent_version = "v2"` dà gli agenti;
- `request_user_input` e le skill di sistema arrivano dalla configurazione.

`run_codex.py`:
1. usa una CODEX_HOME temporanea con quella configurazione e una copia del catalogo senza quei campi;
2. prima di mandare il prompt, controlla con `codex debug prompt-input`, che non chiama il modello, che il modello
   veda 0 tool e solo due messaggi: le istruzioni neutre dell'articolo e il prompt;
3. dopo, conta le chiamate ai tool negli eventi.

`--check` lo verifica con una domanda breve: "quali tool puoi chiamare?". Senza le modifiche il modello riceveva
5.098 token di contesto ed elencava exec, apply_patch, gli agenti e altri strumenti. Con le modifiche riceve 59
token e risponde `NONE`.

**Claude:** dall'API senza tool, oppure da claude.ai con esecuzione di codice e ricerca web disattivate, incollando
tutto il file del prompt come unico messaggio. Salvare la risposta nello stesso formato, in
`answers/<modello>/<gioco>/answer.md`.

## Risultati: GPT-6 Astra (6 novembre 2026)

Una risposta per gioco, con `run_codex.py`:
- `gpt-6-astra`, reasoning `xhigh`;
- 0 tool visibili e 0 chiamate;
- tra 14 e 23 minuti per risposta.

I punteggi sono di `evaluate_code.py`, con lo stesso protocollo delle reti.

| gioco | **codice di Astra** | rete di Astra (pesi) | altri riferimenti |
|---|---|---|---|
| Low Tide (500 ep.) | **188.9 ± 0.7** | 79.9 ± 1.0 (`gpt`) | euristica BFS 164.0, miglior PPO 166.9, `claude` 132.3 |
| Breakout (1000 ep.) | **786.0 ± 2.6** | 512.5 ± 7.6 (`paper_astra`) | PPO 500M passi (articolo) 25.3 |
| Seaquest (1000 ep.) | **187.3 ± 2.8** | 51.4 ± 1.4 (`paper_astra`) | PPO 500M passi (articolo) 88.9 |
| Asterix (1000 ep.) | **83.2 ± 1.8** | 54.8 ± 1.1 (`paper_astra`) | PPO 500M passi (articolo) 55.2 |

**In tutti e quattro i giochi il codice batte la rete scritta dallo stesso modello e il miglior PPO.**

**Controlli superati da tutte e quattro le policy, su 2000 stati:**
- importano solo numpy e la standard library;
- 0 errori e 0 azioni non valide;
- 95-283 µs per chiamata;
- nessuna memoria: stessa azione con gli stati in ordine inverso.

**Cosa fanno, tutte calcolate a ogni passo dalla sola osservazione:**
- **Low Tide.** Calcola, per ogni cella, l'ultimo passo in cui ci si può arrivare restando in tempo per tornare all'asciutto. Per il tempo usa il caso peggiore, cioè intervalli di marea di 3 passi.
  - Le mosse sono controllate anche se la sticky action le ripete.
  - Il percorso tra le conchiglie è scelto con una beam search.
  - C'è un bonus per restare vicino alla battigia, dove compaiono le conchiglie nuove.
  - Annega nello 0.6% degli episodi, contro l'11% dell'euristica BFS; prende 190.4 conchiglie e 1.5 punture.
- **Breakout.** Fisica esatta della palla, ricerca su 5 rimbalzi sulla racchetta e programmazione dinamica sulla posizione della racchetta con le sticky actions. Il 96.7% degli episodi arriva a 10.000 passi.
- **Seaquest.** Modello delle collisioni a breve orizzonte, sub da recuperare, ossigeno, siluri nemici e rischio della sticky action.
- **Asterix.** Pianificazione a breve orizzonte per intercettare i tesori, con un modello veloce e uno lento del movimento. Una ricerca di sicurezza separata tiene conto di sticky actions e nuovi nemici.

**Nota su Seaquest.** Il contatore dei token in `run.json` dà 26.031 token in ingresso e solo 516 di ragionamento, contro 14-21 mila negli altri giochi. Probabilmente conta solo l'ultima richiesta, dopo una compattazione automatica del contesto. Gli eventi mostrano un solo turno, 42 blocchi di ragionamento, nessun tool e nessun errore.
