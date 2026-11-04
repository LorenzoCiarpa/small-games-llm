# Quanto sono lontane tra loro le reti scritte dagli LLM?

**L'ipotesi da mettere alla prova.** I pesi che gli LLM scrivono "a mano" non nascono dal nulla: i modelli hanno
già visto reti neurali per questi giochi e le ricordano. Se fosse così, le reti per questo tipo di compito
dovrebbero essere più o meno tutte uguali, e conoscendone una si risolverebbero tutti i giochi.

Da questa ipotesi seguono previsioni che si possono controllare sui pesi:

1. se i pesi fossero il ricordo di reti addestrate, dovrebbero **somigliare a reti addestrate**;
2. se le reti fossero tutte più o meno uguali, la loro **distanza strato per strato** dovrebbe essere molto più
   piccola di quella tra due reti senza niente in comune;
3. le unità di una rete dovrebbero avere un **gemello** nell'altra, e gli strati dovrebbero calcolare
   **rappresentazioni** simili sugli stessi stati;
4. la rete di un gioco dovrebbe **giocare** discretamente anche agli altri.

I numeri sono in [`results/report.md`](results/report.md) (tabelle) e in `results/results.json` (tutto).

```bash
uv run --group analysis python experiments/weight_distance/analyze.py          # ~35 s su 8 core
uv run --group analysis python experiments/weight_distance/analyze.py --help   # episodi, ripetizioni, ...
```

Il gruppo `analysis` aggiunge solo scipy (algoritmo ungherese). Tutto è con seed fisso, quindi lo stesso comando
dà gli stessi numeri.

## Le reti

| rete | autore | gioco | architettura | punteggio |
|---|---|---|---|---|
| `astra/breakout` | GPT-6 Astra, nell'articolo | Breakout (MinAtar) | 600-32-32-6, tanh | 512.5 (casuale 0.5) |
| `astra/seaquest` | GPT-6 Astra, nell'articolo | Seaquest (MinAtar) | 600-32-32-6, tanh | 51.4 (casuale 0.04) |
| `astra/asterix` | GPT-6 Astra, nell'articolo | Asterix (MinAtar) | 600-32-32-6, tanh | 54.8 (casuale 0.5) |
| `claude/low_tide` | Claude Opus, per questo progetto | Low Tide | 403-46-36-5, relu/tanh | 132.3 (euristica 164.0, casuale 2.0) |
| `gpt/low_tide` | GPT-6 Astra, per questo progetto | Low Tide | 403-43-61-53-5, relu | 79.9 |
| `ppo/{breakout,pong,pacman,freeway}-grid` | **addestrate con PPO**, 10⁹ passi | giochi a griglia degli autori | 500-32-32-5, tanh | — |

Le prime cinque sono scritte da un LLM in una sola risposta, senza strumenti né training: le tre di MinAtar sono
quelle dell'articolo (`weights/<gioco>/paper_astra.json`), le due di Low Tide sono in `weights/low_tide/`.

Le quattro reti PPO **non giocano ai nostri giochi** e servono solo come termine di paragone, per vedere com'è fatta
una rete *addestrata* di questa taglia. Vengono dall'Esperimento 1 dell'articolo ("su quale gioco è stata addestrata
questa rete?"): gli autori le hanno addestrate su loro versioni 10×10 di Breakout, Pong, Pac-Man e Freeway, con 5
canali (agente, oggetti in movimento, ricompense, ostacoli, frame precedente) e 5 azioni. Sono in `reference/`
(solo l'attore, `W` = kernel trasposto), copiate da
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) `exp1_which_game/networks/`
(a9f535cc, dfc6cb65, bdf99279, 0bbcc3ea), CC BY 4.0. Le reti PPO addestrate su MinAtar, con cui l'articolo
confronta quelle di Astra, non sono state pubblicate.

## Cosa misura `analyze.py`, e perché non basta ‖A − B‖

- **Le unità nascoste si possono rimescolare.** Riordinando i 32 neuroni di uno strato nascosto, e insieme le
  colonne corrispondenti dello strato dopo, la rete calcola esattamente la stessa funzione. Con tanh si può anche
  cambiare segno a un neurone: ingressi e uscite col segno opposto. Due copie della stessa rete con i neuroni in
  ordine diverso possono quindi avere una ‖A − B‖ enorme. Per questo, oltre alla distanza **grezza**, lo script
  calcola quella **allineata**: cerca il riordino (con i cambi di segno) dei neuroni di B che la porta più vicina ad A
  (weight matching, come in Git Re-Basin, Ainsworth et al. 2022). La verifica, una rete contro una sua copia
  rimescolata, dà distanza 0.
- **Serve un termine di paragone.** L'allineamento accorcia la distanza anche tra reti che non c'entrano niente.
  Il **modello nullo** è quindi B coi suoi stessi valori sparpagliati a caso dentro ogni strato, poi allineata
  allo stesso modo. Se allineata ≈ nullo, le due reti non hanno in comune più di due reti a caso.
- **La distanza relativa** ‖A − B‖ / √(‖A‖² + ‖B‖²) vale 0 per reti identiche, circa 1 per reti scorrelate e
  √2 per reti opposte.
- **La scala nelle reti scritte a mano è arbitraria.** Ci sono pesi da 1000 e oltre, e una tanh saturata non
  distingue 100 da 1000. Per questo la ‖A − B‖ misura soprattutto dove stanno quei pochi pesi enormi. Le misure
  che non dipendono dalla scala sono quelle che seguono.
- **Unità gemelle**: le unità del primo strato di A e di B sono accoppiate una a una per coseno dei pesi in
  ingresso. Il nullo è lo stesso accoppiamento con gli ingressi di B rimescolati.
- **CKA lineare** (Kornblith et al. 2019) tra le uscite di ogni coppia di strati, calcolata sugli stessi stati di
  gioco. Funziona anche tra reti di forma diversa, come le due di Low Tide. Il riferimento è una rete con la
  stessa forma e pesi casuali, che misura quanta somiglianza viene "gratis" dal leggere gli stessi ingressi.
- **Comportamento**: quanto spesso due reti scelgono la stessa azione sugli stessi stati, rispetto al caso, e il
  **cross-play**, cioè la rete di un gioco che gioca agli altri due.
- **Reti di forma diversa** (sezione 7): ogni rete è riscritta, senza cambiare quello che calcola, in uno spazio
  comune di ingressi (griglia 10×10 a 7 canali più 3 scalari) e di azioni, con gli strati nascosti allargati da
  unità a zero. Il primo strato e l'uscita si allineano in modo esatto. Gli strati intermedi si allineano in modo
  esatto solo tra reti con lo stesso numero di strati nascosti; altrimenti il risultato è approssimato, e accanto
  c'è il limite superiore di von Neumann.

## Risultati

**1. Le reti scritte a mano non somigliano per niente a reti addestrate.** Per le reti PPO "valori distinti" vuol
dire che quasi nessun valore si ripete.

| | scritte da LLM (5 reti) | addestrate con PPO (4 reti) |
|---|---|---|
| pesi non nulli | 3–5% | 100% |
| valori distinti | 40–62 | ~17 000 su ~17 250 |
| non nulli con al più 2 decimali | 94–100% | 0% |
| peso più grande (in valore assoluto) | da 244 a 5122 | meno di 9 |
| quota coperta dagli 8 valori più usati | 34–81% | 0% |

Le reti scritte a mano sono fatte di poche costanti tonde, ripetute. Claude in Low Tide usa soprattutto potenze di 2
(1, 8, 16, 32, 64, 1024); Astra in Asterix usa ±8, ±16, 24 e 1000 come soglie. Sono programmi tradotti in pesi.
Nessuna discesa del gradiente produce una rete così, e nessuna rete addestrata ha questo aspetto: quello che
scrivono non è il ricordo dei pesi di una rete allenata.

**2. Reti per giochi diversi: distanti quanto reti a caso, strato per strato.** Le tre reti di Astra hanno la stessa
forma, gli stessi ingressi e le stesse azioni, quindi sono confrontabili peso per peso. La loro distanza relativa
grezza è circa 1.00 in tutti gli strati, cioè sono ortogonali. Anche dopo l'allineamento resta circa uguale al
nullo (tutta la rete: 1.00 contro 0.99, 0.99 contro 0.98, 1.01 contro 1.08).

L'unica somiglianza vera è tra il primo strato di Seaquest e quello di Asterix:
- distanza allineata 0.79 contro 0.95 del nullo;
- 15 unità su 32 con coseno > 0.5, contro 0 del nullo.

I due giochi in effetti si somigliano: nemici e premi che scorrono su righe e un giocatore da spostare. Già al
secondo strato però non sono più vicine del caso.

Le reti PPO per giochi diversi si comportano allo stesso modo: distanza allineata di tutta la rete 0.87–0.94,
contro 0.94–0.95 del nullo.

**Le quattro reti di Astra, compresa Low Tide** (sezione 7 del report). Gli strati intermedi della rete di Low Tide
non sono uguali a quelli delle reti MinAtar: Low Tide è 403-43-61-53-5, cioè 3 strati nascosti di larghezza
diversa, mentre le reti MinAtar sono 600-32-32-6. Per confrontarle, ogni rete è riscritta senza cambiare quello
che calcola:
- gli ingressi vanno in uno spazio comune: agente → player, meduse → danger, conchiglie → reward, più acqua e
  scalari come canali propri;
- le azioni sono rimesse nello stesso ordine;
- gli strati sono allargati con unità a zero.

Ogni strato è poi allineato nel modo più favorevole. La tabella riporta il coseno dopo l'allineamento, tra
parentesi quello del nullo: 1 vuol dire stessi pesi, 0 niente in comune.

| coppia | primo strato | intermedio | uscita | tutta la rete |
|---|---|---|---|---|
| Breakout – Seaquest | 0.00 (0.02) | 0.37 (0.40) | **−0.82** (−0.11) | −0.00 (0.06) |
| Breakout – Asterix | 0.06 (0.02) | 0.57 (0.46) | −0.20 (0.21) | 0.03 (0.04) |
| Seaquest – Asterix | **0.44** (0.12) | 0.37 (0.50) | −0.04 (−0.22) | −0.02 (−0.22) |
| Low Tide – Breakout | 0.00 (0.06) | 0.40 (0.43)* | 0.02 (0.14) | — |
| Low Tide – Seaquest | 0.14 (0.06) | 0.80 (0.78)* | 0.12 (0.10) | — |
| Low Tide – Asterix | 0.03 (0.06) | 0.70 (0.55)* | 0.07 (0.04) | — |
| *riferimento:* Low Tide, Astra – Claude | 0.26 (0.05) | 0.59 (0.70)* | 0.12 (0.11) | — |

- **Primo strato** (quello che legge lo schermo). La rete di Low Tide non ha niente in comune con le tre di
  MinAtar: coseno da 0.00 a 0.14. Tra le MinAtar si somigliano solo Seaquest e Asterix (0.44). Le due reti di Low
  Tide di Astra e di Claude, cioè stesso gioco e autori diversi, si somigliano di più (0.26) di quanto la rete di
  Astra per Low Tide somigli alle reti di Astra per gli altri giochi.
- **Strati intermedi.** Non dicono niente, perché il nullo è già alto (0.40–0.78). Con così pochi pesi non nulli
  anche valori rimescolati a caso si allineano bene. Con `gpt/low_tide` (*) il confronto è anche approssimato:
  il coseno trovato è un limite inferiore e il limite superiore (0.92–0.97) è troppo largo per dire qualcosa.
- **Uscita.** Breakout e Seaquest sono quasi *opposte* (−0.82). Breakout spinge su sinistra e destra (+90) e
  vieta su, giù e fuoco (−100), Seaquest fa il contrario.
- **Tutta la rete.** Coseno circa 0, come il nullo. In norma 2, per esempio, Breakout e Seaquest distano 3623, con
  norme 3588 e 498: la distanza è grande quanto la rete più grande.

**3. Ogni rete legge cose diverse.**
- Breakout legge la palla (72% delle connessioni del primo strato), la sua scia (23%) e appena la racchetta (5%).
- Seaquest legge pericoli, premi e giocatore.
- Asterix legge scia, pericoli, giocatore e premi.

Sono programmi scritti per quel gioco, non una rete universale.

**4. Nessuna rete sa giocare agli altri giochi.** Nel cross-play (200 episodi per casella) ogni rete segna solo nel
suo gioco: 512.7, 48.3 e 54.5. Su un altro gioco va come il caso, per esempio:
- la rete di Breakout su Asterix fa 0.6, contro 0.5 delle azioni casuali;
- la rete di Seaquest su Breakout fa 0.5.

Sugli stati di un altro gioco le reti scelgono la stessa azione tanto spesso quanto due scelte indipendenti.

**5. Stesso gioco, due LLM diversi (Low Tide): stessi mattoni, resto diverso.** È l'unica coppia di reti scritte
per lo stesso gioco. Le architetture sono diverse, quindi non c'è una ‖A − B‖ diretta.
- **Primo strato.** 3 unità su circa 45 sono quasi identiche (coseno > 0.9) e 7 superano 0.5, contro 0.9 del
  nullo. Le gemelle sono mattoni elementari che entrambe le reti scrivono allo stesso modo:
  - la copia dello scalare "la marea sale";
  - "l'agente è nell'ultima colonna";
  - "l'agente è nell'ultima riga".
- **CKA.** Gli strati finali si somigliano (logits 0.75), quelli intermedi in parte (circa 0.5). I primi strati no:
  al massimo 0.11, contro 0.07 della rete casuale.
- **Azioni.** Le due reti scelgono la stessa azione il 50% delle volte (il caso darebbe 23%). È più di quanto
  ciascuna concordi con l'euristica: 43% Claude, 35% GPT.

Il comportamento converge, i pesi no.

## Cosa se ne può concludere

- **"Le reti sono tutte più o meno uguali": i dati non lo sostengono.** Due reti per giochi diversi sono distanti
  quanto due reti scorrelate, in ogni strato, anche dopo aver tolto la simmetria dei neuroni. Anche le due reti per
  lo stesso gioco condividono solo qualche unità elementare. Hanno in comune lo **stile**:
  - pochi pesi tondi;
  - tanh e relu usate come porte logiche e soglie;
  - unità che rispondono a domande del tipo "dov'è l'agente?".

  È un modo di programmare, non un insieme di pesi.
- **"Le hanno viste e le ricordano": non per i pesi.** Una rete ricordata avrebbe l'aspetto di una rete
  addestrata, e queste ne sono l'opposto. Quello che il modello può aver visto è la conoscenza dei giochi e delle
  strategie (per esempio "segui la palla"), insieme a come calcola un MLP. Questo esperimento non distingue tra una
  strategia *ricordata* e una *ricavata dalle regole*. Il controllo giusto per questo è Low Tide, un gioco nuovo,
  dove entrambe le reti funzionano: 132 e 80 contro 2 del caso, anche se sotto l'euristica (164).

## Limiti e prossimi passi

- **C'è una sola rete per coppia (autore, gioco).** Quindi non si vede quanto variano due risposte dello stesso
  modello allo stesso compito. È il test decisivo per "convergono sempre alla stessa rete": chiedere 3–5 volte
  (conversazioni nuove) la rete di Breakout, a Claude e ad Astra, e confrontare la distanza *dentro* lo stesso
  gioco con quella *tra* giochi. Se le reti per Breakout risultassero quasi identiche tra loro e lontane da quelle
  per gli altri giochi, l'ipotesi del ricordo tornerebbe in gioco. Lo script le confronta da solo se hanno la
  stessa forma: basta aggiungerle in `load_nets()`.
- **Le reti PPO sono di altri giochi.** Il confronto diretto, rete scritta contro rete addestrata sullo stesso
  gioco, richiede di addestrare un 600-32-32-6 con PPO su MinAtar, perché i checkpoint dell'articolo non sono
  pubblici.
- **Il CKA ha un pavimento alto** (fino a 0.4 con reti casuali, sui giochi MinAtar) su questi ingressi sparsi e binari. Va sempre letto
  rispetto alla colonna "rete casuale".
