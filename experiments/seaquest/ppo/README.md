# PPO on Seaquest: which weights does it find, and can they be read?

`train.py` trains with PPO a 600-32-32-6 tanh network, the same architecture as the network that GPT-6 Astra wrote
by hand in the article (`paper_astra.json`), and saves it in `weights/seaquest/ppo.json`. `analyze.py` takes the same
measurements on the two networks. Astra's network is decoded neuron by neuron in
[`decode_astra`](../decode_astra/README.md) and serves as the reference point. The measurements are those of
[PPO on Low Tide](../../low_tide/ppo/README.md) and [PPO on Breakout](../../breakout/ppo/README.md), explained there.

Seaquest is the MinAtar game where the comparison is even: PPO plays **better** than Astra.

**In short.**

- **How it plays:** 55.9 points against Astra's 51.4, and 107 against 54 when always playing the most likely move. But
  it plays in a different way: Astra stays still in column 5 and only chooses the row; PPO moves between columns 3 and
  6, turning right and left.
- **What the weights look like:** dense and distributed.
  - It uses all 425 live inputs, with 13,526 distinct values; Astra 2.6%, with 12 values.
  - Keeping half of the live first-layer weights it loses half of its score; with a quarter it no longer plays. Astra
    holds up with 5%.
  - No neuron computes "low oxygen" or "time to surface", which in Astra are one neuron each (|r| 0.98 and
    1.00 against 0.29). Yet PPO surfaces too, and goes up more often when oxygen is low.
  - A depth-6 tree imitates it in 65% of states (Astra: 87%).
- **What can be read:** the most important first-layer neurons have localized patches in the submarine channel
  (where it is) combined with patches in the enemy channels (what is nearby). They cannot be reduced to a single game
  quantity.

```bash
uv run --group ppo python experiments/seaquest/ppo/train.py          # ~1h50 on one core (200M steps)
uv run --group analysis python experiments/seaquest/ppo/analyze.py   # ~10 min with 4 processes
```

## 1. Training

As for Breakout (`src/small_games/ppo.py`, PPO with clipping and GAE): actor 600-32-32-6 tanh, critic 600-64-64-1,
64 environments, rollouts of 128 steps, Adam lr 5·10⁻⁴ decaying to 0, sticky actions 0.1 as in evaluation, training
seeds far from the evaluation ones. All six actions (MinAtar's minimal action set for Seaquest is complete),
**entropy coefficient 0.05**, 200 million steps.

The first attempt (`ppo_v1`, entropy 0.01, 30M steps) had stopped at 5.4 points. The three runs of
`experiments/ppo_runs.py` differ only in the entropy coefficient:

| run | passes 10 points | training return at 50M / 100M / 200M | evaluation (1000 episodes) |
|---|---|---|---|
| `ppo_ent01` (0.01) | never | 4.4 / 4.4 / 4.5 | 4.4 ± 0.05 |
| **`ppo_ent05` (0.05)** | **at 36M steps** | 24 / 37 / 57 | **55.9 ± 1.3** |
| `ppo_ent10` (0.1) | at 28M steps | 16 / 23 / 34 | 34.4 ± 1.0 |
| Astra's network | | | 51.4 ± 1.4 |

With entropy 0.01 PPO stays stuck for all 200M steps. `weights/seaquest/ppo.json` is `ppo_ent05`: its
curve is still rising at the end, so with more steps it would probably improve further.

## 2. How it plays

| | `paper_astra` | `ppo_v1` | `ppo` |
|---|---|---|---|
| score (200 episodes, sampled) | 48.3 ± 2.9 | 5.4 ± 0.1 | 56.0 ± 3.0 |
| greedy score | 53.9 | 5.4 | 107.4 |

On the states played by the two networks (20 episodes, most likely moves):

| | `paper_astra` | `ppo` |
|---|---|---|
| column of the submarine | always 5 | 3: 14%, 4: 29%, 5: 35%, 6: 18% |
| facing left / right | 99% / 0% | 49% / 50% |
| moves: no-op, left, up, right, down, fire | 0, 0, 9, 0, 9, 81% | 5, 9, 31, 12, 29, 14% |
| goes up, with oxygen < 60 and at least one diver on board / with oxygen ≥ 60 | 72% / 6% | 57% / 27% |

Astra reduced the game to one dimension: still in column 5, it fires almost always and chooses the row. PPO did not
find this simplification: it moves horizontally, turns around, fires much less. PPO too, however, learns to
surface: with little oxygen and some divers on board it goes up twice as often.

## 3. What a PPO weight looks like

| | `paper_astra` | `ppo_v1` | `ppo` |
|---|---|---|---|
| nonzero weights | 3.6% | 100.0% | 100.0% |
| distinct values in the first layer | 12 | 18,735 | 18,950 |
| inputs that make up 90% of a neuron's weight (median, out of 600) | 9 | 354 | 302 |
| largest weight in absolute value | 40.00 | 1.41 | 3.84 |

175 inputs out of 600 are always zero: the gauge channel outside row 9, row 9 of the other channels and a few
cells never visited. The PPO weights on these stay those of the random initialization (see Breakout). Restricted to
the live inputs:

| first layer, only the 425 live inputs out of 600 | `paper_astra` | `ppo_v1` | `ppo` |
|---|---|---|---|
| nonzero weights | 2.6% | 100.0% | 100.0% |
| distinct values | 12 | 13,422 | 13,526 |
| inputs that make up 90% of a neuron's weight (median, out of 425) | 9 | 254 | 240 |
| mean \|w\| on dead / live inputs | 0.012 / 0.158 | 0.052 / 0.125 | 0.064 / 0.295 |

The first-layer weights of the most important neurons (section 5), one panel per channel, red positive and blue
negative.

**Astra's network:** neurons "the submarine is in row r" (a whole row of the player channel), "there is a danger
in row r near column 5" (danger channel, weights centred on column 5) and a threshold on oxygen
(gauge channel).

![first layer of paper_astra.json](results/analysis/fields_paper_astra.png)

**PPO network:**

![first layer of ppo.json](results/analysis/fields_ppo.png)

In the submarine channel (player) every neuron has localized patches: positive in one zone, negative in another
(u29 and u12 for example). It is a position code by zones, not by row. In the danger and projectile channels there
are wider patches, often in the lower half of the screen, sometimes of opposite sign to those of the submarine.
The single patches have a shape; the neuron, which sums them all, has no name.

Astra's first layer puts 47% of its weight on the danger channel, 31% on reward and 22% on player, and ignores the
trail: it does not need to know where objects are going, because it waits for them to pass in front of column 5. PPO
splits the weight between player (25%), danger (30%), projectile (24%) and trail (12%).

## 4. How many weights are needed (pruning)

All layers (200 episodes):

| weights kept | 100% | 50% | 25% | 10% | 5% | 2% |
|---|---|---|---|---|---|---|
| `paper_astra` | 48.3 | 48.3 | 49.8 | 1.0 | 1.0 | 0.7 |
| `ppo_v1` | 5.4 | 5.2 | 5.1 | 1.2 | 0.2 | 0.0 |
| `ppo` | 56.0 | 16.4 | 1.2 | 0.4 | 0.1 | 0.1 |

First layer only, on the live inputs:

| live first-layer weights kept | 100% | 50% | 25% | 10% | 5% |
|---|---|---|---|---|---|
| `paper_astra` | 48.3 | 48.3 | 48.3 | 48.3 | 48.3 |
| `ppo_v1` | 5.4 | 5.3 | 5.3 | 2.4 | 1.0 |
| `ppo` | 55.3 | 31.0 | 1.7 | 1.2 | 0.2 |

(With all the live weights `ppo` scores 55.3 instead of 56.0: a few inputs turn on in evaluation episodes but never
in the states used to find the live ones.)

## 5. How much each neuron counts (ablation)

| | neurons (constant) | causing a loss of more than 10% | more than 50% | maximum loss | reference |
|---|---|---|---|---|---|
| `paper_astra` | 64 (6) | 40 | 29 | 42.1 | 45.7 ± 3.9 |
| `ppo_v1` | 64 (0) | 0 | 0 | 0.2 | 5.4 ± 0.2 |
| `ppo` | 64 (0) | 62 | 14 | 52.8 | 59.6 ± 4.6 |

In both networks almost every neuron counts: in a network with 32 units per layer, removing one breaks the
computation. The difference is in the meaning. In Astra the indispensable neurons are "the submarine is in row r",
"danger in row r", "low oxygen". In PPO the most important one (u23 of the second layer, loss of 53 points out of 60)
has |r| 0.42 with the row of the submarine, and no game quantity explains it better.

## 6. What the neurons compute

### The quantities of Astra's algorithm

| quantity | `paper_astra` r / R² | `ppo_v1` r / R² | `ppo` r / R² |
|---|---|---|---|
| low oxygen (< 60) | **0.98 / 0.93** | 0.20 / 0.48 | 0.29 / 0.19 |
| at least one diver on board | 0.10 / 0.00 | 0.23 / 0.21 | 0.21 / 0.18 |
| time to surface (oxygen < 60 and at least one diver) | **1.00 / 1.00** | 0.17 / 0.39 | 0.29 / 0.19 |
| enemy near in my row (≤ 2) | 0.07 / 0.03 | 0.14 / 0.00 | 0.28 / 0.22 |
| diver to rescue in my row | 0.15 / 0.45 | 0.14 / 0.03 | 0.04 / 0.00 |

"Time to surface" in Astra is a neuron. In PPO it is not there, not even as a linear combination of the neurons of a
layer (R² 0.19), even though the network behaves as if it used it (section 2). Astra's row quantities ("enemy near
in my row") have a low r even in Astra: Astra computes them row by row, for the three candidate rows, not for
"my row" (sections 4 and 5 of `decode_astra`).

### Boolean or graded neurons

| | `paper_astra` layer 0 | `paper_astra` layer 1 | `ppo` layer 0 | `ppo` layer 1 |
|---|---|---|---|---|
| saturated tanh neurons (\|a\| > 0.95) | 99% | 73% | 19% | 73% |
| neurons with \|r\| ≥ 0.8 with a game quantity | 0/32 | 0/26 | 0/32 | 0/32 |
| mean R² of a neuron on the game quantities | 0.12 | 0.15 | 0.26 | 0.28 |

Astra's first layer is made of 32 true/false conditions; PPO's of 32 graded neurons (none saturated
in 90% of states). Here the game vocabulary does not tell the two networks apart: it explains the PPO neurons slightly
better than Astra's (R² 0.26-0.28 against 0.12-0.15), because Astra computes conditions on single rows and cells,
which the vocabulary (distances, rows relative to the submarine) does not contain. In Astra, the mark of an algorithm
is something else: every neuron is a condition on a few inputs, with round weights.

## 7. Can it be written as a rule? (distillation)

| depth | `paper_astra` agreement / leaves / points | `ppo_v1` | `ppo` |
|---|---|---|---|
| 1 | 84.2% / 2 | 36.2% / 2 | 38.7% / 2 |
| 2 | 85.1% / 4 / 0.0 | 44.2% / 4 / 1.1 | 52.8% / 4 / 1.3 |
| 4 | 87.2% / 16 / 0.7 | 46.1% / 16 / 0.8 | 59.6% / 16 / 1.3 |
| 6 | 87.1% / 45 | 50.7% / 37 | 65.2% / 57 |
| 8 | 90.9% / 106 / 3.3 | 52.9% / 67 / 1.8 | 70.4% / 154 / 5.9 |
| 12 | 93.0% / 326 / 9.1 | 53.4% / 120 / 1.6 | 71.8% / 424 / 13.3 |
| the network, greedy | 53.9 | 5.4 | 107.4 |

Astra's 84% at depth 1 comes mostly from the fact that it fires in 81% of states. No tree plays
well: Seaquest punishes every mistake (a wrong move near an enemy costs a life) and the quantities of the
vocabulary are not enough to write the exact rule, not even Astra's. But Astra's tree gets 7% of the moves wrong
with 326 leaves, PPO's gets 28% wrong with 424.

PPO's depth-2 tree (53% agreement) says: "if there is a target ahead within 5 columns and I do not already have a
torpedo in flight in the row, fire; otherwise move towards the central rows". A sensible rule, which however describes
only half of the moves.

## 8. Conclusion

- **Astra:** the weights are the program. True/false conditions on rows and cells, round weights on a few inputs, a
  one-dimensional strategy (still in column 5).
- **PPO:** it plays better than Astra, and its solution does not have this form:
  - it uses all the live inputs, with continuous values, and does not withstand pruning;
  - every neuron counts, none corresponds to a game quantity or to a condition of Astra's algorithm;
  - the behaviour is also more complex: it moves horizontally and cannot be summarized in a short tree.

The only readable trace is the position patches in the submarine channel.

Over the course of training and with the other entropies the picture is the same: see
[`experiments/ppo_over_training`](../../ppo_over_training/README.md).

## Files

- `train.py`: configuration and launch of the training (the one of `ppo.json`).
- `analyze.py`: all the measurements; with `--only-key` only the key quantities and the measurements on the live
  inputs.
- `results/ppo_ent01/`, `results/ppo_ent05/`, `results/ppo_ent10/`: the three 200M-step runs (curve
  `train_log.csv`, `checkpoint_*M.json` every 5M steps, `eval.json`, `final.pt`, `console.log`).
- `results/ppo_v1/`: the first attempt (entropy 0.01, 30M steps).
- `results/analysis/`: `analysis.json` (all the numbers), `trees_*.txt` (trees of depth 1-4), `fields_*.png`
  (images of the first layer).
