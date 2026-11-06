# The Low Tide BFS heuristic written into the weights of an MLP

The `decode_*` folders start from a network written by an LLM and recover its algorithm. Here we do the reverse: we
start from the algorithm of the Low Tide heuristic (`heuristic_policy`, which uses a BFS) and write it into the weights.
The network must stay within the game's limits:

- at most 3 hidden layers of 64 units;
- relu, tanh or linear activations;
- no training.

`build.py` writes `weights/low_tide/bfs_net.json`, a 403 → 64 (relu) → 58 (tanh) → 63 (relu) → 5 network, with 6,806
non-zero parameters out of 33.663. Every weight comes from a formula.

**Result:** 166.7 ± 2.0 points over the 500 protocol episodes.

| network or policy | score |
|---|---|
| BFS heuristic | 164.0 ± 1.5 |
| `claude` | 132.3 ± 0.9 |
| `gpt` | 79.9 ± 0.9 |
| best PPO (`ppo_arch_gpt`, 100M steps) | 166.9 ± 2.7 |
| `bfs_net` | 166.7 ± 2.0 |

**The network is not the exact BFS, which does not fit (section 1).** It is a potential-based version of the same algorithm:
- it goes toward the nearest shell;
- it follows the same tide rules;
- it never steps into a jellyfish.

It makes the same move as the BFS in 78.8% of the states. In almost all the others, it either takes another equally
short step or goes toward a shell that the BFS excludes (section 5).

```bash
uv run python experiments/low_tide/bfs_net/build.py               # writes the weights
uv run python experiments/low_tide/bfs_net/check.py               # verification, ~2 min
uv run evaluate low_tide bfs_net --baseline heuristic
uv run play low_tide --weights bfs_net                            # to watch it play
```

## 1. Why the exact BFS does not fit

**The first layer sees a cell "relative to the agent" only with one neuron per cell.** A first-layer neuron
computes a fixed weighted sum of the inputs.
- The weight of a cell cannot change with the agent's position. A question like "is there a shell two cells
  to my left?" cannot be written with a single neuron for all agent positions.
- Instead, each cell can be given its own neuron: "is there a shell here, and how far is it from the agent?". The
  agent's position is a one-hot, so the distance is one weight per agent cell.
- The cells that matter are the 60 of rows 3-8. Row 9 is never allowed, because the tide reaches
  level 0 only to rise again.
- That leaves 4 neurons for the agent's position and the tide, and none for a separate jellyfish map.

**Going around jellyfish requires one propagation step for each path step.** A BFS that goes around an
obstacle propagates the distance cell by cell. In the protocol episodes the BFS path reaches 26
steps. With 3 layers at most 3 can be done.

**"The nearest" is a minimum over 60 cells.** With two relu layers after the first it cannot be computed exactly.

Without the limits the exact BFS can be written, but it needs 104 layers and 51,103 neurons: see
[`../bfs_exact`](../bfs_exact).

So the network keeps from the BFS what can be written in 3 layers:
- the nearest shell, approximated with a potential;
- the tide rules, exact;
- the jellyfish, but only as "do not enter the cell", without going around them along a path.

## 2. The algorithm the network computes

`policy_logits` in `build.py` is the specification, written directly on the observation.

- **Shells.** A shell at Manhattan distance d is worth 6^(18−d). Shells are summed per direction:
  - **up:** the rows above the agent;
  - **down:** the rows below, but only the allowed ones;
  - **left and right:** the columns to the left and to the right, in any row.

  The direction with the highest sum wins.
  - A shell weighs 6 times a shell one step farther, so the nearest wins: it takes 6 shells
    one step farther to beat it.
  - The tie preferences are those of the BFS (up, down, left, right), as factors 1.06, 1.04, 1.02 and 1.
- **Jellyfish.** The direction that leads into a jellyfish is forbidden, and the next one wins.
- **Tide:**
  - the last allowed row is `9 − L − margin`, with margin 0 if the tide is falling, 1 if it is rising and 2 if it is
    rising and changes within 2 steps. These are the rules of `heuristic_policy`;
  - below that row the agent moves up, and it does not go down below it;
  - with no shells it waits one row above.

Why left and right count the columns in any row:
- A diagonal shell enters both its vertical and its horizontal direction. On ties the
  vertical one wins, as in the BFS, which first changes row and then column.
- If there are more shells on one side, that side wins. This is the main difference from the BFS: at equal
  distance the BFS chooses with a fixed order, while the potential goes where a move brings it closer to more shells.

How much each piece is worth. The versions are in Python, with the most likely move, over the 500 protocol episodes:

| version | score | same move as the BFS | episodes ended by drowning |
|---|---|---|---|
| BFS heuristic | 164.0 ± 1.5 | 100% | 11.0% |
| the nearest shell with the BFS preferences, without jellyfish | 148.8 ± 1.6 | 90.6% | 14.0% |
| potential, without the jellyfish ban | 146.2 ± 1.7 | 79.6% | 12.6% |
| **potential (the specification)** | **166.2 ± 1.9** | 78.5% | 13.8% |
| the network `bfs_net`, with the actions sampled as in the protocol | 166.7 ± 2.0 | 78.8% | 15.2% |

- **The jellyfish:** they are worth 15-20 points. A sting stuns for 3 steps, and near the rising water it makes the agent drown.
- **The ties:** resolved by the potential instead of the fixed order, they are worth a few points more than the BFS.

## 3. Structure of the network

```
403 → 64 (relu) → 58 (tanh) → 63 (relu) → 5 (linear)
```

| layer | what it does |
|---|---|
| 1 (relu, 64) | **reads**: one shell or jellyfish per cell (60), agent's row and column, margin from the water, safety margin |
| 2 (tanh, 64 → 58) | **measures**: potential of each row and column, adjacent jellyfish in each row and column, thresholds on the position and on the tide |
| 3 (relu, 63) | **routes**: potential of each direction, vetoes for adjacent jellyfish, tide events |
| 4 (linear, 5) | **decides**: one logit per move |

The potentials are measured in units of P1 = 6^17 ≈ 1.7·10¹³, the value of an adjacent shell. A saturated tanh
threshold is ±1. I write it as [x] = (t + 1)/2, which is 1 if the condition is true and 0 if it is false.

### Layer 1: one shell per cell, with the distance inside

For cell (R, C), with R between 3 and 8:

```
pre = P1·shell(R,C) + 0.5·P1·jellyfish(R,C) + Σ_agent cells (pot(d) − P1)·agent
pot(d) = 6^(18−d) for 1 ≤ d ≤ 17, otherwise 0          d = |R − r| + |C − c|
```

The agent is a one-hot, so the sum over its cells has a single term, `pot(d) − P1`, which is always ≤ 0. The
neuron's value is:

| what is in the cell | value |
|---|---|
| shell | 6^(18−d): P1 if adjacent, P1/6 at distance 2, and so on |
| jellyfish adjacent to the agent | 0.5·P1 |
| farther jellyfish, or empty cell | 0 |

It is the "gate" of GPT's network: a number passes only if the cell contains something. Here, however, the distance
does not have to be built with ramps, because it comes directly from the weight of the cell the agent is in.

The other 4 neurons:

| neuron | value |
|---|---|
| r | agent's row (weights 0..9 on its cells) |
| c | agent's column |
| m | `9 − r − L`, the dry rows below the agent (like h0 of `claude.json`) |
| v | `relu(2·rising − countdown/5)` (`rising`: the tide is rising) |

v is 1.8 or 1.6 when the margin is 2, from 1.4 to 1.0 when it is 1, 0 when the tide is falling. It approximates the
margin within 0.4, which is enough for the thresholds of the second layer.

### Layer 2: potentials, adjacent jellyfish, thresholds (tanh)

| neurons | pre-activation | value |
|---|---|---|
| potential of row R (6) and column C (10) | 10⁻³ · (sum of the cells)/P1 | tanh in the linear region: 10⁻³ times the potential, relative error ≤ 2.5·10⁻⁶ |
| adjacent jellyfish in row R (6) and column C (10) | 500 · ((sum of the cells)/P1 − 0.45) | ±1 |
| [r > k], k = 1..8, and [c > k], k = 0..8 (17) | 50 · (r − k − 0.5) | ±1 |
| row R allowed (6) | 250 · (m + r − v − R + 0.5) | ±1: R ≤ 9 − L − margin |
| [σ ≥ k], k = 0, 1, 2 (3) | 250 · (m − v − k + 0.5) | ±1, with σ = (last allowed row) − r |

How an adjacent jellyfish is recognized. The row just above the agent has only one adjacent cell, the one in its
column; the same holds for the row below and for the neighboring columns. The potential of that row or column, in
units of P1, therefore falls into three separate bands:

| adjacent cell | potential |
|---|---|
| empty | 0 - 0.4 (the other shells in the row are at least 2 steps away, and together they are worth at most 0.4) |
| jellyfish | 0.5 - 0.9 |
| shell | 1.0 - 1.4 |

The threshold at 0.45 separates the first band from the other two. The shell is distinguished from the jellyfish in
the third layer.

### Layer 3: directions, vetoes, tide (relu)

The direction units let the potential of a row or column through only if it is on the right side
of the agent (M = 4 switches the unit off):

```
up, row R            relu(pot_R − M·(1 − [r > R]))
down, row R          relu(pot_R − M·(1 − [r < R]) − M·(1 − [R allowed]))
left, column C       relu(pot_C − M·(1 − [c > C]))
right, column C      relu(pot_C − M·(1 − [c < C]))
```

The veto for an adjacent jellyfish, for example above:

```
jellyfish above (row R)   relu(0.475·Q_R − pot_R + 0.475 − M·(1 − [r = R + 1]))
```

The value depends on the band of that row's potential:

| band | Q | value |
|---|---|---|
| empty | −1 | ≤ 0 |
| jellyfish | +1 | 0.95 − pot, between 0.05 and 0.45: on |
| shell | +1 | 0.95 − pot ≤ −0.05: off |

A single relu thus asks a non-monotonic question ("jellyfish yes, shell no"), by combining a threshold and a
linear value. The gate `[r = R + 1] = [r > R] − [r > R + 1]` switches it on only for the row just above the agent.

The last 4 neurons are the tide events, 0 or 1:
- σ < 0, it must move up;
- σ ≤ 0, it cannot move down;
- σ = 1, it waits here;
- σ ≥ 2, it moves down to wait.

### Output

```
up    = Λ·1.06·Σ up    − W_veto·(jellyfish above)    + W_force·[σ < 0] + 30·[σ ≤ 0]
down  = Λ·1.04·Σ down  − W_veto·(jellyfish below)    − W_force·[σ ≤ 0] + 30·[σ ≥ 2]
left  = Λ·1.02·Σ left  − W_veto·(jellyfish on the left)
right = Λ·Σ right      − W_veto·(jellyfish on the right)
no-op = 30·[σ = 1]
```

with Λ = 1.4·10¹⁶, W_veto = 10³·Λ and W_force = 10⁶·Λ.
- **Λ:** it is chosen so that even at distance 17 a tie is broken by ~100 logits. The softmax therefore samples the
  specification's move with probability ~1.
- **Waiting:** it is worth 30, below the potential of any shell (≥ ~5000 logits, at distance 17), so it matters
  only when no shell is visible.
- **First attempt:** waiting was worth 2. The softmax chose it only 65% of the time, and the network scored 159
  instead of 166.

**The weights are huge:** 6^17 in the first layer, logits up to ~10²². All the values of the first layer are exact
integers in float64 (< 2^53), so shells at distance 17 (value 6) and adjacent ones (value 1.7·10¹³) coexist
without rounding errors. The reference forward pass works in float64.

## 4. Verification

`check.py` rewrites the weights from the formulas and plays the 500 protocol episodes with the network:

```
weights rewritten from the formulas equal to the file: True
states: 456230 (500 episodes); score 166.66 ± 1.97
network ≠ specification (argmax): 0
potentials by direction, largest relative error against the formula: 2.5e-06; states with a potential that should be 0 and is not: 0
states with a veto (adjacent jellyfish) different from the formula: 0
same move as the BFS heuristic: 78.8%
when the move differs:
    9.7% of the states  no shell reachable for the BFS
    6.2% of the states  another step of a shortest path
    4.8% of the states  a longer step (another shell, or the jellyfish avoided another way)
    0.5% of the states  stays still
```

## 5. Where it departs from the BFS

| share of states | what happens |
|---|---|
| 78.8% | the same move |
| 9.7% | **the BFS sees no reachable shells and waits, while the network moves** |
| 6.2% | **another equally short step** |
| 4.8% | **a longer step** |
| 0.5% | **it stays still**: the BFS has a reachable shell, the network does not see a useful one |

- **The BFS waits, the network moves (9.7%).** The shells are there, but the BFS excludes them:
  - they are in a row that the tide is about to cover: for the network they still count for left and right;
  - or, with the tide rising, they are next to a jellyfish: the BFS also blocks the cells next to jellyfish, the
    network only the jellyfish.
- **Another equally short step (6.2%).** It is the tie resolved by the potential instead of the fixed order.
- **A longer step (4.8%).**
  - The network goes toward a group of shells instead of toward the single nearest one.
  - Or it goes around a jellyfish by choosing the direction with the second-highest potential, while the BFS takes the first free
    path.

The network drowns in 15.2% of the episodes, against 11.0% for the BFS. The reason is that it does not keep its
distance from jellyfish when the tide is rising. It still scores a few points more, because it collects better.

## 6. What is left out

- **Going around jellyfish along a path, and the cells next to jellyfish when the tide is rising.** This would need more
  layers, or neurons for a jellyfish map that the first layer does not have.
- **Row 9.** It is never allowed, so it costs nothing.
- **Shells more than 17 steps away.** They do not exist on this beach.
