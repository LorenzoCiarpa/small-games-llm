# The exact Low Tide BFS in an MLP, without the game's limits

[`bfs_net`](../bfs_net) stays within the game's limits, 3 layers of 64 units, and for this reason computes only a
potential-based version of the BFS (see its section 1). Here the limits are removed, to see **how many neurons are
needed to make exactly the move of `heuristic_policy` in every state.**

`build.py` writes `bfs_exact.json`:
- 104 relu hidden layers, up to 503 units per layer, **51,103 neurons** in total;
- 273,944 non-zero weights, out of 25.5 million;
- no training: every weight comes from a formula.

The game's limits allow at most 3 × 64 = 192 neurons.

**Result:** the network always makes the same move as the heuristic:
- in all 470,889 states of the 500 protocol episodes;
- in all 20,000 purpose-built states, with paths of up to 54 steps.

It therefore plays the same episodes, with the same score episode by episode: **164.0 ± 1.5**, like the heuristic.

The network exceeds the game's limits, so `uv run evaluate` and `uv run play` do not load it. `check.py` reads it with
the repo's own `load_weights`, removing only the limits on layers and units, and evaluates it with the reference
forward pass (`ReferenceMLP`, float64).

```bash
uv run python experiments/low_tide/bfs_exact/build.py     # writes bfs_exact.json (3.8 MB, not versioned)
uv run python experiments/low_tide/bfs_exact/check.py     # verification, ~9 min with 8 processes
```

## 1. The algorithm

The heuristic's BFS starts from the agent and visits the neighbors in the order up, down, left, right. It stops at
the first shell it removes from the queue and takes the first step of the path that led it there.

**An equivalent rule, more convenient for a network:**

> go to the allowed neighbor closest to a shell; on ties, in the order up, down, left, right.

The two rules choose the same move:
- the BFS removes cells from the queue in order of distance;
- at equal distance, it removes them in the order of their paths: first those that start with up, then with down, and so on.

So the first shell it finds is the nearest. Its path starts with the first direction from which a
shortest path starts, i.e. the first direction whose neighbor is at distance d − 1 from a shell.

**The network computes the distance from the shells with a wave:**
- **Start.** The wave starts from the shells in the allowed cells. These are the cells the BFS can visit:
  - rows up to the last safe one;
  - no jellyfish;
  - if the tide is rising, no cells next to a jellyfish.
- **One step per layer.** At each layer the wave reaches the allowed cells next to a cell already reached. After k
  layers it covers exactly the cells at distance ≤ k from a shell.
- **Count.** Each cell counts for how many layers it has been reached: after K layers the count is K + 1 − d, with d
  the distance from the nearest shell (0 if d > K).
- **Readout.** At the end the network reads the count in the agent's four neighbors:
  - logit = 10⁴ · (count + 0.4, 0.3, 0.2 or 0.1) for up, down, left and right;
  - the neighbor closest to a shell wins, and on ties the BFS order.
- **Tide.** These are the heuristic's rules:
  - if the agent is below the last safe row, it moves up;
  - if no neighbor has been reached by the wave, it waits one row above the last safe one.

## 2. Structure

| layers | units | what they compute |
|---|---|---|
| 1 | 405 | copies of agent, shells and jellyfish; number of neighboring jellyfish if the tide is rising; water level; agent's row; countdown ≥ 3 |
| 2 | 405 | tide rising and countdown ≤ 2, i.e. margin 2 instead of 1 |
| 3 | 425 | for each row, whether it is within the last safe row; where the agent is relative to that row |
| 4 | 303 | allowed cells; start of the wave; "agent below the last row"; wait up or down |
| 5-102 | 503 (the last 403) | one wave step each, 98 in total |
| 103 | 363 | the count in the agent's neighbors, one unit per cell and direction |
| 104 | 8 | count per direction; wait only if no neighbor has been reached |
| output | 5 | logits |

**Per cell and per step, 5 units are needed:**
- **2 for the "or" of the neighbors.** With s the number of reached cells around, relu(s) − relu(s − 1) = min(1, s). A single
  relu does not stop at 1.
- **1 for the count.**
- **1 for the mask of allowed cells.** It is needed at every step, so it must be carried forward from layer to layer.
- **1 for the agent's position.** It is needed only at the end, but it too must be carried forward.

**The computation is exact.** All internal values are small integers, between 0 and 100, which float64 represents
without error.
- The only non-integer input is the countdown, k/5 in float32.
- `relu(10 · countdown − 5)` is 0 for k ≤ 2 and at least 1 for k ≥ 3, so `relu(sale − ·)` (`sale`: the tide is
  rising) gives an exact 0/1.

**Even when sampling from the softmax the move is always the same.** The logits have margins ≥ 1000, and exp(−1000)
is 0 in float64. The network therefore plays exactly the same episodes as the heuristic.

## 3. How many neurons are needed

**Depth: one layer per path step.**
- **K = 98 is enough for every possible state.** In 100 cells the nearest shell is at distance ≤ 99, and the
  agent's neighbor needs d − 1 ≤ 98.
- **It is a safe bound, not the minimum.** With walls of jellyfish the longest shortest path is shorter: in the
  constructed states it reaches 54 steps.
- **In the protocol episodes the shell found is at most 26 steps away.** For those states
  K = 25 steps would be enough, i.e. 31 layers and 14,384 neurons. But the network would be wrong in a state with a longer path.

**Width: 5 units per cell**, about 500 per layer.

| wave steps K | hidden layers | neurons | weights ≠ 0 | exact if the shell is within |
|---|---|---|---|---|
| 98 (`bfs_exact.json`) | 104 | 51,103 | 273,944 | always |
| 30 | 36 | 16,899 | 87,420 | 31 steps |
| 25 | 31 | 14,384 | 73,705 | 26 steps: all protocol states |
| game limits (`bfs_net`, potential-based) | 3 | 185 | 6,806 | not exact (78.8% of the moves) |

`build.py --k N` builds the shorter variants.

**Depth can be traded for width, but at a high cost.** One can compute the distances between all pairs of
cells and double the radius at each layer, as in the min-plus matrix product.
- log₂ 99 ≈ 7 doublings are enough.
- But each doubling needs one unit for each triple of cells, about a million.

I did not build it.

## 4. Verification

`check.py`:
1. rewrites the weights from the formulas and checks that they match the file;
2. plays the 500 protocol episodes with the network, sampling from the softmax like `evaluate`. In every state it
   compares the move with the heuristic's. It plays the same episodes with the heuristic and compares the scores;
3. checks 20,000 purpose-built states, including some outside the game's rules:
   - half random, with up to 45% jellyfish and 1-3 shells;
   - a quarter jellyfish mazes;
   - a quarter serpentines.

   They serve to test long paths, which do not occur in the episodes.

```
weights rewritten from the formulas equal to the file: True
network: 104 hidden layers, 51103 neurons, 273944 non-zero weights out of 25472463

protocol episodes: 500, states 470889
  moves different from the heuristic: 0
  sampled action ≠ argmax: 0; smallest logit margin: 1000
  network score 163.97 ± 1.47, heuristic 163.97 ± 1.47, episodes with a different score: 0
  largest distance of the shell found by the BFS: 26 (for these states K = 25 steps, 31 layers, would be enough)

constructed states (random, mazes, serpentines): 20000
  moves different from the heuristic: 0; smallest margin: 1000
  shell distance: median 8, largest 54, states with distance > 26: 1879

(540 s)
```
