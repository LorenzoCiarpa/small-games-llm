# How to read the Low Tide network `gpt.json`

`weights/low_tide/gpt.json` is the network hand-written by GPT-6 Astra for Low Tide, without training and without
running code. It scores 79.9 ± 1.0 points over 500 episodes. For comparison, `claude.json` scores 132.3, the
heuristic (BFS) 164.0 and a random policy 2.0.

This analysis decodes it neuron by neuron, as
[`decode_claude`](../decode_claude/README.md) does for Claude's network. This one too is an
algorithm written in the form of weights, but it is a different algorithm, built with different tricks:

- **view:** it looks at cells one by one, but only in a window of 6×4 cells;
- **shells:** for each one it computes an attraction that depends on the distance;
- **move:** it sums the attractions by direction and adds a push that keeps the agent inside the window.

The reading is verified against the real network on 186,768 game states. The match is exact, up to
rounding (see [Verification](#7-verification)):

```bash
uv run python experiments/low_tide/decode_gpt/check.py            # ~1 min
uv run python experiments/low_tide/decode_gpt/check.py --episodes 500
```

## 1. Game and positions

The game and the observation are described in sections 1 and 2 of
[`decode_claude`](../decode_claude/README.md#2-how-to-read-the-positions). In brief:

```
index = channel*100 + row*10 + column         agent 0..99, water 100..199, shells 200..299, jellyfish 300..399
400 rising tide, 401 countdown, 402 stun
```

- **Rows:** row 0 is at the top (land), row 9 at the bottom (sea).
- **Water:** with level L it covers rows `10−L` to 9.
- **Actions:** 0 no-op, 1 up (row−1), 2 down (row+1), 3 left, 4 right.
- **m (margin):** the dry rows between the agent and the water, `m = 9 − row − L`.
- **Shoreline:** the last dry row, `9 − L`.

## 2. Network structure

```
403 → 43 (relu) → 61 (relu) → 53 (relu) → 5 (linear)       23,612 parameters, 1,174 non-zero
```

| layer | what it does |
|---|---|
| 1 (43) | **reads**: 24 cells of the window, the agent's position as "ramps", tide level and direction |
| 2 (61) | **measures**: attraction of each shell (6 − distance), proximity of each jellyfish, tide danger |
| 3 (53) | **sorts by direction**: sums the attractions of the shells above, below, to the left, to the right |
| 4 (5) | **decides**: push toward the window + attractions − adjacent jellyfish ± emergency rules |

All hidden layers are relu. The network uses neither tanh nor ±1 conditions like Claude's. It uses two
piecewise-linear tricks, which are worth seeing first.

### Trick 1: ramps

A **ramp** is `relu(x − a)`: it is 0 up to `a`, then grows by 1 at each step. With the ramps of an integer
coordinate (the agent's row or column) two things can be built.

- **A step:** `relu(x − a) − relu(x − a − 1)` is 1 if `x ≥ a + 1`, otherwise 0.
- **A tent, that is, minus the absolute value:** `relu(x − a) − 2·relu(x − R)` equals `(R − a) − |x − R|` for
  `x ≥ a`. This is how the network computes the **distance** between the agent and a cell.

### Trick 2: the gate

It is written `relu(v + 64·g − 64)`, with g ∈ {0, 1} and 0 ≤ v ≤ 64:

- with `g = 1` it equals `v`;
- with `g = 0` it equals 0.

It is a "let v through only if g is true", that is, an AND between a number and a condition. The gate g is usually
a step built from the ramps.

## 3. First layer: what it reads (43 neurons)

| neurons | inputs | value |
|---|---|---|
| h0-h23 | shell (+1) and jellyfish (−1) of one cell, bias 1 | 2 shell, 1 empty, 0 jellyfish |
| h24-h30 | agent in row r, weights `r − 2 − k` | ramps on the row: `relu(r − 2 − k)`, k = 0..6 |
| h31-h34 | agent in column c, weights `c − 5 − k` | ramps on the column: `relu(c − 5 − k)`, k = 0..3 |
| h35 | water, column 0, rows 3..9 (weight 1) + 10·rising tide | `L + 10·[rising]` |
| h36 | agent in row 0 (weight 2) or 1 (weight 1) | `relu(2 − r)` |
| h37-h41 | water in row 5, 6, 7, 8, 9 of column 0 + rising tide, bias −1 | rising **and** L ≥ 5, 4, 3, 2, 1 |
| h42 | rising tide | 0 or 1 |

**The window.** h0-h23 look at the 24 cells of rows 3-8 and columns 6-9, one per neuron: the cell in row
R and column C is neuron `4·(R−3) + (C−6)`. For example, h0 has weights `[236: +1, 336: −1]`, that is, shell and
jellyfish in row 3, column 6.

**The rest of the beach is invisible:**

- columns 0-5;
- row 9;
- in all, 46 of the 70 cells of the tide zone.

**What it does not read:** countdown (401) and stun (402), like Claude's network.

## 4. Second layer: what it measures (61 neurons)

### n0-n23: the attraction of each shell

For cell (R, C) of the window:

```
pre = 32·h_cell + h24 − 2·h(25+R−3) + h31 − 2·h(32+C−6) − 32·h(37+R−3) − 60 − (R−3) − (C−6)
```

- **Rows:** `h24 − 2·h(25+R−3)` is a tent, that is, `(R − 2) − |r − R|`.
- **Columns:** `h31 − 2·h(32+C−6)` is another tent, `(C − 5) − |c − C|`.
- **Bias:** it cancels the constants.

With a shell in the cell (`h_cell = 2`) what remains is

```
n = relu(6 − d)        d = |r − R| + |c − C|   (Manhattan distance)
```

The value is therefore 5 for an adjacent shell, 1 for one at distance 5, 0 from distance 6 onward.

- **Empty cell** (`h_cell = 1`): pre = 32 − 58 − d, negative.
- **Jellyfish** (`h_cell = 0`): pre even more negative.

Two details:

- **The ramps start from row 2 and column 5.** An agent in rows 0-1 is seen as if it were in row
  2, one in columns 0-4 as if it were in column 5. This happens in less than 5% of states.
- **The term `−32·h(37+R−3)` switches the shell off when the tide threatens it.** For row R the neuron used is
  "rising and L ≥ 8 − R", that is: the tide is rising and the shell's row has margin ≤ 1. The network ignores the
  shells that the water is about to reach.

### n24-n47: nearby jellyfish

It is the same computation with `−32·h_cell` instead of `+32` and without the tide term. Only jellyfish pass
(`h_cell = 0`):

```
n = relu(2 − d)         1 if the jellyfish is adjacent to the agent, 0 if it is farther away
```

### n48-n58: the ramps, copied

n48-n54 copy h24-h30, n55-n58 copy h31-h34. In an MLP each layer sees only the previous one, so the
agent's position has to be "passed forward" if it is needed later. The relu does not change a value that is already
≥ 0.

### n59, n60: the tide danger

```
n59 = relu(h24 − h36 + h35 − 15)
n60 = relu(h24 − h36 + h35 − 14)
```

- **The row:** `h24 − h36 = relu(r − 2) − relu(2 − r) = r − 2` for every row. Two opposite ramps reconstruct r.
- **The tide:** `h35 = L + 10·[rising]`. With the tide falling the sum stays below zero. With the tide rising it
  equals `r + L − 7 = 2 − m`, because `r + L = 9 − m`.

So:

```
n59 = relu(2 − m) if the tide is rising, otherwise 0      (2 at m = 0, 1 at m = 1)
n60 = relu(3 − m) if the tide is rising, otherwise 0      (3, 2, 1 at m = 0, 1, 2)
```

## 5. Third layer: the directions (53 neurons)

Each neuron sums the attractions of one row or one column of the window and lets them through a gate
(trick 2) only if that row or column is in the right direction. The gate is a step built from the copied
ramps.

| neurons | sum | gate | direction |
|---|---|---|---|
| p0-p5 | attractions of row R = 3..8 | `[r ≥ R + 1]` | shells **above** |
| p6-p11 | attractions of row R | `[r ≤ R − 1]` | shells **below** |
| p12-p15 | attractions of column C = 6..9 | `[c ≥ C + 1]` | shells **to the left** (p15 always 0) |
| p16-p19 | attractions of column C | `[c ≤ C − 1]` | shells **to the right** |
| p20-p39 | the same for nearby jellyfish | | adjacent jellyfish above, below, left, right |
| p40-p50 | copies of the ramps (n48-n58) | | |
| p51, p52 | copies of n59, n60 | | |

For example, p0 has weights `n0..n3: +1, n49: +64, n50: −64` and bias −64. The step `n49 − n50 = relu(r−3) − relu(r−4)`
equals `[r ≥ 4]`, so p0 lets the shells of row 3 through only if the agent is lower down.

A diagonal shell counts in **two** directions. For example, a shell up and to the left pushes both up and
left.

## 6. Last layer: the rule (5 logits)

Rewritten as a function of the agent's position (r, c), with `[x]` = 1 if true and 0 if false:

```
no-op = −4
up    = −12   + 12·[r≥4] + 0.75·[r≥7] + 12·[r=9]           + 0.9·(shells above) − 25·(jellyfish above) + 1000·n59
down  =  12.75 − 12·[r≥3] − 0.75·[r≥6] − 12·[r≥8]          + 0.9·(shells below) − 25·(jellyfish below) − 1000·n60
left  = −12   + 12·[c≥7] + 0.75·[c=9]                      + 0.9·(shells left)  − 25·(jellyfish left)
right =  12.75 − 12·[c≥6] − 0.75·relu(c−7) − 11.25·relu(c−8) + 0.9·(shells right) − 25·(jellyfish right)
```

"Shells above" is the sum of `6 − d` over the visible shells above the agent (section 5). "Jellyfish above" is 1
if there is a jellyfish in the cell directly above.

### The push toward the window

Without shells and without danger, the logits depend only on the position:

| row r | ≤ 2 | 3 | 4-5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| up | −12 | −12 | 0 | 0 | 0.75 | 0.75 | 12.75 |
| down | **12.75** | 0.75 | 0.75 | 0 | 0 | −12 | −12 |

| column c | ≤ 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|
| left | −12 | −12 | 0 | 0 | 0.75 |
| right | **12.75** | 0.75 | 0.75 | 0 | −12 |

It is a potential field:

- **Outside the window** (rows 3-8, columns 6-9) the push is strong, about 12.75 versus −12, and brings the agent
  back inside.
- **Inside the window** the moves that stay in the window are worth 0 or 0.75, those that would leave it −12 (up
  from row 3, down from row 8, left from column 6, right from column 9). The agent wanders, with "no-op" always
  at −4.

The push keeps the agent in the area that the network can see.

### Shells: a gradient

Each visible shell at distance d adds `0.9·(6 − d)` to the moves that bring the agent closer to it. An adjacent
shell is worth 4.5, one at distance 5 is worth 0.9. Several shells in the same direction add up. It is like
following the gradient of an attraction field: the agent goes where the total attraction is strongest, not toward
the nearest shell.

### Jellyfish and emergencies

- **Adjacent jellyfish:** −25 to the move that leads onto it. It is an almost absolute ban.
- **Rising tide and m ≤ 1:** +1000 (m = 1) or +2000 (m = 0) to "up". It flees, and this beats everything.
- **Rising tide and m ≤ 2:** −1000·(3 − m) to "down". Moving down is forbidden.
- **Shells with margin ≤ 1 while the tide is rising:** ignored (section 4).

### The algorithm

```
if the tide is rising and m ≤ 1:   up
otherwise, for each move (up, down, left, right):
    score = push toward the window (rows 3-8, columns 6-9)
          + 0.9 × Σ (6 − distance) over the visible shells in that direction
            (excluding those that the rising tide is about to reach)
          − 25 if there is a jellyfish in the destination cell
    down forbidden if the tide is rising and m ≤ 2
no-op = −4; the move is sampled from the softmax of the scores
```

When the tide is falling there is no rule: the network does not follow the shoreline and does not know that new
shells appear there.

## 7. Verification

`check.py` plays 200 episodes with the network (seeds 10000-10199, sticky 0.1, sampling from the softmax like
`evaluate`). In each state it recomputes, **directly from the game state and without using the weights**, the 43
neurons of the first layer and the 5 logits with the formulas of section 6, and compares them with the true forward
pass.

| check | result |
|---|---|
| states visited | 186,768 |
| first layer: states in which the reading is wrong | **0** |
| logits: maximum difference | 1.1·10⁻¹³ (the network is all relu and linear: the formulas are exact) |
| move with the highest logit differs, excluding near-ties (< 0.5) | **0** |

The near-ties are 61,158 states. They are mostly the agent inside the empty window, where the logits are 0 or 0.75
and the network wanders.

Where the agent is:

| column | 0-4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|
| % of states | 0.4% | 1.1% | 14.3% | 27.7% | 36.1% | 20.5% |

| row | 0-1 | 2 | 3 | 4 | 5 | 6 | 7 | 8-9 |
|---|---|---|---|---|---|---|---|---|
| % of states | 3.7% | 10.4% | 25.7% | 21.6% | 18.7% | 13.5% | 5.1% | 1.2% |

The agent spends 98.6% of the time in columns 6-9. On average only **9.4%** of the shells on the grid are inside
the window: it collects those it sees, and the others pile up where it does not look.

## 8. Limits visible in the weights

- **It sees only 24 of the 70 cells of the tide zone** (34%): rows 3-8, columns 6-9. Columns 0-5 and row 9 do not
  exist for the network, and the push toward the window keeps it there even when the window is empty. This is why
  it collects 86.6 shells per episode, against Claude's 154.
  - The most likely cause is the limit of 64 units per layer. With one neuron per cell for the shells and one for
    the jellyfish, 24 cells already take up 48 neurons; the whole tide zone would require 140.
  - Claude avoided the problem with neurons that summarize an entire row or column (the binary comparisons).
- **It does not exploit the tide cycle.** It has no rules for the falling tide, so it does not follow the shoreline
  where new shells appear.
- **It does not read countdown (401) and stun (402).**
- **It drowns in 15.6% of episodes, but only because of sticky actions.** Without sticky actions
  (`uv run evaluate low_tide gpt --sticky 0`) it drowns in 0 episodes out of 500 and scores 90.5 points.
  - **Mechanisms:** they are the same as for Claude's network. The sticky actions repeat a "down" while the tide
    is rising. Of the 65 drownings over 500 episodes, 34 cases are sticky only and 31 are sticky + a jellyfish
    that stuns on the shoreline.
  - **Why more often:** with the tide rising, the network chooses "down" at m = 3 in 3.4% of steps, against 1.3%
    for Claude, that is, 2.6 times more. It is the push toward the window: with the tide high, row 3 is close to
    the water, but the network wants to stay in rows 3-8 (from row 2 "down" is worth 12.75). Each descent to m = 3
    is an opportunity for two consecutive sticky actions.
- **Jellyfish: it does better than Claude** (6.7 stings per episode against 21.7). It sees adjacent jellyfish in
  the whole window, not only on the shoreline, and spends little time on the shoreline, where new ones appear.

## 9. Comparison with `claude.json`

| | `claude.json` | `gpt.json` |
|---|---|---|
| architecture | 403-46-36-5, relu and tanh | 403-43-61-53-5, relu only |
| non-zero weights | 953 | 1,174 |
| shells | the whole tide zone, but only in a cross (row and column) and without distance | only 24 cells, with Manhattan distance |
| jellyfish | only on the shoreline, only adjacent | adjacent, inside the window |
| tide | full strategy: follows the shoreline when falling, 2 rows of margin when rising | emergencies only: up at m ≤ 1, never down at m ≤ 2 |
| main trick | binary comparisons (powers of 2), saturated tanh as a boolean | relu ramps (steps, absolute value), gate `relu(v + 64g − 64)` |
| underlying idea | priority rules ("if… then…") | potential field + attraction gradient |
| score (sticky 0.1) | 132.3 ± 0.9 | 79.9 ± 1.0 |
| shells / stings / drownings | 154 / 21.7 / 4.2% | 86.6 / 6.7 / 15.6% |
| without sticky actions | 140.7, no drownings | 90.5, no drownings |

The two networks are two different programs for the same game:

- **Claude** sees little of each thing, but everywhere, and has a strategy tied to the tide cycle.
- **GPT** sees a small area well, with true distances, but stays locked in that area.
