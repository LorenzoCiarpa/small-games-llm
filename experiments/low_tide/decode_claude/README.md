# How to read the Low Tide network `claude.json`

`weights/low_tide/claude.json` is the network hand-written by Claude Opus for Low Tide, without training and without
running code. It scores 132.3 ± 0.9 points over 500 episodes; the heuristic (BFS) scores 164.0, a random policy 2.0.

This analysis decodes it neuron by neuron. The result is that the network **is not a black box**: it is a
rule-based algorithm written in the form of weights. The first layer measures a few quantities of the game, the
second turns them into true or false conditions, the last combines them into a priority rule among the moves.

The reading is verified against the real network on 197,920 game states (see [Verification](#7-verification)):

```bash
uv run python experiments/low_tide/decode_claude/check.py            # ~1 min
uv run python experiments/low_tide/decode_claude/check.py --episodes 500
```

## 1. The game in brief

The grid is 10×10 and is a beach. The tide rises and falls by one row every 3-5 steps, between a low level
(0-2) and a high level (5-7).

- **Falling tide:** each uncovered row fills with shells (25% per cell) and jellyfish (8% per cell).
- **Rising tide:** whatever is on the submerged row is erased.
- **Shell:** +1.
- **Jellyfish:** −1 and 3 steps of stun, during which actions are ignored.
- **Water:** the agent cannot enter it. If the tide submerges the row the agent is on, it drowns and the episode
  ends.
- **Sticky actions:** with probability 0.1 the environment repeats the previous action instead of the chosen one.

The full rules are in `src/small_games/games/low_tide/env.py`.

## 2. How to read the positions

The observation has 403 numbers: four binary 10×10 planes, then three scalars.

```
index = channel*100 + row*10 + column

agent      0..99      (exactly one cell is 1)
water      100..199   (rows 10-L to 9 are 1, L = tide level)
shells     200..299
jellyfish  300..399
400  the tide is rising (1) or falling (0)
401  countdown: steps to the next level change / 5
402  stun: steps left / 3
```

**Row 0 is at the top (land), row 9 at the bottom (sea).** Within each plane, indices 0-9 are row 0 and indices
90-99 are row 9. The same holds for all four planes: index 245 is the shell in row 4, column 5.

```
           col 0 1 2 3 4 5 6 7 8 9
 row 0   ┐                              land: always dry
 row 1   │ rows 0-2: never water,       the agent starts in row 1
 row 2   ┘ never shells or jellyfish
 row 3   ┐
 ...     │ tide zone: shells and jellyfish appear here
 row 9   ┘                              sea: the first row to get wet
```

- **Water:** it fills from the bottom. With level L, rows `10-L` to 9 are wet. The level never exceeds 7, so
  rows 0-2 stay dry.
- **Actions:** 0 no-op, 1 up (row−1, toward land), 2 down (row+1, toward the sea), 3 left, 4 right.

Two names I use throughout the document:

- **shoreline:** the last dry row, `9 − L`, the one just above the water;
- **m (margin):** the dry rows between the agent and the water, `m = 9 − row − L`. With m = 0 the agent is on the
  shoreline; if the tide rises by one row at that moment, it drowns.

## 3. Network structure

```
403 → 46 (relu) → 36 (tanh) → 5 (linear)        20,461 parameters, 953 non-zero
```

| layer | what it does |
|---|---|
| 1 (relu, 46) | **measures**: margin, tide direction, shells in the row and column, the agent's column, jellyfish on the shoreline |
| 2 (tanh, 36) | **conditions**: each neuron is saturated, +1 means true and −1 false; ANDs are built with huge weights |
| 3 (linear, 5) | **rule**: each logit is a weighted sum of conditions; the move with the highest priority wins |

One idea to keep in mind: in the first layer a weight does not "reward" or "punish" a cell. It serves to
**measure** or to **ask a question**. The weight says how much the input counts in the sum, the bias acts as a
threshold, and the relu answers "yes, and by how much" if the sum exceeds the threshold, otherwise 0. The negative
sign only serves to subtract. What decides whether something is good or bad is the last layer.

## 4. First layer: what it measures (46 neurons, relu)

| neurons | question | value |
|---|---|---|
| h0 | how many dry rows are there between me and the water? | m, from 0 to 9 |
| h1 | is the tide rising? | 0 or 1 |
| h2-h11 | is there a shell **above** me, in my column? (one per column 0-9) | ≥ 1 if yes, 0 if no |
| h12-h18 | is there a shell **to the left**, in my row? (one per row 3-9) | ≥ 1 if yes, 0 if no |
| h19-h25 | is there a shell **to the right**, in my row? (one per row 3-9) | ≥ 1 if yes, 0 if no |
| h26-h35 | am I in column c? (one-hot) | 0 or 1 |
| h36-h45 | is there a jellyfish **on the shoreline** in column k? | ≥ 1 if yes, 0 if no |

### h0: the margin, with a ruler

```
water, column 0, rows 3..9            weight −1  → sum = −L  (counts the wet rows)
agent in row r, any column            weight −r  → sum = −r  (all of row 4 weighs −4, all of row 7 −7)
bias 9
h0 = 9 − r − L = m
```

- **Weights on the agent:** they are a ruler. It does not matter which cell the agent is in, only the row matters.
- **Weights on the water:** column 0 is enough, because the water is the same in all columns. Rows 3..9 are
  enough, because the level does not exceed 7.

Example: agent in row 4, level 3 (water in rows 7, 8, 9). h0 = 9 − 4 − 3 = 2, and indeed rows 5 and 6
are dry. The agent never enters the water, so h0 is never negative and the relu does not come into play here.

### h1: the tide direction

It has weight 1 on input 400 and bias 0: it copies it.

### h2-h11: shell above, with a binary comparison

Take h2, which looks at column 0 (h3 at column 1, and so on):

```
shells,     rows 3, 4, 5, ..., 9       weights 64, 32, 16, 8, 4, 2, 1       (2^(9−row))
agent,      rows 4, 5, 6, 7, 8, 9      weights 193, 225, 241, 249, 253, 255 (256 − (2^(10−row) − 1))
bias −256
```

With the agent in row r of the same column, the agent's weight and the bias together equal `−(2^(10−r) − 1)`,
that is, minus the sum of the weights of all rows from r to 9. So

`pre-activation = (weights of the shells present) − (sum of the weights of the rows from r down)`.

A power of 2 is larger than all smaller powers put together. So **one** shell above the agent (rows 3..r−1) is
enough to beat all those below put together: the neuron turns on (≥ 1) if and only if there is a shell above the
agent, in its column.

The neuron stays at 0 in two cases:

- **the agent is in another column, or in a row ≤ 3:** its weight is missing, and the bias −256 outweighs any sum
  of shells (at most 127);
- **there are no shells above the agent:** the sum does not reach the threshold.

A single neuron thus computes an AND between "I am in this column" and "there is a shell higher up than me".

### h12-h18 and h19-h25: shell to the left or to the right

It is the same trick, rotated onto the columns, with one neuron for each row from 3 to 9 (h12 and h19 for row 3,
h18 and h25 for row 9). The bias is −1024.

- **h12-h18 (left):** shell in column k with weight `2^(9−k)` (512 on the left, 1 on the right); agent in
  column j (1..9) with weight `1024 − (2^(10−j) − 1)`. It turns on if there is a shell in a column < j.
- **h19-h25 (right):** shell in column k with weight `2^k` (1 on the left, 512 on the right); agent in column
  j (0..8) with weight `1024 − (2^(j+1) − 1)`. It turns on if there is a shell in a column > j.

### h26-h35: the agent's column

Weight +1 on all 10 cells of column c of the agent plane, bias 0. It is 1 if the agent is in that column, 0
otherwise.

### h36-h45: jellyfish on the shoreline, with the threshold built from the water

For column k (h36 is column 0):

```
jellyfish, rows 3..9   weights 8, 16, 32, ..., 512    (2^row)
water,     rows 3..9   weights 4, 8, 16, ..., 256     (2^(row−1))
bias −511
```

With level L the wet rows are `10−L..9` and their sum is `512 − 2^(9−L)`. So

`pre-activation = (jellyfish) + 1 − 2^(9−L)`.

- **Jellyfish on the shoreline:** it is in row `9−L`, weighs exactly `2^(9−L)` and turns the neuron on.
- **Jellyfish higher up:** added together they are worth at most `2^(9−L) − 8` and are not enough.

The water moves the threshold together with the shoreline, and the neuron sees **only** the jellyfish on the
shoreline. With L = 0 the threshold is 511 and one jellyfish in row 9 (weight 512) is enough. With L = 7 the
shoreline is row 2, where there are never any jellyfish, and the neuron stays off.

### What the first layer does not look at

- **Countdown (401) and stun (402):** no weights.
- **Rows 0-2 of water, shells and jellyfish:** no weights, but they are always empty.

The only information truly ignored is time: the network does not know when the tide will move, nor how long it will
stay stunned.

## 5. Second layer: the conditions (36 neurons, tanh)

Each neuron has weights such that the pre-activation is at least ±2.5, so tanh is about **+1 (true)** or
**−1 (false)** (tanh 2.5 = 0.987). ANDs are built with very large weights (1000, 5120): if one piece is missing,
the sum stays far below zero.

| neuron | pre-activation | condition |
|---|---|---|
| g0 | 5·Σ h12..18 − 2.5 | shell to the left in my row |
| g1 | 5·Σ h19..25 − 2.5 | shell to the right in my row |
| g2 | 5·Σ h2..11 + 1000·h1 − 1002.5 | shell above **and** rising tide |
| g3 | 12·h1 − 6·h0 − 3 | rising **and** m ≤ 1 |
| g4 | 18·h1 − 6·h0 − 3 | rising **and** m ≤ 2 |
| g5 | 6·h0 − 60·h1 − 3 | falling **and** m ≥ 1 |
| g6 | −6·h0 + 3 | m = 0 (I am on the shoreline) |
| g7 | 6·h0 + 60·h1 − 75 | rising **and** m ≥ 3 |
| g8-g16 | 5·h(36+i) + 5120·h(27+i) − 5120·h0 − 5122.5 | jellyfish **directly to the left**: I am in column i+1, jellyfish in column i, m = 0 |
| g17-g25 | 5·h(37+i) + 5120·h(26+i) − 5120·h0 − 5122.5 | jellyfish **directly to the right**: I am in column i, jellyfish in column i+1, m = 0 |
| g26-g35 | 5·h(36+i) + 5120·h(26+i) − 5122.5 | jellyfish **below me**, on the shoreline of my column |

The thresholds on m (g3, g4, g5, g7) are built by giving h0 a weight of −6 or +6 and choosing the bias so that the
sign change falls between two integer values. For example, with the tide rising g4 equals 15 − 6·m, positive for
m ≤ 2 and negative for m ≥ 3.

## 6. Last layer: the rule (5 logits)

```
no-op = 0
up    =   23.5 + 4.5·g2 + 20·g3
down  = −160   + 10.5·g5 − 20·g4 − 2.5·g6 + 3·g7 − 15·Σ g26..35
left  = −128   + 6·g0  − 15·Σ g8..16
right = −126.5 + 7.5·g1 − 15·Σ g17..25
```

The biases look odd because **neurons that are off are worth −1, not 0**. For example, the ten "jellyfish below"
neurons, when off, give −15 × (−10) = +150, which offsets almost all of the −160. At most one of them is on (the
one for the agent's column), and turning it on subtracts 30.

Rewritten with `[x]` = +1 if x is true, −1 if it is false:

```
no-op = 0
up    = 23.5 + 4.5·[rising and shell above] + 20·[rising and m ≤ 1]
down  = −10 + 10.5·[falling and m ≥ 1] − 20·[rising and m ≤ 2] − 2.5·[m = 0] + 3·[rising and m ≥ 3]   (−30 if jellyfish below)
left  = 7   + 6·[shell to the left]                                                                   (−30 if jellyfish directly to the left)
right = 8.5 + 7.5·[shell to the right]                                                                (−30 if jellyfish directly to the right)
```

Here are the values in the various cases. "No-op" is always 0. Left is 13 with a shell to the left and 1 without;
right is 16 with a shell to the right and 1 without.

| tide | m | up | down | chosen move |
|---|---|---|---|---|
| falling | 0 | −1 | −6 | right if there is a shell to the right (16), then left (13), otherwise wander |
| falling | ≥ 1 | −1 | **20** (−10 with jellyfish below) | **down**: follows the receding water |
| rising | 0 | **39** (48 with shell above) | −46 | **up**: beats everything |
| rising | 1 | **39** (48) | −41 | **up** |
| rising | 2 | −1 (8 with shell above) | −41 | right (16) > left (13) > up toward a shell (8) > wander |
| rising | ≥ 3 | −1 (8) | 5 (−25 with jellyfish below) | right (16) > left (13) > up (8) > **down** (5) |

**Wander** means that left ≈ 1.08, right ≈ 1.10, no-op 0 and up −1. The policy samples from the softmax of the
logits, so it goes left or right with about 40% probability each, stays still 14% of the time and moves up 5%.

### The algorithm

```
m = dry rows between me and the water

if the tide is FALLING:
    m ≥ 1 → down                        follow the receding water
                                        (if there is a jellyfish below, on the shoreline: step sideways)
    m = 0 → walk along the shoreline:
            shell to the right → right, else shell to the left → left, else wander
            never toward a jellyfish right next to me

if the tide is RISING:
    m ≤ 1 → up                          flee: it has priority over everything
    m = 2 → shells in the row (first right, then left), then shell above, else wander;
            never down
    m ≥ 3 → shells in the row, then shell above, else down (to go back to m = 2)
```

The strategy follows the rules of the game:

- **Falling tide:** new shells appear only on the row just uncovered, that is, on the shoreline. The network
  sticks to it and sweeps it horizontally.
- **Rising tide:** the water advances by one row every 3-5 steps. The network keeps a margin of 2 rows and moves
  back up as soon as the margin drops to 1. Meanwhile it collects what it finds in its row.

## 7. Verification

`check.py` plays 200 episodes with the network (seeds 10000-10199, sticky 0.1, sampling from the softmax like
`evaluate`). In each state it recomputes, **directly from the game state and without using the weights**, the 46
neurons of the first layer and the 5 logits with the formulas of section 6, and compares them with the true forward
pass.

| check | result |
|---|---|
| states visited | 197,920 |
| first layer: states in which the reading (on/off, value of h0) is wrong | **0** |
| logits: maximum difference | 0.315 (tanh never reaches exactly ±1) |
| move with the highest logit differs, excluding near-ties (< 0.5) | **0** |

The near-ties are 49,550 states. Almost all of them are left versus right when there are no shells (1.08 versus
1.10); there the network samples and so wanders.

Which move has the highest logit, by situation:

| tide | m | % of states | no-op | up | down | left | right |
|---|---|---|---|---|---|---|---|
| rising | 0 | 0.3% | | 100% | | | |
| rising | 1 | 14.0% | | 100% | | | |
| rising | 2 | 32.1% | | 2% | | 26% | 72% |
| rising | ≥ 3 | 3.5% | | 3% | 44% | 27% | 26% |
| falling | 0 | 32.9% | 1% | | | 36% | 64% |
| falling | 1 | 15.8% | | | 93% | 2% | 5% |
| falling | 2 | 1.0% | | | 92% | 1% | 7% |
| falling | ≥ 3 | 0.3% | | | 90% | 1% | 9% |

The network spends two thirds of the time in two states: on the shoreline while the tide is falling (33%) and at
m = 2 while it is rising (32%). The rest are the transitions between the two.

## 8. Limits visible in the weights

The weights also explain why the network scores 132 and not 164 like the heuristic.

- **It does not read countdown (401) and stun (402).** It does not know when the tide will move, so it always keeps
  a margin of 2 rows, even when it would have time to go down and collect.
- **It sees shells only in a cross**, that is, in its row and its column, and **without distance**:
  - a diagonal shell does not exist for it;
  - a shell 1 cell away is worth as much as one 8 cells away;
  - with shells on both sides it goes right 95% of the time (16 versus 13), even if the one on the left is
    closer.
- **It considers shells above only when the tide is rising.**
- **It sees jellyfish only on the shoreline.** When the tide is rising and it moves at m = 2, or moves up toward a
  shell, it walks onto jellyfish without seeing them: 21.7 stings per episode on average.
- **It drowns in 4.2% of episodes, but only because of sticky actions.** Without sticky actions
  (`uv run evaluate low_tide claude --sticky 0`) it drowns in 0 episodes out of 500 and scores 140.7 points. In the
  17 drownings over the 500 episodes of the protocol there are two mechanisms, and in both the environment repeats
  a "down" instead of an "up":
  - **10 cases:** the tide is rising and the network, at m = 3, moves down to return to m = 2. The sticky actions
    repeat "down" twice in a row, against the network's choices (at the last step, at m = 1, the network chooses
    up), and the agent reaches m = 0 just as the tide rises. A margin of 2 rows withstands one repetition, not two.
  - **7 cases:** with the tide rising, a repeated "down" takes the agent onto a jellyfish on the shoreline. In 5
    cases out of 7 the tide had just reversed and the network had chosen up. The agent stays stunned for 3 steps at
    m = 0 and the tide rises before it can move.

GPT's network for the same game is decoded in [`decode_gpt`](../decode_gpt/README.md), with
a comparison between the two.
