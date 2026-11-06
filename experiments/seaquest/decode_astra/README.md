# How to read the Seaquest network `paper_astra.json`

`weights/seaquest/paper_astra.json` is the network that GPT-6 Astra wrote by hand for Seaquest (MinAtar)
in the article [Weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/), without training.
Here it scores 51.4 ± 1.4 points over 1000 episodes (53.1 ± 1.4 in the article); a random policy scores 0.04.

This analysis decodes it neuron by neuron. The result is a very simple algorithm:

- **the submarine never moves horizontally.** It stays in column 5, facing left, where the game starts,
  and fires to the left;
- **at every step it chooses the row.** It compares three rows, its own, the one above and the one below, and moves to
  the one with the highest value. The value rewards divers to rescue and targets in the line of fire, and forbids
  rows with a danger near column 5;
- **it surfaces** when oxygen drops below 60 and it has at least one diver on board.

It works because in Seaquest everything moves horizontally, along the rows: whoever stays still in one column and
chooses the row well sees both the enemies to hit and the divers to collect pass in front of the bow.

The reading is checked against the real network on 263,355 game states (see [Verification](#8-verification)):

```bash
uv run python experiments/seaquest/decode_astra/check.py               # ~1.5 min
uv run python experiments/seaquest/decode_astra/check.py --examples    # the situations of section 6
uv run python experiments/seaquest/decode_astra/check.py --ablations   # families of neurons switched off (section 9)
```

## 1. The game in brief

It is the official MinAtar Seaquest (`minatar/environments/seaquest.py`, version 1.0.15). The grid is 10×10: row
0 is the surface, rows 1-8 are the sea, row 9 contains only two indicators.

- **Submarine:** it takes up two cells of the same row, bow and stern. It starts in row 0, column 5, facing left
  (bow in column 5, stern in column 6). Left and right move it by one column and turn it that way;
  up and down move it by one row (rows 0-8). Only the bow counts for collisions.
- **Fire:** a torpedo leaves the bow in the direction the submarine faces and advances one cell per step.
  After each shot 5 reload steps are needed; meanwhile "fire" does nothing, that is, the submarine stays still.
- **Enemies:** fish and submarines appear at the left or right edge, in a random row 1-8, and cross the
  screen horizontally. At first they move one step every 6, then more and more often (up to one every 3). Enemy
  submarines fire torpedoes every 11 steps, which advance one cell per step.
- **Points:** +1 for each enemy hit by a torpedo.
- **Death:** touching a fish, a submarine or an enemy torpedo.
- **Divers to rescue:** one appears every 31 steps, in a row 1-8, and crosses the screen (one step every 6).
  Passing over it loads it on board, up to 6.
- **Oxygen:** 200 steps. It drops by 1 at every step underwater; at 0 the episode ends.
- **Surfacing (row 0):**
  - with no divers on board the episode ends;
  - with 1-5 divers one is lost and oxygen refills;
  - with 6 divers all are lost, ⌊oxygen/20⌋ points are gained (from 0 to 10) and oxygen refills.

  Every surfacing raises the difficulty: enemies appear more often and move faster.
- **Sticky actions:** with probability 0.1 the environment repeats the previous action instead of the chosen one.

A detail of the MinAtar code that matters later: after surfacing with 6 divers the counter drops to −1, not to
0, so the first diver collected afterwards only brings it back to 0.

## 2. How positions are read

The observation has 600 numbers in {0, 1}: the 10×10 grid with 6 channels per cell.

```
index = (row*10 + column)*6 + channel            row 0 at the top, column 0 on the left

channel 0  player      bow and stern of the submarine
channel 1  danger      fish, enemy submarines, enemy torpedoes
channel 2  reward      fish, enemy submarines, divers to rescue
channel 3  projectile  your torpedoes
channel 4  trail       the cell behind enemies and divers (tells which way they are going), and the stern
channel 5  gauge       row 9: oxygen and divers on board
```

For example, index 330 = (5·10 + 5)·6 + 0 is the player channel of cell (row 5, column 5), and 557 = (9·10 + 2)·6
+ 5 is the gauge channel of cell (9, 2). I write cells as (row, column).

**The channels are ORs.** An enemy turns on both danger and reward; a diver to rescue only reward; an enemy torpedo
only danger. The network uses this overlap to tell objects apart (section 4).

**Row 9** contains two bars in the same gauge channel:

- oxygen, from the left: columns 0 to ⌊oxygen/20⌋ − 1;
- divers on board, from the right: with n divers, columns 9 − n to 8.

Cell (9, 2) is on if and only if oxygen is at least 60, because the diver bar never reaches column 2.
Cell (9, 8) is on if there is at least one diver on board, or if oxygen is at least 180.

**Actions:** 0 no-op, 1 left, 2 up, 3 right, 4 down, 5 fire.

Two names I use throughout the document:

- **r**, the row of the submarine;
- **[surface]**, the condition "oxygen < 60 and at least one diver on board", that is, it is time to go back to the
  surface.

## 3. Structure of the network

```
600 → 32 (tanh) → 32 (tanh) → 6 (linear)        20,486 parameters, 803 nonzero
```

| layer | what it does |
|---|---|
| 1 (tanh, 32) | **reads**: the row of the submarine, the rows with a diver to rescue, the rows with a danger near column 5, the rows with a target in the line of fire, whether it is time to surface |
| 2 (tanh, 32) | **evaluates**: for the row of the submarine, the value of staying, going up or going down; 6 neurons are empty |
| 3 (linear, 6) | **chooses**: fire = stay, up, down; no-op, left and right are always −100 |

**The network never moves horizontally.** The logits of no-op, left and right have only the bias −100, while up,
down and fire are at least −24. The softmax never chooses left or right, and sticky actions only repeat actions
already taken (at the first step, "no-op"). So the submarine stays in column 5 for the whole episode, facing left, as at the
start: in all 263,355 states of the verification. "Fire" acts as "stay here": it fires if the torpedo is loaded,
otherwise it does nothing.

One idea to keep in mind: with tanh a saturated neuron is **+1 (true)** or **−1 (false)**, not 0. In the next layer
the neurons that are off count −1 times their weight, and the biases also serve to compensate for this. The whole
first layer is made of true-or-false conditions (with one exception, the nearby danger). The second layer instead is
**not** saturated: it works in the central part of tanh, where the output grows with the input, and compares scores.

## 4. First layer: what it reads (32 neurons)

| neurons | inputs | question | value |
|---|---|---|---|
| h0-h8 | player, row r, weight 6, bias −6 | is the submarine in row r = 0..8? | ±1 |
| h9-h15 | reward +8 and danger −8 over the whole row j = 2..8, bias −4 | is there a diver to rescue in row j? | ±1 |
| h16-h23 | danger and reward in columns 2-8 of row j = 1..8, bell-shaped weights, bias −3 | is there a danger near column 5 in row j? | ±1, sometimes intermediate |
| h24 | gauge (9, 2) weight −6, (9, 8) weight +6, bias −3 | [surface]: oxygen < 60 and at least one diver on board? | ±1 |
| h25-h31 | danger in columns 0-3 of row j = 2..8, weight 6, bias −3 | is there something in the line of fire in row j? | ±1 |

### h0-h8: the row of the submarine

h5 has weight 6 on indices 300, 306, ..., 354, that is, the player channel of the whole row 5, and bias −6.

- **Submarine in row 5:** bow and stern turn on two cells, 6 + 6 − 6 = 6, tanh = +1.
- **Elsewhere:** −6, tanh = −1.

It is the row of the submarine in "one-hot" code (±1). Row 9 is missing: the submarine never goes there.

### h9-h15: divers to rescue, with a subtraction between channels

h12 looks at row 5: weight +8 on the reward channel (indices 302, 308, ..., 356) and −8 on the danger channel (301, 307,
..., 355) of all 10 cells, bias −4. For each cell it counts reward − danger:

| object in the cell | danger | reward | contribution |
|---|---|---|---|
| fish or enemy submarine | 1 | 1 | 0 |
| diver to rescue | 0 | 1 | **+8** |
| enemy torpedo | 1 | 0 | −8 |

The reward channel mixes enemies and divers, but subtracting danger cancels the enemies and leaves only the divers.
With a diver in the row the pre-activation is 8 − 4 = 4 (tanh = +1), without one it is −4 (−1). A side effect: an
enemy torpedo in the same row cancels the diver. Row 1 has no neuron of this kind, so divers in row 1 are invisible.

### h16-h23: nearby danger, with a bell of weights

h20 looks at row 5, only columns 2-8. Bias −3.

| column | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| danger weight (index) | 2 (313) | 4 (319) | 7 (325) | 7 (331) | 7 (337) | 5 (343) | 2 (349) |
| reward weight (index) | −1.8 (314) | −3.6 (320) | −1 (326) | 0 | −1 (338) | −3 (344) | −1.8 (350) |
| **enemy** (danger + reward) | 0.2 | 0.4 | **6** | **7** | **6** | 2 | 0.2 |
| **enemy torpedo** (danger only) | 2 | **4** | **7** | **7** | **7** | **5** | 2 |
| **diver** (reward only) | −1.8 | −3.6 | −1 | 0 | −1 | −3 | −1.8 |

The neuron is tanh(sum − 3). With a single object in the row:

- **on (+1):** an enemy in columns 4-6, that is, in front of the bow, on the bow or on the stern, or an enemy
  torpedo in columns 3-7 (in column 3 tanh 1 = 0.76, in column 7 tanh 2 = 0.96);
- **half on:** an enemy in column 7, or a torpedo in column 2 or 8: tanh(−1) = −0.76, that is, 12% of the
  way from −1 to +1. In the second layer this is enough to change the choice (section 6);
- **off:** an enemy in columns 2, 3 and 8 (0.2-0.4) and everything in columns 0, 1 and 9.

The bell reads as a sensible rule for the game. An enemy moves one step every 3-6, so it is dangerous only
when it is one cell from the bow. An enemy torpedo moves one step per turn, so it is dangerous even at two cells. An
enemy in column 3 is in the line of fire and can be hit, so it counts little (0.4); one in column 7 is behind
the stern, where torpedoes do not reach, and counts more (2).

The negative weights on the reward channel serve to tell enemies and torpedoes apart, but they have a side effect: **a
diver in the row lowers the danger.** For example, a diver in column 3 (−3.6) and an enemy in column 6 (+6) give
6 − 3.6 − 3 = −0.6: the neuron stays almost off, and the row looks safe.

### h24: time to surface

Weight −6 on index 557, cell (9, 2), and +6 on 593, cell (9, 8). Bias −3.

| oxygen | divers on board | (9, 2) | (9, 8) | pre-activation | h24 |
|---|---|---|---|---|---|
| < 60 | ≥ 1 | 0 | 1 | +3 | **+1** |
| < 60 | 0 (or −1) | 0 | 0 | −3 | −1 |
| 60-179 | ≥ 1 | 1 | 1 | −3 | −1 |
| 60-179 | 0 | 1 | 0 | −9 | −1 |
| ≥ 180 | any | 1 | 1 | −3 | −1 |

Two cells of the bar are enough: (9, 2) says "oxygen ≥ 60", (9, 8) "at least one diver on board". With oxygen
below 60 the second cannot be turned on by oxygen, so the ambiguity of the shared bar disappears. h24 is
[surface].

### h25-h31: targets in the line of fire

h28 has weight 6 on the danger channel of cells (5, 0), (5, 1), (5, 2), (5, 3), that is, indices 301, 307, 313, 319,
and bias −3. It is +1 if those cells contain at least one enemy or enemy torpedo. These are the columns that a torpedo
fired from the bow in column 5 crosses: an enemy there is a target.

### What the first layer does not look at

- **The projectile and trail channels:** no weights. The network does not know which way enemies and divers are going
  (MinAtar shows it only with the trail) and does not know whether one of its torpedoes is already in flight.
- **Row 1** for divers and targets, **row 0** entirely (there are never enemies or divers there).
- **The oxygen bar:** only 2 cells out of 10.
- **Columns 0, 1 and 9** for the nearby danger: an enemy arriving from the right becomes visible in column 7.

## 5. Second layer: the value of the rows (32 neurons, 26 used)

23 of the 32 neurons follow **a single pattern**. Each one belongs to a submarine row ρ and to a move, and looks at
the row the move takes the submarine to, the **arrival row** k:

| neurons | move | submarine row ρ | arrival row k |
|---|---|---|---|
| g0-g7 | fire (stay) | 1..8 | ρ |
| g8-g14 | up | 2..8 | ρ − 1 |
| g16-g23 | down | 0..7 | ρ + 1 |

```
pre = 10·h(row ρ)                                   the gate: is the submarine in row ρ?
    + Σ_j a(|j − k|) · h(diver in row j)            j = 2..8
    + Σ_j c(|j − k|) · h(target in row j)           j = 2..8
    − 3 · h(danger in row k)
    − 0.3 · h(danger in the row after k)            only up and down: k − 1 for up, k + 1 for down
    + bias

a(d) = 0.6, 0.45, 0.34, 0.255, 0.19, 0.145, 0.11, 0.08     (≈ 0.6 · 0.75^d)
c(d) = 0.35, 0.12, 0.04, 0.01                               (≈ 0.35 · 3^−d)
```

For example g4 (fire with the submarine in row 5) has weight 10 on h5, 0.6 on h12 (diver in row 5), 0.45 on h11 and
h13 (rows 4 and 6), and so on down to 0.255 on h9 and h15 (rows 2 and 8); 0.35 on h28 (target in row 5), 0.12 on
h27 and h29; −3 on h20 (danger in row 5); bias −10.32.

**The gate.** If the submarine is not in row ρ, the term 10·h(row ρ) is −10 instead of +10 and the
pre-activation drops below −12: the neuron is stuck at −1, whatever is on the screen. In every state, at most three
of the 23 neurons are alive, those of the submarine's row. It is a table indexed by the row, built with
an AND.

**The bias.** In all 23 neurons the bias is chosen so that, with the gate open and everything else off
(−1), the pre-activation is exactly **−0.70**. For g4: 10 − (sum of the a weights = 2.69) − (sum of the
c weights = 0.69) + 3 (the danger that is off counts −3 · −1) − 10.32 = −0.70. The biases that differ from neuron to
neuron (−11.96, −11.09, −10.63, ...) only compensate for the fact that, near the edges, the sum of the a and c weights
is smaller.

**The value of a row.** Rewriting with [x] = 1 if true and 0 if false (h = 2[x] − 1), the live neuron is
tanh(V), with

```
V(k) = −0.70 + Σ_divers 1.2·0.75^d + Σ_targets 0.7·3^−d − 6·[danger in row k]
```

where d is the distance in rows between the diver (or the target) and the arrival row k.

- **Divers:** a diver in the arrival row is worth +1.2, one row away +0.9, two rows away +0.68, and so on down to 0.16
  at seven rows. It is an attraction that decreases with distance: all divers on the screen pull, the nearest ones the
  most.
- **Targets:** +0.7 in the row, +0.24 one row away, +0.08 two rows away. They attract only from close by.
- **Danger:** −6 sends tanh(V) to −1 whatever the attractions. It is a veto.
- **Danger in the row after** (only up and down): −0.6. A small caution: moving towards a row next to a
  danger costs a little.

These neurons **are not saturated**: tanh(−0.70) = −0.60, tanh(0.5) = 0.46. Here tanh only acts as a compressor:
it flattens very high values (many divers) and very low ones (danger).

### Going back to the surface

- **g8-g14 (up):** they also have a weight on h24. For g9-g14 it is 0.75, that is, +1.5 to V when [surface] holds.
  For g8 (up from row 2 to row 1) it is 4.75 and the bias is lower: V is −8.7 without [surface] and +0.8 with it.
  Without a need to surface, the network never goes up to row 1.
- **g15** = tanh(6·h1 + 6·h24 − 6): submarine in row 1 **and** [surface].
- **g24** = tanh(3·h1 + 3·h2 − 3·h24): submarine in row 1 or 2 **and not** [surface].
- **g25** = tanh(3·h1 − 3·h24 − 3): submarine in row 1 **and not** [surface].
- **g26-g31:** no weights and no bias. They are 0 and do not count.

## 6. Last layer: the rule (6 logits)

```
no-op = left = right = −100
fire  = 241.8 + 30·Σ g0..g7
up    = 238   + 30·Σ g8..g14 + 40·g15 − 12·g24
down  = 244   + 30·Σ g16..g23 + 4·g25
```

The biases are large because **neurons that are off are −1**. With the submarine in row r, 7 of the 8 "fire" neurons
are off and give −210, which brings 241.8 down to 31.8. The same holds for up (6 off out of 7: 238 − 180 = 58) and
down (7 out of 8: 244 − 210 = 34). What remains:

| row r | fire (stay) | up | down |
|---|---|---|---|
| 0 | 1.8 | 0 | 30 + 30·tanh(D) |
| 1 | 31.8 + 30·tanh(F) | **80** if [surface], otherwise −24 | 38 + 30·tanh(D) (30 + … if [surface]) |
| 2 | 31.8 + 30·tanh(F) | 30 + 30·tanh(U) if [surface], otherwise **−24** | 30 + 30·tanh(D) |
| 3-7 | 31.8 + 30·tanh(F) | 30 + 30·tanh(U) | 30 + 30·tanh(D) |
| 8 | 31.8 + 30·tanh(F) | 30 + 30·tanh(U) | 0 |

with

```
F = V(r)                                               stay
U = V(r − 1) − 0.6·[danger in r − 2] + 1.5·[surface]   go up
D = V(r + 1) − 0.6·[danger in r + 2]                   go down
```

So the three moves have the same form, 30 + 30·tanh(value of the arrival row), and **the row with the highest
value wins**. Staying has 1.8 extra and wins ties. The exceptions are at the edges:

- **row 0:** fire and up are 1.8 and 0, down up to 60. It dives, unless row 1 has a danger;
- **row 1:** up is 80 if [surface] and beats everything; otherwise −24, because surfacing without divers is death.
  Down has 6.2 extra to bring the submarine back down;
- **row 2:** going up is forbidden unless there is a need to surface;
- **row 8:** it cannot go down.

The scale 30 makes the choice almost deterministic when the values differ, and a lottery when they are equal.
Here are the logits of the real network in some hand-built situations (`check.py --examples`; submarine in row 5 unless
stated, objects as (row, column)):

| situation | up | down | fire | p(up, down, fire) |
|---|---|---|---|---|
| empty sea, row 5 | 11.7 | 11.7 | 13.5 | 0.12, 0.12, **0.75** |
| diver in my row, (5, 1) | 35.6 | 35.6 | 45.4 | 0, 0, **1** |
| diver one row below, (6, 1) | 29.1 | 43.6 | 37.4 | 0, **1**, 0 |
| diver three rows below, (8, 1) | 20.4 | 29.1 | 25.9 | 0, **0.96**, 0.04 |
| left target in my row, (5, 2) | 16.8 | 16.8 | 31.2 | 0, 0, **1** |
| left target one row above, (4, 2) | 29.4 | 13.2 | 18.6 | **1**, 0, 0 |
| enemy at (5, 6), on the stern | 11.7 | 11.7 | 1.8 | **0.5, 0.5**, 0 |
| enemy at (5, 7), half on | 11.7 | 11.7 | 5.2 | **0.5, 0.5**, 0 |
| enemy torpedo at (5, 3) | 16.8 | 16.8 | 1.8 | **0.5, 0.5**, 0 |
| enemy at (6, 5), one row below | 11.7 | 0.0 | 13.5 | 0.14, 0, **0.86** |
| enemy at (7, 5), two rows below | 11.7 | 4.1 | 13.5 | 0.14, 0, **0.86** |
| enemy at (6, 6) and divers at (6, 3), (6, 8): the enemy is "covered" | 29.1 | **42.8** | 37.4 | 0, **1**, 0 |
| diver (7, 1), target (4, 2), enemy (6, 6) | **43.6** | 0.0 | 38.0 | **1**, 0, 0 |
| oxygen 50, 2 divers on board | **49.7** | 11.7 | 13.5 | **1**, 0, 0 |
| oxygen 50, no divers on board | 11.7 | 11.7 | 13.5 | 0.12, 0.12, **0.75** |
| oxygen 50, 2 divers, enemy at (4, 5) | 0.0 | 11.7 | 13.5 | 0, 0.14, **0.86** |
| row 2 | −23.9 | 11.7 | 13.5 | 0, 0.14, **0.86** |
| row 1 | −23.9 | **19.6** | 13.4 | 0, **1**, 0 |
| row 1, oxygen 50, 2 divers | **79.9** | 11.7 | 13.4 | **1**, 0, 0 |
| row 0 (surface) | −0.1 | **11.6** | 1.8 | 0, **1**, 0 |
| row 0, enemy at (1, 5) | −0.1 | 0.0 | 1.8 | **0.12, 0.13**, 0.75 |

The formulas, with the real values of the first-layer neurons, reproduce the logits of the network within 0.063
(section 8). The numbers above are slightly lower than those of the formulas with inputs of exactly ±1 (13.5 instead
of 13.7 in the empty sea), because an "off" neuron with pre-activation −3 is tanh(−3) = −0.995 and not −1.

## 7. The algorithm

```
the submarine always stays in column 5, facing left; "fire" means "I stay here and fire"

value(row k) = Σ divers in rows 2-8                    1.2 · 0.75^(distance in rows from k)
             + Σ targets in columns 0-3 of rows 2-8    0.7 · 3^−(distance from k), up to 3 rows
             − 6   if row k has a danger near the bow
                   (enemy in columns 4-6, enemy torpedo in columns 3-7)
             − 0.7 if there is a "half" danger (enemy in column 7, torpedo in column 2 or 8)

at the surface (row 0):   down, unless row 1 has a danger
                          (then it waits, but with fire 1.8, up 0, down 0: it is a lottery, section 9)
if oxygen < 60 and there is at least one diver on board:
    in row 1:  up (surfaces)
    elsewhere: "up" gets +1.5 of value
choose among:
    stay and fire    value(r)
    up               value(r − 1) − 0.6 if there is a danger in r − 2      (never above row 2,
                                                                           except to surface)
    down             value(r + 1) − 0.6 if there is a danger in r + 2      (not from row 8)
the move with the highest value; on a tie (within 0.1) stay and fire
```

The strategy follows the rules of the game:

- **Everything flows horizontally.** A diver in the submarine's row sooner or later reaches column 5 and is
  collected, unless it is already moving away. An enemy in the submarine's row, on the left, sooner or later
  meets a torpedo. Choosing the row is enough.
- **Dangers are a veto,** attractions a graded score.
- **Oxygen** dictates the return to the surface, and the network goes back only if it has a diver, that is, only if
  surfacing is not fatal.

## 8. Verification

`check.py` plays the same 200 episodes as `uv run evaluate seaquest paper_astra --episodes 200` (seeds 10000-10199,
sticky 0.1, actions sampled from the softmax with the same generator). In every state it recomputes, **directly from
the game state and without using the weights**:

- the 32 neurons of the first layer, with the formulas of section 4;
- the 6 logits, with the formulas of section 6;
- the action of the simplified rule of section 7, made only of true-or-false conditions;

and compares them with the real forward pass.

| check | result |
|---|---|
| states visited | 263,355 (score 48.3 ± 2.9, the same as `evaluate`) |
| submarine in column 5, facing left | **100%** of states |
| first layer, states where the reconstruction differs (> 1e-9) | **0** |
| logits, maximum difference | 0.063 (some second-layer neurons are taken as exactly ±1) |
| move with the highest logit differs, excluding near-ties < 0.5 | **0** (near-ties: 4,238) |
| simplified rule, same move as the network | **97.4%** of states; **99.1%** where the network has a margin > 3 (76% of states) |

The differences of the simplified rule come from two simplifications: the nearby danger treated as on,
half on or off (in the network the contributions add up, divers included), and the comparison between values instead
of the one between tanh values, which flattens high values.

Which move has the highest logit, by submarine row:

| row | % of states | up | down | fire |
|---|---|---|---|---|
| 0 | 0.9% | | 86% | 14% |
| 1 | 1.9% | 32% | 40% | 28% |
| 2 | 13.6% | 5% | 10% | 85% |
| 3 | 15.8% | 9% | 10% | 81% |
| 4 | 16.2% | 10% | 9% | 80% |
| 5 | 15.9% | 10% | 9% | 81% |
| 6 | 13.6% | 10% | 8% | 82% |
| 7 | 11.8% | 10% | 6% | 84% |
| 8 | 10.4% | 8% | | 92% |

The submarine spends 97% of the time in rows 2-8, spread almost uniformly. It reaches row 1 only
to surface, or because of a sticky action; in row 2 "up" wins only with [surface].

## 9. Limits visible in the weights

In the 200 episodes of the verification the network scores 48.3 points per episode: 42.9 from enemies hit and 5.4
from surfacings with 6 divers. Every episode ends with a death (none reaches the limit of 10,000 steps), on average
after 1,317 steps. `check.py` classifies the 200 deaths as follows:

| how it dies | deaths | |
|---|---|---|
| **sticky: moved, danger already on:** a repetition of up or down took it into a row whose danger was already on one step before | 115 | 57% |
| **dive from the surface (softmax):** in row 0, with a danger in row 1, the softmax chooses "down" | 23 | 12% |
| **trapped: no safe move:** danger in the row where it ends up and no logit above 10 | 16 | 8% |
| **sticky: moved, danger not on:** moved into a row whose danger was not yet on (12 are enemy torpedoes from the left) | 15 | 8% |
| oxygen ran out | 11 | 6% |
| **sticky: held still** while the network wanted to escape | 8 | 4% |
| **danger not on, diver in the row:** the danger was not on because of the diver | 6 | 3% |
| **sticky: surfaced with no divers on board:** a repeated "up" took it to the surface | 6 | 3% |

- **Danger is a veto only for the arrival row.** Moving towards a row *next to* a danger costs
  only 0.6 of value, less than a diver two rows away (0.68). So the network does not avoid placing itself one row from
  an enemy. With sticky actions, the next move repeats the previous one with probability 0.1, and takes it onto
  the enemy. Sticky actions are involved in 72% of deaths (144 out of 200). Without sticky actions
  (`uv run evaluate seaquest paper_astra --sticky 0`, the same 200 seeds) the network scores **110.1 ± 2.9** points
  instead of 48.3 ± 2.9.
- **At the surface it draws lots.** In row 0 the logits are fire 1.8, up 0 and down up to 60. If row 1 has a
  danger, down drops to 0 and the three moves end up almost even: at every waiting step the network dives onto the
  danger with probability 0.13 (table of section 6, last row). In the other rows this does not happen, because
  there a safe move is worth at least 11.7. Always choosing the move with the highest logit instead of sampling
  (`uv run evaluate seaquest paper_astra --greedy`, 200 episodes) the network scores 53.9 ± 3.1.
- **Its world is rows 2-8 of column 5.** In row 2 it cannot go up (row 1 is forbidden unless it has to
  surface), in row 8 it cannot go down, and it never moves sideways. If danger arrives in its row and
  in the only adjacent row, it has no escape route: of the 16 "trapped" deaths, 14 are in rows 1, 2 and 8.
- **It fires only to the left.** Enemies arriving from the right cannot be hit, only avoided, and they are the ones
  that kill it most often: 86 deaths, against 60 for enemies from the left (the rest are enemy torpedoes). The
  danger detector sees them only from column 7.
- **Divers cover enemies.** The negative weights on the reward channel in h16-h23 lower the danger when the row
  also contains a diver. A row with a diver and an enemy in column 6 looks safe and attracts (table of section 6:
  with two divers in the row the network goes down onto the enemy with probability 1). 6 deaths.
- **It does not read the trail, so it does not know where things are going.** A diver that has already passed column
  5 and is moving away attracts as much as one that is approaching: on average 20% of the visible divers. It collects
  27.7 of the 43.4 divers that appear in an episode.
- **It surfaces late.** It goes back to the surface only with oxygen below 60, so a surfacing with 6 divers is worth
  at most ⌊59/20⌋ = 2 points instead of 10. Out of 1,645 surfacings, 562 are with 6 divers, and they give 5.4 points
  per episode.
- **Without divers on board there is no plan B.** With oxygen below 60 and no divers, h24 stays off and the network
  does not surface, which is right because surfacing would be fatal. But it does not look for divers with more
  urgency: the attraction stays the same. 11 deaths from oxygen.
- **Unused capacity:** 6 neurons of the second layer (g26-g31) are empty, 3 outputs out of 6 are constant, the
  projectile and trail channels have no weights at all, and row 1 is invisible for divers and targets.

### How much each family of neurons counts

`check.py --ablations` replays the same 200 episodes with one first-layer family fixed at −1 (off), that is,
as if that thing were never on the screen:

| variant | score | steps per episode | deaths from oxygen |
|---|---|---|---|
| full network | 48.3 ± 2.9 | 1,317 | 6% |
| without divers (h9-h15) | 39.3 ± 2.7 | 1,011 | 49% |
| without left targets (h25-h31) | 24.8 ± 1.6 | 1,432 | 4% |
| without nearby danger (h16-h23) | 3.2 ± 0.3 | 147 | 0% |
| without surfacing (h24) | 4.3 ± 0.1 | 199 | 96% |
| without divers, targets and danger | 1.3 ± 0.1 | 123 | 2% |

- **Danger and surfacing are indispensable.** Without the first the network dies on average after 147 steps; without
  the second it stays underwater until the last oxygen.
- **Without targets the score halves.** The network also lives a little longer, but it hits only what
  happens to pass in front of it.
- **Divers are needed above all for oxygen.** Without the attraction towards divers, when oxygen runs out it often has
  none on board, and half the episodes end this way.

All of the network's behaviour is explained by the formulas of sections 4-6, which match the real network in
every visited state. The simplified rule of section 7 gets the move wrong in 2.6% of states. The mismatches
come from the cases where the details that the rule rounds off matter: several objects in the same row that add up
in the danger detector, and high values that tanh flattens.
