# How to read the Asterix network `paper_astra.json`

`weights/asterix/paper_astra.json` is the network that GPT-6 Astra wrote by hand, without training, for MinAtar
Asterix in the article "Weight-space programming" (Experiment 5). It scores 54.8 ± 1.1 points over 1000 episodes; a
random policy scores 0.5.

This analysis decodes it neuron by neuron, as [`decode_claude`](../../low_tide/decode_claude/README.md) and
[`decode_gpt`](../../low_tide/decode_gpt/README.md) do for Low Tide. This network too is an algorithm written in the
form of weights, and a fairly simple one:

- **it never moves horizontally:** left, right and fire have bias −1000. The player stays in column 5 and only
  chooses the row. All enemies and treasures cross column 5, so the game becomes one-dimensional;
- **for each move (no-op, up, down) it computes an alarm:** the enemies that are coming toward column 5 in the row
  where the move takes the player, and in the next one;
- **among the moves without an alarm it chooses the one that goes toward the treasures:** no-op if a treasure is
  coming in its row, otherwise toward the rows with incoming treasures, with an attraction that decreases with the
  distance in rows.

The reading is checked against the real network on 171,311 game states. The first layer matches exactly; the final
rule, outside near-ties, chooses the same move as the network in all states but 13, and played in place of the
network it gets the same score (see [Verification](#8-verification)):

```bash
uv run python experiments/asterix/decode_astra/check.py                  # ~40 s
uv run python experiments/asterix/decode_astra/check.py --regola 1000    # + the rule in place of the network
uv run python experiments/asterix/decode_astra/check.py --varianti 1000  # + the network with one piece removed
```

## 1. The game in brief

The grid is 10×10. The rules are in `.venv/lib/python3.12/site-packages/minatar/environments/asterix.py`.

- **Player:** starts in row 5, column 5. It moves one cell per step in the four directions, but stays in rows 1-8:
  "up" from row 1 and "down" from row 8 do not move it.
- **Enemies and treasures:** they appear on the left edge (column 0, and move right) or on the right edge (column 9,
  and move left), in a free row between 1 and 8. **Each row holds at most one entity.** One entity in three is a
  treasure. They cross the whole row in a straight line, then leave.
- **Treasure touched:** +1. **Enemy touched:** the episode ends. There are no lives and no other ways to die.
- **Order within a step:** first the player moves and collisions are checked, then (if it is their turn) the
  entities move and collisions are checked again.
- **Speed:** at the start the entities advance one cell every 5 steps and one appears every 10. Every ~100 steps the
  difficulty goes up. From step ~816 the entities move at every step, from step ~918 one appears at every step. The
  timer is not in the observation: looking at a frame, one cannot tell when the entities will move.
- **Fire:** in Asterix it does nothing.
- **Sticky actions:** with probability 0.1 the environment repeats the previous action instead of the chosen one.

## 2. How to read the positions

The observation has 600 numbers in {0, 1}: the 10×10 grid with 6 channels per cell.

```
index = (row*10 + column)*6 + channel

channel 0 player      1 danger (the enemies)   2 reward (the treasures)
        3 projectile  4 trail                  5 gauge                   (3 and 5 are always empty in Asterix)
```

**Row 0 is at the top, column 0 on the left.** For example the player in the starting cell (5, 5) is index
`55*6 + 0 = 330`; an enemy at (4, 3) is `43*6 + 1 = 259`; the trail at (4, 5) is `45*6 + 4 = 274`.

```
           col 0 1 2 3 4 5 6 7 8 9
 row 0        . . . . . . . . . .      always empty
 row 1   ┐    ← entities moving left appear in column 9
 ...     │    rows 1-8: at most one entity per row; the player is here
 row 8   ┘    → entities moving right appear in column 0
 row 9        . . . . . . . . . .      always empty
                        ↑
                    column 5: the player never leaves it
```

- **Trail:** it is the cell the entity comes from (the one behind it), so it tells which direction the entity is
  moving. An enemy at (4, 3) with the trail at (4, 2) is moving right, that is toward column 5.
- **Actions:** 0 no-op, 1 left, 2 up (row − 1), 3 right, 4 down (row + 1), 5 fire.

Three names I use throughout the document:

- **r:** the player's row;
- **incoming:** an entity that is moving toward column 5, or that is on it. One that has already left column 5
  behind is **moving away**;
- **d:** for an incoming entity, the cells it still needs to reach column 5 (0 if it is already there). An enemy
  at (4, 3) moving right has d = 2.

## 3. Network structure

```
600 → 32 (tanh) → 32 (tanh) → 6 (linear)        20,486 parameters, 637 nonzero
```

| layer | what it does |
|---|---|
| 1 (tanh, 32) | **measures**, row by row: is there an incoming enemy near column 5, above, below or in the player's row? Is there an incoming treasure, and how many rows above or below? |
| 2 (tanh, 32) | **summarizes**: three alarms, one per move (sum of the dangers), and four thresholds on the treasure of each row |
| 3 (linear, 6) | **decides**: each move is worth 61 without an alarm and 0 or less with an alarm, plus the attraction of the treasures; left, right and fire are worth −1000 |

The nonzero weights are 468 in the first layer, 53 in the second and 46 in the last, plus 70 biases. Left, right
and fire have no weights, only the bias −1000: they are never sampled, and the sticky actions can only repeat an
action already taken. In the first layer **all the weights on the player are in column 5**: the network looks at
the player only there, and indeed it never takes it away from there. The 32 units of the first layer all have bias
−8.

## 4. First layer: what it measures (32 neurons, tanh)

| neurons | question (k = row looked at) | used for |
|---|---|---|
| h0-h7 | is there an incoming enemy in row k = 1..8, one or two rows **above** me? | alarm of "up" |
| h8-h15 | is there an incoming enemy in row k, one or two rows **below** me? | alarm of "down" |
| h16-h23 | is there an incoming enemy in row k, which is **mine**? | alarm of "no-op" |
| h24-h31 | is there an incoming **treasure** in row k? How many rows above or below me? | attraction |

Each neuron looks at a single row. It adds up three things: what is in the row (enemy or treasure, and whether it is
incoming), where the player is relative to that row, and the bias −8.

### 4.1 The enemy: incoming or not, with the trail

Take h3, which looks at row 4 from the "above me" point of view. Its weights on row 4:

```
column             3     4     5     6     7
danger (enemy)     +5    +6    +11   +6    +5        indices 259, 265, 271, 277, 283
trail                    −5    −6    −5              indices 268, 274, 280
```

The trick is in the trail. **The trail of an enemy that is moving away is on the cell on the side of column 5, and
its weight cancels that of the enemy:**

- enemy at (4, 4) moving right (incoming, d = 1): the trail is at (4, 3), weight 0 → **+6**;
- enemy at (4, 4) moving left (moving away): the trail is at (4, 5), weight −6 → 6 − 6 = **0**;
- enemy at (4, 3) moving right (incoming, d = 2): trail at (4, 2), weight 0 → **+5**;
- enemy at (4, 5), in column 5: 11 − 5 (the trail is at (4, 4) or (4, 6)) → **+6**.

The same scheme holds for all the danger neurons. The contribution of the enemy, E, is:

| enemy in row k | E in h0-h15 (rows above and below) | E in h16-h23 (my row) |
|---|---|---|
| incoming, d ≤ 1 (also d = 0) | 6 | 6 |
| incoming, d = 2 | 5 | 4 |
| incoming, d = 3 | 0 | 3 |
| farther, or moving away, or none | 0 | 0 |

In its own row the network looks one cell further (columns 2-8 instead of 3-7, weights in `h19`: 3, 4, 9, 10, 9,
4, 3 on the enemy and −3, −4, −9, −4, −3 on the trail).

### 4.2 Which row counts: the weight of the player

Each neuron has a weight on the player only in the cells (r, 5) from which that row is relevant:

- **h3 (row 4, above):** player at (5, 5) **+4** (index 330), at (6, 5) **+3** (index 390). That is: row 4 is the
  destination row of "up" if I am in row 5, the next row if I am in row 6.
- **h11 (row 4, below):** player at (3, 5) +4, at (2, 5) +3.
- **h19 (row 4, mine):** player at (4, 5) +4.

The pre-activation is therefore `−8 + E + P`:

| incoming enemy | destination row (P = 4) | next row (P = 3) | my row (P = 4) | row that does not concern me (P = 0) |
|---|---|---|---|---|
| d ≤ 1 | 2 | 1 | 2 | −2 |
| d = 2 | 1 | 0 | 0 | −3 (above, below) / −4 (mine) |
| d = 3 | −4 | −5 | −1 | −8 (above, below) / −5 (mine) |
| none | −4 | −5 | −4 | −8 |

In the rest of the document it is convenient to use `1 + h`, that is how far the neuron is above its "off" value
(−1). It is its **contribution to the alarm**:

| pre-activation | ≤ −3 | −2 | −1 | 0 | 1 | 2 | ≥ 3 |
|---|---|---|---|---|---|---|---|
| 1 + tanh(pre) | ≈ 0 | 0.04 | 0.24 | 1.00 | 1.76 | 1.96 | 2.00 |

Example, player in row 5, enemy at (4, 3) moving right: h3 = tanh(−8 + 5 + 4) = tanh(1) = 0.76, contribution 1.76
to the alarm of "up".

### 4.3 The edges

Rows 1 and 8 have different weights, because there one of the two vertical moves does not move the player.

- **h0 (row 1, above):** player at (1, 5) **+8**, at (2, 5) +6, at (3, 5) +3 (indices 90, 150, 210). With the player
  in row 1, "up" leaves it where it is: h0 looks at row 1 itself, and even **without enemies** it is tanh(0) = 0,
  that is contribution 1. It is a **wall**. With an incoming enemy at d ≤ 2 in row 1 it rises to 2.
- **Player in row 2:** the weight 6 instead of 4 makes row 1 more dangerous, and without enemies h0 is tanh(−2):
  contribution 0.04, that is "up" toward the edge costs 1.2 points.
- **h16 and h23 (my row, if it is row 1 or 8):** player +7 instead of +4. An enemy at d ≤ 3 gives at least 1.96, and
  **without enemies** the contribution is 0.24: staying still on the edge costs a little.
- **h15 (row 8, below)** does the same for row 8 and "down": player at (8, 5) +8, (7, 5) +6, (6, 5) +3.
- **h7 (row 8, above) and h8 (row 1, below)** have no weights on the player: there is no row below 8 from which to
  look at row 8 as "above me". They are dead neurons, with pre-activation at most −2.

### 4.4 h24-h31: the treasure, with the distance in rows

Take h27, row 4. On the treasures and on the trail of row 4 it has a **staircase**:

```
column    0    1    2    3    4    5    6    7    8    9
treasure  +8   +8   +16  +16  +24  +24  +24  +16  +16  +8
trail          −8   −8   −16  −16  −24  −16  −16  −8   −8
```

An **incoming** treasure has its trail on the side of the edge, where the weight of the trail is 8 less than its
own; one that **is moving away** has its trail on the side of column 5, where the weight of the trail equals its
own:

- treasure at (4, 2) moving right: 16 − 8 (trail at (4, 1)) = **8**;
- treasure at (4, 2) moving left: 16 − 16 (trail at (4, 3)) = **0**.

It gives 8 for every incoming treasure and 0 for every treasure moving away, in all columns.

The player has weights along all of column 5: **+0.8, +1.4, +1.5, +1.6** if it is 1, 2, 3, 4 rows below row 4
(that is, the treasure is above it), **−0.8, −1.4, −1.5** if it is above, 0 if it is in row 4. In general, at n rows
of distance, `t = 0.8` for n = 1 and `t = 1.2 + 0.1·n` for n ≥ 2, with the + sign if the treasure is above.

So, with an incoming treasure, `h = tanh(−8 + 8 ± t) = tanh(±t)`:

| incoming treasure in row k | h |
|---|---|
| in my row | 0 |
| 1, 2, 3, ..., 7 rows above | 0.66, 0.89, 0.91, 0.92, 0.94, 0.95, 0.96 |
| 1, 2, 3, ... rows below | −0.66, −0.89, −0.91, ... |
| no incoming treasure | ≈ −1 (pre-activation ≤ −6) |

The neuron encodes three things together: whether there is an incoming treasure, whether it is above or below, and
how far it is. The second layer separates them with four thresholds.

### 4.5 Side effects of the trail

The trail channel is shared by enemies and treasures, so the trail of a treasure enters the danger neurons (weights
from −3 to −9) and that of an enemy enters the treasure neurons (from −8 to −24). Almost always it pushes further
down a neuron that is already off, and changes nothing. The exception is the edge:

- with the player in row 1 and a treasure in row 1 whose trail is in columns 4-6, the **wall** of h0 drops from
  tanh(0) to tanh(−5) or tanh(−6): it disappears;
- if the trail is in columns 3-7, the 0.24 cost of staying still on the edge (h16) disappears as well.

The same holds in row 8. These are treasures that have just left column 5 behind (or incoming treasures at d = 1,
but there "no-op" wins anyway): 284 states out of 171,311 in the 200 episodes of `check.py`. At those moments the
player is no longer pushed away from the edge: it stays, goes up or goes down as in the middle.

### What the first layer does not look at

- **The player outside column 5.** It never goes there.
- **Enemies far from column 5:** more than 2 cells away in the rows above and below, more than 3 in its own row.
  Enemies more than 2 rows away.
- **Treasures that are moving away.**
- **The projectile and gauge channels:** they are always empty.

## 5. Second layer: alarms and thresholds (32 neurons, tanh)

| neuron | pre-activation | meaning |
|---|---|---|
| g0 | 7 + Σ h0..h7 | alarm of "up" |
| g1 | 7 + Σ h8..h15 | alarm of "down" |
| g2 | 7 + Σ h16..h23 | alarm of "no-op" |
| A_k = g(4k−2), k = 2..8 | 21.6 + 24·h(23+k) | incoming treasure in row k: above, in my row or at most 2 rows below |
| B_k = g(4k−1), k = 1..8 | 8 + 20·h(23+k) | incoming treasure in row k: above or in my row |
| C_k = g(4k), k = 1..7 | −8 + 20·h(23+k) | incoming treasure in row k: above |
| D_k = g(4k+1), k = 1..7 | −21.6 + 24·h(23+k) | incoming treasure in row k: at least 3 rows above (graded) |

For example, for row 4 the four neurons are g14 (A), g15 (B), g16 (C), g17 (D). A_1, C_8 and D_8 are missing: a
treasure in row 1 cannot be below the player, one in row 8 cannot be above it.

### The alarms: the bias 7 offsets seven neurons that are off

A danger neuron that is off is about −1, so eight off neurons sum to −8. The bias 7 offsets seven of them:

`g = tanh(7 + Σ h) = tanh(−1 + Σ (1 + h)) = tanh(A − 1)`

where **A = Σ (1 + h)** is the alarm of the move: the sum of the contributions of section 4.2. With A = 0,
g = −0.76; with A = 1, g = 0; with A = 2, g = 0.76; with A ≥ 3, g ≥ 0.96. In practice, during play, A always falls
in one of these groups: 0-0.15, 0.24-0.35 (only for "no-op"), 1.00-1.11, 1.76 or more.

With the tables of section 4, the alarm of each move is:

| move | where the alarm comes from | contribution |
|---|---|---|
| up | incoming enemy in the row above, d ≤ 1 / d = 2 | 1.96 / 1.76 |
| | incoming enemy two rows above, d ≤ 1 / d = 2 | 1.76 / 1.00 |
| | if the row above is row 1: enemy at d ≤ 2 / none | 2.00 / 0.04 |
| up from row 1 (wall) | no enemy in row 1 / enemy at d ≤ 2 | 1.00 / 2.00 |
| down | the same, downward | |
| no-op (rows 2-7) | incoming enemy in my row, d ≤ 1 / d = 2 / d = 3 | 1.96 / 1.00 / 0.24 |
| no-op (rows 1 and 8) | incoming enemy in my row, d ≤ 3 / none | 1.96-2.00 / 0.24 |

To these the **residuals** are added. An incoming enemy at d ≤ 1 brings to −2 (contribution 0.04) its neurons that
do not have the weight of the player. If the enemy is far from the player this happens in all three groups: the
three alarms rise together by 0.04, the three logits drop together by about 1.2 points and the choice does not
change. If the enemy is close, the residual goes only to the moves it does not concern (a case where it matters is
in section 9).

### The thresholds on the treasure

With h = tanh(±t) from section 4.4 the four thresholds fall like this:

| incoming treasure in row k | A | B | C | D |
|---|---|---|---|---|
| none | −0.98 | −1 | −1 | −1 |
| 3 or more rows below | ≤ −0.12 | −1 | −1 | −1 |
| 2 rows below | 0.34 | −1 | −1 | −1 |
| 1 row below | 1 | −1 | −1 | −1 |
| in my row | 1 | 1 | −1 | −1 |
| 1 row above | 1 | 1 | 1 | −1 |
| 2 rows above | 1 | 1 | 1 | −0.34 |
| 3, 4, 5, 6, 7 rows above | 1 | 1 | 1 | 0.12, 0.48, 0.69, 0.81, 0.87 |

A and D are graded near the threshold, B and C are boolean. The next layer uses them in pairs.

## 6. Last layer: the rule (6 logits)

```
no-op = 17    − 80·g2 + 16·Σ(k=1..7) (B_k − C_k) + 16·B_8
up    = 0.15  − 80·g0 +  6·Σ(k=1..7) (C_k − D_k)
down  = −0.69 − 80·g1 +  6·Σ(k=2..8) (A_k − B_k)
left = right = fire = −1000
```

The pairs are read with the threshold table:

- **B_k − C_k** is 2 only with the treasure incoming **in my row**, 0 in all other cases. No-op gets +32. B_8 alone
  is −1 without a treasure: 16·B_8 = −16, which with the 17 of the bias leaves 1.
- **C_k − D_k** is 2 with the treasure 1 row above, 1.34 at 2 rows, then it drops as D turns on.
- **A_k − B_k** does the same for the treasures below. When off, the A_k are −0.984 and not −1: seven of them give
  6 × 7 × 0.016 = 0.69, which the bias −0.69 of down cancels.

Rewritten with the alarm A of each move:

```
no-op = 1    − 80·tanh(A_noop − 1) + 32·[incoming treasure in my row]
up    = 0.15 − 80·tanh(A_up − 1)   + Σ a(n)   over the incoming treasures n rows above
down  = 0    − 80·tanh(A_down − 1) + Σ a(n)   over the incoming treasures n rows below

a(n) = 12, 8.0, 5.3, 3.1, 1.8, 1.1, 0.8   for n = 1, 2, ..., 7
```

The alarm term, as A varies:

| A | 0 | 0.04 | 0.24 | 1 | 1.76 | 1.96 | 2 | 3.52 |
|---|---|---|---|---|---|---|---|---|
| −80·tanh(A − 1) | **60.9** | 59.7 | 51.4 | **0** | −51.4 | −59.7 | **−60.9** | −79.0 |

Here are the real logits of the network in some cases, with the player in row 5 (in rows 1 and 8 where indicated):

| situation | no-op | up | down | move |
|---|---|---|---|---|
| nothing incoming | 61.9 | 61.1 | 60.9 | wanders: no-op 56%, up 24%, down 20% |
| incoming treasure in my row | **93.9** | 61.1 | 60.8 | no-op |
| incoming treasure 1 row above | 61.9 | **73.1** | 60.8 | up |
| incoming treasure 2 / 3 / 4 rows above | 61.9 | 69.1 / 66.3 / 64.2 | 60.8 | up (100% / 98% / 88%) |
| treasure 1 row above and one 2 rows below | 61.9 | **73.1** | 68.7 | up (99%) |
| treasure 2 rows above and one 1 row below | 61.9 | 69.1 | **72.7** | down (97%) |
| incoming enemy in my row, d = 1 | −58.7 | 59.8 | 59.7 | up or down |
| incoming enemy in my row, d = 2 | 1.0 | 60.9 | 60.7 | up or down |
| incoming enemy in my row, d = 3 | 52.4 | 61.1 | 60.9 | up or down |
| incoming enemy in my row, d = 4 | 61.9 | 61.1 | 60.9 | as "nothing": it does not see it |
| incoming enemy in the row above, d ≤ 1 | 60.7 | −59.5 | 59.7 | no-op (73%) or down |
| incoming enemy two rows above, d ≤ 1 | 60.7 | −51.2 | 59.7 | no-op or down |
| incoming enemy two rows above, d = 2 | 61.9 | 0.1 | 60.7 | no-op or down |
| enemy in my row d = 1, treasure 1 row above | −58.7 | **71.8** | 59.6 | up |
| enemy at d = 1 above, in my row and below | −61.1 | −62.0 | −62.1 | no way out: wanders |
| enemy at d = 2 above, at d = 1 in my row and below | −60.0 | **−54.4** | −61.1 | up: the lesser evil |
| row 1, nothing incoming | 52.4 | 0.1 | **60.9** | down: it moves away from the edge |
| row 8, nothing incoming | 52.4 | **61.1** | 0.0 | up |

The hierarchy comes from the magnitudes:

- **Safety first.** A move with alarm 1 loses 61 points, one with alarm 1.76 or more loses more than 110, and the
  treasures give at most 32. As long as there is a move without an alarm, the network does not choose one with alarm
  1 or more. The 0.24 alarm of "no-op" (enemy at d = 3, or the edge) instead costs only 9.6 points.
- **Then the treasure in my row:** no-op reaches 93.9, more than "up" or "down" can get even with treasures in all
  rows (61 + 32.1).
- **Then the treasures in the other rows:** the side with the largest total attraction wins. Several treasures in
  the same direction add up, like the gradient of an attraction field.
- **Nothing in sight:** the three logits differ by less than 1 point and the network samples: it stays still more
  often.
- **At the edges:** staying still costs 9.6 points and the move against the wall is worth 0: the network goes back
  toward the middle, unless the way is blocked or a treasure arrives in the row.

## 7. The algorithm

```
stay in column 5 (left, right, fire: −1000)

for each enemy incoming toward column 5:  d = cells it still has to cover (0 if it is already in column 5)

alarm(no-op) = enemy in my row:                 d ≤ 1 → 1.96,  d = 2 → 1,  d = 3 → 0.24
               (in rows 1 and 8:                d ≤ 2 → 2,     d = 3 → 1.96,  otherwise 0.24)
alarm(up)    = enemy in the row above:          d ≤ 1 → 1.96,  d = 2 → 1.76
             + enemy two rows above:            d ≤ 1 → 1.76,  d = 2 → 1
               (from row 1 "up" does not move:  1, or 2 with an enemy in row 1 at d ≤ 2;
                toward row 1:                   2 with an enemy at d ≤ 2, otherwise 0.04)
alarm(down)  = the same, downward

score(m)     = base(m) − 80·tanh(alarm(m) − 1)          base: no-op 1, up 0.15, down 0
               (61 without alarm, 0 with alarm 1, from −51 to −80 with alarm 1.76 or more)
score(no-op) += 32                  if a treasure is incoming in my row
score(up)    += a(n)                for each treasure incoming n rows above     a = 12, 8, 5.3, 3.1, 1.8, 1.1, 0.8
score(down)  += a(n)                for each treasure incoming n rows below

sample the move from the softmax of the scores
```

In words: the player acts as the **goalkeeper of column 5**. Everything passes through there, so it only needs to
choose the row: it waits for the treasures that arrive in its row, moves toward the rows where others arrive, and
escapes vertically when an enemy approaches column 5 in its row or in the one it is moving to.

This rule leaves out two details: the effects of the trail at the edges (section 4.5), which the rule in `check.py`
includes as an exception, and the 0.04 residuals (section 5), which it ignores.

## 8. Verification

`check.py` plays 200 episodes with the network as `evaluate` does (seeds 10000-10199, same generator for sampling,
sticky 0.1): the score, 54.5, is the same as `uv run evaluate asterix paper_astra --episodes 200`. In every state it
recomputes, **directly from the game state and without using the weights**, the pre-activations of the first layer
(section 4, with the effects of the trail), the state of the second layer, the logits with the formulas of section 6
and those of the rule of section 7. Then it compares them with the real forward pass.

| check | result |
|---|---|
| states visited | 171,311 |
| states with the player outside column 5 | **0** |
| first layer: states in which a pre-activation differs from the reading | **0** (maximum deviation 1.8·10⁻¹⁵) |
| second layer: states in which the state of a neuron differs (g0-g2: off < −0.3, halfway, on > 0.3; g3-g31: sign) | **0** |
| logits with the formulas of section 6: maximum difference | 1.0·10⁻¹³ |
| rule of section 7: maximum difference of the logits | 9.1 (the 0.04 residuals added up, section 5) |
| rule: mean distance between its softmax and that of the network | 0.007 |
| rule: move with the highest logit is different, near-ties (< 1) excluded | **13** out of 133,412 |

The near-ties are 37,899 states: 29,820 are "nothing incoming" (61.9 vs 61.1 vs 60.9), 6,528 have incoming
treasures in other rows, but with small or similar attractions on the two sides. There the network samples and
wanders.

Played in place of the network on the same 1000 seeds of `evaluate` (`--regola 1000`), the rule of section 7 scores
**55.7 ± 1.1** points, against **54.8 ± 1.1** for the network.

Mean probability of each move according to the network, by situation (the first one that applies counts, in this
order):

| situation | % of states | no-op | up | down |
|---|---|---|---|---|
| no move without an alarm (all ≥ 0.5) | 0.8% | 22% | 42% | 36% |
| incoming enemy in my row, d ≤ 3 | 4.8% | 7% | 46% | 46% |
| incoming treasure in my row | 31.4% | 100% | | |
| row 1 or 8 | 7.5% | 64% | 17% | 20% |
| incoming treasure in another row | 21.2% | 33% | 33% | 34% |
| nothing incoming | 34.3% | 73% | 14% | 13% |

- **Treasure in my row:** a third of the time the network waits for a treasure that is coming straight at it.
- **Row 1 or 8:** "no-op" at 64% is the case in which the way toward the middle has an alarm. The edge costs 0.24,
  less than any alarm, and the network waits there.
- **Nothing incoming:** the 73% of "no-op" is more than the 56% of the empty case because in 58% of these states
  "up" or "down" has an alarm of at least 0.5, and the choice is left between no-op and the other move.

The player is in each of rows 2-7 between 13% and 15% of the time, in rows 1 and 8 7.5% and 7.0% of the time.

## 9. Limits visible in the weights

### How much each piece counts

`check.py --varianti 1000` plays the network with one piece removed at a time (1000 episodes, `evaluate` protocol):

| variant | score | mean length |
|---|---|---|
| the network as it is | 54.8 ± 1.1 | 870 |
| without alarms (g0-g2 fixed at the "off" value) | 2.7 ± 0.1 | 94 |
| without the no-op alarm (g2) | 12.9 ± 0.3 | 354 |
| above and below look only at the destination row (weights 3 on the player in h0-h15 removed) | 17.4 ± 0.6 | 398 |
| without treasures (g3-g31 removed from the last layer) | 22.9 ± 0.6 | 873 |
| without the attraction toward the treasures above and below | 34.3 ± 0.7 | 900 |
| without the +32 to no-op for the treasure in my row | 46.3 ± 1.0 | 821 |

- **Surviving counts more than seeking treasures.** Without any signal on the treasures the network still scores
  22.9: it stays alive as long as before (873 steps against 870) and takes the treasures that come at it by chance.
  Without alarms it dies in about a hundred steps.
- **Without the no-op alarm (g2)** the network does not escape from the enemies that come into its row: the length
  drops from 870 to 354 steps and the score to 12.9.
- **Looking two rows ahead is essential:** with only the destination row the score drops from 54.8 to 17.4.

### The edge is where it dies

Rows 1 and 8 are 14.5% of the states but 63.5% of the deaths: 127 out of 200 in the 200 episodes of `check.py`
(row 1: 71, row 8: 56; the other rows between 5 and 16 each).

| cause of death (200 episodes) | deaths | of which on the edge | of which entering a row with an enemy already in column 5 |
|---|---|---|---|
| no move without an alarm, but one of the three survived the step | 90 | 89 | 68 |
| sticky: the environment repeated the previous move | 68 | 8 | 31 |
| no move without an alarm, and none survived | 42 | 30 | 17 |

A typical case: the player is in row 1, an enemy comes toward it in row 1 at d = 2, and in row 2 there is an enemy
**already in column 5**. The alarms are no-op 2.0, up 2.0 (the wall), down 1.96, and the real logits of the network
are no-op −61.0, up −62.0, down −59.9: with 68% probability the player moves down into the enemy. Staying still, or
"up", would have kept it alive at least for that step, because an enemy 2 cells away does not reach column 5 in a
single step. The weights explain why:

- **an enemy in column 5 counts as much as one 1 cell away.** The weight of the enemy in column 5 is 11 (10 in its
  own row), higher than the others, but its trail is always in column 4 or 6, with weight −5 (−4), and brings it
  back to 6: the same value as an enemy at d = 1. Yet entering an occupied cell kills at once, while an enemy at
  d = 1 hits only if the entities move in that step;
- **the alarm saturates at 2.** On the edge the weights of the player are larger (7 and 8 instead of 4), so an enemy
  2 cells away in its own row already gives 2.0, more than an enemy in column 5 of the adjacent row (1.96);
- **the residuals go in the wrong direction.** The enemy in column 5 of row 2 also brings to −2 the neurons "row 2,
  above me" and "row 2, mine", and adds 0.04 to the alarms of up and of no-op: the move that enters the enemy gains
  one more point;
- **the network does not know when the entities move.** The timer is not in the observation, and the network treats
  almost the same way an enemy that can arrive in this step and one that will arrive in 2 or 3.

On the edge the player has only one escape route. Staying still costs 9.6 points, enough to make it leave again when
the adjacent row is free, but if the adjacent row has an alarm it stays there (section 8: on the edge, with neither
enemies nor treasures in its row, it stays still 64% of the time), and there the next trap is more likely.

### Sticky actions

68 deaths out of 200 (34%) come when the environment repeats the previous move instead of the chosen one, 31 of
these into an enemy already in column 5. Without sticky actions (`uv run evaluate asterix paper_astra --sticky 0`)
the network scores **68.9 ± 1.2** points over 1000 episodes, with episodes of 998 steps on average instead of 870.

### Treasures

In the 200 episodes of `check.py` the network collects 68.8% of the treasures that appear (10,906 out of 15,851).
The attraction sums the incoming treasures row by row, but it does not look at how far they are from column 5 nor
whether the player will get there in time: a treasure that has just appeared in column 0 weighs as much as one at
one cell. Beyond 4 rows of distance it is worth less than 2 points: the move toward the treasure becomes the most
likely one, not a certain one.

### Other

- **No lateral movement.** It is a design choice: everything passes through column 5, so no treasure is
  unreachable. But the only possible dodge is vertical.
- **Two dead neurons:** h7 (row 8, above) and h8 (row 1, below) have no weights on the player and never exceed −2.
  They are the price of having written the families symmetrically.

### What remains unexplained

Nothing substantial. Every neuron of the first layer has an exact formula in game terms (deviation 10⁻¹⁵), and so
do the second layer and the logits. Two things blur the simple rule:

- **the effects of the trail at the edges** (section 4.5): 284 states out of 171,311 in the 200 episodes of
  `check.py`. The rule includes them as an exception;
- **the 0.04 residuals per enemy** (section 5), which the rule ignores: they shift the logits by up to 9.1 points,
  but almost always by equal amounts for the three moves. The mean distance between the two softmaxes is 0.007 and
  the rule plays like the network (55.7 against 54.8).
