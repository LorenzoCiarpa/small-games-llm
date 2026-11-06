# How to read the Breakout network `paper_astra.json`

`weights/breakout/paper_astra.json` is the network that GPT-6 Astra wrote by hand for Breakout (MinAtar) in the
article "Weight-space programming" (Experiment 5), without training. It scores 512.5 ± 7.6 points over 1000 episodes
(the article reports 517.4 ± 7.4); a random policy scores 0.5.

This analysis decodes it neuron by neuron, as [`decode_claude`](../../low_tide/decode_claude/README.md) and
[`decode_gpt`](../../low_tide/decode_gpt/README.md) do for the two Low Tide networks. The result is that the network
is a classic algorithm written in the form of weights: **predict the column where the ball will land and move the
paddle there**.

- **The first layer does not compute the trajectory; it looks it up in a table.** The ball occupies a single cell,
  so its 100 weights into a neuron are a "cell → landing column" table. Astra computed the bounces off walls,
  ceiling and bricks for every cell and every direction, and wrote the 400 results into the weights.
- **It uses 6 neurons out of 32 in the first layer and 8 out of 32 in the second.** The other 50 have all their
  weights at zero.

The reading is verified in two ways (see [Verification](#8-verification)). Rewriting the weights from the formulas
in this document gives back the file, parameter by parameter. During play, the reading matches the real network in
all 1,456,882 visited states:

```bash
uv run python experiments/breakout/decode_astra/check.py                            # ~2 min
uv run python experiments/breakout/decode_astra/check.py --sticky 0 --episodes 10   # without sticky actions
```

## 1. The game in brief

The rules are those of MinAtar (`minatar/environments/breakout.py`), which the repo uses without changes.

- **Grid:** 10×10. The paddle is one cell in row 9 and moves one column per step. It starts in column 4.
- **Bricks:** 30, in rows 1-3. Each is worth +1. When they are all gone, the three rows reappear the next time the
  ball reaches the bottom.
- **Ball:** it moves only diagonally, one row and one column per step, so it has 4 directions. It starts in row 3,
  from column 0 heading down-right or from column 9 heading down-left (at random).
- **Walls and ceiling:** if the ball would leave the grid, it stays on the edge and that component of the direction
  is reversed. For this reason it spends **two steps in a row** in column 0 (or 9, or in row 0).
- **Brick:** if the ball would enter a cell with a brick, the brick disappears, the ball stays in its row (it only
  changes column) and the vertical direction is reversed. Exception: right after a brick, a second brick at the next
  step is passed through.
- **Bottom:** at every step the paddle moves first, then the ball. When the ball in row 8 would go down into row 9:
  - if the paddle is below the ball's column, the ball bounces and keeps its horizontal direction;
  - if the paddle is below the column the ball is heading to, the ball bounces "off the edge" and also reverses its
    horizontal direction;
  - otherwise the ball is lost and the episode ends.
- **Actions:** 0 no-op, 1 left, 3 right. Actions 2 (up), 4 (down) and 5 (fire) do nothing in Breakout, like
  "no-op".
- **Sticky actions:** with probability 0.1 the environment repeats the action executed at the previous step instead
  of the chosen one.
- **Limit:** the episode is cut off at 10,000 steps.

The ball and the paddle have the same horizontal speed, one column per step.

## 2. How positions are read

The observation has 600 numbers in {0, 1}: 10×10 cells times 6 channels.

```
index = (row*10 + column)*6 + channel          row 0 at the top, column 0 on the left

channel 0  paddle         always in row 9: indices 540, 546, ..., 594
channel 1  -              always empty in Breakout
channel 2  bricks
channel 3  ball
channel 4  trail          the cell where the ball was at the previous step
channel 5  -              always empty in Breakout
```

For example, the ball in row 5, column 6 is index (56)·6 + 3 = 339, the trail in row 6, column 7 is
(67)·6 + 4 = 406, the paddle in column 9 is (99)·6 + 0 = 594. I write (r, c) for the cell in row r and column c.

```
           col 0 1 2 3 4 5 6 7 8 9
 row 0                                 ceiling
 row 1   ┐
 row 2   │ bricks
 row 3   ┘
 row 4   ┐
 ...     │ the ball flies
 row 8   ┘ the last row before the bottom
 row 9                                 the paddle
```

**The ball's direction** is not in the observation, but the trail gives it: it is ball − trail. I call the 4
directions down-right, down-left, up-right, up-left ("down" means toward the paddle).

Three names I use throughout the document:

- **catch columns:** when the ball is in row 8 in column x and is heading toward x + dx, the paddle catches it if
  after its move it is in x or in x + dx. Against a wall there is only one catch column.
- **T:** the landing column the network predicts; **p:** the paddle's column; **Δ = T − p**.
- **flight:** the ball's path from one bounce on the paddle to the next.

## 3. Network structure

```
600 → 32 (tanh) → 32 (tanh) → 6 (linear)       20,486 parameters, 815 nonzero
```

| layer | neurons used | weights ≠ 0 | what it does |
|---|---|---|---|
| 1 (tanh, 32) | 6: h0-h5 | 769 + 1 bias | **measure**: for each of the 4 directions, Δ = where the ball would land − paddle (h0-h3); the ball's direction (h4, h5) |
| 2 (tanh, 32) | 8: g0-g7 | 24 + 8 bias | **conditions**: "the ball goes in direction k and lands to the left (or to the right) of the paddle", as ±1 booleans |
| 3 (linear, 6) | 2 logits out of 6 | 8 + 5 bias | **rule**: left if a "left" condition is true, right if a "right" one is true, otherwise no-op |

- **Empty neurons:** h6-h31 and g8-g31 have no weights at all, neither incoming nor outgoing. Their value is
  tanh(0) = 0.
- **What the first layer reads:** the ball (553 weights, 72%), the trail (180, 23%), the paddle (36, 5%).
  **No weights on the bricks.**

## 4. First layer (6 neurons, tanh)

| neuron | question | value |
|---|---|---|
| h0 | if the ball were going down-right, where would it land relative to the paddle? | tanh(0.02·Δ) ≈ Δ/50 |
| h1 | the same, down-left | tanh(0.02·Δ) |
| h2 | the same, up-right | tanh(0.02·Δ) |
| h3 | the same, up-left | tanh(0.02·Δ) |
| h4 | is the ball going right? | +1 yes, −1 no |
| h5 | is the ball going down? | +1 yes, −1 no |

### h0-h3: where the ball will land, in a table

Each neuron h0-h3 is dedicated to one direction, but it **does not know** which direction the ball is going: it
computes where the ball would land if it were going in its direction. The second layer chooses which of the four to
listen to.

```
ball in (r, c)             weight 0.02 · T_k(r, c)     a table of 100 values per neuron
paddle in column p         weight −0.02 · p            (column 0 has weight 0)
bias 0
h_k = tanh(0.02 · (T_k − p))
```

- **The table:** the ball is in a single cell, so of the 100 weights on the ball channel only one counts, the one
  for its cell. It is a "cell → landing column" table, with the column multiplied by 0.02.
- **The paddle:** it too is in a single cell, so its weights subtract its column.
- **The 0.02 scale:** with |Δ| ≤ 9 the pre-activation stays between −0.18 and 0.18, where tanh is almost a straight
  line (tanh 0.18 = 0.178). Here tanh does not act as a switch as in the other layers: it lets Δ/50 through with its
  sign.

**What is in the table.** T_k(r, c) is the column the ball is in when, having started from (r, c) in direction k,
it reaches row 8 on the way down. Astra computed it by simulating the game rules:

- **walls and ceiling** exactly as in MinAtar, including the extra step on the edge;
- **if the ball is going up,** it bounces as soon as the row above it is a brick row (1-3), **as if all the bricks
  were there**: from rows 4-8 it bounces in row 4, from rows 2 and 3 immediately, from rows 0 and 1 off the
  ceiling; then it goes down with no obstacles;
- **if the ball is going down and is already in rows 6-8,** T is the midpoint of the two catch columns (for example
  7.5 if columns 7 and 8 catch it), so the paddle stops in either of the two. From rows 0-5 T is the column in
  row 8.

The tables are the weights of the ball channel multiplied by 50. The two for the rightward directions:

```
h0 (down-right)                               h2 (up-right)
        col 0   1   2   3   4   5   6   7   8   9         col 0  1  2  3  4  5  6  7  8  9
row 0       8   9   9   8   7   6   5   4   3   2   row 0     9  9  8  7  6  5  4  3  2  1
row 1       7   8   9   9   8   7   6   5   4   3   row 1     9  8  7  6  5  4  3  2  1  0
row 2       6   7   8   9   9   8   7   6   5   4   row 2     7  8  9  9  8  7  6  5  4  3
row 3       5   6   7   8   9   9   8   7   6   5   row 3     6  7  8  9  9  8  7  6  5  4
row 4       4   5   6   7   8   9   9   8   7   6   row 4     5  6  7  8  9  9  8  7  6  5
row 5       3   4   5   6   7   8   9   9   8   7   row 5     6  7  8  9  9  8  7  6  5  4
row 6     2.5 3.5 4.5 5.5 6.5 7.5 8.5   9 8.5 7.5   row 6     7  8  9  9  8  7  6  5  4  3
row 7     1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5   9 8.5   row 7     8  9  9  8  7  6  5  4  3  2
row 8     0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5   9   row 8     9  9  8  7  6  5  4  3  2  1
row 9       0   1   2   3   4   5   6   7   8   9   row 9     0  1  2  3  4  5  6  7  8  9
```

h1 and h3 are the same tables mirrored: T_left(r, c) = 9 − T_right(r, 9 − c). Row 9 (T = c) is not needed: the ball
gets there only when it is lost. The zeros of the table are the weights missing from the file: h0 has 99 weights
on the ball, h1 84, h2 98, h3 82, plus 9 on the paddle each.

Three cells, with the real indices:

- **h0, ball in (6, 7), index 405:** weight 0.18, T = 9. The ball goes to (7, 8), then to (8, 9); there the next
  column would be 10, so the ball stays in 9 and only column 9 catches it. Midpoint (9 + 9)/2 = 9.
- **h3, ball in (5, 6), index 339:** weight 0 (missing), T = 0. The ball goes to (4, 5), tries to enter row 3,
  bounces and stays in row 4 moving to (4, 4), then goes down to (5, 3), (6, 2), (7, 1), (8, 0).
- **h2, ball in (8, 0), index 483:** weight 0.18, T = 9. This is the ball just returned in the left corner: it goes
  up to (4, 4), bounces into (4, 5) and goes down to (8, 9), in the opposite corner.

### h4, h5: the direction, from ball and trail

```
h4: ball in column c    weight 20·c   (but column 0: 10, column 9: 170)   trail in column c  weight −20·c    bias 0
h5: ball in row r       weight 10·r   (but row 8: 70)                     trail in row r     weight −10·r    bias 5
```

They are two rulers. Ball − trail is the displacement of the last step, ±1 in column and ±1 in row: h4 is ±20 and
h5 is ±10 + 5, enough to saturate tanh. The exceptions in the weights are for the cases where ball and trail have
the same column or the same row, that is, for bounces:

| h4 | pre-activation | reading | case |
|---|---|---|---|
| ball to the right of the trail | +20 (+10 if the ball is in column 9) | right | |
| ball to the left of the trail | −20 (−10 if the ball is in column 0) | left | |
| ball and trail in column 0 | 10 − 0 = **+10** | right | bounce off the left wall: now it goes right |
| ball and trail in column 9 | 170 − 180 = **−10** | left | bounce off the right wall |

| h5 | pre-activation | reading | case |
|---|---|---|---|
| ball below the trail | +15 (+5 if the ball is in row 8) | down | |
| ball above the trail | −5 | up | |
| same row r ≠ 8 | 10r − 10r + 5 = **+5** | down | ceiling, or brick hit from below: now it goes down |
| same row 8 | 70 − 80 + 5 = **−5** | up | bounce off the paddle: now it goes up |

The reading is wrong in two cases, and each time for a single step:

- **edge bounce off the paddle:** the ball reverses its horizontal direction but has just moved in the old one, and
  h4 reads the old one;
- **brick hit from above** (the ball is going down in rows 0-2, above a brick): the ball stays in its row and now
  goes up, but h5 reads "down".

This happens in 0.56% of the states, and in none of the flights in which the network loses the ball (sections 8
and 9).

### Example: ball in (5, 6) going up to the left, paddle in column 9

Active inputs: ball 339, trail 406 (row 6, column 7), paddle 594, plus the bricks, which nobody reads.

```
h0 = tanh(0.18 − 0.18) =  0             down-right: T = 9
h1 = tanh(0.06 − 0.18) = −0.119         down-left:  T = 3
h2 = tanh(0.14 − 0.18) = −0.040         up-right:   T = 7
h3 = tanh(0    − 0.18) = −0.178         up-left:    T = 0, Δ = −9
h4 = tanh(120 − 140)   = −1             goes left
h5 = tanh(50 − 60 + 5) = −0.9999        goes up
```

### What the first layer does not look at

- **The bricks** (channel 2): no weights. The network does not know where they are; for balls going up it imagines
  them all in place.
- **Channels 1 and 5**, which are always empty in Breakout.
- **When the ball will land:** the network knows where it will land, not when. If the paddle is not at the
  predicted landing point it moves right away, one column per step, even when it would have time.

## 5. Second layer: the conditions (8 neurons, tanh)

Two neurons per direction, one per side:

```
g(2k)   = tanh(−415 − 1000·h_k + 200·sx·h4 + 200·sy·h5)       the ball goes in direction k and lands to the LEFT
g(2k+1) = tanh(−415 + 1000·h_k + 200·sx·h4 + 200·sy·h5)       the ball goes in direction k and lands to the RIGHT
```

| neurons | k | h_k | sx (sign on h4) | sy (sign on h5) |
|---|---|---|---|---|
| g0, g1 | down-right | h0 | + | + |
| g2, g3 | down-left | h1 | − | + |
| g4, g5 | up-right | h2 | + | − |
| g6, g7 | up-left | h3 | − | − |

The bias −415 is −400 − 15, and it is an AND of three conditions:

- **the direction is right:** h4 and h5 are ±1, so the two 200 terms give +400 only if the direction read is k. If
  even one of the two signs is wrong the sum is at most −415 + 0 + 178 = −237, and the neuron is off;
- **Δ exceeds the threshold:** with the right direction what remains is −15 ∓ 1000·h_k, and
  1000·h_k = 1000·tanh(0.02·Δ) is ±10 for Δ = ±0.5 and ±19.997 for Δ = ±1. The threshold falls at |Δ| = 0.75, and
  Δ is always a multiple of 0.5:

| Δ = T − p | ≤ −1 | −0.5, 0, +0.5 | ≥ +1 |
|---|---|---|---|
| g(2k), "left" | **from +5 up, on** | from −25 to −5, off | off |
| g(2k+1), "right" | off | from −25 to −5, off | **from +5 up, on** |

The pre-activation is always at least 4.99 away from zero, so every neuron is ±1 within 10⁻⁴: it is a boolean. In
every state at most one of the 8 is on.

In the earlier example (ball up-left, Δ = −9): g6 = tanh(−415 + 178 + 200 + 200) = tanh(163) = +1. The other seven
are off, all below −190.

## 6. Last layer: the rule (6 logits)

```
no-op = 0
left  = 90 + 30·(g0 + g2 + g4 + g6)
right = 90 + 30·(g1 + g3 + g5 + g7)
up = down = fire = −100
```

The bias 90 looks odd because **off neurons are −1, not 0**. With the four "left" neurons off,
left = 90 − 120 = −30; with one on, left = 90 + 30·(1 − 3) = +30. Rewritten with `[x]` = +1 if x is true, −1 if it
is false, and T the predicted column for the direction read:

```
no-op = 0
left  = 30·[T ≤ p − 1]
right = 30·[T ≥ p + 1]
up = down = fire = −100
```

| where the ball lands | left | right | no-op | move |
|---|---|---|---|---|
| to the left of the paddle (T ≤ p − 1) | **30** | −30 | 0 | left |
| above the paddle (\|T − p\| ≤ 0.5) | −30 | −30 | **0** | no-op |
| to the right of the paddle (T ≥ p + 1) | −30 | **30** | 0 | right |

The gap between the first and the second logit is always 30, so the softmax picks the move with probability at
least 1 − 2·10⁻¹³: the sampled network plays like its argmax. Indeed `evaluate --greedy` gives the same score as
`evaluate` on the same 200 episodes (512.67). Up, down and fire are never chosen; in Breakout they would act as
"no-op".

## 7. The algorithm

```
direction = ball − trail                     (walls, ceiling, bricks and paddle handled in the weights)
T = column where the ball will land in row 8, taken from a precomputed table:
    if going up:   bounce off the first brick row above it (as if they were all there) or off the ceiling,
                   then down to row 8, bouncing off the walls
    if going down: down to row 8, bouncing off the walls
                   (from rows 6-8: halfway between the two catch columns)
if T is left of the paddle by at least 1 column:   left
if T is right of the paddle by at least 1 column:  right
otherwise:                                         no-op
```

The network redoes the prediction at every step from the cell the ball is in, so it corrects itself when the ball
deviates (a gap between the bricks, a brick hit from above).

**Why imagine all the bricks.** For a ball going up, "all the bricks are there" is the case in which it comes back
down **soonest**. If bricks are missing, the ball goes higher and comes back later, so the network has more time to
correct, not less. In the data the real flight is never shorter than the predicted one (section 8). We do not know
whether Astra chose it for this reason.

The price is wasted moves: while the ball is going up, the paddle heads toward a landing point that is not the real
one. Without sticky actions the extra time is always enough and the network loses no ball. With sticky actions
these wasted moves play a part in 29 of the 102 lost balls (section 9).

## 8. Verification

`check.py` runs two checks.

- **Static:** it rewrites all the weights from the formulas of sections 4-6, with the table T computed by
  simulating the rules of section 4, and compares them with the file.
- **In play:** it plays 200 episodes with the network, the same as
  `uv run evaluate breakout paper_astra --episodes 200` (seeds 10000-10199, sticky 0.1, actions sampled from the
  softmax). In every state it recomputes, **from the game state and without using the weights**, the neurons of the
  two hidden layers and the 6 logits with the formulas of sections 4-6, and compares them with the real forward
  pass.

| check | result |
|---|---|
| weights rewritten from the formulas of sections 4-6, against the file | **0** different parameters out of 20,486 (815 nonzero in both) |
| states visited | 1,456,882 |
| h0-h3: states where the value is wrong (by more than 10⁻⁹) | **0** (maximum difference 3·10⁻¹⁷) |
| h4, h5 (direction): states where the sign is wrong | **0** |
| h6-h31, g8-g31 different from 0 | **0** |
| g0-g7: states where on/off is wrong | **0** |
| logits: maximum difference | 0.003 (tanh never reaches exactly ±1) |
| different highest-logit move | **0** (near-ties below 0.5: none) |

The network stays still in 50.7% of the states and goes left or right in 24.7% each. The score over the 200
episodes is 512.66 ± 16.87, the same as `evaluate`.

**The network against the game.** The script also simulates the real flight, with the real bricks, and checks the
prediction. "Correct" means that the columns where the network would stay still are catch columns.

| phase (direction read) | states | correct prediction |
|---|---|---|
| the ball goes up | 722,118 | 42.6% |
| the ball goes down, rows 0-2 | 89,868 | 88.4% |
| the ball goes down, rows 3-8 | 644,896 | **100%** |

- **Ball going up:** the prediction is correct only when the bricks are really where the network imagines them. But
  the real flight is never shorter than the predicted one: 0 states.
- **Ball going down:** from row 3 downward the prediction is always exact. In rows 0-2 it is wrong when there is
  still a brick below the ball that sends it back up.
- **Direction read different from the real one:** 8,115 states (0.56%): 2,232 after an edge bounce and 5,883 after
  a brick hit from above. None of these states falls in a flight that ended with the ball lost.

Without sticky actions (`check.py --sticky 0 --episodes 10`, 100,000 states) the reading still matches in all
states. Every episode reaches the limit of 10,000 steps with 834 points and the network never loses the ball.
`uv run evaluate breakout paper_astra --sticky 0 --episodes 200` gives the same: 834 in all 200 episodes.

## 9. Limits visible in the weights

Without sticky actions the network never loses the ball, and the score is limited only by the 10,000-step cap.

- **The 10,000-step limit.** Without sticky actions every episode reaches the limit with 834 points: in 10,000 steps
  there are 835 flights of 12 steps, and each one breaks about one brick. With sticky 0.1, 98 episodes out of 200
  reach the limit, with 702.5 points on average. The other 102 end earlier with the ball lost, with 330.3 points on
  average.
- **The lost balls.** There are 102 in 113,332 flights, and in each one the environment replaced at least 2 actions
  during the flight (2 replacements in 42 cases, 3 in 39, 4 in 18, 5 in 3). To understand which flights are at risk
  we need the **margin**: the actions the paddle has before the ball reaches the bottom, minus the columns it has to
  cover.

  | margin at takeoff | flights | lost balls | % of flights |
  |---|---|---|---|
  | 1 | 5,319 (4.7%) | 46 | **0.86%** |
  | 2-4 | 8,237 (7.3%) | 8 | 0.10% |
  | 5-9 | 62,127 (54.8%) | 27 | 0.04% |
  | ≥ 10 | 37,649 (33.2%) | 21 | 0.06% |

  - **Margin 1 means from one corner to the other.** All 5,319 flights with margin 1 have 9 columns to cover: the
    ball returned in a corner goes up to row 4, bounces off the bricks of row 3 and comes down in the opposite
    corner. Ball and paddle move at the same speed, so the paddle has 10 actions for 9 columns. These flights are
    4.7% of the total but account for 45% of the lost balls.
  - **In these flights a replacement costs only at the start.** The paddle was still in the corner (or had just
    arrived there), and the repeated action does not move it: every replacement before the first move costs one
    column. When the paddle is already running toward the landing point, repeating the previous move changes
    nothing. Example, the episode with seed 10004 (37 points, `uv run play breakout --weights paper_astra --seed 10004`):

    ```
    step   ball    direction   T  paddle  chosen  executed
     429   (8, 0)  down-left   0    0     no-op   no-op     the ball reaches the corner, the paddle is there
     430   (8, 0)  up-right    9    0     right   no-op     sticky
     431   (7, 1)  up-right    9    0     right   no-op     sticky
     432   (6, 2)  up-right    9    0     right   right
     ...                                                    the paddle chases, one column per step
     439   (8, 9)  down-right  9    7     right   right     the paddle reaches 8, not 9: ball lost
    ```

  - **Three kinds of lost ball:**

    | flight | lost balls |
    |---|---|
    | from one corner to the other (margin 1), the network's choices always right | 46 |
    | larger margin, the network's choices always right: sticky only | 27 |
    | the network made at least one move that does not bring the paddle closer to the real landing point, then the sticky actions | 29 |

    The network's wrong moves all fall in the two phases in which the prediction can be wrong: 201,115 while the
    ball is going up, 3,835 while it is going down in rows 0-2, none from row 3 down. It makes them in 43.5% of the
    flights, but on their own they are never enough to lose the ball (without sticky actions, section 8). In the 29
    lost flights the prediction was wrong for a few steps, because the ball went through a gap or hit a brick from
    above. The paddle chased the wrong landing point, then the sticky actions took away the rest of the margin.
- **The network does not choose where to send the ball.** Which of the two catch columns the paddle uses, and
  therefore whether the ball keeps or reverses its horizontal direction, is a side effect: the logits depend only on
  Δ. There is nothing in the weights that avoids corner-to-corner flights, and without bricks in the observation the
  network could not even aim.
- **It does not see the bricks.** While the ball is going up, the prediction is correct in only 42.6% of the
  states. As seen in section 7 it is always wrong in the direction of extra time: without sticky actions it costs no
  ball, with sticky actions it plays a part in 29 lost balls out of 102.
- **Almost all of the network is empty.** 50 hidden neurons out of 64 have no weights. The whole algorithm fits in
  14 neurons and 815 numbers, 400 of which are the trajectory table.
