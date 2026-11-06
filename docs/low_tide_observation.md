# Low Tide: what the network sees

At each step the network receives a vector of **403 numbers** (float32) and returns **5 logits**, one per action.
This document explains what each number represents, which values it can take and which game rules
determine them. The source is `src/small_games/games/low_tide/env.py`. The constraints listed here were verified on
600 episodes (about 310,000 observations, played randomly and by the heuristic), with no exception.

## The game in brief

- The beach is a 10×10 grid. **Row 0** is at the top, on the land side; **row 9** is at the bottom, on the sea side.
  Columns go from 0 (left) to 9 (right).
- **The water** covers the bottom `livello` (level) rows. The level goes from 0 to 7, so rows 0–2 are always
  dry. Rows 3–9 are the tidal zone.
- **The tide** changes by one row every 3, 4 or 5 steps (drawn at random).
  - It rises up to a high level, drawn from 5, 6 and 7, then falls down to a low level, drawn from 0, 1 and 2,
    and so on.
  - When it **falls**, the row that is uncovered is randomly filled with shells (25% per cell) and jellyfish (8% per
    cell).
  - When it **rises**, the row that gets covered loses everything that was on it.
- Cell effects:
  - stepping on a **shell** gives +1 and removes it;
  - stepping on a **jellyfish** gives −1, removes it and stuns the agent for 3 steps;
  - if the rising tide covers the agent's row, the agent **drowns** and the episode ends.
- **Moving** into the water or off the grid has no effect: the agent stays where it is.
- **Actions:** 0 no-op, 1 up (row − 1), 2 down (row + 1), 3 left (column − 1), 4 right (column + 1).
- **Sticky actions:** with probability 0.1 the previous action is repeated instead of the chosen one.
- An episode lasts at most 1000 steps.

## How the vector is laid out

| indices | what | possible values |
|---|---|---|
| 0–99 | **agent** plane | 0 or 1; exactly one 1 |
| 100–199 | **water** plane | 0 or 1; whole rows starting from the bottom |
| 200–299 | **shells** plane | 0 or 1 |
| 300–399 | **jellyfish** plane | 0 or 1 |
| 400 | **the tide is rising** | 0 or 1 |
| 401 | tide **countdown** | 0.2, 0.4, 0.6, 0.8, 1.0 |
| 402 | **stun** | 0, 1/3, 2/3, 1 |

The first 400 numbers are four 10×10 grids, written one after the other. Within each grid the order is by rows:

```
index = 100 · plane + 10 · row + column          plane: 0 agent, 1 water, 2 shells, 3 jellyfish

plane  = index // 100
row    = (index % 100) // 10
column = index % 10
```

Examples for the cell in row 3 and column 6: agent → 36, water → 136, shell → 236, jellyfish → 336.

> **Warning: the order differs from that of the MinAtar games.** In Breakout, Seaquest and Asterix the index is
> `(row · 10 + column) · 6 + channel`: the channels of a cell are adjacent. In Low Tide the cells of the same
> plane are adjacent. Whoever writes the weights must use the right formula for each game.

## The four planes

### Agent (indices 0–99)

- There is always **exactly one 1**, in the agent's cell: index `10 · row + column`.
- At the start of every episode the agent is in **row 1**, in a random column.
- It can be on any dry row, i.e. from 0 to `9 − livello`. With the tide at 0 it can reach
  row 9.
- It is never in the water, except in the **last observation** of an episode that ended by drowning. It is the only
  case in which agent and water have a 1 in the same cell.
- It is never on top of a shell or a jellyfish, because it collects it, or is stung by it, in the same step in which
  it gets there.

### Water (indices 100–199)

- The water always occupies **whole rows starting from the bottom**. With level L, rows `10 − L` to 9 are all 1
  and the others all 0. There are no half-flooded rows.
- So **level = (sum of the plane) / 10**, an integer between 0 and 7. The lowest dry row is `9 − livello`.
- Rows 0–2 are never water. At the start of the episode the level is between 2 and 5.
- Frequency of the levels in the measured observations:

  | level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | share | 3% | 10% | 17% | 20% | 20% | 17% | 10% | 3% |

### Shells (indices 200–299)

- One 1 for each shell.
- Where they can be:
  - only on **dry rows of the tidal zone** (3–9), never in rows 0–2;
  - never in the water;
  - never in the same cell as a jellyfish;
  - never under the agent.
- When they **appear**:
  - at the start, in all dry rows of the tidal zone, with probability 25% per cell;
  - every time the tide falls, in the row just uncovered, with the same probability.
- When they **disappear**: when the agent collects them (+1), or when the rising tide covers their row.
- **They do not move.** On average about 3.5 are visible per observation, with a measured maximum of 27.

### Jellyfish (indices 300–399)

- The rules are the same as for shells, with probability **8% per cell**: they appear in uncovered rows, do not
  move and disappear when covered by the water.
- They also disappear when the agent steps on them. That step gives −1 and starts the stun.
- On average about 2.6 are visible per observation, with a measured maximum of 13.

## The three scalars

### 400: the tide is rising

- It is **1** if the next level change will raise the water, **0** if it will lower it. It is 1 about half of the time.
- It changes value only in the step in which the level reaches its target. For example, if the tide is rising
  toward 6, as soon as the water reaches 6 the value switches to 0.
- **The target is not in the observation.** The network does not know whether the tide will rise to 5, 6 or 7,
  nor whether it will fall to 0, 1 or 2. It finds out only when this value changes.

### 401: countdown

- It is **(steps left until the next level change) / 5**:

  | value | 1.0 | 0.8 | 0.6 | 0.4 | 0.2 |
  |---|---|---|---|---|---|
  | steps to the change | 5 | 4 | 3 | 2 | 1 |

- **0.2 means that the level changes at the end of the next step.**
- It is never 0. When it reaches zero the level changes and a new interval (3, 4 or 5 steps) is drawn
  immediately. So after a change it reads 0.6, 0.8 or 1.0, and it goes down by 0.2 at each step.
- At the start of the episode it is 0.6, 0.8 or 1.0.
- Measured frequencies: 0.2, 0.4 and 0.6 about 25% each, 0.8 17%, 1.0 8%.

### 402: stun

- It is **(remaining stun steps) / 3**, so 0, 1/3, 2/3 or 1.
- In the step in which the agent steps on a jellyfish, the observation shows **1**. The next 3 actions are
  ignored and the value goes down like this:

  | stun read | 1 | 2/3 | 1/3 | 0 |
  |---|---|---|---|---|
  | the next action | ignored | ignored | ignored | executed |

- So, if the stun is greater than 0, the agent stays still in the next step whatever action the
  network chooses. It is non-zero in about 2% of the measured observations.

## The order of events in a step

`step(action)` runs, in order:

1. **Sticky action:** with probability 0.1 the chosen action is replaced by the previous one.
2. **Movement.** If the agent is stunned, it does not move and the stun decreases by 1. Otherwise it moves, unless
   the target cell is off the grid or in the water.
3. **Shell or jellyfish** in the target cell: +1, or −1 and stun set to 3.
4. **Tide:** the countdown decreases by 1. If it reaches 0, the level changes by one row: the covered row
   is emptied, the uncovered one is populated, and a new interval is drawn.
5. **Drowning:** if the water now covers the agent's row, the episode ends.
6. **Truncation:** after 1000 steps the episode stops.

The observation returned by `step` shows the state *after* all these events, including any tide
change.

The most dangerous situation follows from these rules: **rising tide (400 = 1) with the countdown at 0.2
(401).** At the end of the next step the water will cover row `9 − livello`, i.e. the lowest dry row. If
the agent is there at the end of the step, it drowns. To survive it must move up in that step, but it cannot if it is
stunned (402 > 0) or if the sticky action repeats a different action (10% of the steps).

## What the network does not see

The observation contains almost all of the game state, but not:

- **the tide's target**: how far it will rise or fall in the current cycle;
- **the previous action**, which the sticky action could repeat;
- **the current step** and how many are left until 1000;
- **the score**;
- **the future**: where shells and jellyfish will appear.

## Example: the start of the episode with seed 10000

It is episode 0 of `evaluate` and the one you see when launching `uv run play low_tide`. Legend: A agent,
o shell, j jellyfish, ~ water, . sand.

```
row 0    . . . . . . . . . .
row 1    . . . . . . A . . .
row 2    . . . . . . . . . .
row 3    . o o . . . . o j .
row 4    j o j . . o . o o .
row 5    j o . . . . o . . o
row 6    . . . j . . . . . .
row 7    o o j o j . . . o .
row 8    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
row 9    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
```

The non-zero inputs are:

| inputs | how many | detail |
|---|---|---|
| agent | 1 | index 16 = row 1, column 6 |
| water | 20 | indices 180–199 = rows 8 and 9, so level 2 |
| shells | 14 | 231, 232, 237 (row 3), 241, 245, 247, 248 (row 4), ... |
| jellyfish | 7 | 338 (row 3, col. 8), 340, 342 (row 4), 350 (row 5), 363, 372, 374 |
| scalars | | 400 = **1** (the tide is rising), 401 = **0.8** (changes in 4 steps), 402 = **0** (not stunned) |

The hidden state in this example is that the tide is rising toward level 6. The network will find out only when
value 400 goes back to 0.

## Reading the observation in Python

```python
import numpy as np

grid = obs[:400].reshape(4, 10, 10)          # [plane, row, column]
agent, water, shells, jellyfish = grid
row, column = np.argwhere(agent)[0]
level = int(water.sum()) // 10                 # 0..7; lowest dry row = 9 - level
rising = obs[400] == 1.0
steps_to_change = round(obs[401] * 5)          # 1..5
stun_steps = round(obs[402] * 3)               # 0..3
```

## How a network reads it: two examples

A network does not receive variables like `level` or `row`: it must derive them with weighted sums of these
403 numbers. Each unit computes `activation(Σ weight · input + bias)`.

- In `weights/low_tide/gpt.json` unit 0 of the first layer has weight +1 on input 236 (shell in row 3,
  column 6), −1 on 336 (jellyfish in the same cell) and bias 1. So it computes `relu(1 + shell − jellyfish)`:
  it is 2 if the cell has a shell, 1 if it is empty, 0 if it has a jellyfish. In
  practice it is a judgment of that cell.
- A unit with weight 1 on inputs 0–9, i.e. row 0 of the agent plane, is 1 if and only if the agent is in row 0.
  With 10 such units, one per row, the network knows which row the agent is in. For the water level one cell per
  row of the water plane is enough, because rows flood whole: for example input 170 (water in row 7,
  column 0) is 1 if and only if the level is at least 3.
