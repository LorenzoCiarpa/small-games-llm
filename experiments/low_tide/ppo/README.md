# PPO on Low Tide: what weights does it find, and can they be read?

`train.py` trains with PPO a 403-64-64-5 tanh network: two hidden layers of 64 units, the maximum that Low Tide
allows. It saves it to `weights/low_tide/ppo.json`, the same format as the hand-written networks. `analyze.py` takes
the same measurements on the PPO network and on the two hand-written networks, `claude.json` and `gpt.json`. For the
latter we know the algorithm neuron by neuron ([`decode_claude`](../decode_claude/README.md),
[`decode_gpt`](../decode_gpt/README.md)), so they serve as a reference point: they show what weights that *are* an
algorithm look like under these measurements.

**In short.**

- **How it plays:** PPO learns to play as well as the hand-written networks. Always playing the most probable move,
  it scores 139 points, more than the 132 of Claude's network.
- **What the weights are like:** they cannot be read as an algorithm.
  - They are dense, with 25,000 distinct values in the first layer.
  - No neuron matters on its own.
  - No neuron computes the quantities that Claude's algorithm uses (the margin from the water, "there is a shell to
    the right").
  - The behavior does not compress into a short rule: a decision tree with 442 leaves imitates it in only 77% of the
    states.
- **The only readable trace:** the most important first-layer neurons weigh mostly the **agent's row** combined
  with the **tide direction**. It is a blurred version of the "danger" that Claude's network computes exactly.
- **With 100M steps instead of 20M** (`ppo_100m.json`, section 7): 162.7 points, as much as the BFS heuristic
  (164.0), and 169 greedy. The weights stay dense and unreadable, and tolerate pruning even less.

```bash
uv run --group ppo python experiments/low_tide/ppo/train.py          # ~16 min on one core
uv run --group analysis python experiments/low_tide/ppo/analyze.py   # ~30 min with 2 processes
```

## 1. Training

The algorithm is in `src/small_games/ppo.py`: standard PPO with clipping and GAE, like CleanRL's `ppo.py`.

| | |
|---|---|
| network (actor) | 403-64-64-5, tanh; exported as is |
| critic | 403-64-64-1, tanh; not exported |
| environments | 64 in parallel, rollouts of 128 steps (8192 steps per update) |
| optimization | 4 epochs, 8 minibatches, Adam lr 5·10⁻⁴ decaying to 0, clip 0.2 |
| returns | γ 0.99, λ 0.95; an episode cut off at 1000 steps is closed with the estimated value, not as the end of the game |
| exploration | entropy coefficient 0.01 |
| duration | 20 million steps, 16 minutes |
| environment | the evaluation one: sticky actions 0.1, 1000 steps; seeds from 1,000,000 up, never the evaluation ones |

Mean episode return during training:

| steps | 1.1M | 4.8M | 9.7M | 14.7M | 19.6M |
|---|---|---|---|---|---|
| return | 65 | 87 | 97 | 114 | 130 |

The curve is still rising at the end: with more steps the network would probably improve further.

### Score

These are 500 protocol episodes (seeds 10000-10499, sticky actions 0.1).

- **Sampled:** the move is drawn from the softmax of the logits, as in the protocol.
- **Greedy:** always the most probable move.

| network | sampled | greedy | without sticky, greedy | shells | stings | drownings (sampled) |
|---|---|---|---|---|---|---|
| heuristic (BFS) | 164.0 | | | | | |
| `claude.json` | 132.3 ± 0.9 | 129.6 | 136.8 | 154.0 | 21.7 | 4.2% |
| **`ppo.json`** | **125.7 ± 1.9** | **139.2** | **153.2** | 147.2 | 21.5 | 25.2% |
| `gpt.json` | 79.9 ± 1.0 | 76.3 | 93.0 | 86.6 | 6.7 | 15.6% |

PPO collects almost as many shells as Claude's network, but drowns much more often. When sampling it sometimes
picks unlikely moves and drowns in 25% of the episodes; playing greedy it drowns in 16% and beats Claude's network.
Without sticky actions and playing greedy it still drowns in 4.8% of the episodes; the Claude and GPT networks,
under the same conditions, never do.

## 2. What PPO weights look like

| | `claude.json` | `gpt.json` | `ppo.json` |
|---|---|---|---|
| nonzero weights | 4.3% | 4.6% | 100% |
| distinct values in the first layer | 34 | 9 | 25,004 |
| inputs that make up 90% of a neuron's weight (median, first layer) | 9 out of 403 | 2 | 240 |
| largest weight (in absolute value) | 1023 | 1000 | 1.25 |

Every hand-written first-layer neuron looks at a few chosen inputs, with round and often huge values: powers of 2,
±1000. Every PPO neuron looks at more than half of the observation, with small weights that are all different.

**The live inputs.** 91 inputs out of 403 are zero in all the states played by the four networks: rows 0-2 of the
water, shell and jellyfish planes (the tide never rises that far, and leaves nothing there) and one cell of the
agent plane. PPO receives no gradient on their weights, which
stay those of the random initialization: small, but nonzero and all different. Restricted to the live inputs, the
picture does not change:

| first layer, only the 312 live inputs out of 403 | `claude` | `gpt` | `ppo` | `ppo_100m` |
|---|---|---|---|---|
| nonzero weights | 5.1% | 3.4% | 100.0% | 100.0% |
| distinct values | 34 | 9 | 19,582 | 19,731 |
| inputs that make up 90% of a neuron's weight (median, out of 312) | 9 | 2 | 186 | 172 |
| mean \|w\| on dead / live inputs | 0.530 / 11.021 | 0.007 / 0.083 | 0.057 / 0.128 | 0.060 / 0.231 |

The images show the first-layer weights of the 8 most important neurons (those whose ablation loses the most
points, section 4): one row per neuron, one panel per observation plane (red positive, blue negative).

**Claude's network:** you can see the binary comparisons of section 4 of `decode_claude`, for example "shell to the
right in row r": the agent in row r with weights increasing from the right, the shells of the same row with powers
of 2.

![first layer of claude.json](results/analysis/fields_claude.png)

**PPO network:**

![first layer of ppo.json](results/analysis/fields_ppo.png)

In the agent plane there is a structure: the weights depend mostly on the **row**, positive at the top and negative
at the bottom or vice versa. Together with the weight on the "the tide is rising" input, in the last panel, it is a
blurred measure of "how close am I to the water, and is the water rising?". In the water, shell and jellyfish
planes the weights look like noise: small, of varying sign, with no recognizable shape. Every neuron mixes a bit of
everything.

## 3. How many weights are needed (pruning)

Only the largest weights in absolute value are kept, the same fraction in every layer, and the network is put back
to play. These are 200 episodes.

| weights kept | 100% | 50% | 25% | 10% | 5% | 2% |
|---|---|---|---|---|---|---|
| `claude.json` | 132.1 | 132.1 | 132.1 | 34.3 | 34.3 | 1.0 |
| `gpt.json` | 79.7 | 79.7 | 79.7 | 42.8 | 1.2 | 1.2 |
| `ppo.json` | 125.5 | 103.9 | 11.5 | 5.3 | 6.6 | 2.6 |

The hand-written networks are already sparse: removing the 75% smallest weights removes only zeros. The PPO network
already loses points with half of the weights, and with a quarter it no longer plays. Its information is spread over
all the weights.

The same on the first layer only, and only on the live inputs: the weights of the dead inputs set to zero, then the
stated fraction of the largest live weights is kept; the other layers stay whole.

| live first-layer weights kept | 100% | 50% | 25% | 10% | 5% |
|---|---|---|---|---|---|
| `claude` | 131.9 | 131.9 | 131.9 | 131.9 | 131.9 |
| `gpt` | 79.7 | 79.7 | 79.7 | 79.7 | 79.7 |
| `ppo` | 125.5 | 94.9 | 14.0 | 8.9 | 3.0 |
| `ppo_100m` | 155.2 | 18.4 | 4.3 | 2.3 | 1.5 |

## 4. How much each neuron matters (ablation)

A neuron is replaced, at every step, with the value it had in another state taken at random (resample ablation),
and we measure how many points are lost. These are 100 episodes per neuron; the standard error of the reference is
±1.5 for Claude, ±2.0 for GPT and ±4.4 for PPO.

| | neurons | neurons that lose more than 10% | maximum loss | share of the 5 worst neurons in the total loss |
|---|---|---|---|---|
| `claude.json` | 82 | 7 | 129 points out of 133 | 72% |
| `gpt.json` | 157 | 35 | 78 out of 82 | 23% |
| `ppo.json` | 128 | **0** | 12 out of 124 | 20% |

In Claude's network a few neurons carry the strategy: without "the tide is rising" the network scores 3.8 points,
without "rising and margin ≤ 1" 9.6, without the margin from the water 56. The seven neurons above 10% are exactly
the variables and the conditions of the algorithm. In the PPO network no neuron matters more than the statistical
noise: the computation is distributed.

## 5. What the neurons compute

### The quantities of Claude's algorithm

Claude's algorithm relies on a few quantities: the margin from the water, the danger ("the tide is rising and the
margin is ≤ 2"), "there is a shell to the right or to the left in my row". For each quantity and each network two
things are measured, on the states visited by the network itself:

- **one neuron:** the highest correlation between the quantity and a single neuron (|r|). It is 1 if a neuron *is*
  the quantity;
- **the whole layer:** how well a linear combination of all the neurons of a layer reconstructs it (R², on episodes
  not used for the combination). It is high if the quantity is there, even if spread over many neurons.

For each quantity the best layer is reported:

| quantity | `claude.json` (r / R²) | `gpt.json` (r / R²) | `ppo.json` (r / R²) |
|---|---|---|---|
| margin from the water | **1.00 / 1.00** | 0.62 / **1.00** | 0.33 / 0.48 |
| danger (rising and margin ≤ 2) | **0.90 / 1.00** | **1.00 / 1.00** | 0.77 / 0.71 |
| shell to the right in the row | **1.00 / 1.00** | 0.64 / 0.64 | 0.29 / 0.27 |
| shell to the left in the row | **1.00 / 1.00** | 0.41 / 0.46 | 0.49 / 0.41 |

GPT's network does not compute "there is a shell to the right" because its algorithm is a different one: an
attraction that decreases with distance (`decode_gpt`). It does, however, compute the margin and the danger exactly.
The PPO network has none of these quantities cleanly, not even spread over the whole layer. The danger is the
closest (r 0.77).

### Boolean or graded neurons

| | first layer | second layer |
|---|---|---|
| `claude.json`: saturated tanh neurons (\|a\| > 0.95) | relu | 100% |
| `ppo.json`: saturated tanh neurons | 3% | 24% |

In Claude's network the second layer is made of true/false conditions. In the PPO network almost no neuron works as
a switch.

### With the game's vocabulary

The game has a vocabulary of 17 variables, in `src/small_games/features.py`: row, column, level, rising tide,
countdown, stun, margin, shells in the four directions, nearest shell, adjacent jellyfish. The PPO neuron best
explained by a single one of these has |r| = 0.80; only 1 neuron out of 128 exceeds 0.8. In each of the two layers,
for 57 neurons out of 64 the variable that explains them best is the water **level**, the **rising tide**, the row
or the column. For the other 7 it is a shell variable, always with low correlations. A linear combination of all 17
variables explains on average 35-44% of the variance of a PPO neuron. Every neuron is a mixture.

## 6. Can it be written as a rule? (distillation)

A decision tree on the 17 variables learns to imitate the network's most probable move. It is trained on 70% of the
episodes and the agreement is measured on the remaining 30%. Then the tree plays on its own in place of the
network, 200 episodes.

| depth | `claude.json` agreement / leaves / points | `gpt.json` | `ppo.json` |
|---|---|---|---|
| 4 | 90.4% / 16 / 15.6 | 82.9% / 16 / 49.2 | 59.9% / 16 / 4.7 |
| 6 | **99.6%** / 47 | 92.2% / 53 | 64.2% / 53 |
| 8 | 99.6% / 73 / 117.0 | 95.2% / 115 / 61.9 | 72.6% / 136 / 84.6 |
| 12 | 99.6% / 92 / 117.0 | 96.5% / 242 / 68.3 | 77.0% / 442 / 112.3 |
| the network itself, greedy | 127.4 | 74.0 | 138.7 |

- **Claude:** the behavior is a 47-leaf rule, on the same variables as the algorithm found by hand.
- **GPT:** the behavior is almost a rule. The rest is the distance-based attraction, which these variables describe
  only in part.
- **PPO:** it does not come close. With 442 leaves the tree still gets one move in four wrong. It still plays
  reasonably well (112 points): it imitates the important parts, not the network.

The depth 2-4 trees are in `results/analysis/trees_<network>.txt`.

## 7. With five times more steps: `ppo_100m`

The curve of section 1 was still rising at 20M steps. `ppo_100m` is the same configuration taken to 100M steps
(`experiments/ppo_runs.py`, 55 minutes), saved in `weights/low_tide/ppo_100m.json`.

| steps | 10M | 20M | 50M | 75M | 100M |
|---|---|---|---|---|---|
| training return | 99 | 117 | 124 | 141 | 160 |

On the 500 protocol episodes it scores **162.7 ± 2.6**, as much as the BFS heuristic (164.0), and 169 playing
greedy. It collects 173 shells per episode, against Claude's 154, and takes half the stings (10.6), but still drowns
in 32% of the episodes: it takes more risks, and it pays off.

The same measurements, next to the 20M-step network (the full tables are in the previous sections and in
`results/analysis/analysis.json`):

| | `ppo` (20M) | `ppo_100m` |
|---|---|---|
| score (200 episodes) / greedy | 125.5 / 138.7 | 155.5 / 169.1 |
| live inputs that make up 90% of a neuron's weight (median, out of 312) | 186 | 172 |
| all layers, 50% of the weights | 103.9 | 29.6 |
| first layer, 50% of the live weights | 94.9 | 18.4 |
| neurons that lose more than 10% / maximum loss | 0 / 11.8 | 7 / 24.8 |
| margin from the water (r / R²) | 0.33 / 0.48 | 0.37 / 0.62 |
| danger (r / R²) | 0.77 / 0.70 | 0.75 / 0.75 |
| shell to the right / to the left (r) | 0.29 / 0.45 | 0.30 / 0.46 |
| saturated tanh neurons, first / second layer | 3% / 24% | 16% / 44% |
| depth-6 tree: agreement | 64.2% | 64.5% |
| depth-12 tree: agreement / leaves / points | 77.0% / 442 / 112.3 | 74.5% / 516 / 136.1 |

More training does not make the network more readable:

- **pruning:** it holds up even less. With half of the weights the 20M network scores 104, the 100M one 30: it has
  learned to use more of them;
- **neurons:** a few more matter (7 above 10%, the worst 25 points out of 152), and they are still the ones tied to
  the rising tide (|r| 0.29-0.49 with "rising");
- **Claude's quantities:** the margin from the water is reconstructed a bit better from the whole layer (R² 0.62),
  but no neuron computes it; the others do not change;
- **rule:** the tree imitates it less than before.

The first-layer image (`results/analysis/fields_ppo_100m.png`) is like the 20M one: in the agent plane the weights
depend mostly on the row, together with the weight on "the tide is rising"; the other planes are noise.

## 8. Conclusion

- **The hand-written networks:** the weights are the program. Every neuron has a name ("margin", "shell to the
  right in row 5"), a few neurons carry the whole strategy, and the behavior is a short rule on the game
  quantities.
- **PPO:** it finds a solution that plays as well as those (better, if greedy), but that does not have this form:
  - the weights are dense and continuous;
  - the computation is distributed over all the neurons;
  - the "natural" quantities of the game do not appear as neurons, and only partly as combinations of neurons;
  - the behavior cannot be summarized in a short rule.

A partial description can be given. The most important first-layer neurons combine the agent's row with the tide
direction, that is, a blurred version of "am I about to end up underwater?". Almost all neurons mostly follow
level, tide and position. The collection strategy, on the other hand, cannot be reduced to
quantities with a name.

## Files

- `train.py`: training configuration and launch.
- `analyze.py`: all the measurements; with `--only-key` only the key quantities.
- `results/ppo/`:
  - `train_log.csv`: the training curve;
  - `checkpoint_*.json`: the weights at 10, 25 and 50% of training;
  - `eval.json`: the evaluations;
  - `final.pt`: actor and critic in PyTorch format.
- `results/ppo_100m/`: the 100M-step run, with the same files and `checkpoint_*M.json` every 5M steps.
- `results/analysis/`:
  - `analysis.json`: all the numbers;
  - `trees_*.txt`: the trees;
  - `fields_*.png`: the first-layer images.
