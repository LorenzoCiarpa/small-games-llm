# small-games-llm

Test bed for the idea of the post [Weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/):
an LLM writes by hand, without training, without tools and without running code, the weights of an MLP that plays a game.
Here are the new game **Low Tide** (never seen before, never used to train any model) and the three games
of the article (MinAtar **Breakout**, **Seaquest**, **Asterix**), all behind the same interface, so
`evaluate` and `play` work on any registered game.

```bash
uv sync                    # creates .venv from uv.lock (Python 3.12)
uv run evaluate --help     # at the end: for each game, the available baselines and weights
uv run evaluate --list     # (also `play --list`) protocols, allowed networks, what each baseline does
```

**Baselines**: reference policies that are not networks and serve as a scale for the scores.
`random` (uniform random action) and `astra_code` (the function `policy(obs, rng)` written as code by GPT-6 Astra,
without tools, in [`experiments/code_policies/`](experiments/code_policies)) exist for all games; Low Tide also
has `heuristic` (BFS to the nearest shell, avoiding jellyfish and rows that flood). **Greedy**, instead, applies only to networks: argmax of the logits instead of sampling
from the softmax (the article's protocol uses sampling).

## Evaluating

```bash
uv run evaluate low_tide claude gpt                     # file names in weights/low_tide/
uv run evaluate low_tide weights/low_tide/claude.json   # or any path
uv run evaluate low_tide --baseline heuristic --baseline astra_code
uv run evaluate breakout paper_astra --baseline random
```

Each game has its own protocol (episodes, seeds, sticky actions, step limit), which can be changed with
`--episodes`, `--seed0`, `--sticky` and `--max-steps`. `--greedy` uses the argmax instead of sampling from the softmax.
Episode k uses seed `seed0 + k` (see *Reproducibility*); `--workers` (default: all cores) does not change the
results. The score always uses the reference forward pass (`small_games/weights.py`), never code written
by the model.

## Playing or watching

```bash
uv run play low_tide                          # you play
uv run play low_tide --weights claude         # you watch a network
uv run play low_tide --baseline heuristic     # you watch a baseline
uv run play seaquest --weights paper_astra --speed 20
```

Keys: the game's own (arrows, space to stay still, `F` to fire in the MinAtar games);
`1 2 3 ...` change who plays (you, network, baseline) without restarting; `P` pause, `N` one step, `+ -` speed,
`T` turn-based or real time, `R` restart with the same seed, `E` go to seed + 1, `S` sticky on/off,
`G` greedy/sampled (network only), `C` coordinates.

## Reproducibility

An episode depends only on its seed: the game, the sticky actions and the generator the policy samples with
(`policy_rng`, the same in `evaluate` and in `play`) all restart from that seed. Therefore:

- the same `evaluate` command always gives the same numbers, with any `--workers`;
- in `play`, `R` (or relaunching with the same `--seed`) repeats the identical episode;
- `play` starts from the game's seed0 (10000 for all), i.e. from episode 0 of `evaluate`, and `E` moves to
  episodes 1, 2, ...; in general `play <game> --seed s` shows exactly episode `s - seed0`, for example
  `uv run play breakout --weights paper_astra --seed 10002` is episode 2.

This holds as long as the same policy plays the whole episode with the same settings: changing who plays, `G` or `S`
mid-game changes what follows (and from that moment it no longer matches `evaluate`).

## The games

| game | observation | actions | allowed network | protocol |
|---|---|---|---|---|
| `low_tide` | 403 = 4 planes 10×10 + 3 scalars | 5 | ≤ 3 hidden layers of ≤ 64 units, relu/tanh/linear | 500 episodes, seed 10000.., sticky 0.1, 1000 steps |
| `breakout`, `seaquest`, `asterix` | 600 = 10×10 grid × 6 channels | 6 | ≤ 2 hidden layers of ≤ 32 units, tanh (the article: 600-32-32-6) | 1000 episodes, seed 10000.., sticky 0.1, 10000 steps |

What the 403 inputs of Low Tide represent, and which values they can take, is explained in
[`docs/low_tide_observation.md`](docs/low_tide_observation.md).

The MinAtar games are the official ones from the package `minatar==1.0.15` (the version used by the article), not a
reimplementation. The observation is the article's: index `(row * 10 + column) * 6 + channel`,
channels player, danger, reward, projectile, trail, gauge, each the OR of some of the game's channels, with
the same mapping written in the article's prompts (see `games/minatar_games.py`). Actions: 0 no-op, 1 left, 2 up,
3 right, 4 down, 5 fire.

**Integration check.** The networks written by GPT-6 Astra in the article (`weights/<game>/paper_astra.json`)
give the same scores here as in the article, within about one standard error:

| | here (1000 episodes) | article |
|---|---|---|
| Breakout | 512.5 ± 7.6 | 517.4 ± 7.4 |
| Seaquest | 51.4 ± 1.4 | 53.1 ± 1.4 |
| Asterix | 54.8 ± 1.1 | 53.2 ± 1.0 |

**Low Tide** (500 episodes): heuristic 164.0 ± 1.5, `claude` 132.3 ± 0.9, `gpt` 79.9 ± 1.0, random 2.0, `astra_code` 188.9 ± 0.7;
`bfs_net`, the heuristic hand-written as weights in this repo, 166.7 ± 2.0.

## Experiments

Experiments on a single game are in `experiments/<game>/`; those that compare several games are directly in
`experiments/`.

- [`experiments/weight_distance/`](experiments/weight_distance): how far apart are the networks written by the
  LLMs? Layer-by-layer distances (raw, after aligning the neurons, and against a null model), twin units,
  CKA, actions on the same states and cross-play between games, with four networks trained with PPO as a point of
  comparison. `uv run --group analysis python experiments/weight_distance/analyze.py`
- [`experiments/ppo_over_training/`](experiments/ppo_over_training): do PPO networks become sparser or more
  readable as training goes on, or with a different entropy? No: the main measures on all runs
  and all checkpoints, next to the hand-written networks.
  `uv run --group analysis python experiments/ppo_over_training/analyze.py`
- [`experiments/ppo_warm_start/`](experiments/ppo_warm_start): PPO that starts from the hand-written networks instead
  of random weights. It measures how much it improves, after how many steps and how much time is saved. It also
  measures the gradient at that point and how much the weights move. `uv run --group ppo --group analysis python
  experiments/ppo_warm_start/analyze.py`
- [`experiments/code_policies/`](experiments/code_policies): the same LLM writes the algorithm as Python code
  instead of as weights. The prompt is the article's, with only the part about the network replaced by a function
  `policy(obs, rng)`, and the model really runs without tools (`run_codex.py`).
  - GPT-6 Astra's code beats its own network and the best PPO in all four games: Low Tide 188.9,
    Breakout 786.0, Seaquest 187.3, Asterix 83.2.
  - Astra's networks score 79.9, 512.5, 51.4 and 54.8.

  `uv run python experiments/code_policies/evaluate_code.py`
- [`experiments/ppo_runs.py`](experiments/ppo_runs.py): launches, monitors and stops all PPO training runs in the
  background (10 runs from scratch, ~2 hours on 8 cores, plus the 9 of experiments/ppo_warm_start/).

**Low Tide**

- [`experiments/low_tide/decode_claude/`](experiments/low_tide/decode_claude): how to read the network `claude.json`,
  neuron by neuron, and which game algorithm it represents. The reading is verified against the real network on
  ~200,000 states. `uv run python experiments/low_tide/decode_claude/check.py`
- [`experiments/low_tide/decode_gpt/`](experiments/low_tide/decode_gpt): the same analysis for `gpt.json`, with a
  comparison between the two networks. `uv run python experiments/low_tide/decode_gpt/check.py`
- [`experiments/low_tide/bfs_net/`](experiments/low_tide/bfs_net): the reverse of the previous two. It starts
  from the BFS heuristic and writes it into the weights of an MLP with the same limits (403 → 64 → 58 → 63 → 5),
  without training.
  - The exact BFS does not fit. A potential-based version fits (the nearest shell weighs more than all the others,
    never stepping into a jellyfish), which scores 166.7 points against the heuristic's 164.0.
  - Network and specification choose the same move in all 456,230 states checked.

  `uv run python experiments/low_tide/bfs_net/check.py`
- [`experiments/low_tide/bfs_exact/`](experiments/low_tide/bfs_exact): the same BFS, exact, removing the game's
  limits, to see how many neurons are needed.
  - A wave starts from the shells and advances one cell per layer.
  - It needs 104 layers and 51,103 neurons, against the 192 allowed.
  - It makes the same move as the heuristic in every state checked, so it gets the same score episode by
    episode.

  `uv run python experiments/low_tide/bfs_exact/check.py`
- [`experiments/low_tide/ppo/`](experiments/low_tide/ppo): a network trained with PPO (403-64-64-5) plays as well
  as Claude's, and with 100M steps (`ppo_100m.json`) as well as the heuristic, but its weights do not read as
  an algorithm. The same measures on the networks: weights, pruning, ablation, quantities of Claude's algorithm in
  the neurons, distillation into decision trees.
  `uv run --group ppo python experiments/low_tide/ppo/train.py`, then `uv run --group analysis python
  experiments/low_tide/ppo/analyze.py`

**Breakout, Seaquest, Asterix**: the article's GPT-6 Astra networks (`paper_astra.json`), read neuron by
neuron like the Low Tide ones, each with a `check.py` that verifies the reading against the real network.

- [`experiments/breakout/decode_astra/`](experiments/breakout/decode_astra): predicts the column where the
  ball will arrive, from a "cell → arrival column" table written in the weights, and moves the paddle there.
- [`experiments/seaquest/decode_astra/`](experiments/seaquest/decode_astra): the submarine stays in column 5 and
  chooses only the row, based on divers to rescue, targets in the line of fire and dangers; it surfaces when oxygen is low.
- [`experiments/asterix/decode_astra/`](experiments/asterix/decode_astra): a "goalkeeper" in column 5 that
  chooses only the row, avoiding incoming enemies and moving toward the treasures.

And the PPO networks with the same architecture (600-32-32-6), compared with Astra's using the Low Tide measures.
`uv run --group ppo python experiments/<game>/ppo/train.py`, then `uv run --group analysis python
experiments/<game>/ppo/analyze.py`.

- [`experiments/breakout/ppo/`](experiments/breakout/ppo): PPO leaves the plateau at ~6 points only with more entropy
  and reaches 63, against Astra's 512.
- [`experiments/seaquest/ppo/`](experiments/seaquest/ppo): PPO plays better than Astra (55.9 against 51.4), but in a
  different way: it moves horizontally instead of staying in column 5.
- [`experiments/asterix/ppo/`](experiments/asterix/ppo): PPO reaches 24.6, half of Astra, still rising.

### Does PPO find networks as sparse and readable as the hand-written ones?

No, in none of the four games. Even where it plays as well as or better than the hand-written networks (Low Tide,
Seaquest), PPO finds dense weights, almost all of which are needed, and neurons without a name. The measures on the
first layer are restricted to the live inputs, those that are not always zero (on the others PPO leaves the initial weights):

| game | network | score | live weights ≠ 0 | distinct values | inputs for 90% of a neuron's weight | score with 25% of live weights | neurons that alone remove > 50% | key quantities: best \|r\| | depth-6 tree: agreement |
|---|---|---|---|---|---|---|---|---|---|
| Low Tide | `claude` (hand-written) | 132.1 | 5% | 34 | 9 of 312 | 131.9 (100%) | 3 | 0.97 | 100% |
| | `gpt` (hand-written) | 79.7 | 3% | 9 | 2 of 312 | 79.7 (100%) | 12 | 0.67 | 92% |
| | `ppo` (20M steps) | 125.5 | 100% | 19,582 | 186 of 312 | 14.0 (11%) | 0 | 0.47 | 64% |
| | `ppo_100m` | 155.5 | 100% | 19,731 | 172 of 312 | 4.3 (3%) | 0 | 0.47 | 64% |
| Breakout | `paper_astra` (hand-written) | 512.7 | 10% | 53 | 77 of 220 | 512.7 (100%) | 14 | 0.73 | 89% |
| | `ppo` | 62.4 | 100% | 7,027 | 136 of 220 | 0.4 (1%) | 10 | 0.34 | 72% |
| Seaquest | `paper_astra` (hand-written) | 48.3 | 3% | 12 | 9 of 425 | 48.3 (100%) | 29 | 0.46 | 87% |
| | `ppo` | 56.0 | 100% | 13,526 | 240 of 425 | 1.7 (3%) | 14 | 0.22 | 65% |
| Asterix | `paper_astra` (hand-written) | 54.5 | 5% | 33 | 10 of 315 | 54.5 (100%) | 14 | 0.57 | 92% |
| | `ppo` | 26.8 | 100% | 10,029 | 167 of 315 | 7.3 (27%) | 2 | 0.11 | 49% |

Scores over 200 episodes; "distinct values" and "inputs" refer to the first layer; pruning keeps the largest 25% of
the live weights of the first layer; the key quantities are those of the hand-written algorithm (mean over the
game's quantities); the tree imitates the network's most likely move using the game's variables.

- **The hand-written networks** are sparse (3-10% of the weights), with few round values, and pruning removes only
  zeros. The neurons are the quantities and conditions of the algorithm, and the behavior can be written as a short tree.
- **The PPO networks** use every live input with a different weight, do not survive pruning, and no neuron computes
  the quantities of the hand-written algorithm, not even as a linear combination. On Low Tide no neuron matters on
  its own; in the MinAtar games, with 32 units per layer, many matter, but none has a recognizable meaning.
- **The readable traces** in PPO are position encodings: the agent's row together with the tide on Low Tide, the
  paddle's column on Breakout, position blobs of the submarine and of the player on Seaquest and Asterix.
- **With more training or a different entropy** the picture does not change: the network improves by using more
  weights, not fewer ([`experiments/ppo_over_training/`](experiments/ppo_over_training)).
- **The behavior** is different too: in Seaquest and Asterix Astra reduces the game to one dimension (it stays in
  column 5 and chooses the row), PPO does not.

### Are the hand-written networks a good starting point for PPO?

Astra's networks on the MinAtar games are; GPT's and Claude's on Low Tide are not. Each run uses the configuration of
the game's best from-scratch run and changes only its initial actor; for Low Tide the comparison is with a
from-scratch run with the same architecture as the hand-written network
([`experiments/ppo_warm_start/`](experiments/ppo_warm_start)).

| game | hand-written network | PPO from the hand-written network | PPO from scratch | gradient at fixed weights: hand-written network / PPO from scratch at end of run |
|---|---|---|---|---|
| Breakout | 512.5 | 512.5 (identical) | 63.2 | 4·10⁻¹³ / 0.085 |
| Seaquest | 51.4 | **82.5** (+60%) | 55.9 | 0.37 / 0.12 |
| Asterix | 54.8 | 50.8 (−7%) | 24.6 | 1.14 / 0.11 |
| Low Tide | `gpt` 79.9 | 147.3 (+84%) | **166.9** | 1.32 / 0.41 |
| Low Tide | `claude` 132.3 | 31.0 (−77%) | **158.4** | 28.3 / 0.35 |

- **Breakout:** Astra's network is a plateau. The softmax is saturated, the gradient is practically zero and in 200M
  steps the weights move by 7·10⁻⁶. PPO does not change it, but it is worth 8 times the best from-scratch run.
- **Seaquest:** PPO makes the network worse for 5M steps, returns to its level at 20M and ends at +60%. To reach the
  final level of the from-scratch run it saves 150M steps.
- **Asterix:** PPO does not improve it, but it stays at twice the from-scratch run.
- **Low Tide:** in the first million steps PPO erases the hand-written networks (GPT drops to 1, Claude to 6), then
  learns more slowly than from scratch.
  - The weights of the hand-written networks are huge: ‖θ‖ from 500 to 45,000, against 11-17 for a random
    initialization. With the same learning rate each update changes the policy much more: the KL is 20-45 times that
    of a from-scratch run.
  - Training only the critic first is not enough.

## Structure

The full map, with the link between the sections of the article and the folders, is in
[`STRUCTURE.md`](STRUCTURE.md).

```
src/small_games/
  core.py            Game, Protocol, NetSpec, GridSpec and the game registry
  weights.py         weight format (JSON), per-game validation, reference MLP
  policy.py          sampling, MLP policy, random baseline, lookup of files in weights/
  evaluate.py        `evaluate` CLI
  play.py            `play` CLI (pygame)
  render.py          shared drawing: panel, generic renderer for channel grids
  ppo.py             PPO for any registered game, with checkpoints and exact resume (group `ppo`)
  analysis.py        measures on the weights: sparsity, pruning, ablation, probes, trees (group `analysis`)
  features.py        the variables of each game (the vocabulary for probes and trees)
  games/
    low_tide/        env.py (rules), baselines.py (heuristic: do NOT give it to the model), render.py
    minatar_games.py Breakout, Seaquest, Asterix
weights/<game>/<name>.json
experiments/<game>/<name>/   experiments on the weights of one game (scripts, data, results)
experiments/<name>/           experiments that compare several games
```

## Adding a game

1. Write a module (or a package) in `src/small_games/games/`: it is imported automatically.
2. The environment must have `reset(seed) -> (obs, info)`, `step(a) -> (obs, r, terminated, truncated, info)` and the
   attributes `sticky_prob`, `last_action`, `max_steps`, `t` (details in `core.py`).
3. Call `register(Game(...))` with name, `make_env`, `obs_dim`, action names, `Protocol`, `NetSpec`, and
   optionally baselines, keys, `episode_stats` (keys of `info` averaged by `evaluate`).
4. For `play` a `GridSpec` is enough (generic renderer, one shape per channel); for a custom look, set
   `renderer` to a subclass of `small_games.render.Renderer`, like `games/low_tide/render.py`.
5. Put the weights in `weights/<name>/`.

## Weight format

The same for all games; every entry not listed in a sparse block is 0, and `y = act(W @ x + b)`:

```json
{"layers": [
  {"in": 600, "out": 32, "activation": "tanh",
   "W": {"sparse": [[out, in, valore], ...]},  "b": {"dense": [...]}},
  ...
  {"in": 32, "out": 6, "activation": "linear", "W": {"dense": [[...], ...]}, "b": {"sparse": [[out, valore], ...]}}
]}
```

## Credits and licenses

- MinAtar (Kenny Young and Tian Tian, GPL-3.0) is a dependency installed from PyPI; it is not included in the repository.
- `weights/{breakout,seaquest,asterix}/paper_astra.json` are the article's `hand.npz` networks
  ([RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), `exp5_write_from_scratch/networks/`,
  CC BY 4.0), converted to this format (W = transposed kernel).
