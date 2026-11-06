# Repository structure

This file explains how the repository is organized and where the code and results of each part of the article
live. The READMEs inside the experiment folders are written in Italian; the numbers and commands in them are
the reference for the article.

```
small-games-llm/
├── src/small_games/          the Python package: games, reference forward pass, evaluate, play, PPO
│   ├── games/low_tide/       Low Tide, the new game (env.py) and its hand-written baselines (baselines.py)
│   ├── games/minatar_games.py   Breakout, Seaquest, Asterix: the official MinAtar 1.0.15 games with the
│   │                            article's 6-channel observation
│   ├── weights.py            the weight-file format and the reference MLP forward pass used for every score
│   ├── evaluate.py, play.py  the `uv run evaluate` and `uv run play` commands
│   ├── ppo.py                the PPO trainer (CleanRL style, PyTorch CPU)
│   └── analysis.py, features.py   tools for the weight analyses (pruning, ablation, probes, ...)
├── weights/<game>/           every network, one JSON file each (format in src/small_games/weights.py)
├── experiments/<game>/<name>/   experiments on one game
├── experiments/<name>/       experiments that compare several games
└── docs/                     the description of the Low Tide observation
```

## Weight files

| file | what it is |
|---|---|
| `weights/low_tide/claude.json` | written by Claude Opus 5.5 from the prompt in `experiments/low_tide/weights_prompt/` |
| `weights/low_tide/gpt.json` | written by GPT-6 Astra from the same prompt |
| `weights/<minatar game>/paper_astra.json` | the networks written by GPT-6 Astra in the original article, converted to JSON |
| `weights/low_tide/bfs_net.json` | the BFS heuristic of Low Tide written by hand as weights, within the game's limits |
| `weights/<game>/ppo*.json` | networks trained with PPO: `ppo` (reference run), `ppo_100m` (Low Tide, 100M steps), `ppo_ent*` (entropy sweep), `ppo_v1` (first MinAtar attempt), `ppo_arch_*` (PPO from scratch with the shape of a hand-written network), `ppo_from_*` (PPO started from a hand-written network; `_cw` = with critic warm-up) |

## From the article to the code

| article section | folder | main command |
|---|---|---|
| The games and the protocol | `src/small_games/`, `docs/low_tide_observation.md` | `uv run evaluate <game> <weights>` |
| Writing weights for a new game | `experiments/low_tide/weights_prompt/`, `weights/low_tide/` | `uv run evaluate low_tide claude gpt --baseline heuristic` |
| What the written weights look like | `experiments/weight_distance/` | `uv run --group analysis python experiments/weight_distance/analyze.py` |
| Reading the algorithm back from the weights | `experiments/low_tide/decode_claude/`, `experiments/low_tide/decode_gpt/`, `experiments/<minatar game>/decode_astra/` | `uv run python experiments/<game>/decode_*/check.py` |
| PPO from scratch, and whether its weights read as algorithms | `experiments/<game>/ppo/`, `experiments/ppo_over_training/`, `experiments/ppo_runs.py` | `uv run --group ppo python experiments/ppo_runs.py status` |
| Hand-written networks as a starting point for PPO | `experiments/ppo_warm_start/` | `uv run --group ppo --group analysis python experiments/ppo_warm_start/analyze.py` |
| From an algorithm to weights: the BFS in an MLP | `experiments/low_tide/bfs_net/` (within the limits), `experiments/low_tide/bfs_exact/` (exact, without limits) | `uv run python experiments/low_tide/bfs_net/check.py` |
| Asking for the algorithm as code | `experiments/code_policies/` | `uv run python experiments/code_policies/evaluate_code.py` |

## Inside an experiment folder

Every experiment folder follows the same pattern:
- `README.md`: the question, the method, the results and the command that reproduces them;
- one or more scripts (`analyze.py`, `check.py`, `build.py`, `train.py`);
- `results/`: tables, figures and logs written by the scripts.

PPO runs keep their training log, checkpoints every 5M steps and final weights in
`experiments/<game>/ppo/results/<run>/`. The final weights are also copied to `weights/<game>/<run>.json`.

## Not in the repository

- `experiments/code_policies/prompts/`: the prompts contain the MinAtar source code (GPL-3.0). Regenerate them with
  `uv run python experiments/code_policies/make_prompts.py`.
- `experiments/low_tide/bfs_exact/bfs_exact.json` (3.8 MB): regenerate it with
  `uv run python experiments/low_tide/bfs_exact/build.py`.
- PPO training states (`state.pt`), used only to resume a run.
