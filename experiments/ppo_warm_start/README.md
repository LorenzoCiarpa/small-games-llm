# PPO starting from the hand-written networks: are they good initializations?

The networks written by the LLMs are readable algorithms (see `experiments/<game>/decode_*`). Here they are the
starting point of the PPO actor, and the questions are:

- does training improve them, and by how much?
- after how many steps?
- how much time does it save compared to starting from scratch?
- what gradient does PPO see at that point? Is it close to zero, that is, a plateau where training does not move them?
- how much do the weights move?

**The runs.** Each run uses the configuration of the game's best run from scratch and changes only the starting
point of the actor. The critic starts random.

- **Breakout, Seaquest, Asterix** (`ppo_from_astra`): starts from `paper_astra.json`, has the 600-32-32-6
  tanh architecture and runs for 200M steps.
  - The ten runs from scratch in `experiments/ppo_runs.py` varied the entropy coefficient, not the learning rate,
    which is 5e-4 in all of them.
  - The best ones, in all three games, are those with entropy 0.05 (`ppo_ent05`), which serve as the reference.
- **Low Tide** (`ppo_from_gpt`, `ppo_from_claude`): starts from `gpt.json` (43-61-53 relu) or from `claude.json` (46-36
  relu/tanh).
  - It uses the configuration of `ppo_100m`: entropy 0.01, lr 5e-4, 100M steps.
  - The hand-written networks have a different architecture, so the reference is a run from scratch with the same
    architecture (`ppo_arch_gpt`, `ppo_arch_claude`).
  - There is also a `_cw` variant, in which the first 2M steps train only the critic: the actor stays the
    hand-written network until the critic has learned its value.

The common hyperparameters are: 64 environments × 128 steps, 4 epochs × 8 minibatches, Adam (eps 1e-5), gradient
clipping at 0.5 and lr decaying linearly to 0.

**In short: Astra's networks on the MinAtar games yes, GPT's and Claude's networks on Low Tide no.**

- **Breakout:** Astra's network is a true plateau. The gradient is 4·10⁻¹³, against 0.06-0.16 for PPO from scratch.
  In 200M steps the weights move by 7·10⁻⁶, and the final network plays exactly like the starting one (512.5).
  - It is not an optimum: the softmax is saturated (entropy 0, probability 1 in float32), so PPO does not explore
    and sees no signal.
  - The starting point is still worth 8 times the best run from scratch (63.2).
- **Seaquest:** the score drops from 51 to 33 in the first 5M steps. It returns to Astra's level at 20M steps and
  ends at **82.5 (+60%)**, above anything PPO from scratch reaches (55.9).
  - To reach 95% of the final score from scratch it saves 150M steps, about 1.4 hours on one core.
- **Asterix:** the score drops from 55 to 39 and slowly climbs back up to **50.8 (−7%)**. PPO does not improve
  Astra's network, but the starting point remains twice the run from scratch (24.6).
- **Low Tide:** PPO destroys the hand-written networks in the first million steps.
  - GPT drops from 80 to 1 and climbs back up to **147.3 (+84%)**. However, it stays below the run from scratch with
    the same architecture (166.9), which reaches 147 at 90M steps: starting from GPT wastes time instead of saving it.
  - Claude drops from 132 to 6 and never recovers: it ends at **31.0 (−77%)**, against 158.4 from scratch.
  - Training only the critic first does not solve the problem. Claude ends at 66.6, better, but still worse than
    starting from scratch and than the starting network. GPT gets stuck on a policy that never scores: 0.0 from 4M
    steps to the end.
- **Gradient:** apart from Breakout, no hand-written network sits on a plateau. The actor's gradient is 3-80 times
  the one PPO sees during its own training.
- **Weight movement:** in absolute value they move as much as in a run from scratch (21-44 against 40-55 in L2 norm).
  - Relative to their weights, which are huge (‖θ‖ from 500 to 45,000 against 11-17 for a random initialization),
    it is little: 0.07-9%.
  - It is enough, however, to change the policy: at the end they choose the same move as the starting network in
    only 15-68% of states.
  - 42-80% of the zero weights of the hand-written network become non-zero: the network is no longer sparse or
    readable.

**Why it collapses on Low Tide.** With weights this large the network sits in a very steep region. The same Adam
step, about lr on each weight, changes the policy much more than when starting from small weights.

- In the first million steps the KL between the policy before and after an update (median) is:
  - **Low Tide:** 0.23 starting from GPT and 0.070 from Claude, against 0.004-0.005 from scratch. That is 20-45
    times more, and the starting policy is erased before the critic is of any use.
  - **Seaquest and Asterix:** 0.0045 and 0.0057, against 0.0012-0.0014 from scratch, that is, 3-5 times. Here
    Astra's networks survive.
- On the MinAtar games entropy also matters: the 0.05 bonus, chosen for the runs from scratch, pushes Astra's
  almost deterministic networks toward randomness.
  - On Asterix the entropy part of the initial gradient (0.95) is almost as large as the policy-gradient part
    (1.06).
  - Entropy rises from 0.19 to 0.69 on Seaquest and from 0.31 to 0.69 on Asterix.
  - The initial drop in the two games probably comes from here: it is a hypothesis, we have not isolated it with a
    run at lower entropy.

```bash
uv run python experiments/ppo_runs.py start breakout/ppo_from_astra seaquest/ppo_from_astra asterix/ppo_from_astra \
    low_tide/ppo_from_gpt low_tide/ppo_arch_gpt low_tide/ppo_from_gpt_cw \
    low_tide/ppo_from_claude low_tide/ppo_arch_claude low_tide/ppo_from_claude_cw   # ~2 hours on 9 cores
uv run --group ppo python experiments/ppo_warm_start/probe.py          # gradient at fixed weights, ~1 min per network
uv run --group ppo --group analysis python experiments/ppo_warm_start/analyze.py --workers 8   # a few minutes
```

`analyze.py` writes to `results/`:

- `tables.md`: all the tables, including the complete ones;
- `curves.png`: the plot;
- `summary.json`: all the numbers;
- `scores.json`: the cache of the evaluations.

The final weights are in `weights/<game>/ppo_from_*.json` and `ppo_arch_*.json`. Logs, checkpoints every 5M steps
and the final evaluation are in `experiments/<game>/ppo/results/<run>/`.

## The measurements

- **Scores.**
  - Start and end use the full protocol: 1000 episodes for MinAtar, 500 for Low Tide.
  - Along the run, each checkpoint (every 5M steps) is evaluated on 200 episodes. The times ("back to the level
    at", "+10% at", "passes the threshold at") are the first checkpoint that gets there, so they have a resolution
    of 5M steps.
  - The training curve (mean of the episodes finished in the last 1M steps) also catches the collapses between two
    checkpoints, but it is noisier. On Breakout the policy does not change, yet the curve oscillates by ±15%.
  - The first million steps of this curve are not used: only the short episodes finish there, and those are the
    ones that went badly.
- **Savings.** The threshold is 95% of the final score of the run from scratch. The saving is the difference
  between the steps that the run from scratch and the run from the hand-written network need to pass it. The hours
  are those of the run from scratch: one core per run, ~1.8 h per 200M MinAtar steps and ~0.85 h per 100M Low Tide
  steps.
- **Gradient during training.** `ppo.py` logs two norms of the actor's gradient, before clipping:
  - **on the minibatch** at every update;
  - **on the full batch** of 8192 steps every `log_every` updates, split into a policy-gradient part and an
    entropy part.

  It also logs the fraction of minibatches cut by clipping and ‖θ − θ₀‖.
- **Gradient at fixed weights** (`probe.py`).
  - For each network PPO runs 2M steps with the actor frozen: the critic learns the value of that policy, and the
    log records the actor's gradient without ever applying it.
  - The reported value is the mean over the last quarter, with the critic by then adapted.
  - This compares on equal terms:
    - the hand-written networks;
    - PPO's random initialization;
    - the PPO networks halfway through and at the end of training;
    - the final networks of the runs from the hand-written weights.
- **Movement.**
  - ‖θ_end − θ₀‖ over all the actor's weights, in absolute terms, relative to ‖θ₀‖ and per layer.
  - How many zero weights in the hand-written network are now non-zero (|w| > 10⁻³).
  - Agreement between the most probable move of the final network and that of the hand-written network, on 40,000
    states played by the hand-written network or by the final network.

## Results

![score, distance from the start and gradient during training](results/curves.png)

One row for each hand-written network: the dashed line is its score. For Low Tide there is also the 64-64 run from
scratch (`ppo_100m`).

- **First column:** the checkpoint score (line with dots) and the training curve (thin line).
- **Second column:** the distance of the weights from the start. For the runs from scratch on the MinAtar games,
  launched before `ppo.py` logged it, it is computed from the checkpoints.
- **Third column:** the norm of the actor's gradient on the full batch, median over 1M steps. For the same MinAtar
  runs it comes from `probe.py`, measured at fixed weights.

### How much it improves

| game | start | initial collapse (training curve) | worst checkpoint | back to the hand-written network's level at | +10% at | best checkpoint | end | end − start | from scratch, same arch. |
|---|---|---|---|---|---|---|---|---|---|
| Breakout | Astra 512.5 | - (the policy does not change) | 512.7 | does not drop | never | 512.7 | 512.5 ± 7.6 | +0.0 (0%) | 63.2 ± 1.0 |
| Seaquest | Astra 51.4 | 36.9 at 4.9M | 33.3 at 5M | 20M | 25M | 87.6 at 190M | 82.5 ± 2.1 | **+31.0 (+60%)** | 55.9 ± 1.3 |
| Asterix | Astra 54.8 | 36.0 at 8.9M | 38.9 at 15M | 140M (with noise) | never | 55.4 at 175M | 50.8 ± 1.0 | −4.0 (−7%) | 24.6 ± 0.5 |
| Low Tide | GPT 79.9 | 1.3 at 1.1M | 46.1 at 5M | 15M | 15M | 149.5 at 100M | 147.3 ± 2.9 | **+67.3 (+84%)** | 166.9 ± 2.7 |
| Low Tide | GPT, 2M critic only | 0.0 at 4.2M | 0.0 at 5M | never | never | 79.7 at 0 | 0.0 ± 0.0 | −79.9 (−100%) | 166.9 ± 2.7 |
| Low Tide | Claude 132.3 | 6.3 at 5.7M | 5.4 at 5M | never | never | 132.1 at 0 | 31.0 ± 0.3 | −101.2 (−77%) | 158.4 ± 2.9 |
| Low Tide | Claude, 2M critic only | 38.4 at 4.3M | 39.8 at 5M | never | never | 132.1 at 0 | 66.6 ± 0.6 | −65.7 (−50%) | 158.4 ± 2.9 |

### How much time is saved

| game | start | threshold (95% of the final score from scratch) | from the hand-written network, passes it at | from scratch, passes it at | saving | from scratch, reaches the hand-written network at | from scratch, reaches the final score of this run at |
|---|---|---|---|---|---|---|---|
| Breakout | Astra | 60.1 | 0 | 105M | 105M steps ≈ 0.9 h | never (in 200M) | never |
| Seaquest | Astra | 53.1 | 20M | 170M | **150M steps ≈ 1.4 h** | 160M | never |
| Asterix | Astra | 23.4 | 0 | 160M | 160M steps ≈ 1.4 h | never (in 200M) | never |
| Low Tide | GPT | 158.5 | never | 95M | none: it does not get there | 5M | 90M |
| Low Tide | GPT, 2M critic only | 158.5 | never | 95M | none | 5M | 0 |
| Low Tide | Claude | 150.4 | never | 95M | none | 30M | 5M |
| Low Tide | Claude, 2M critic only | 150.4 | never | 95M | none | 30M | 5M |

On Breakout and Asterix the saving is a lower bound: the run from scratch never reaches the starting network, and
the run from the hand-written weights already starts above the threshold.

### The gradient: plateau or not?

Actor gradient on the full batch, at fixed weights and with the critic adapted, before clipping. Full table
in `results/tables.md`.

| game | network | entropy | ‖θ‖ | ‖g‖ | of which policy gradient / entropy | ‖g‖ / ‖θ‖ |
|---|---|---|---|---|---|---|
| Breakout | Astra (hand-written) | 0.000 | 3588 | **4.2·10⁻¹³** | 1.2·10⁻¹⁴ / 4.2·10⁻¹³ | 10⁻¹⁶ |
| | random initialization | 1.099 | 11.3 | 0.060 | 0.060 / 0.0001 | 0.005 |
| | PPO from scratch at 5M / 50M / 200M | 0.53-0.76 | 174-181 | 0.164 / 0.144 / 0.085 | | 0.0005-0.0009 |
| Seaquest | Astra (hand-written) | 0.143 | 498 | **0.367** | 0.344 / 0.096 | 0.0007 |
| | random initialization | 1.792 | 11.3 | 0.055 | | 0.005 |
| | PPO from scratch at 5M / 50M / 200M | 1.23-1.48 | 21-57 | 0.084 / 0.126 / 0.115 | | 0.002-0.004 |
| | from the hand-written network, end | 0.685 | 500 | 0.131 | | 0.0003 |
| Asterix | Astra (hand-written) | 0.286 | 1760 | **1.14** | 1.06 / 0.95 | 0.0006 |
| | random initialization | 1.609 | 11.3 | 0.015 | | 0.001 |
| | PPO from scratch at 5M / 50M / 200M | 1.19-1.28 | 102-114 | 0.076 / 0.100 / 0.111 | | 0.001 |
| | from the hand-written network, end | 0.693 | 1451 | 0.289 | | 0.0002 |
| Low Tide | GPT (hand-written) | 0.620 | 1600 | **1.32** | 1.31 / 0.075 | 0.0008 |
| | random, GPT's arch. | 1.609 | 16.7 | 0.025 | | 0.0015 |
| | from scratch, GPT's arch., end | 0.414 | 43.7 | 0.409 | | 0.009 |
| | from the hand-written network, end | 0.291 | 1601 | 1.09 | | 0.0007 |
| | Claude (hand-written) | 0.351 | 45621 | **28.3** | 28.3 / 1.09 | 0.0006 |
| | random, Claude's arch. | 1.609 | 12.8 | 0.042 | | 0.003 |
| | from scratch, Claude's arch., end | 0.363 | 44.4 | 0.352 | | 0.008 |
| | from the hand-written network, end | 0.493 | 45621 | 1.88 | | 0.00004 |
| | 64-64 from scratch (`ppo_100m`) at 5M / 50M / 100M | 0.38-0.45 | 27-57 | 0.237 / 0.235 / 0.252 | | 0.004-0.009 |

- **Astra's Breakout** is a plateau in the literal sense: a gradient 10¹¹ times smaller than that of any other
  network.
  - Adam has eps = 10⁻⁵, and a gradient this small leads it to steps of about lr × 10⁻⁸ per weight.
  - During the run the norm stays at 4·10⁻¹³ from the first to the last update.
- **The other hand-written networks** have a gradient 3-80 times larger than the one PPO sees at the end of its own
  training:
  - Seaquest: 0.37 against 0.115;
  - Asterix: 1.14 against 0.111;
  - GPT: 1.32 against 0.409;
  - Claude: 28.3 against 0.352.
- Divided by ‖θ‖ the gradient is small (10⁻⁴-10⁻³), because the weights are huge. But Adam does not look at the
  ratio: it moves each weight by about lr. What matters is how much the policy changes, and that is a lot, as the
  KL above shows.
- **With the random critic** of the first update the gradient is even larger: 18.6 for GPT, against 0.076
  from scratch.
  - Clipping (0.5 on the joint norm of actor and critic) cuts 100% of the minibatches in the first million steps of
    the runs from Astra (Seaquest and Asterix) and from GPT, and 79% in the one from Claude.
  - For Adam, clipping changes little, because Adam normalizes the scale.

### How much the weights move

| game | start | ‖θ₀‖ | ‖θ_end − θ₀‖ | relative | per layer | from scratch: ‖θ_end − θ₀‖ (relative) | zeros of the hand-written network now ≠ 0 | same move as the hand-written network (on its states / on its own) |
|---|---|---|---|---|---|---|---|---|
| Breakout | Astra | 3588 | 7·10⁻⁶ | 0.00% | 0 · 0 · 0 | 51.2 (29%) | 0% | 100% / 100% |
| Seaquest | Astra | 498 | 43.9 | 8.8% | 34.7% · 8.1% · 0.5% | 55.5 (491%) | 51% | 37% / 56% |
| Asterix | Astra | 1451 | 32.6 | 2.3% | 13.8% · 7.1% · 0.1% | 52.0 (52%) | 42% | 68% / 70% |
| Low Tide | GPT | 1600 | 27.5 | 1.7% | 36.9% · 2.2% · 1.6% · 0.2% | 39.9 (239%) | 80% | 31% / 36% |
| Low Tide | GPT, 2M critic only | 1600 | 20.8 | 1.3% | 28.2% · 1.8% · 0.9% · 0.3% | 39.9 (239%) | 68% | 15% / 0% |
| Low Tide | Claude | 45621 | 40.8 | 0.09% | 0.2% · 0.0% · 12.9% | 42.3 (330%) | 60% | 17% / 18% |
| Low Tide | Claude, 2M critic only | 45621 | 32.9 | 0.07% | 0.2% · 0.0% · 8.9% | 42.3 (330%) | 60% | 18% / 24% |

- **The first layer** is the one that moves the most in Astra's and GPT's networks. It contains almost all the
  weights (600×32 out of 20,486 in the MinAtar games, 403×43 out of 23,612 in GPT), and Adam moves each one by
  about lr.
- **Claude** is the exception: the first two layers have norms 11,772 and 44,075, and in proportion it is the output
  that moves.
- **The policy changes a lot:** Seaquest ends at +60% while choosing Astra's move in only 37% of its states.
- **Breakout** is identical to the start: the weights and therefore the moves.

## All scores

Full protocol (1000 MinAtar episodes, 500 Low Tide), mean ± standard error. All PPO runs have lr 5e-4.

| model | Breakout | Seaquest | Asterix |
|---|---|---|---|
| random | 0.5 ± 0.0 | 0.1 ± 0.0 | 0.5 ± 0.0 |
| **Astra, hand-written** (`paper_astra`) | **512.5 ± 7.6** | 51.4 ± 1.4 | **54.8 ± 1.1** |
| PPO from scratch, entropy 0.01, 30M steps (`ppo_v1`) | 5.8 ± 0.0 | 5.4 ± 0.1 | 9.1 ± 0.2 |
| PPO from scratch, entropy 0.01, 200M (`ppo_ent01`) | 34.7 ± 0.5 | 4.4 ± 0.0 | 19.4 ± 0.4 |
| PPO from scratch, entropy 0.05, 200M (`ppo_ent05`) | 63.2 ± 1.0 | 55.9 ± 1.3 | 24.6 ± 0.5 |
| PPO from scratch, entropy 0.1, 200M (`ppo_ent10`) | 52.2 ± 1.4 | 34.4 ± 1.0 | 15.5 ± 0.4 |
| PPO from Astra, entropy 0.05, 200M (`ppo_from_astra`) | 512.5 ± 7.6 | **82.5 ± 2.1** | 50.8 ± 1.0 |

The architectures are all 600-32-32-6 tanh.

| model (Low Tide) | architecture | score |
|---|---|---|
| random | | 1.9 ± 0.1 |
| heuristic | | 164.0 ± 1.5 |
| **Claude, hand-written** (`claude`) | 46-36 relu/tanh | 132.3 ± 0.9 |
| **GPT, hand-written** (`gpt`) | 43-61-53 relu | 79.9 ± 0.9 |
| PPO from scratch, 20M steps (`ppo`) | 64-64 tanh | 125.7 ± 1.9 |
| PPO from scratch, 100M (`ppo_100m`) | 64-64 tanh | 162.7 ± 2.6 |
| PPO from scratch, 100M (`ppo_arch_gpt`) | 43-61-53 relu | **166.9 ± 2.7** |
| PPO from scratch, 100M (`ppo_arch_claude`) | 46-36 relu/tanh | 158.4 ± 2.9 |
| PPO from GPT, 100M (`ppo_from_gpt`) | 43-61-53 relu | 147.3 ± 2.9 |
| PPO from GPT, 2M critic only + 98M (`ppo_from_gpt_cw`) | 43-61-53 relu | 0.0 ± 0.0 |
| PPO from Claude, 100M (`ppo_from_claude`) | 46-36 relu/tanh | 31.0 ± 0.3 |
| PPO from Claude, 2M critic only + 98M (`ppo_from_claude_cw`) | 46-36 relu/tanh | 66.6 ± 0.6 |

All Low Tide PPO runs have entropy 0.01.

## What remains open

- **Learning rate.** The learning rate (5e-4) is the one used in the runs from scratch. With weights this large, an
  lr 10-100 times smaller, or a KL-limited step, might keep the starting policy instead of erasing it. It is the most
  direct test of the steep-region hypothesis.
- **Entropy on the MinAtar games.** A lower entropy (0.01 or 0) would tell whether the initial drop of Seaquest and
  Asterix comes from the bonus.
- **Breakout.** It does not improve by construction: without entropy it does not explore. The network would need
  to be softened first, for example by dividing the logits by a temperature, so as to give it probabilities other
  than 0 and 1 without changing its most probable move.
