# How far apart are the networks written by LLMs?

**The hypothesis to test.** The weights that LLMs write "by hand" do not come from nothing: the models have
already seen neural networks for these games and remember them. If that were so, the networks for this kind of task
should be more or less all the same, and knowing one of them would solve all the games.

From this hypothesis follow predictions that can be checked on the weights:

1. if the weights were the memory of trained networks, they should **resemble trained networks**;
2. if the networks were all more or less the same, their **layer-by-layer distance** should be much
   smaller than the one between two networks with nothing in common;
3. the units of one network should have a **twin** in the other, and the layers should compute similar
   **representations** on the same states;
4. the network for one game should also **play** the others reasonably well.

The numbers are in [`results/report.md`](results/report.md) (tables) and in `results/results.json` (everything).

```bash
uv run --group analysis python experiments/weight_distance/analyze.py          # ~35 s on 8 cores
uv run --group analysis python experiments/weight_distance/analyze.py --help   # episodes, repetitions, ...
```

The `analysis` group only adds scipy (Hungarian algorithm). Everything uses a fixed seed, so the same command
gives the same numbers.

## The networks

| network | author | game | architecture | score |
|---|---|---|---|---|
| `astra/breakout` | GPT-6 Astra, in the article | Breakout (MinAtar) | 600-32-32-6, tanh | 512.5 (random 0.5) |
| `astra/seaquest` | GPT-6 Astra, in the article | Seaquest (MinAtar) | 600-32-32-6, tanh | 51.4 (random 0.04) |
| `astra/asterix` | GPT-6 Astra, in the article | Asterix (MinAtar) | 600-32-32-6, tanh | 54.8 (random 0.5) |
| `claude/low_tide` | Claude Opus, for this project | Low Tide | 403-46-36-5, relu/tanh | 132.3 (heuristic 164.0, random 2.0) |
| `gpt/low_tide` | GPT-6 Astra, for this project | Low Tide | 403-43-61-53-5, relu | 79.9 |
| `ppo/{breakout,pong,pacman,freeway}-grid` | **trained with PPO**, 10⁹ steps | the authors' grid games | 500-32-32-5, tanh | — |

The first five are written by an LLM in a single answer, without tools or training: the three MinAtar ones are
those of the article (`weights/<game>/paper_astra.json`), the two Low Tide ones are in `weights/low_tide/`.

The four PPO networks **do not play our games** and serve only as a reference point, to see what a *trained*
network of this size looks like. They come from Experiment 1 of the article ("which game was this network trained
on?"): the authors trained them on their own 10×10 versions of Breakout, Pong, Pac-Man and Freeway, with 5
channels (agent, moving objects, rewards, obstacles, previous frame) and 5 actions. They are in `reference/`
(actor only, `W` = transposed kernel), copied from
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) `exp1_which_game/networks/`
(a9f535cc, dfc6cb65, bdf99279, 0bbcc3ea), CC BY 4.0. The PPO networks trained on MinAtar, with which the article
compares Astra's, have not been published.

## What `analyze.py` measures, and why ‖A − B‖ is not enough

- **Hidden units can be shuffled.** If the 32 neurons of a hidden layer are reordered, together with the
  corresponding columns of the next layer, the network computes exactly the same function. With tanh one can also
  flip the sign of a neuron: inputs and outputs with the opposite sign. Two copies of the same network with the
  neurons in a different order can therefore have a huge ‖A − B‖. For this reason, besides the **raw** distance, the
  script computes the **aligned** one: it looks for the reordering (with sign flips) of the neurons of B that brings it
  closest to A (weight matching, as in Git Re-Basin, Ainsworth et al. 2022). The check, a network against a shuffled
  copy of itself, gives distance 0.
- **A reference point is needed.** Alignment shortens the distance even between networks that have nothing to do with
  each other. The **null model** is therefore B with its own values scattered at random within each layer, then
  aligned in the same way. If aligned ≈ null, the two networks have no more in common than two random networks.
- **The relative distance** ‖A − B‖ / √(‖A‖² + ‖B‖²) is 0 for identical networks, about 1 for uncorrelated networks
  and √2 for opposite networks.
- **The scale in hand-written networks is arbitrary.** There are weights of 1000 and more, and a saturated tanh does
  not distinguish 100 from 1000. For this reason ‖A − B‖ mostly measures where those few huge weights are. The
  measures that do not depend on scale are the ones that follow.
- **Twin units**: the first-layer units of A and B are paired one to one by the cosine of their incoming
  weights. The null is the same pairing with the inputs of B shuffled.
- **Linear CKA** (Kornblith et al. 2019) between the outputs of each pair of layers, computed on the same game
  states. It also works between networks of different shape, such as the two Low Tide ones. The reference is a network
  with the same shape and random weights, which measures how much similarity comes "for free" from reading the same
  inputs.
- **Behaviour**: how often two networks choose the same action on the same states, compared with chance, and
  **cross-play**, that is, the network for one game playing the other two.
- **Networks of different shape** (section 7): each network is rewritten, without changing what it computes, in a
  common space of inputs (10×10 grid with 7 channels plus 3 scalars) and of actions, with the hidden layers widened by
  zero units. The first layer and the output are aligned exactly. The intermediate layers are aligned exactly only
  between networks with the same number of hidden layers; otherwise the result is approximate, and the von Neumann
  upper bound is given next to it.

## Results

**1. The hand-written networks do not resemble trained networks at all.** For the PPO networks "distinct values"
means that almost no value repeats.

| | written by LLMs (5 networks) | trained with PPO (4 networks) |
|---|---|---|
| nonzero weights | 3–5% | 100% |
| distinct values | 40–62 | ~17,000 out of ~17,250 |
| nonzero with at most 2 decimals | 94–100% | 0% |
| largest weight (in absolute value) | from 244 to 5122 | less than 9 |
| share covered by the 8 most used values | 34–81% | 0% |

The hand-written networks are made of a few round constants, repeated. Claude in Low Tide uses mostly powers of 2
(1, 8, 16, 32, 64, 1024); Astra in Asterix uses ±8, ±16, 24 and 1000 as thresholds. They are programs translated into
weights. No gradient descent produces a network like this, and no trained network looks like this: what they write is
not the memory of the weights of a trained network.

**2. Networks for different games: as far apart as random networks, layer by layer.** Astra's three networks have the
same shape, the same inputs and the same actions, so they can be compared weight by weight. Their raw relative
distance is about 1.00 in all layers, that is, they are orthogonal. Even after alignment it stays about equal to the
null (whole network: 1.00 against 0.99, 0.99 against 0.98, 1.01 against 1.08).

The only real similarity is between the first layer of Seaquest and that of Asterix:
- aligned distance 0.79 against 0.95 for the null;
- 15 units out of 32 with cosine > 0.5, against 0 for the null.

The two games are in fact similar: enemies and rewards that scroll along rows and a player to move. Already at the
second layer, however, the networks are no closer than chance.

The PPO networks for different games behave the same way: aligned distance of the whole network 0.87–0.94,
against 0.94–0.95 for the null.

**Astra's four networks, Low Tide included** (section 7 of the report). The intermediate layers of the Low Tide network
are not the same as those of the MinAtar networks: Low Tide is 403-43-61-53-5, that is, 3 hidden layers of different
width, while the MinAtar networks are 600-32-32-6. To compare them, each network is rewritten without changing what
it computes:
- the inputs go into a common space: agent → player, jellyfish → danger, shells → reward, plus water and
  scalars as channels of their own;
- the actions are put back in the same order;
- the layers are widened with zero units.

Each layer is then aligned in the most favourable way. The table reports the cosine after alignment, and in
parentheses that of the null: 1 means the same weights, 0 nothing in common.

| pair | first layer | intermediate | output | whole network |
|---|---|---|---|---|
| Breakout – Seaquest | 0.00 (0.02) | 0.37 (0.40) | **−0.82** (−0.11) | −0.00 (0.06) |
| Breakout – Asterix | 0.06 (0.02) | 0.57 (0.46) | −0.20 (0.21) | 0.03 (0.04) |
| Seaquest – Asterix | **0.44** (0.12) | 0.37 (0.50) | −0.04 (−0.22) | −0.02 (−0.22) |
| Low Tide – Breakout | 0.00 (0.06) | 0.40 (0.43)* | 0.02 (0.14) | — |
| Low Tide – Seaquest | 0.14 (0.06) | 0.80 (0.78)* | 0.12 (0.10) | — |
| Low Tide – Asterix | 0.03 (0.06) | 0.70 (0.55)* | 0.07 (0.04) | — |
| *reference:* Low Tide, Astra – Claude | 0.26 (0.05) | 0.59 (0.70)* | 0.12 (0.11) | — |

- **First layer** (the one that reads the screen). The Low Tide network has nothing in common with the three
  MinAtar ones: cosine from 0.00 to 0.14. Among the MinAtar ones only Seaquest and Asterix are similar (0.44). The two
  Low Tide networks by Astra and by Claude, that is, same game and different authors, are more similar to each other
  (0.26) than Astra's Low Tide network is to Astra's networks for the other games.
- **Intermediate layers.** They say nothing, because the null is already high (0.40–0.78). With so few nonzero weights
  even values shuffled at random align well. With `gpt/low_tide` (*) the comparison is also approximate:
  the cosine found is a lower bound and the upper bound (0.92–0.97) is too loose to say anything.
- **Output.** Breakout and Seaquest are almost *opposite* (−0.82). Breakout pushes left and right (+90) and
  forbids up, down and fire (−100); Seaquest does the opposite.
- **Whole network.** Cosine about 0, like the null. In 2-norm, for example, Breakout and Seaquest are 3623 apart, with
  norms 3588 and 498: the distance is as large as the larger network.

**3. Each network reads different things.**
- Breakout reads the ball (72% of the first-layer connections), its trail (23%) and barely the paddle (5%).
- Seaquest reads dangers, rewards and player.
- Asterix reads trail, dangers, player and rewards.

They are programs written for that game, not a universal network.

**4. No network can play the other games.** In cross-play (200 episodes per cell) each network scores only in
its own game: 512.7, 48.3 and 54.5. On another game it does as well as chance, for example:
- the Breakout network on Asterix scores 0.6, against 0.5 for random actions;
- the Seaquest network on Breakout scores 0.5.

On the states of another game the networks choose the same action as often as two independent choices would.

**5. Same game, two different LLMs (Low Tide): same building blocks, the rest different.** It is the only pair of
networks written for the same game. The architectures are different, so there is no direct ‖A − B‖.
- **First layer.** 3 units out of about 45 are almost identical (cosine > 0.9) and 7 exceed 0.5, against 0.9 for the
  null. The twins are elementary building blocks that both networks write in the same way:
  - the copy of the scalar "the tide is rising";
  - "the agent is in the last column";
  - "the agent is in the last row".
- **CKA.** The final layers are similar (logits 0.75), the intermediate ones partly (about 0.5). The first layers are
  not: at most 0.11, against 0.07 for the random network.
- **Actions.** The two networks choose the same action 50% of the time (chance would give 23%). That is more than
  each one agrees with the heuristic: 43% Claude, 35% GPT.

The behaviour converges, the weights do not.

## What can be concluded

- **"The networks are all more or less the same": the data do not support it.** Two networks for different games are
  as far apart as two uncorrelated networks, in every layer, even after removing the neuron symmetry. Even the two
  networks for the same game share only a few elementary units. What they have in common is the **style**:
  - few round weights;
  - tanh and relu used as logic gates and thresholds;
  - units that answer questions like "where is the agent?".

  It is a way of programming, not a set of weights.
- **"They have seen them and remember them": not for the weights.** A remembered network would look like a
  trained network, and these are the opposite of one. What the model may have seen is knowledge of the games and of
  strategies (for example "follow the ball"), together with how an MLP computes. This experiment does not distinguish
  between a *remembered* strategy and one *derived from the rules*. The right control for this is Low Tide, a new game,
  where both networks work: 132 and 80 against 2 for chance, even if below the heuristic (164).

## Limits and next steps

- **There is a single network per (author, game) pair.** So one cannot see how much two answers of the same
  model to the same task vary. It is the decisive test for "they always converge to the same network": ask 3–5 times
  (new conversations) for the Breakout network, to Claude and to Astra, and compare the distance *within* the same
  game with the distance *between* games. If the Breakout networks turned out almost identical to each other and far
  from those for the other games, the memory hypothesis would be back in play. The script compares them on its own if
  they have the same shape: just add them in `load_nets()`.
- **The PPO networks are for other games.** The direct comparison, written network against network trained on the
  same game, requires training a 600-32-32-6 with PPO on MinAtar, because the checkpoints of the article are not
  public.
- **CKA has a high floor** (up to 0.4 with random networks, on the MinAtar games) on these sparse, binary inputs. It
  must always be read against the "rete casuale" (random network) column.
