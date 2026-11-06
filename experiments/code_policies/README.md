# An algorithm written as code instead of as weights

The `decode_*` folders show that the networks hand-written by the LLMs are algorithms put into weights. Here
the same LLM is asked to write the algorithm directly as Python code, under the same conditions:
- one conversation per game and a single answer;
- no tools: no code is run and there is no calculator;
- no feedback.

The comparison is between the same model writing the weights and the same model writing the code, on the same game.

## The prompts

```bash
uv run python experiments/code_policies/make_prompts.py    # writes prompts/<game>.txt
```

For Breakout, Seaquest and Asterix the prompt is the one of Experiment 5 of the article
(`exp5_write_from_scratch/runs/<game>/message.txt` in
[lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), CC BY 4.0).

**Identical to the original, word for word:**
- the game description;
- the exact source code of MinAtar 1.0.15;
- the input (600 numbers, index `(row * 10 + column) * 6 + channel`);
- the channel mapping.

**Only the parts about the network change:**

| section | weights prompt | code prompt |
|---|---|---|
| task | the weights of a network | a Python function, the algorithm you want |
| output | 6 logits, action sampled from the softmax | the action, an integer |
| network / function | 600-32-32-6 tanh, every weight starts at 0 | `policy(obs, rng)`, rules below |
| how it is evaluated | 1000 episodes, sticky 0.1, 10,000 steps | the same, with the action returned by the function |
| how it is written | an `edits` block | a `python` block with the complete file |

For Low Tide the prompt has the same structure:
- the game description;
- the source of `env.py` (not `baselines.py`, which contains the heuristic);
- the input (403 numbers), the output (5 actions) and the protocol (500 episodes, 1000 steps).

If the prompt used to have Claude and GPT write the Low Tide weights was different, it is worth putting its game
part here, to compare the two on equal terms.

**Rules for the function**, written in the prompt:
- **Only the current observation.** No memory between one call and the next, like the networks it is compared with.
  Constant tables computed at load time are fine.
- **Only the standard library and numpy.** The game is not imported.
- **At most about 1 ms per call.**
- **Errors.** If the function raises an error or returns an invalid value, the action is 0.

The `prompts/` folder is not versioned, because it contains the MinAtar source code (GPL-3.0).

## How to use them

**Astra, with Codex without tools:**

```bash
uv run python experiments/code_policies/run_codex.py low_tide --dry-run   # check what the model would see
uv run python experiments/code_policies/run_codex.py low_tide             # gpt-6-astra, xhigh, as in the article
uv run python experiments/code_policies/run_codex.py breakout             # ... one game at a time
```

Each call is a new conversation. The answer goes in `answers/<model>/<game>/`:
- `answer.md`, the answer;
- `policy.py`, the last `python` block;
- `events.jsonl`, the raw events;
- `run.json`, timings, tokens and checks.

**How the tools are removed.** In Codex 0.164 the article's configuration (`codex_no_tools.toml`) is not enough.
The tools of gpt-6-astra come from Codex's model catalog:
- `tool_mode = "code_mode_only"` gives code mode, which contains apply_patch, MCP and a clock;
- `multi_agent_version = "v2"` gives the agents;
- `request_user_input` and the system skills come from the configuration.

`run_codex.py`:
1. uses a temporary CODEX_HOME with that configuration and a copy of the catalog without those fields;
2. before sending the prompt, checks with `codex debug prompt-input`, which does not call the model, that the model
   sees 0 tools and only two messages: the article's neutral instructions and the prompt;
3. afterwards, counts the tool calls in the events.

`--check` verifies it with a short question: "which tools can you call?". Without the changes the model received
5,098 tokens of context and listed exec, apply_patch, the agents and other tools. With the changes it receives 59
tokens and answers `NONE`.

**Claude:** from the API without tools, or from claude.ai with code execution and web search turned off, pasting
the whole prompt file as the only message. Save the answer in the same format, in
`answers/<model>/<game>/answer.md`.

## Results: GPT-6 Astra (6 November 2026)

One answer per game, with `run_codex.py`:
- `gpt-6-astra`, reasoning `xhigh`;
- 0 visible tools and 0 calls;
- between 14 and 23 minutes per answer.

The scores come from `evaluate_code.py`, with the same protocol as the networks.

| game | **Astra's code** | Astra's network (weights) | other references |
|---|---|---|---|
| Low Tide (500 ep.) | **188.9 ± 0.7** | 79.9 ± 1.0 (`gpt`) | BFS heuristic 164.0, best PPO 166.9, `claude` 132.3 |
| Breakout (1000 ep.) | **786.0 ± 2.6** | 512.5 ± 7.6 (`paper_astra`) | PPO 500M steps (article) 25.3 |
| Seaquest (1000 ep.) | **187.3 ± 2.8** | 51.4 ± 1.4 (`paper_astra`) | PPO 500M steps (article) 88.9 |
| Asterix (1000 ep.) | **83.2 ± 1.8** | 54.8 ± 1.1 (`paper_astra`) | PPO 500M steps (article) 55.2 |

**In all four games the code beats the network written by the same model and the best PPO.**

**Checks passed by all four policies, on 2000 states:**
- they import only numpy and the standard library;
- 0 errors and 0 invalid actions;
- 95-283 µs per call;
- no memory: same action with the states in reverse order.

**What they do, all computed at each step from the observation alone:**
- **Low Tide.** It computes, for each cell, the last step at which it can be reached while still having time to return to dry ground. For timing it uses the worst case, i.e. tide intervals of 3 steps.
  - Moves are also checked for the case in which the sticky action repeats them.
  - The path between the shells is chosen with a beam search.
  - There is a bonus for staying near the shoreline, where new shells appear.
  - It drowns in 0.6% of the episodes, against 11% for the BFS heuristic; it collects 190.4 shells and 1.5 stings.
- **Breakout.** Exact ball physics, search over 5 bounces on the paddle and dynamic programming on the paddle position with the sticky actions. 96.7% of the episodes reach 10,000 steps.
- **Seaquest.** Short-horizon collision model, divers to pick up, oxygen, enemy torpedoes and sticky-action risk.
- **Asterix.** Short-horizon planning to intercept the treasures, with a fast and a slow model of movement. A separate safety search accounts for sticky actions and new enemies.

**Note on Seaquest.** The token counter in `run.json` gives 26,031 input tokens and only 516 reasoning tokens, against 14-21 thousand in the other games. It probably counts only the last request, after an automatic context compaction. The events show a single turn, 42 reasoning blocks, no tools and no errors.
