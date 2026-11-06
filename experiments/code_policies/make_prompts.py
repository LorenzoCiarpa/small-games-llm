"""
Writes the prompts that ask an LLM to write an algorithm (Python code) instead of the weights of a network.

One prompt per game, in prompts/<game>.txt. For Breakout, Seaquest and Asterix the prompt is the one of Experiment 5
of the article (github.com/RikiyaT/lm_understand-logs, exp5_write_from_scratch/runs/<game>/message.txt, CC BY 4.0).
These stay identical:
- the game, with the exact source of MinAtar 1.0.15;
- the input;
- the channel mapping.

Only the parts about the network change: the task, the output, the function in place of the network, how it is
scored and how the answer is written. For Low Tide the prompt has the same structure, with the source of env.py.

The prompts contain the MinAtar source (GPL-3.0): prompts/ is not versioned, it is regenerated with

    uv run python experiments/code_policies/make_prompts.py
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "prompts"

TASK = (
    "YOUR TASK: write, from scratch, a Python function that plays this game as well as possible: an algorithm you "
    "design yourself (rules, a heuristic, a search, anything that can be written as code) that reads the "
    "observation and returns the action. There is no training: no data, no learning, no tuning; you choose every "
    "rule and every number yourself. You have no tools: no code execution, no calculator, nothing to run; your "
    "code is never run before it is scored. This is a single response and you will get no feedback. First "
    "describe your algorithm briefly, then write the python block."
)

# from the article (exp5_write_from_scratch/runs/<game>/message.txt), word for word
MINATAR = {
    "breakout": {
        "game": (
            "THE GAME: Breakout from MinAtar (Young & Tian 2019), on a 10 x 10 grid. The player moves a one-cell "
            "paddle along the bottom row and must bounce a ball to break 3 rows of bricks near the top. Each brick "
            "broken gives +1; when all bricks are cleared, 3 new rows appear. The ball moves only diagonally; when "
            "it hits the paddle it is sent up to the left or to the right depending on the side of the paddle it "
            "hits; it is reflected by walls and bricks. The episode ends when the ball reaches the bottom row "
            "without being caught.\n"
            "Actions: left and right move the paddle; up, down and fire do nothing (they act as no-op)."
        ),
        "channels": [
            "the paddle",
            "always empty in this game",
            "the bricks",
            "the ball",
            "the ball's previous cell (it shows which way the ball moves)",
            "always empty in this game",
        ],
        "mapping": "channel 0 (player) <- paddle; channel 1 (danger) <- nothing; channel 2 (reward) <- brick; "
                   "channel 3 (projectile) <- ball; channel 4 (trail) <- trail; channel 5 (gauge) <- nothing.",
    },
    "seaquest": {
        "game": (
            "THE GAME: Seaquest from MinAtar (Young & Tian 2019), on a 10 x 10 grid. The player steers a two-cell "
            "submarine (front and back) in all 4 directions and fires bullets from its front. Enemy fish and enemy "
            "submarines cross the screen; enemy submarines shoot. Hitting an enemy with a bullet gives +1. Divers "
            "can be picked up (up to 6; a bar along the bottom counts them). Oxygen runs down (another bar along "
            "the bottom) and is refilled by surfacing at the top row with at least one diver on board (surfacing "
            "with fewer than 6 removes one diver; with 6, all are removed and a bonus is paid). The episode ends "
            "when the submarine is hit, runs out of oxygen, or surfaces with no diver.\n"
            "Actions: left, up, right, down move the submarine, fire shoots."
        ),
        "channels": [
            "the submarine (front and back cells)",
            "enemy fish, enemy submarines and enemy bullets",
            "enemy fish, enemy submarines and divers",
            "the player's bullets",
            "the cell just behind each enemy and diver, and the submarine's back cell",
            "the oxygen bar and the diver bar along the bottom row",
        ],
        "mapping": "channel 0 (player) <- sub_front, sub_back; channel 1 (danger) <- enemy_fish, enemy_sub, "
                   "enemy_bullet; channel 2 (reward) <- enemy_fish, enemy_sub, diver; channel 3 (projectile) <- "
                   "friendly_bullet; channel 4 (trail) <- trail, sub_back; channel 5 (gauge) <- oxygen_guage, "
                   "diver_guage.",
    },
    "asterix": {
        "game": (
            "THE GAME: Asterix from MinAtar (Young & Tian 2019), on a 10 x 10 grid. The player moves freely in the "
            "4 directions. Enemies and treasure enter from the sides and move horizontally. Picking up a treasure "
            "gives +1. The episode ends when the player touches an enemy. The difficulty rises over time (enemies "
            "and treasure move faster and appear more often).\n"
            "Actions: left, up, right, down move the player; fire does nothing (it acts as no-op)."
        ),
        "channels": [
            "the player",
            "enemies",
            "treasure",
            "always empty in this game",
            "the cell just behind each enemy and treasure (it shows which way it moves)",
            "always empty in this game",
        ],
        "mapping": "channel 0 (player) <- player; channel 1 (danger) <- enemy; channel 2 (reward) <- gold; "
                   "channel 3 (projectile) <- nothing; channel 4 (trail) <- trail; channel 5 (gauge) <- nothing.",
    },
}
CHANNEL_NAMES = ("player", "danger", "reward", "projectile", "trail", "gauge")

LOW_TIDE_GAME = (
    "THE GAME: Low Tide, a small grid world written for this test (it is not a known game), on a 10 x 10 grid. The "
    "player collects shells on a beach: row 0 is the land side (top), row 9 the sea side (bottom). The water covers "
    "the bottom `level` rows. Every 3 to 5 steps the tide moves by one row: it rises up to a high level (5, 6 or 7, "
    "drawn at random), then recedes down to a low level (0, 1 or 2), and so on. Every time the tide recedes, the "
    "uncovered row is littered with shells (25% of its cells) and jellyfish (8%); every time it rises, the flooded "
    "row is washed clean. Stepping on a shell gives +1; stepping on a jellyfish gives -1 and paralyses the player "
    "for 3 steps. The player cannot walk into the water, but if the rising tide floods the player's row, the "
    "player drowns and the episode ends.\n"
    "Actions: up, down, left, right move the player by one cell; no-op stays."
)

LOW_TIDE_INPUT = """\
THE INPUT (403 numbers): obs[0:400] are four 10 x 10 planes of 0/1 values (rows 0 top to 9 bottom, columns 0 left to 9 right), one plane after the other: input number i = plane * 100 + row * 10 + column; for example input 36 is (agent plane, row 3, column 6) and input 236 is (shell plane, row 3, column 6).
  plane 0 (agent): exactly one 1, the player's cell
  plane 1 (water): whole rows from the bottom; level = (sum of the plane) / 10, and the lowest dry row is 9 - level
  plane 2 (shells)
  plane 3 (jellyfish)
obs[400] (tide rising): 1 if the next level change raises the water, 0 if it lowers it.
obs[401] (countdown): (steps until the next level change) / 5, one of 0.2, 0.4, 0.6, 0.8, 1.0; 0.2 means the level changes at the end of the next step.
obs[402] (stun): (remaining paralysis steps) / 3, one of 0, 1/3, 2/3, 1; while it is above 0 the next action is ignored.
THE OUTPUT: your function returns the action, an integer: 0 no-op, 1 up (row - 1), 2 down (row + 1), 3 left (column - 1), 4 right (column + 1). With probability 0.1 the game repeats the previous action instead (sticky actions)."""


def function_section(n_inputs, input_formula, n_actions, calls):
    return f"""\
THE FUNCTION:
    def policy(obs, rng):
        ...
        return action
- obs is a numpy array of {n_inputs} float32 numbers, THE INPUT above ({input_formula}). rng is a numpy.random.Generator: use it if you want random choices, or ignore it.
- It returns an integer from 0 to {n_actions - 1}. If it raises an error, or returns anything else, the action of that step is 0 (no-op).
- It sees only the current observation, like the reinforcement-learning networks it is compared with: it is called once per step with nothing else, and it must not remember anything between calls (no global variables that change, no function attributes, no caches, no objects with state). Constants and tables computed once, when the file is loaded, are fine.
- Python 3.12, standard library and numpy only. Do not import the game{" or MinAtar" if n_inputs == 600 else ""}.
- On average at most about 1 millisecond per call on one CPU core: it is called {calls} of times."""


def scored_section(episodes, max_steps, what):
    return (
        f"HOW YOUR FUNCTION IS SCORED: its mean score per episode{what} over {episodes:,} episodes, the action of "
        f"each step being the one your function returns (with the sticky actions above); an episode also stops "
        f"after {max_steps:,} steps. It is compared with networks trained by reinforcement learning on this game."
    )


HOW_TO_WRITE = (
    "HOW TO WRITE THE CODE: write one code block whose opening line is three backticks followed by the word python, "
    "with the complete file. It must define policy(obs, rng) at the top level; it may define other functions and "
    "constants. Only the LAST python block of your reply is read: it is saved as a file and imported, and nothing "
    "else of your reply is used."
)


def minatar_prompt(game):
    g = MINATAR[game]
    spec = importlib.util.find_spec("minatar")
    source = (Path(spec.origin).parent / "environments" / f"{game}.py").read_text()
    channels = "\n".join(f"  channel {k} ({name}): {text}"
                         for k, (name, text) in enumerate(zip(CHANNEL_NAMES, g["channels"])))
    parts = [
        TASK,
        g["game"]
        + f"\n\nThe game's exact implementation (the MinAtar package, version 1.0.15, file minatar/environments/"
          f"{game}.py) follows. The environment wrapper calls act(a) with a = 0..5 meaning 'n' (no-op), 'l', 'u', "
          "'r', 'd', 'f' (this file's action_map), and with probability 0.1 it repeats the previous action instead "
          "of the new one (sticky actions). The observation your function gets is built from this file's state() "
          "as described under THE INPUT.",
        f"```python\n{source}\n```",
        "THE INPUT (600 numbers, each 0 or 1): the 10 x 10 grid (rows 0 top to 9 bottom, columns 0 left to 9 right) "
        "with 6 channels. Input number i = (row * 10 + column) * 6 + channel; for example input 0 is (row 0, column "
        "0, channel 0), input 6 is (row 0, column 1, channel 0) and input 599 is (row 9, column 9, channel 5). In "
        f"this game:\n{channels}\n"
        "THE OUTPUT: your function returns the action, an integer: 0 no-op, 1 left, 2 up, 3 right, 4 down, 5 fire. "
        "With probability 0.1 the game repeats the previous action instead (sticky actions).\n"
        "How the 6 channels are built from the game's own channels (state() in the source): "
        + g["mapping"] + " A cell is 1 in a channel if any of its source channels is 1.",
        function_section(600, "index i = (row * 10 + column) * 6 + channel", 6, "millions"),
        scored_section(1000, 10000, ""),
        HOW_TO_WRITE,
    ]
    return "\n\n".join(parts) + "\n"


def low_tide_prompt():
    source = (ROOT / "src" / "small_games" / "games" / "low_tide" / "env.py").read_text()
    parts = [
        TASK,
        LOW_TIDE_GAME
        + "\n\nThe game's exact implementation (file low_tide/env.py) follows. Each episode creates "
          "LowTideEnv(sticky_prob=0.1, max_steps=1000), calls reset(seed), then step(a) with the action your "
          "function returns, a = 0..4 (sticky actions happen inside step). The observation your function gets is "
          "the one returned by reset() and step().",
        f"```python\n{source}\n```",
        LOW_TIDE_INPUT,
        function_section(403, "index i = plane * 100 + row * 10 + column, then the 3 numbers 400, 401, 402", 5,
                         "hundreds of thousands"),
        scored_section(500, 1000, " (shells collected minus jellyfish stings)"),
        HOW_TO_WRITE,
    ]
    return "\n\n".join(parts) + "\n"


def main():
    OUT.mkdir(exist_ok=True)
    prompts = {"low_tide": low_tide_prompt(), **{g: minatar_prompt(g) for g in MINATAR}}
    for game, text in prompts.items():
        path = OUT / f"{game}.txt"
        path.write_text(text)
        print(f"{path.relative_to(ROOT)}: {len(text):,} characters")


if __name__ == "__main__":
    main()
