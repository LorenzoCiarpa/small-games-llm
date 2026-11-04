"""Low Tide: the new game written for this project (see env.py for the rules)."""

from ...core import Game, NetSpec, Protocol, register
from .baselines import heuristic_policy, safe_policy
from .env import ACTION_NAMES, MAX_STEPS, OBS_DIM, STICKY_PROB, LowTideEnv


def make_env(sticky_prob=STICKY_PROB, max_steps=MAX_STEPS):
    return LowTideEnv(sticky_prob=sticky_prob, max_steps=max_steps)


GAME = register(Game(
    name="low_tide",
    title="Low Tide",
    summary="collect shells on a beach between the tides, avoiding jellyfish and the rising water",
    make_env=make_env,
    obs_dim=OBS_DIM,
    action_names=("fermo", "su", "giù", "sinistra", "destra"),
    protocol=Protocol(episodes=500, seed0=10_000, sticky_prob=STICKY_PROB, max_steps=MAX_STEPS),
    net=NetSpec(max_hidden_layers=3, max_hidden_units=64, hidden_activations=("relu", "tanh", "linear")),
    baselines={"heuristic": heuristic_policy, "safe": safe_policy},
    human_keys={"space": 0, "up": 1, "down": 2, "left": 3, "right": 4},
    human_help="frecce per muoverti, spazio per restare fermo",
    renderer="small_games.games.low_tide.render:LowTideRenderer",
    episode_stats=("shells", "stings", "drowned"),
))

assert len(ACTION_NAMES) == GAME.n_actions
