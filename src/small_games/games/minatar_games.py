"""
The three games of the article (Experiment 5 of "Weight-space programming"):
Breakout, Seaquest and Asterix from MinAtar (Young & Tian 2019), exactly as the
article scored them.

  * The games are the official MinAtar package, version 1.0.15, run through
    its own `minatar.Environment` wrapper (sticky actions included). Nothing
    of the game logic is reimplemented here.
  * Observation: 600 numbers in {0, 1}, the 10 x 10 grid in 6 channels,
        index = (row * 10 + col) * 6 + channel     (row 0 top, col 0 left)
        channel 0 player, 1 danger, 2 reward, 3 projectile, 4 trail, 5 gauge
    Each channel is the OR of some of the game's own channels (state()); the
    mapping of each game is the one written in the article's prompts.
  * Actions: 0 no-op, 1 left, 2 up, 3 right, 4 down, 5 fire (MinAtar's
    full action set, the same for the three games).
  * Protocol: 1000 episodes, actions sampled from softmax(logits), sticky
    actions 0.1, episodes cut at 10,000 steps.
  * Network: 600 -> 32 -> 32 -> 6 with tanh, the article's architecture.

Seeding: reset(seed) seeds MinAtar's generator, which the wrapper shares
between the game and the sticky actions (MinAtar's own design), and forgets
the previous action, so an episode depends only on its seed and the policy.
"""

import numpy as np

from ..code_policy import code_baseline
from ..core import Game, GridSpec, NetSpec, Protocol, register

CHANNELS = ("player", "danger", "reward", "projectile", "trail", "gauge")
H = W = 10
OBS_DIM = H * W * len(CHANNELS)

STICKY_PROB = 0.1
MAX_STEPS = 10_000

# for every game: our channel -> the MinAtar channels OR-ed into it, and what
# the channel shows in that game (None: always empty)
SOURCES = {
    "breakout": {
        "player": (["paddle"], "the paddle"),
        "danger": ([], None),
        "reward": (["brick"], "the bricks"),
        "projectile": (["ball"], "the ball"),
        "trail": (["trail"], "the ball's previous cell"),
        "gauge": ([], None),
    },
    "seaquest": {
        "player": (["sub_front", "sub_back"], "the submarine (front and back)"),
        "danger": (["enemy_fish", "enemy_sub", "enemy_bullet"], "enemy fish, subs and torpedoes"),
        "reward": (["enemy_fish", "enemy_sub", "diver"], "enemy fish, enemy subs and divers"),
        "projectile": (["friendly_bullet"], "your torpedoes"),
        "trail": (["trail", "sub_back"], "behind enemies, divers and the submarine"),
        "gauge": (["oxygen_guage", "diver_guage"], "oxygen and divers on board (bottom row)"),
    },
    "asterix": {
        "player": (["player"], "the player"),
        "danger": (["enemy"], "the enemies"),
        "reward": (["gold"], "the treasures"),
        "projectile": ([], None),
        "trail": (["trail"], "the cell behind enemies and treasures"),
        "gauge": ([], None),
    },
}

ACTION_NAMES = ("no-op", "left", "up", "right", "down", "fire")


class MinAtarEnv:
    """A MinAtar game with the article's 6-channel observation and a step cap."""

    def __init__(self, game, sticky_prob=STICKY_PROB, max_steps=MAX_STEPS, ramping=True):
        from minatar import Environment  # imports matplotlib/seaborn: only when needed

        self.game = game
        self.env = Environment(game, sticky_action_prob=sticky_prob, difficulty_ramping=ramping)
        self.max_steps = max_steps
        own = self.env.env.channels                  # MinAtar name -> index in state()
        self._map = np.zeros((len(own), len(CHANNELS)), dtype=np.float32)
        for k, ch in enumerate(CHANNELS):
            for src in SOURCES[game][ch][0]:
                self._map[own[src], k] = 1.0
        self.t = 0
        self.score = 0.0

    @property
    def sticky_prob(self):
        return self.env.sticky_action_prob

    @sticky_prob.setter
    def sticky_prob(self, p):
        self.env.sticky_action_prob = p

    @property
    def last_action(self):
        return int(self.env.last_action)

    def _obs(self):
        s = self.env.state().reshape(H * W, -1).astype(np.float32)
        return (s @ self._map > 0).astype(np.float32).ravel()

    def _info(self):
        return {"t": self.t, "score": self.score}

    def reset(self, seed=None):
        if seed is not None:
            self.env.seed(seed)
        self.env.reset()
        self.env.last_action = 0
        self.t = 0
        self.score = 0.0
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        if not 0 <= action < len(ACTION_NAMES):
            raise ValueError(f"invalid action {action}")
        r, terminal = self.env.act(action)
        self.t += 1
        self.score += r
        truncated = not terminal and self.t >= self.max_steps
        return self._obs(), float(r), bool(terminal), truncated, self._info()


SUMMARIES = {
    "breakout": "bounce the ball with the paddle and break the bricks",
    "seaquest": "a submarine shoots enemies, rescues divers and surfaces for oxygen",
    "asterix": "pick up treasure and dodge the enemies crossing the screen",
}
HELP = {
    "breakout": "left/right arrows, space to stay still",
    "seaquest": "arrows to move, F to fire, space to stay still",
    "asterix": "arrows to move, space to stay still",
}


def _make_env_factory(name):
    def make_env(sticky_prob=STICKY_PROB, max_steps=MAX_STEPS):
        return MinAtarEnv(name, sticky_prob=sticky_prob, max_steps=max_steps)
    return make_env


for _name in ("breakout", "seaquest", "asterix"):
    register(Game(
        name=_name,
        title=f"{_name.capitalize()} (MinAtar)",
        summary=SUMMARIES[_name],
        make_env=_make_env_factory(_name),
        obs_dim=OBS_DIM,
        action_names=ACTION_NAMES,
        protocol=Protocol(episodes=1000, seed0=10_000, sticky_prob=STICKY_PROB, max_steps=MAX_STEPS),
        net=NetSpec(max_hidden_layers=2, max_hidden_units=32, hidden_activations=("tanh",)),
        baselines={"astra_code": code_baseline(_name, len(ACTION_NAMES))},
        human_keys={"space": 0, "left": 1, "up": 2, "right": 3, "down": 4, "f": 5},
        human_help=HELP[_name],
        grid=GridSpec(H, W, CHANNELS, tuple(SOURCES[_name][ch][1] for ch in CHANNELS), layout="hwc"),
    ))
