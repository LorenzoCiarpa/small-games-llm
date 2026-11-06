"""
Game descriptions and the game registry.

A game is a `Game` record: how to build its environment, the shape of the
observation and of the action space, the scoring protocol, the limits on the
networks it accepts, its baseline policies and how a human plays it.

Adding a game: put a module (or a package) in `small_games/games/` that calls
`register(Game(...))` at import time. Every module in that folder is imported
automatically, so `evaluate` and `play` pick the new game up without changes.

The environment returned by `Game.make_env` must follow this interface:

    env.reset(seed=None) -> (obs, info)
    env.step(action)     -> (obs, reward, terminated, truncated, info)
    env.sticky_prob      probability of repeating the previous action (settable)
    env.last_action      the action actually executed at the last step
    env.max_steps        the episode is truncated after this many steps
    env.t                steps done in the current episode

`obs` is a flat float vector of length `Game.obs_dim`; `info` is a dict, and
the keys listed in `Game.episode_stats` are averaged by `evaluate` at the end
of each episode.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

PolicyFn = Callable[[np.ndarray, np.random.Generator], int]


@dataclass(frozen=True)
class Protocol:
    """How a policy is scored: `episodes` episodes with seeds seed0, seed0+1, ..."""
    episodes: int
    seed0: int
    sticky_prob: float = 0.1
    max_steps: int = 1000


@dataclass(frozen=True)
class NetSpec:
    """Limits on the MLP a weight file may describe (the output layer is always linear)."""
    max_hidden_layers: int
    max_hidden_units: int
    hidden_activations: tuple[str, ...] = ("relu", "tanh", "linear")


@dataclass(frozen=True)
class GridSpec:
    """How the observation encodes a grid of binary channels.

    layout "hwc": index = (row * width + col) * n_channels + channel
    layout "chw": index = (channel * height + row) * width + col
    `labels[k]` describes channel k in this game (None: always empty). Used by
    the generic renderer of `play`.
    """
    height: int
    width: int
    channels: tuple[str, ...]
    labels: tuple[str | None, ...]
    layout: str = "hwc"

    @property
    def size(self) -> int:
        return self.height * self.width * len(self.channels)

    def planes(self, obs: np.ndarray) -> np.ndarray:
        """The grid part of `obs` as an array (channel, row, col)."""
        g = np.asarray(obs[: self.size]) > 0.5
        n = len(self.channels)
        if self.layout == "hwc":
            return g.reshape(self.height, self.width, n).transpose(2, 0, 1)
        if self.layout == "chw":
            return g.reshape(n, self.height, self.width)
        raise ValueError(f"unknown grid layout {self.layout!r}")


@dataclass(frozen=True)
class Game:
    name: str                              # used on the command line
    title: str
    summary: str
    make_env: Callable[..., Any]           # make_env(sticky_prob=..., max_steps=...) -> env
    obs_dim: int
    action_names: tuple[str, ...]          # display labels, shown by `play`
    protocol: Protocol
    net: NetSpec
    baselines: dict[str, PolicyFn] = field(default_factory=dict)  # "random" is always added
    human_keys: dict[str, int] = field(default_factory=dict)      # pygame key name -> action
    human_help: str = ""
    grid: GridSpec | None = None
    renderer: str | None = None            # "module:Class" of a custom renderer, else generic
    episode_stats: tuple[str, ...] = ()

    @property
    def n_actions(self) -> int:
        return len(self.action_names)


_REGISTRY: dict[str, Game] = {}
_discovered = False


def register(game: Game) -> Game:
    if game.name in _REGISTRY:
        raise ValueError(f"game {game.name!r} registered twice")
    _REGISTRY[game.name] = game
    return game


def _discover() -> None:
    global _discovered
    if _discovered:
        return
    _discovered = True
    from . import games
    for mod in pkgutil.iter_modules(games.__path__):
        importlib.import_module(f"{games.__name__}.{mod.name}")


def list_games() -> list[str]:
    _discover()
    return sorted(_REGISTRY)


def get_game(name: str) -> Game:
    _discover()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown game {name!r}; available: {', '.join(sorted(_REGISTRY))}") from None
