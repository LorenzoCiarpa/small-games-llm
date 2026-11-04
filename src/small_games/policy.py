"""Policies shared by `evaluate` and `play`: the MLP of a weight file and the baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import Game, PolicyFn, get_game, list_games
from .weights import ReferenceMLP, load_weights

WEIGHTS_DIR = Path("weights")
POLICY_SEED_SALT = 0x5EED


def policy_rng(seed):
    """The generator a policy samples from in the episode with this seed.

    `evaluate` and `play` use the same one, so the episode with seed s is the
    same in both (same game, same sticky actions, same sampled actions).
    """
    return np.random.default_rng([seed, POLICY_SEED_SALT])


def softmax(logits):
    p = np.exp(logits - np.max(logits))
    return p / p.sum()


def sample_action(logits, rng, greedy=False):
    """An action drawn from softmax(logits), or the argmax if greedy."""
    if greedy:
        return int(np.argmax(logits))
    c = np.cumsum(np.exp(logits - np.max(logits)))
    return int(np.searchsorted(c, rng.random() * c[-1], side="right"))


class MLPPolicy:
    def __init__(self, mlp: ReferenceMLP, greedy=False):
        self.mlp = mlp
        self.greedy = greedy

    def __call__(self, obs, rng):
        return sample_action(self.mlp.logits(obs), rng, self.greedy)


class RandomPolicy:
    """A uniformly random action at every step."""

    def __init__(self, n_actions):
        self.n_actions = n_actions

    def __call__(self, obs, rng):
        return int(rng.integers(self.n_actions))


def baselines(game: Game) -> dict[str, PolicyFn]:
    """The game's baselines, with "random" always first."""
    return {"random": RandomPolicy(game.n_actions), **game.baselines}


def available_weights(game: Game) -> list[str]:
    folder = WEIGHTS_DIR / game.name
    return sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []


def resolve_weights(game: Game, arg: str) -> Path:
    """A path to a weight file, or the name of a file in weights/<game>/ (without .json)."""
    p = Path(arg)
    if p.is_file():
        return p
    q = WEIGHTS_DIR / game.name / f"{arg}.json"
    if q.is_file():
        return q
    names = ", ".join(available_weights(game)) or "none"
    raise FileNotFoundError(f"no weight file {arg!r} (in weights/{game.name}/: {names})")


def load_mlp(game: Game, arg: str) -> tuple[Path, ReferenceMLP]:
    path = resolve_weights(game, arg)
    return path, ReferenceMLP(load_weights(path, game))


def _doc(fn):
    doc = getattr(fn, "__doc__", None) or ""
    return doc.strip().splitlines()[0] if doc.strip() else ""


def games_overview(verbose=False) -> str:
    """Per game: its baselines and the weight files in weights/<game>/ (for --help and --list)."""
    lines = []
    for name in list_games():
        g = get_game(name)
        bl = baselines(g)
        w = ", ".join(available_weights(g)) or "-"
        if not verbose:
            lines.append(f"  {name:10s} baseline: {', '.join(bl):28s} weights: {w}")
            continue
        p = g.protocol
        lines += [f"{name}  ({g.title}): {g.summary}",
                  f"  obs {g.obs_dim}, actions {g.n_actions} ({', '.join(g.action_names)})",
                  f"  protocol: {p.episodes} episodes from seed {p.seed0}, sticky {p.sticky_prob}, "
                  f"max {p.max_steps} steps",
                  f"  network: <= {g.net.max_hidden_layers} hidden layers of <= {g.net.max_hidden_units} "
                  f"units, activations {', '.join(g.net.hidden_activations)}",
                  "  baselines:"]
        width = max(len(b) for b in bl)
        lines += [f"    {b:{width}s}  {_doc(fn)}" for b, fn in bl.items()]
        lines += [f"  weights/{name}/: {w}", ""]
    return "\n".join(lines).rstrip()
