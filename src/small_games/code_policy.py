"""
Policies written as code by an LLM, used as baselines (experiments/code_policies/).

Each answer is a file with a function `policy(obs, rng) -> action`. `CodePolicy` wraps it with the rules of the
prompt: an exception or an invalid action counts as action 0. The file is imported lazily, in the process that plays,
so the object can be sent to the evaluation workers.

The code comes from an LLM: read it before running it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# relative to the repository root, like WEIGHTS_DIR in policy.py
CODE_DIR = Path("experiments/code_policies/answers")


def load_module(path, name="llm_policy"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CodePolicy:
    """The function policy(obs, rng) of a file; errors and invalid actions become 0. Picklable."""

    def __init__(self, path, n_actions):
        self.path, self.n_actions, self._fn = str(path), n_actions, None

    def __getstate__(self):
        return {"path": self.path, "n_actions": self.n_actions, "_fn": None}

    def __call__(self, obs, rng):
        if self._fn is None:
            self._fn = load_module(self.path).policy
        try:
            a = int(self._fn(obs.copy(), rng))
            return a if 0 <= a < self.n_actions else 0
        except Exception:
            return 0


def code_baseline(game_name: str, n_actions: int, model: str = "gpt-6-astra") -> CodePolicy:
    """The code answer of `model` for a game, as a baseline."""
    pol = CodePolicy(CODE_DIR / model / game_name / "policy.py", n_actions)
    pol.__doc__ = (f"the function policy(obs, rng) written as code by {model}, without tools "
                   f"(experiments/code_policies/answers/{model}/{game_name}/)")
    return pol
