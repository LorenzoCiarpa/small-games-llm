"""
Weight files and the reference forward pass.

Format (JSON), the same for every game:
{
  "layers": [
    {"in": <obs_dim>, "out": 32, "activation": "relu",
     "W": {"sparse": [[out_index, in_index, value], ...]}   or  {"dense": [[...], ...]}  (out x in)
     "b": {"sparse": [[out_index, value], ...]}             or  {"dense": [...]}
    },
    ...
    {"in": 32, "out": <n_actions>, "activation": "linear", "W": ..., "b": ...}
  ]
}
Every entry not listed in a sparse block is 0.  y = act(W @ x + b).
Scores always use the reference forward pass below, never the model's own code.
"""

from __future__ import annotations

import json

import numpy as np

from .core import Game


def _parse_matrix(spec, rows, cols, name):
    if not isinstance(spec, dict) or len(spec) != 1 or next(iter(spec)) not in ("sparse", "dense"):
        raise ValueError(f"{name}: expected {{'sparse': ...}} or {{'dense': ...}}")
    if "dense" in spec:
        m = np.asarray(spec["dense"], dtype=np.float64)
        if m.shape != (rows, cols):
            raise ValueError(f"{name}: dense shape {m.shape}, expected {(rows, cols)}")
        return m
    m = np.zeros((rows, cols), dtype=np.float64)
    seen = set()
    for k, entry in enumerate(spec["sparse"]):
        if len(entry) != 3:
            raise ValueError(f"{name}: sparse entry #{k} {entry} is not [out, in, value]")
        i, j, v = int(entry[0]), int(entry[1]), float(entry[2])
        if not (0 <= i < rows and 0 <= j < cols):
            raise ValueError(f"{name}: sparse entry #{k} index ({i},{j}) out of range {(rows, cols)}")
        if (i, j) in seen:
            raise ValueError(f"{name}: duplicate sparse entry ({i},{j})")
        seen.add((i, j))
        m[i, j] = v
    return m


def _parse_vector(spec, n, name):
    if not isinstance(spec, dict) or len(spec) != 1 or next(iter(spec)) not in ("sparse", "dense"):
        raise ValueError(f"{name}: expected {{'sparse': ...}} or {{'dense': ...}}")
    if "dense" in spec:
        v = np.asarray(spec["dense"], dtype=np.float64)
        if v.shape != (n,):
            raise ValueError(f"{name}: dense length {v.shape}, expected {n}")
        return v
    v = np.zeros(n, dtype=np.float64)
    seen = set()
    for k, entry in enumerate(spec["sparse"]):
        if len(entry) != 2:
            raise ValueError(f"{name}: sparse entry #{k} {entry} is not [index, value]")
        i, x = int(entry[0]), float(entry[1])
        if not 0 <= i < n:
            raise ValueError(f"{name}: sparse index {i} out of range {n}")
        if i in seen:
            raise ValueError(f"{name}: duplicate sparse entry {i}")
        seen.add(i)
        v[i] = x
    return v


def load_weights(path, game: Game):
    """Read a weight file and check it against the game's input, output and NetSpec."""
    with open(path) as f:
        data = json.load(f)
    spec = game.net
    layers = data["layers"]
    if not 1 <= len(layers) <= spec.max_hidden_layers + 1:
        raise ValueError(f"{len(layers)} layers: allowed 1 to {spec.max_hidden_layers + 1}")
    parsed, prev_out = [], game.obs_dim
    for li, L in enumerate(layers):
        n_in, n_out, act = int(L["in"]), int(L["out"]), L["activation"]
        last = li == len(layers) - 1
        if n_in != prev_out:
            raise ValueError(f"layer {li}: in={n_in}, but previous output is {prev_out}")
        if last:
            if n_out != game.n_actions or act != "linear":
                raise ValueError(f"last layer must have out={game.n_actions} and activation 'linear'")
        else:
            if act not in spec.hidden_activations:
                raise ValueError(f"layer {li}: activation {act!r} not in {spec.hidden_activations}")
            if not 1 <= n_out <= spec.max_hidden_units:
                raise ValueError(f"layer {li}: {n_out} units, max {spec.max_hidden_units}")
        Wm = _parse_matrix(L["W"], n_out, n_in, f"layer {li} W")
        bv = _parse_vector(L["b"], n_out, f"layer {li} b")
        if not (np.all(np.isfinite(Wm)) and np.all(np.isfinite(bv))):
            raise ValueError(f"layer {li}: non-finite values")
        parsed.append((Wm, bv, act))
        prev_out = n_out
    return parsed


class ReferenceMLP:
    def __init__(self, layers):
        self.layers = layers

    def logits(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        for Wm, b, act in self.layers:
            x = Wm @ x + b
            if act == "relu":
                x = np.maximum(x, 0.0)
            elif act == "tanh":
                x = np.tanh(x)
        return x

    def n_params(self):
        return sum(Wm.size + b.size for Wm, b, _ in self.layers)

    def n_nonzero(self):
        return sum(int(np.count_nonzero(Wm) + np.count_nonzero(b)) for Wm, b, _ in self.layers)

    def describe(self, obs_dim):
        sizes = [obs_dim] + [Wm.shape[0] for Wm, _, _ in self.layers]
        return (" -> ".join(map(str, sizes)) + " | " + ", ".join(a for _, _, a in self.layers)
                + f" | {self.n_params()} parameters, {self.n_nonzero()} non-zero")
