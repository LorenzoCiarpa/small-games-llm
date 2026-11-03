"""
Low Tide policy written by Claude: a plain MLP whose weights are read from
weights.json. There is no game logic here: the behaviour lives entirely in
the weights (403 -> 46 relu -> 36 tanh -> 5 linear).
"""

import json

import numpy as np


def _matrix(spec, rows, cols):
    if "dense" in spec:
        return np.asarray(spec["dense"], dtype=np.float64).reshape(rows, cols)
    m = np.zeros((rows, cols), dtype=np.float64)
    for i, j, v in spec["sparse"]:
        m[int(i), int(j)] = float(v)
    return m


def _vector(spec, n):
    if "dense" in spec:
        return np.asarray(spec["dense"], dtype=np.float64).reshape(n)
    v = np.zeros(n, dtype=np.float64)
    for i, x in spec["sparse"]:
        v[int(i)] = float(x)
    return v


_ACTIVATIONS = {
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
    "linear": lambda x: x,
}


class Policy:
    def __init__(self, weights_path):
        with open(weights_path) as f:
            data = json.load(f)
        self.layers = []
        for layer in data["layers"]:
            n_in, n_out = int(layer["in"]), int(layer["out"])
            W = _matrix(layer["W"], n_out, n_in)
            b = _vector(layer["b"], n_out)
            self.layers.append((W, b, _ACTIVATIONS[layer["activation"]]))

    def logits(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        for W, b, act in self.layers:
            x = act(W @ x + b)
        return x

    def act(self, obs, rng):
        z = self.logits(obs)
        p = np.exp(z - z.max())
        p /= p.sum()
        return int(rng.choice(len(p), p=p))
