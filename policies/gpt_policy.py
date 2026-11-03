import json
import numpy as np


class Policy:
    def __init__(self, weights_path):
        with open(weights_path, "r", encoding="utf-8") as f:
            specification = json.load(f)

        self.layers = []

        for layer in specification["layers"]:
            if "dense" in layer["W"]:
                W = np.asarray(layer["W"]["dense"], dtype=np.float64)
            else:
                W = np.zeros(
                    (layer["out"], layer["in"]),
                    dtype=np.float64,
                )
                for out_index, in_index, value in layer["W"]["sparse"]:
                    W[out_index, in_index] = value

            if "dense" in layer["b"]:
                b = np.asarray(layer["b"]["dense"], dtype=np.float64)
            else:
                b = np.zeros(layer["out"], dtype=np.float64)
                for index, value in layer["b"]["sparse"]:
                    b[index] = value

            self.layers.append((W, b, layer["activation"]))

    def logits(self, obs):
        x = np.asarray(obs, dtype=np.float64)

        for W, b, activation in self.layers:
            x = W @ x + b

            if activation == "relu":
                x = np.maximum(x, 0.0)
            elif activation == "tanh":
                x = np.tanh(x)

        return x
