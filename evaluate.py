"""
Evaluation harness for Low Tide policies.
DO NOT give this file to the model that writes the weights: it contains a
hand-written heuristic baseline.

Usage
    python evaluate.py weights.json                      # score a weight file
    python evaluate.py weights.json --policy policy.py   # also check that the
                                                         # model's code is a plain MLP
    python evaluate.py --baseline random|safe|heuristic  # reference scores

Scoring protocol (the one described in the prompt):
    episodes 500, seeds 10_000 .. 10_499, sticky actions 0.1,
    max 1000 steps, actions sampled from softmax(logits).
    The score always uses the REFERENCE forward pass below, never the
    model's own code.

Weight file format (JSON):
{
  "layers": [
    {"in": 403, "out": 32, "activation": "relu",
     "W": {"sparse": [[out_index, in_index, value], ...]}   or  {"dense": [[...], ...]}  (out x in)
     "b": {"sparse": [[out_index, value], ...]}             or  {"dense": [...]}
    },
    ...
    {"in": 32, "out": 5, "activation": "linear", "W": ..., "b": ...}
  ]
}
Every entry not listed in a sparse block is 0.  y = act(W @ x + b).
"""

import argparse
import collections
import importlib.util
import json
import math
import sys

import numpy as np

from low_tide_env import (LowTideEnv, OBS_DIM, N_ACTIONS, H, W, CH_WATER,
                          CH_SHELL, CH_JELLY, IDX_RISING, IDX_COUNTDOWN,
                          IDX_STUN, MAX_INTERVAL, STUN_STEPS, MOVES)

ALLOWED_ACTIVATIONS = ("relu", "tanh", "linear")
MAX_HIDDEN_LAYERS = 3
MAX_HIDDEN_UNITS = 64

EVAL_EPISODES = 500
EVAL_SEED0 = 10_000


# ------------------------------------------------------------------ weights
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


def load_weights(path):
    with open(path) as f:
        data = json.load(f)
    layers = data["layers"]
    if not 1 <= len(layers) <= MAX_HIDDEN_LAYERS + 1:
        raise ValueError(f"{len(layers)} layers: allowed 1 to {MAX_HIDDEN_LAYERS + 1}")
    parsed, prev_out = [], OBS_DIM
    for li, L in enumerate(layers):
        n_in, n_out, act = int(L["in"]), int(L["out"]), L["activation"]
        last = li == len(layers) - 1
        if n_in != prev_out:
            raise ValueError(f"layer {li}: in={n_in}, but previous output is {prev_out}")
        if act not in ALLOWED_ACTIVATIONS:
            raise ValueError(f"layer {li}: activation {act!r} not in {ALLOWED_ACTIVATIONS}")
        if last:
            if n_out != N_ACTIONS or act != "linear":
                raise ValueError("last layer must have out=5 and activation 'linear'")
        elif not 1 <= n_out <= MAX_HIDDEN_UNITS:
            raise ValueError(f"layer {li}: {n_out} units, max {MAX_HIDDEN_UNITS}")
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


def sample_action(logits, rng, greedy=False):
    if greedy:
        return int(np.argmax(logits))
    z = logits - np.max(logits)
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(N_ACTIONS, p=p))


# ------------------------------------------------------------------ baselines
def random_policy(obs, rng):
    return int(rng.integers(N_ACTIONS))


def safe_policy(obs, rng):
    """Walk to row 0 and stay there: never drowns, never scores."""
    return 1


def _decode(obs):
    g = obs[:400].reshape(4, H, W)
    ar, ac = map(int, np.argwhere(g[0] > 0.5)[0])
    level = int(g[CH_WATER].sum() // W)
    rising = obs[IDX_RISING] > 0.5
    countdown = int(round(obs[IDX_COUNTDOWN] * MAX_INTERVAL))
    stun = int(round(obs[IDX_STUN] * STUN_STEPS))
    return g, ar, ac, level, rising, countdown, stun


def heuristic_policy(obs, rng):
    """BFS to the nearest shell, avoiding jellyfish and rows about to flood."""
    g, ar, ac, level, rising, countdown, stun = _decode(obs)
    deepest_ok = H - level - 1                  # lowest dry row
    if rising:
        deepest_ok -= 1 if countdown > 2 else 2 # keep a margin from rising water
    if ar > deepest_ok:
        return 1
    blocked = g[CH_JELLY] > 0.5
    if rising:                                  # a sting near the water is lethal:
        near = np.zeros_like(blocked)           # also avoid cells next to jellyfish
        near[1:] |= blocked[:-1]; near[:-1] |= blocked[1:]
        near[:, 1:] |= blocked[:, :-1]; near[:, :-1] |= blocked[:, 1:]
        blocked = blocked | near
        blocked[ar, ac] = False
    start = (ar, ac)
    prev = {start: None}
    q = collections.deque([start])
    goal = None
    while q:
        cur = q.popleft()
        if g[CH_SHELL][cur] > 0.5 and cur != start:
            goal = cur
            break
        for a in (1, 2, 3, 4):
            dr, dc = MOVES[a]
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr <= deepest_ok and 0 <= nc < W and (nr, nc) not in prev and not blocked[nr, nc]:
                prev[(nr, nc)] = (cur, a)
                q.append((nr, nc))
    if goal is None:
        # wait just above the waterline for the next receding row
        wait_row = max(0, deepest_ok - 1)
        return 2 if ar < wait_row else (1 if ar > wait_row else 0)
    node = goal
    while prev[node][0] != start:
        node = prev[node][0]
    return prev[node][1]


# ------------------------------------------------------------------ evaluation
def evaluate(policy_fn, episodes=EVAL_EPISODES, seed0=EVAL_SEED0, sticky=0.1):
    env = LowTideEnv(sticky_prob=sticky)
    rng = np.random.default_rng(12345)
    returns, shells, stings, drowned, lengths = [], [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed0 + ep)
        total = 0.0
        while True:
            obs, r, term, trunc, info = env.step(policy_fn(obs, rng))
            total += r
            if term or trunc:
                break
        returns.append(total)
        shells.append(info["shells"])
        stings.append(info["stings"])
        drowned.append(info["drowned"])
        lengths.append(info["t"])
    R = np.array(returns)
    return {
        "mean_return": float(R.mean()),
        "stderr": float(R.std(ddof=1) / math.sqrt(len(R))) if len(R) > 1 else 0.0,
        "mean_shells": float(np.mean(shells)),
        "mean_stings": float(np.mean(stings)),
        "drown_rate": float(np.mean(drowned)),
        "mean_length": float(np.mean(lengths)),
    }


def collect_states(n=3000, seed=777):
    """Real game states (from a mix of policies) for the equivalence check."""
    env = LowTideEnv(sticky_prob=0.1)
    rng = np.random.default_rng(seed)
    states, ep = [], 0
    while len(states) < n:
        obs, _ = env.reset(seed=seed + ep)
        ep += 1
        pol = heuristic_policy if ep % 2 else random_policy
        while len(states) < n:
            states.append(obs)
            obs, _, term, trunc, _ = env.step(pol(obs, rng))
            if term or trunc:
                break
    return np.array(states)


def check_policy_code(path, weights_path, ref):
    spec = importlib.util.spec_from_file_location("model_policy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pol = mod.Policy(weights_path)
    states = collect_states()
    rng = np.random.default_rng(0)
    rand = (rng.random((1000, OBS_DIM)) < 0.15).astype(np.float32)
    worst = 0.0
    for x in np.concatenate([states, rand]):
        a = np.asarray(pol.logits(x), dtype=np.float64).reshape(-1)
        b = ref.logits(x)
        if a.shape != b.shape:
            return False, f"Policy.logits returned shape {a.shape}, expected {b.shape}"
        worst = max(worst, float(np.max(np.abs(a - b))))
    ok = worst < 1e-4
    return ok, f"max |logits(policy.py) - logits(reference)| = {worst:.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?")
    ap.add_argument("--policy", help="the model's policy.py, checked against the reference MLP")
    ap.add_argument("--baseline", choices=("random", "safe", "heuristic"))
    ap.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    ap.add_argument("--greedy", action="store_true", help="argmax instead of sampling (extra info)")
    args = ap.parse_args()

    if args.baseline:
        fn = {"random": random_policy, "safe": safe_policy, "heuristic": heuristic_policy}[args.baseline]
        res = evaluate(fn, args.episodes)
        print(f"baseline {args.baseline}: {json.dumps(res, indent=2)}")
        return
    if not args.weights:
        ap.error("give a weights file or --baseline")

    ref = ReferenceMLP(load_weights(args.weights))
    sizes = [OBS_DIM] + [Wm.shape[0] for Wm, _, _ in ref.layers]
    print("architecture:", " -> ".join(map(str, sizes)),
          "|", ", ".join(a for _, _, a in ref.layers))
    print(f"parameters: {ref.n_params()}  non-zero: {ref.n_nonzero()}")

    if args.policy:
        ok, msg = check_policy_code(args.policy, args.weights, ref)
        print(("OK   " if ok else "FAIL ") + "policy.py equivalence:", msg)

    res = evaluate(lambda o, r: sample_action(ref.logits(o), r, args.greedy), args.episodes)
    mode = "greedy" if args.greedy else "sampled"
    print(f"score ({mode}, {args.episodes} episodes): {json.dumps(res, indent=2)}")


if __name__ == "__main__":
    main()
