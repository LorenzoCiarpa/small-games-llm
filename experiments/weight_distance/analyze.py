"""
How far apart are the networks that LLMs wrote by hand? (Italian write-up in README.md)

    uv run --group analysis python experiments/weight_distance/analyze.py
    uv run --group analysis python experiments/weight_distance/analyze.py --episodes 50   # quicker

Run from the repository root. Writes results/report.md (the tables) and
results/results.json (every number). Everything is seeded: the same command
gives the same numbers.

Networks:
  astra/{breakout,seaquest,asterix}  600-32-32-6 tanh, written by GPT-6 Astra (the article)
  claude/low_tide, gpt/low_tide      403 -> ... -> 5, written for Low Tide by Claude Opus and
                                     GPT-6 Astra (weights/low_tide/)
  ppo/{breakout,pong,pacman,freeway}-grid
                                     500-32-32-5 tanh, trained with PPO for 1e9 steps (the
                                     article's Experiment 1): the reference for what a
                                     *trained* network of this size looks like

Measures:
  1. fingerprint of every layer: density, distinct values, round values, norms
  2. which inputs (channels) the first layer reads
  3. weight-space distance between networks of the same shape, layer by layer:
     raw, after aligning the hidden units (the best signed permutation, since
     reordering hidden units or flipping the sign of a tanh unit does not change
     the function), and against a null model (the same weights shuffled)
  4. first-layer units: does a unit of one network have a twin in the other?
  5. representations (linear CKA) of every layer pair on the same states
  6. behaviour: same action on the same states, and cross-play (the network
     of one game playing another game)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from small_games.core import get_game
from small_games.evaluate import evaluate
from small_games.policy import MLPPolicy, RandomPolicy, policy_rng
from small_games.weights import ReferenceMLP, load_weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 1234

# input layout of each observation space: channel names, index layout, trailing scalars
SPACES = {
    "minatar": dict(channels=("player", "danger", "reward", "projectile", "trail", "gauge"), layout="hwc",
                    scalars=()),
    "gridworld": dict(channels=("agent", "moving", "reward", "obstacle", "previous"), layout="hwc", scalars=()),
    "low_tide": dict(channels=("agent", "water", "shell", "jelly"), layout="chw",
                     scalars=("rising", "countdown", "stun")),
}
ODD = {"tanh", "linear"}          # f(-x) = -f(x): flipping a unit's sign is a symmetry


# ====================================================================== networks
@dataclass
class Net:
    name: str
    author: str
    game: str
    space: str
    layers: list                  # [(W out x in, b, activation)]

    @property
    def shape(self):
        return [self.layers[0][0].shape[1]] + [W.shape[0] for W, _, _ in self.layers]

    @property
    def arch(self):
        acts = sorted({a for _, _, a in self.layers[:-1]})
        return "-".join(map(str, self.shape)) + " " + "/".join(acts)

    def forward(self, X):
        """Post-activation output of every layer for a batch X (n x inputs)."""
        outs, h = [], X
        for W, b, act in self.layers:
            h = h @ W.T + b
            if act == "relu":
                h = np.maximum(h, 0.0)
            elif act == "tanh":
                h = np.tanh(h)
            outs.append(h)
        return outs

    def policy(self):
        return MLPPolicy(ReferenceMLP(self.layers))


def load_nets():
    nets = []
    for game in ("breakout", "seaquest", "asterix"):
        layers = load_weights(ROOT / "weights" / game / "paper_astra.json", get_game(game))
        nets.append(Net(f"astra/{game}", "GPT-6 Astra", game, "minatar", layers))
    for who, author in (("claude", "Claude Opus"), ("gpt", "GPT-6 Astra")):
        layers = load_weights(ROOT / "weights" / "low_tide" / f"{who}.json", get_game("low_tide"))
        nets.append(Net(f"{who}/low_tide", author, "low_tide", "low_tide", layers))
    for game in ("breakout", "pong", "pacman", "freeway"):
        with np.load(HERE / "reference" / f"ppo_{game}.npz", allow_pickle=False) as z:
            layers = [(z[f"W{k}"].astype(np.float64), z[f"b{k}"].astype(np.float64), act)
                      for k, act in ((1, "tanh"), (2, "tanh"), (3, "linear"))]
        nets.append(Net(f"ppo/{game}-grid", "PPO", game, "gridworld", layers))
    return nets


def flat(layer):
    W, b, _ = layer
    return np.concatenate([W.ravel(), b])


# ====================================================================== 1. fingerprint
def fingerprint(net):
    rows = []
    L = len(net.layers)
    for k, (W, b, act) in enumerate(net.layers):
        p = flat((W, b, act))
        nz = p[p != 0]
        row = dict(layer=k + 1, shape=f"{W.shape[1]}->{W.shape[0]}", activation=act, params=int(p.size),
                   nonzero=int(nz.size), density=nz.size / p.size, norm=float(np.linalg.norm(p)),
                   max_abs=float(np.abs(p).max()),
                   distinct=int(np.unique(np.round(nz, 6)).size),
                   round2=float(np.mean(np.abs(nz * 100 - np.round(nz * 100)) < 1e-3)) if nz.size else float("nan"),
                   stable_rank=float(np.linalg.norm(W) ** 2 / np.linalg.norm(W, 2) ** 2) if W.any() else 0.0)
        if k < L - 1:   # hidden layer: units that read something and are read by the next layer
            reads = W.any(axis=1) | (b != 0)
            read_by = net.layers[k + 1][0].any(axis=0)
            row["used_units"] = int(np.sum(reads & read_by))
            row["units"] = int(W.shape[0])
        rows.append(row)
    p = np.concatenate([flat(l) for l in net.layers])
    nz = p[p != 0]
    vals, counts = np.unique(np.round(nz, 6), return_counts=True)
    top = sorted(zip(counts.tolist(), vals.tolist()), reverse=True)[:8]
    total = dict(params=int(p.size), nonzero=int(nz.size), density=nz.size / p.size,
                 distinct=int(vals.size), round2=float(np.mean(np.abs(nz * 100 - np.round(nz * 100)) < 1e-3)),
                 top_values=[(v, c) for c, v in top],
                 top8_share=float(sum(c for c, _ in top) / nz.size))
    return dict(layers=rows, total=total)


# ====================================================================== 2. input usage
def input_groups(space, n_in):
    s = SPACES[space]
    C, names = len(s["channels"]), list(s["channels"]) + list(s["scalars"])
    g = np.empty(n_in, dtype=int)
    grid = n_in - len(s["scalars"])
    idx = np.arange(grid)
    g[:grid] = idx % C if s["layout"] == "hwc" else idx // (grid // C)
    g[grid:] = C + np.arange(len(s["scalars"]))
    return names, g


def input_usage(net):
    W = net.layers[0][0]
    names, g = input_groups(net.space, W.shape[1])
    mass = np.abs(W).sum(axis=0)
    conn = (W != 0).sum(axis=0)
    share = {n: float(mass[g == i].sum() / mass.sum()) for i, n in enumerate(names)}
    conn_share = {n: float(conn[g == i].sum() / conn.sum()) for i, n in enumerate(names)}
    used = W.any(axis=0)
    return dict(share=share, conn_share=conn_share, inputs_read=int(used.sum()), inputs=int(W.shape[1]))


def describe_unit(net, u):
    """What a first-layer unit reads: channels and number of inputs."""
    w = net.layers[0][0][u]
    names, g = input_groups(net.space, w.size)
    idx = np.nonzero(w)[0]
    return f"{len(idx)} ingressi: " + ", ".join(sorted({names[g[i]] for i in idx}))


# ====================================================================== 3. weight-space distance
def permute(layers, perms, signs):
    """The same network with hidden units reordered (perms) and sign-flipped (signs)."""
    out, L = [], len(layers)
    for k, (W, b, act) in enumerate(layers):
        if k > 0:
            W = W[:, perms[k - 1]] * signs[k - 1][None, :]
        if k < L - 1:
            W, b = W[perms[k]] * signs[k][:, None], b[perms[k]] * signs[k]
        out.append((W, b, act))
    return out


def _scores(A, B, perms, signs, k):
    """S[i, j]: inner product gained by putting B's unit j of hidden layer k at A's position i."""
    WB = B[k][0]
    if k > 0:
        WB = WB[:, perms[k - 1]] * signs[k - 1][None, :]
    S = A[k][0] @ WB.T + np.outer(A[k][1], B[k][1])
    WB2 = B[k + 1][0]
    if k + 1 < len(B) - 1:
        WB2 = WB2[perms[k + 1]] * signs[k + 1][:, None]
    return S + A[k + 1][0].T @ WB2


def inner(A, B):
    return sum(float(flat(a) @ flat(b)) for a, b in zip(A, B))


def align(A, B, rng, restarts=8):
    """B with its hidden units reordered (and, for odd activations, sign-flipped) to be as
    close as possible to A in L2: weight matching (Ainsworth et al. 2022, Git Re-Basin),
    coordinate ascent over layers with the Hungarian algorithm, best of several restarts."""
    H = len(A) - 1
    sizes = [A[k][0].shape[0] for k in range(H)]

    def assign(S, k):
        if B[k][2] in ODD:        # the symmetries are those of B, the network that is transformed
            sg, C = np.where(S >= 0, 1.0, -1.0), np.abs(S)
        else:
            sg, C = np.ones_like(S), S
        _, col = linear_sum_assignment(C, maximize=True)
        return col, sg[np.arange(len(col)), col]

    best = None
    for r in range(restarts):
        if r == 0:      # layer by layer from the input, on incoming weights only
            perms, signs = [], []
            for k in range(H):
                WB = B[k][0] if k == 0 else B[k][0][:, perms[k - 1]] * signs[k - 1][None, :]
                p, s = assign(A[k][0] @ WB.T + np.outer(A[k][1], B[k][1]), k)
                perms.append(p)
                signs.append(s)
        else:           # identity, then random starts
            perms = [np.arange(n) if r == 1 else rng.permutation(n) for n in sizes]
            signs = [np.ones(n) for n in sizes]
        obj = inner(A, permute(B, perms, signs))
        for _ in range(100):
            for k in rng.permutation(H):
                perms[k], signs[k] = assign(_scores(A, B, perms, signs, k), k)
            new = inner(A, permute(B, perms, signs))
            if new <= obj + 1e-12:
                break
            obj = new
        if best is None or obj > best[0]:
            best = (obj, permute(B, perms, signs))
    return best[1]


def distances(A, B):
    """Per layer and in total: L2 distance, norms, relative distance and cosine."""
    def one(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        d = np.linalg.norm(a - b)
        return dict(norm_a=float(na), norm_b=float(nb), l2=float(d),
                    rel=float(d / np.sqrt(na ** 2 + nb ** 2)), cos=float(a @ b / (na * nb)))
    rows = [one(flat(a), flat(b)) for a, b in zip(A, B)]
    rows.append(one(np.concatenate([flat(a) for a in A]), np.concatenate([flat(b) for b in B])))
    return rows                                   # last row: the whole network


def shuffled(layers, rng):
    """Same values, same sparsity, every entry moved to a random place: no structure left."""
    return [(rng.permutation(W.ravel()).reshape(W.shape), rng.permutation(b), act) for W, b, act in layers]


def scrambled(layers, rng):
    """A random signed permutation of the hidden units: the same function."""
    H = len(layers) - 1
    perms = [rng.permutation(layers[k][0].shape[0]) for k in range(H)]
    signs = [rng.choice([-1.0, 1.0], layers[k][0].shape[0]) if layers[k][2] in ODD
             else np.ones(layers[k][0].shape[0]) for k in range(H)]
    return permute(layers, perms, signs)


def weight_distance(A, B, rng, null_reps, restarts):
    raw = distances(A.layers, B.layers)
    al = distances(A.layers, align(A.layers, B.layers, rng, restarts))
    nulls = [distances(A.layers, align(A.layers, shuffled(B.layers, rng), rng, restarts))
             for _ in range(null_reps)]
    null = [dict(rel=float(np.mean([n[i]["rel"] for n in nulls])), rel_sd=float(np.std([n[i]["rel"] for n in nulls])),
                 cos=float(np.mean([n[i]["cos"] for n in nulls])))
            for i in range(len(raw))]
    return dict(a=A.name, b=B.name, raw=raw, aligned=al, null=null)


def self_check(nets, rng, restarts):
    """Aligning a network with a scrambled copy of itself must give distance 0."""
    return max(distances(n.layers, align(n.layers, scrambled(n.layers, rng), rng, restarts))[-1]["rel"]
               for n in nets)


# ====================================================================== 4. first-layer units
def unit_match(A, B, rng, null_reps):
    """Match the first-layer units of A and B by the cosine of their input weights."""
    WA, WB = A.layers[0][0], B.layers[0][0]
    signed = A.layers[0][2] in ODD and B.layers[0][2] in ODD
    ua, ub = WA[WA.any(axis=1)], WB[WB.any(axis=1)]
    ua = ua / np.linalg.norm(ua, axis=1, keepdims=True)
    ub = ub / np.linalg.norm(ub, axis=1, keepdims=True)

    ia, ib = np.where(WA.any(axis=1))[0], np.where(WB.any(axis=1))[0]

    def matched(X, Y, pairs=False):
        C = X @ Y.T
        C = np.abs(C) if signed else C
        r, c = linear_sum_assignment(C, maximize=True)
        order = np.argsort(-C[r, c])
        return [(float(C[r[k], c[k]]), int(ia[r[k]]), int(ib[c[k]])) for k in order] if pairs else C[r, c][order]

    best = matched(ua, ub, pairs=True)
    m = np.array([x for x, _, _ in best])
    null = [matched(ua, ub[:, rng.permutation(ub.shape[1])]) for _ in range(null_reps)]
    twins = [dict(cos=x, unit_a=i, unit_b=j, reads_a=describe_unit(A, i), reads_b=describe_unit(B, j))
             for x, i, j in best if x > 0.9]
    return dict(a=A.name, b=B.name, units_a=int(len(ua)), units_b=int(len(ub)), pairs=int(len(m)),
                mean=float(m.mean()), median=float(np.median(m)),
                over_09=int(np.sum(m > 0.9)), over_05=int(np.sum(m > 0.5)),
                null_mean=float(np.mean([n.mean() for n in null])),
                null_over_05=float(np.mean([np.sum(n > 0.5) for n in null])),
                signed=signed, top=[float(x) for x in m[:5]], twins=twins)


# ====================================================================== states, CKA, behaviour
def collect_states(game_name, policy, episodes, seed0, every=4, max_steps=2000, cap=2000):
    game = get_game(game_name)
    env = game.make_env(sticky_prob=game.protocol.sticky_prob, max_steps=max_steps)
    X = []
    for seed in range(seed0, seed0 + episodes):
        rng = policy_rng(seed)
        obs, _ = env.reset(seed=seed)
        done, t = False, 0
        while not done and len(X) < cap:
            if t % every == 0:
                X.append(np.asarray(obs, dtype=np.float64).copy())
            obs, _, term, trunc, _ = env.step(policy(obs, rng))
            done, t = term or trunc, t + 1
    return np.array(X)


def cka(X, Y):
    """Linear CKA (Kornblith et al. 2019): 1 = the same representation up to rotation and scale."""
    X, Y = X - X.mean(0), Y - Y.mean(0)
    xx, yy = np.linalg.norm(X.T @ X), np.linalg.norm(Y.T @ Y)
    if xx == 0 or yy == 0:
        return float("nan")
    return float(np.linalg.norm(X.T @ Y) ** 2 / (xx * yy))


def random_net(net, rng):
    """Same shape and activations, Gaussian weights with variance 1/fan_in: the CKA floor."""
    return Net("random", "-", net.game, net.space,
               [(rng.normal(0, 1 / np.sqrt(W.shape[1]), W.shape), np.zeros_like(b), act) for W, b, act in net.layers])


def cka_matrix(A, B, X, rng, reps=5):
    ha, hb = A.forward(X), B.forward(X)
    M = [[cka(x, y) for y in hb] for x in ha]
    floor = [float(np.nanmean([cka(x, random_net(A, rng).forward(X)[i]) for _ in range(reps)]))
             for i, x in enumerate(ha)]
    return dict(a=A.name, b=B.name, matrix=M, random_floor=floor)


def probs(net_or_fn, X, n_actions, rng=None):
    """Action distributions on the states X: softmax of a network, one-hot of a baseline."""
    if isinstance(net_or_fn, Net):
        z = net_or_fn.forward(X)[-1]
        p = np.exp(z - z.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True), z
    a = np.array([net_or_fn(x, rng) for x in X])
    P = np.eye(n_actions)[a]
    return P, P


def agreement(pa, za, pb, zb):
    """P(same action) when both sample, and when both take the argmax (ties split evenly),
    each with its chance level (independent choices with the same action frequencies)."""
    ga = (za == za.max(axis=1, keepdims=True)).astype(float)
    gb = (zb == zb.max(axis=1, keepdims=True)).astype(float)
    ga /= ga.sum(axis=1, keepdims=True)
    gb /= gb.sum(axis=1, keepdims=True)
    return dict(sampled=float(np.mean(np.sum(pa * pb, axis=1))),
                sampled_chance=float(pa.mean(0) @ pb.mean(0)),
                greedy=float(np.mean(np.sum(ga * gb, axis=1))),
                greedy_chance=float(ga.mean(0) @ gb.mean(0)),
                tv=float(np.mean(0.5 * np.abs(pa - pb).sum(axis=1))))


# ====================================================================== 7. networks of different shapes
# Every network is rewritten in one joint space, without changing what it computes:
#  - inputs: a 10 x 10 grid with 7 channels (the 6 MinAtar ones + Low Tide's water) and Low Tide's
#    3 scalars; Low Tide's agent -> player, jellyfish -> danger, shells -> reward. A network has zero
#    weights on the inputs it does not have.
#  - actions: noop, left, up, right, down, fire (Low Tide has no fire: a row of zeros).
#  - hidden layers: padded with zero units to the same width (a unit with no weights changes nothing).
# Then each layer of B is aligned on its own to the same layer of A (rows and columns permuted, and
# sign-flipped for tanh units): the most favourable alignment for every layer, so the distances are a
# lower bound (one consistent alignment for the whole network can only be farther).
JOINT_CHANNELS = ("player", "danger", "reward", "projectile", "trail", "gauge", "water")
JOINT_SCALARS = ("rising", "countdown", "stun")
N_JOINT = 100 * len(JOINT_CHANNELS) + len(JOINT_SCALARS)
LOW_TIDE_ROLE = (0, 6, 2, 1)                 # agent, water, shell, jelly -> joint channel
ACTION_MAP = {"minatar": (0, 1, 2, 3, 4, 5),  # own action k -> joint action (noop, left, up, right, down, fire)
              "low_tide": (0, 2, 4, 1, 3)}    # Low Tide: noop, up, down, left, right


def joint_inputs(space, n_in):
    C = len(JOINT_CHANNELS)
    i = np.arange(n_in)
    if space == "minatar":                   # (cell * 6 + channel), same first 6 channels
        return (i // 6) * C + i % 6
    if space == "low_tide":                  # (channel * 100 + cell), then 3 scalars
        idx = np.empty(n_in, dtype=int)
        idx[:400] = (i[:400] % 100) * C + np.array(LOW_TIDE_ROLE)[i[:400] // 100]
        idx[400:] = 100 * C + np.arange(n_in - 400)
        return idx
    raise ValueError(space)


def to_joint(net):
    (W, b, act), (Wo, bo, acto) = net.layers[0], net.layers[-1]
    W1 = np.zeros((W.shape[0], N_JOINT))
    W1[:, joint_inputs(net.space, W.shape[1])] = W
    rows = list(ACTION_MAP[net.space])
    Wout, bout = np.zeros((6, Wo.shape[1])), np.zeros(6)
    Wout[rows], bout[rows] = Wo, bo
    return [(W1, b, act)] + list(net.layers[1:-1]) + [(Wout, bout, acto)]


def _assign(S, signed):
    """Hungarian on S (maximise); with signed, each pair may also flip sign."""
    sg, C = (np.where(S >= 0, 1.0, -1.0), np.abs(S)) if signed else (np.ones_like(S), S)
    _, col = linear_sum_assignment(C, maximize=True)
    return col, sg[np.arange(len(col)), col]


def _signature_match(XA, XB, rng=None):
    """Row i of XA -> row of XB with the closest sorted absolute values (ties broken at random with rng)."""
    order = np.arange(len(XB)) if rng is None else rng.permutation(len(XB))
    sa, sb = -np.sort(-np.abs(XA), axis=1), -np.sort(-np.abs(XB[order]), axis=1)
    D = (sa ** 2).sum(1)[:, None] + (sb ** 2).sum(1)[None, :] - 2 * sa @ sb.T
    return order[linear_sum_assignment(D)[1]]


def align_layer(A, B, rows_free, cols_free, row_sign, col_sign, rng, restarts=8):
    """A and B are (W, b) of one layer. Pad both with zeros to the same shape, then permute (and
    sign-flip) B's rows and/or columns to bring it as close as possible to A. Returns padded A, aligned B.

    With only rows or only columns free this is one assignment problem, solved exactly. With both free
    it is a quadratic assignment problem: alternate the two assignments, starting from rows and columns
    matched by their sorted absolute values (which do not depend on the other side's order or signs),
    then restart from random tie-breaks of that match and from perturbations of the best so far."""
    (WA, bA), (WB, bB) = A, B
    R, C = max(WA.shape[0], WB.shape[0]), max(WA.shape[1], WB.shape[1])
    WA = np.pad(WA, ((0, R - WA.shape[0]), (0, C - WA.shape[1])))
    WB = np.pad(WB, ((0, R - WB.shape[0]), (0, C - WB.shape[1])))
    bA, bB = np.pad(bA, (0, R - bA.size)), np.pad(bB, (0, R - bB.size))
    MA, MB = np.column_stack([WA, bA]), np.column_stack([WB, bB])

    def apply(rp, rs, cp, cs):
        return (WB[rp][:, cp] * rs[:, None] * cs[None, :], bB[rp] * rs)

    def climb(rp, rs, cp, cs):
        obj = -np.inf
        for _ in range(100):
            if rows_free:          # rows (units of this layer), the bias goes with its row
                rp, rs = _assign(MA @ np.column_stack([WB[:, cp] * cs[None, :], bB]).T, row_sign)
            if cols_free:          # columns (units of the previous layer)
                cp, cs = _assign(WA.T @ (WB[rp] * rs[:, None]), col_sign)
            W2, b2 = apply(rp, rs, cp, cs)
            new = float(np.sum(WA * W2) + bA @ b2)
            if new <= obj + 1e-12:
                break
            obj = new
        return obj, rp, rs, cp, cs

    one, ones_c = np.ones(R), np.ones(C)
    if not (rows_free and cols_free):
        _, rp, rs, cp, cs = climb(np.arange(R), one, np.arange(C), ones_c)
        return (WA, bA), apply(rp, rs, cp, cs)
    best = climb(_signature_match(MA, MB), one, _signature_match(WA.T, WB.T), ones_c)
    for r in range(1, restarts):
        if r % 2:                  # another tie-break of the signature match
            start = (_signature_match(MA, MB, rng), one, _signature_match(WA.T, WB.T, rng), ones_c)
        else:                      # the best so far with a few columns swapped
            cp, cs = best[3].copy(), best[4].copy()
            for _ in range(rng.integers(1, 4)):
                i, j = rng.choice(C, 2, replace=False)
                cp[[i, j]], cs[[i, j]] = cp[[j, i]], cs[[j, i]]
            start = (best[1], best[2], cp, cs)
        cand = climb(*start)
        if cand[0] > best[0]:
            best = cand
    return (WA, bA), apply(*best[1:])


def layer_compare(A, B, rows_free, cols_free, row_sign, col_sign, rng, null_reps, restarts):
    (WA, bA), (WB, bB) = align_layer(A, B, rows_free, cols_free, row_sign, col_sign, rng, restarts)
    a, b = np.concatenate([WA.ravel(), bA]), np.concatenate([WB.ravel(), bB])
    # raw: zero-padded, no alignment
    (_, _), (WR, bR) = align_layer(A, B, False, False, False, False, rng, 1)
    raw = np.concatenate([WR.ravel(), bR])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    nulls = []
    for _ in range(null_reps):
        Bs = (rng.permutation(B[0].ravel()).reshape(B[0].shape), rng.permutation(B[1]))
        _, (WS, bS) = align_layer(A, Bs, rows_free, cols_free, row_sign, col_sign, rng, restarts)
        s = np.concatenate([WS.ravel(), bS])
        nulls.append(float(a @ s / (na * np.linalg.norm(s))))
    return dict(norm_a=float(na), norm_b=float(nb), l2_raw=float(np.linalg.norm(a - raw)),
                l2=float(np.linalg.norm(a - b)), rel=float(np.linalg.norm(a - b) / np.sqrt(na ** 2 + nb ** 2)),
                cos_raw=float(a @ raw / (na * nb)), cos=float(a @ b / (na * nb)),
                cos_null=float(np.mean(nulls)) if nulls else float("nan"),
                cos_null_sd=float(np.std(nulls)) if nulls else float("nan"))


def pad_hidden(layers, widths):
    """The same network with hidden layer k widened to widths[k] by units with no weights."""
    out, L = [], len(layers)
    for k, (W, b, act) in enumerate(layers):
        rows = widths[k] if k < L - 1 else W.shape[0]
        cols = W.shape[1] if k == 0 else widths[k - 1]
        out.append((np.pad(W, ((0, rows - W.shape[0]), (0, cols - W.shape[1]))), np.pad(b, (0, rows - b.size)), act))
    return out


def vn_bound(A, B):
    """Upper bound on the cosine after any permutation and sign flip of rows and columns
    (von Neumann's trace inequality: <A, P B Q> <= sum of sigma_i(A) sigma_i(B))."""
    MA, MB = np.column_stack(A), np.column_stack(B)
    sa, sb = np.linalg.svd(MA, compute_uv=False), np.linalg.svd(MB, compute_uv=False)
    n = max(sa.size, sb.size)
    return float(np.pad(sa, (0, n - sa.size)) @ np.pad(sb, (0, n - sb.size)) / (np.linalg.norm(MA) * np.linalg.norm(MB)))


def cross_shape(A, B, rng, null_reps, restarts):
    """Two networks of any shape, in the joint space.
    - first layer and output layer: each aligned on its own (one side only: exact);
    - same number of hidden layers: hidden layers widened to the same size, then one alignment for
      the whole network (as in section 3: exact);
    - different number of hidden layers: every middle layer of A against every one of B, rows and
      columns aligned on their own (approximate), with the von Neumann upper bound.
    The network with more tanh layers (more symmetries) is the one that gets aligned."""
    if sum(a in ODD for _, _, a in A.layers[:-1]) > sum(a in ODD for _, _, a in B.layers[:-1]):
        A, B = B, A
    JA, JB = to_joint(A), to_joint(B)
    out = dict(a=A.name, b=B.name,
               first=layer_compare(JA[0][:2], JB[0][:2], True, False, JB[0][2] in ODD, False,
                                   rng, null_reps, restarts) | dict(layer_a=1, layer_b=1),
               output=layer_compare(JA[-1][:2], JB[-1][:2], False, True, False, JB[-2][2] in ODD,
                                    rng, null_reps, restarts) | dict(layer_a=len(JA), layer_b=len(JB)))
    if len(JA) == len(JB):
        widths = [max(JA[k][0].shape[0], JB[k][0].shape[0]) for k in range(len(JA) - 1)]
        PA, PB = pad_hidden(JA, widths), pad_hidden(JB, widths)
        raw, al = distances(PA, PB), distances(PA, align(PA, PB, rng, restarts))
        nulls = [distances(PA, align(PA, pad_hidden(to_joint(Net("", "", B.game, B.space, shuffled(B.layers, rng))),
                                                    widths), rng, restarts)) for _ in range(null_reps)]
        whole = []
        for i in range(len(raw)):
            cn = [n[i]["cos"] for n in nulls]
            whole.append(dict(raw[i] | dict(cos_raw=raw[i]["cos"], l2_raw=raw[i]["l2"]),
                              l2=al[i]["l2"], rel=al[i]["rel"], cos=al[i]["cos"],
                              cos_null=float(np.mean(cn)) if cn else float("nan"),
                              cos_null_sd=float(np.std(cn)) if cn else float("nan")))
        out["whole"] = whole                      # per layer, then the whole network
        out["middle"] = whole[1] | dict(layer_a=2, layer_b=2, method="exact")
        out["middle_all"] = []
    else:
        mids = []
        for ka in range(1, len(JA) - 1):
            for kb in range(1, len(JB) - 1):
                m = layer_compare(JA[ka][:2], JB[kb][:2], True, True, JB[kb][2] in ODD, JB[kb - 1][2] in ODD,
                                  rng, null_reps, restarts * 8)
                mids.append(m | dict(layer_a=ka + 1, layer_b=kb + 1, method="approx",
                                     bound=vn_bound(JA[ka][:2], JB[kb][:2])))
        out["middle"] = max(mids, key=lambda m: m["cos"])
        out["middle_all"] = mids
    return out


def two_sided_check(net, rng, restarts, reps=5):
    """How well the approximate (rows and columns) alignment recovers a middle layer from a copy of
    the network with its hidden units scrambled: the worst cosine (1 = perfect)."""
    J, worst = to_joint(net), 1.0
    for _ in range(reps):
        S = to_joint(Net("", "", net.game, net.space, scrambled(net.layers, rng)))
        for k in range(1, len(J) - 1):
            (WA, bA), (WB, bB) = align_layer(J[k][:2], S[k][:2], True, True, S[k][2] in ODD, S[k - 1][2] in ODD,
                                             rng, restarts * 8)
            a, b = np.concatenate([WA.ravel(), bA]), np.concatenate([WB.ravel(), bB])
            worst = min(worst, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    return worst


# ====================================================================== report
def fmt(x, d=2):
    if isinstance(x, str):
        return x
    if x != x:
        return "-"
    return f"{x:.{d}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def report(R):
    nets = R["nets"]
    md = ["# Distanza tra le reti: risultati", "",
          f"Generato da `analyze.py` ({R['seconds']:.0f} s, seed {SEED}). Il commento ai risultati è nel README.", ""]

    md += ["## 1. Com'è fatta ogni rete", "",
           "Per strato: parametri (pesi + bias), quota non nulla, quanti valori diversi compaiono tra i non nulli, "
           "quota di non nulli con al più 2 decimali, norma L2, rango stabile (‖W‖²_F / ‖W‖²_2: quante direzioni "
           "contano davvero), unità nascoste usate (che leggono qualcosa e sono lette dallo strato dopo).", ""]
    rows = []
    for n in nets:
        for r in R["fingerprint"][n["name"]]["layers"]:
            rows.append([n["name"] if r["layer"] == 1 else "", r["layer"], r["shape"], r["activation"], r["params"],
                         f"{100 * r['density']:.1f}%", r["distinct"], f"{100 * r['round2']:.0f}%" if r["round2"] == r["round2"] else "-",
                         fmt(r["norm"]), fmt(r["max_abs"]), fmt(r["stable_rank"], 1),
                         f"{r['used_units']}/{r['units']}" if "units" in r else ""])
    md += [table(["rete", "strato", "forma", "attivazione", "parametri", "non nulli", "valori distinti",
                  "≤ 2 decimali", "norma", "max |w|", "rango stabile", "unità usate"], rows), ""]
    rows = []
    for n in nets:
        t = R["fingerprint"][n["name"]]["total"]
        top = ", ".join(f"{v:g} (×{c})" for v, c in t["top_values"][:6])
        rows.append([n["name"], n["author"], f"{100 * t['density']:.1f}%", t["distinct"],
                     f"{100 * t['round2']:.0f}%", f"{100 * t['top8_share']:.0f}%", top])
    md += ["Tutta la rete: i valori più usati e quanta parte dei pesi non nulli coprono gli 8 più frequenti.", "",
           table(["rete", "autore", "non nulli", "valori distinti", "≤ 2 decimali", "8 valori più usati",
                  "valori più usati (conteggio)"], rows), ""]

    md += ["## 2. Cosa legge il primo strato", "",
           "Per ogni canale d'ingresso: quota delle connessioni non nulle di W1 che partono da quel canale e, tra "
           "parentesi, quota della massa Σ|W1|. Le due misure possono essere molto diverse: nelle reti scritte a "
           "mano la scala dei pesi è arbitraria (la rete di Breakout legge la racchetta con pesi da 0.1 e la palla "
           "con pesi da 100), quindi la massa sottostima i canali letti con pesi piccoli. Ultima colonna: quanti "
           "ingressi sono letti da almeno un'unità.", ""]
    for space in ("minatar", "low_tide", "gridworld"):
        group = [n for n in nets if n["space"] == space]
        names = list(SPACES[space]["channels"]) + list(SPACES[space]["scalars"])
        rows = [[n["name"]] + [f"{100 * R['inputs'][n['name']]['conn_share'][c]:.0f}% "
                               f"({100 * R['inputs'][n['name']]['share'][c]:.0f}%)" for c in names]
                + [f"{R['inputs'][n['name']]['inputs_read']}/{R['inputs'][n['name']]['inputs']}"] for n in group]
        md += [table(["rete"] + names + ["ingressi letti"], rows), ""]

    md += ["## 3. Distanza nello spazio dei pesi, strato per strato", "",
           "Solo tra reti con la stessa forma. Per ogni strato (pesi e bias insieme) e per tutta la rete:",
           "",
           "- **distanza relativa** ‖A − B‖ / √(‖A‖² + ‖B‖²): 0 = identici, 1 = scorrelati (come due vettori "
           "ortogonali), √2 = opposti; il coseno dice la stessa cosa (1 = stessa direzione, 0 = scorrelati);",
           "- **grezza**: così come sono scritte; **allineata**: dopo aver riordinato (e, per tanh, cambiato di "
           "segno) le unità nascoste di B nel modo che la avvicina di più ad A, che non cambia la funzione di B;",
           "- **nullo**: la stessa allineata, ma con B sostituita da B coi valori rimescolati a caso dentro ogni "
           "strato (stessi valori, stessa sparsità, nessuna struttura), media su "
           f"{R['params']['null_reps']} rimescolamenti. Se *allineata* ≈ *nullo*, le due reti non hanno in comune "
           "più di quanto ne avrebbero due reti a caso coi loro stessi valori.",
           "",
           "Attenzione: nelle reti scritte a mano la norma è dominata da pochi pesi enormi (fino a 1000), quindi "
           "‖A − B‖ di uno strato o di tutta la rete misura soprattutto dove stanno quei pochi pesi; per tanh "
           "saturata 100 e 1000 fanno quasi la stessa cosa. Il coseno e le sezioni 4–6 non dipendono dalla scala.",
           "",
           f"Verifica dell'allineamento: una rete contro una sua copia con le unità rimescolate dà distanza "
           f"{R['self_check']:.1e} (deve essere 0).", ""]
    for group, label in (("minatar", "Reti scritte da GPT-6 Astra (600-32-32-6, tanh), giochi diversi"),
                         ("gridworld", "Reti addestrate con PPO (500-32-32-5, tanh), giochi diversi")):
        rows = []
        for d in R["weights"][group]:
            for i, nm in enumerate(["strato 1", "strato 2", "strato 3", "tutta"]):
                rows.append([f"{d['a']} vs {d['b']}" if i == 0 else "", nm, fmt(d["raw"][i]["l2"]),
                             fmt(d["raw"][i]["norm_a"]) + " / " + fmt(d["raw"][i]["norm_b"]),
                             fmt(d["raw"][i]["rel"]), fmt(d["aligned"][i]["l2"]), fmt(d["aligned"][i]["rel"]),
                             fmt(d["aligned"][i]["cos"]),
                             f"{fmt(d['null'][i]['rel'])} ± {fmt(d['null'][i]['rel_sd'])}"])
        md += [f"**{label}**", "",
               table(["coppia", "strato", "‖A−B‖ grezza", "‖A‖ / ‖B‖", "rel. grezza", "‖A−B‖ allineata",
                      "rel. allineata", "coseno allineato", "rel. nullo"], rows), ""]

    md += ["## 4. Unità del primo strato: hanno un gemello nell'altra rete?", "",
           "Ogni unità del primo strato è il vettore dei suoi pesi in ingresso. Le unità di A e di B sono accoppiate "
           "una a una (algoritmo ungherese) massimizzando il coseno (in valore assoluto per tanh). Nullo: lo "
           f"stesso con gli ingressi di B rimescolati, media su {R['params']['null_reps']} rimescolamenti.", ""]
    rows = [[f"{u['a']} vs {u['b']}", f"{u['units_a']} / {u['units_b']}", fmt(u["mean"]), fmt(u["median"]),
             u["over_09"], u["over_05"], fmt(u["null_mean"]), fmt(u["null_over_05"], 1),
             ", ".join(fmt(x) for x in u["top"])] for u in R["units"]]
    md += [table(["coppia", "unità attive", "coseno medio", "mediana", "> 0.9", "> 0.5", "nullo: medio",
                  "nullo: > 0.5", "le 5 coppie migliori"], rows), ""]
    twins = [(u, t) for u in R["units"] for t in u["twins"]]
    if twins:
        md += ["Le coppie con coseno > 0.9 (unità quasi identiche):", ""]
        md += [f"- {u['a']} unità {t['unit_a']} ({t['reads_a']}) ≈ {u['b']} unità {t['unit_b']} ({t['reads_b']}): "
               f"coseno {t['cos']:.2f}" for u, t in twins]
        md += [""]

    md += ["## 5. Rappresentazioni (CKA lineare) sugli stessi stati", "",
           "CKA tra le uscite di due strati sugli stessi stati: 1 = la stessa rappresentazione a meno di rotazioni "
           "e scala, 0 = nulla in comune. *Rete casuale*: CKA tra lo strato di A e lo stesso strato di una rete "
           "con la stessa forma e pesi gaussiani (media su 5): quanto viene \"gratis\" dal fatto di leggere gli "
           "stessi ingressi.", ""]
    for c in R["cka"]:
        la = len(c["matrix"])
        lb = len(c["matrix"][0])
        rows = [[f"{c['a']} {i + 1}"] + [fmt(v) for v in c["matrix"][i]] + [fmt(c["random_floor"][i])]
                for i in range(la)]
        md += [f"**{c['a']} (righe) vs {c['b']} (colonne)**, {c['states']} stati: {c['states_from']}", "",
               table(["strato"] + [f"{c['b']} {j + 1}" for j in range(lb)] + ["A vs rete casuale"], rows), ""]

    md += ["## 6. Comportamento", "",
           "**Stessa azione sugli stessi stati.** Probabilità che le due policy scelgano la stessa azione "
           "campionando (e il caso: due scelte indipendenti con le stesse frequenze di azioni), e scegliendo "
           "l'argmax (pareggi divisi in parti uguali); TV = distanza in variazione totale media tra le due "
           "distribuzioni (0 = identiche, 1 = disgiunte).", ""]
    rows = []
    for a in R["agreement"]:
        rows.append([a["states"], f"{a['a']} vs {a['b']}", fmt(a["sampled"]), fmt(a["sampled_chance"]),
                     fmt(a["greedy"]), fmt(a["greedy_chance"]), fmt(a["tv"])])
    md += [table(["stati", "coppia", "stessa azione (campionata)", "caso", "stessa azione (argmax)", "caso", "TV"],
                 rows), ""]
    games = R["crossplay"]["games"]
    rows = [[p] + [f"{R['crossplay']['scores'][p][g]['mean_return']:.1f} ± {R['crossplay']['scores'][p][g]['stderr']:.1f}"
                   for g in games] for p in R["crossplay"]["players"]]
    md += [f"**Cross-play** ({R['crossplay']['episodes']} episodi per casella, protocollo dei giochi: sticky 0.1, "
           "azioni campionate). Righe: chi gioca; colonne: il gioco. Se \"conoscendone una si risolvono tutte\", "
           "fuori dalla diagonale i punteggi dovrebbero restare alti.", "",
           table(["rete \\ gioco"] + games, rows), ""]

    md += ["## 7. Le reti di Astra per i quattro giochi, anche con forme diverse", "",
           "Le tre reti MinAtar sono 600-32-32-6; quella di Low Tide (`gpt/low_tide`) è 403-43-61-53-5, con un "
           "strato nascosto in più e le azioni in un altro ordine. Per confrontarle, ogni rete è riscritta, *senza "
           "cambiare quello che calcola*, in uno spazio comune:",
           "",
           "- ingressi: griglia 10×10 con 7 canali (i 6 di MinAtar più l'acqua di Low Tide) e i 3 scalari di Low "
           "Tide; in Low Tide agente → player, meduse → danger, conchiglie → reward. Una rete ha peso 0 sugli "
           "ingressi che non ha;",
           "- azioni: fermo, sinistra, su, destra, giù, fuoco (Low Tide non ha fuoco: riga di zeri);",
           "- strati nascosti: allargati con unità a zero quando servono (un'unità senza pesi non cambia niente).",
           "",
           "Poi:",
           "",
           "- **primo strato** e **uscita**: ogni strato è allineato da solo, riordinando (e cambiando di segno, "
           "per tanh) le unità nascoste; è l'allineamento più favorevole possibile per quello strato, calcolato "
           "in modo esatto;",
           "- **strato intermedio, reti con lo stesso numero di strati nascosti** (le tre MinAtar e "
           "`claude/low_tide`): un unico allineamento per tutta la rete, come nella sezione 3 (esatto);",
           "- **strato intermedio con `gpt/low_tide`**, che ha due strati intermedi: ognuno contro quello "
           "dell'altra rete, riordinando righe e colonne insieme. Questo problema non si risolve in modo esatto: "
           "il coseno trovato è un limite inferiore, e accanto c'è il **limite superiore** di von Neumann "
           "(nessun riordino può superarlo). Si tiene la coppia di strati più simile.",
           "",
           "Coseno 1 = stessi pesi a meno di un fattore di scala, 0 = niente in comune; *nullo* = lo stesso "
           f"allineamento con i valori di B rimescolati a caso (media ± dev. std. su {R['params']['null_reps']}). "
           "`claude/low_tide` è incluso come riferimento: stesso gioco di `gpt/low_tide`, autore diverso.",
           "",
           f"Verifiche: ogni rete contro una sua copia con le unità rimescolate dà coseno minimo "
           f"{R['cross_shape_check']:.6f} con i metodi esatti (deve essere 1); il metodo approssimato ritrova gli "
           f"strati intermedi di `gpt/low_tide` rimescolati con coseno minimo {R['two_sided_check']:.2f}.", ""]

    def mid(c):
        m = c["middle"]
        txt = f"{fmt(m['cos'])} ({fmt(m['cos_null'])})"
        return txt + (f", ≤ {fmt(m['bound'])}" if m.get("method") == "approx" else "")
    rows = [[f"{c['a']} vs {c['b']}", f"{fmt(c['first']['cos'])} ({fmt(c['first']['cos_null'])})", mid(c),
             f"{c['middle']['layer_a']} / {c['middle']['layer_b']}",
             f"{fmt(c['output']['cos'])} ({fmt(c['output']['cos_null'])})",
             f"{fmt(c['whole'][-1]['cos'])} ({fmt(c['whole'][-1]['cos_null'])})" if "whole" in c else "-"]
            for c in R["cross_shape"]]
    md += ["**Coseno allineato per strato** (tra parentesi il nullo; \"≤\" = limite superiore).", "",
           table(["coppia", "primo strato", "intermedio", "strati intermedi confrontati", "uscita",
                  "tutta la rete"], rows), ""]
    for key, label in (("first", "Primo strato"), ("middle", "Strato intermedio"), ("output", "Strato di uscita")):
        rows = []
        for c in R["cross_shape"]:
            d = c[key]
            rows.append([f"{c['a']} vs {c['b']}", f"{d['layer_a']} / {d['layer_b']}",
                         f"{fmt(d['norm_a'])} / {fmt(d['norm_b'])}", fmt(d["l2_raw"]), fmt(d["l2"]), fmt(d["rel"]),
                         fmt(d["cos_raw"]), fmt(d["cos"]), f"{fmt(d['cos_null'])} ± {fmt(d['cos_null_sd'])}"])
        md += [f"**{label}**", "",
               table(["coppia", "strati", "‖A‖ / ‖B‖", "‖A−B‖ grezza", "‖A−B‖ allineata", "rel. allineata",
                      "coseno grezzo", "coseno allineato", "coseno nullo"], rows), ""]
    return "\n".join(md)


# ====================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=200, help="episodes per cell of the cross-play table")
    ap.add_argument("--state-episodes", type=int, default=30, help="episodes per policy to collect states")
    ap.add_argument("--null-reps", type=int, default=20, help="shuffles for the null models")
    ap.add_argument("--restarts", type=int, default=8, help="restarts of the weight matching")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    nets = load_nets()
    by = {n.name: n for n in nets}
    R = dict(params=vars(args) | {"out": str(args.out)},
             nets=[dict(name=n.name, author=n.author, game=n.game, space=n.space, arch=n.arch) for n in nets])

    print("1-2. fingerprints and input usage", file=sys.stderr)
    R["fingerprint"] = {n.name: fingerprint(n) for n in nets}
    R["inputs"] = {n.name: input_usage(n) for n in nets}

    print("3. weight-space distances", file=sys.stderr)
    R["self_check"] = self_check(nets, rng, args.restarts)
    R["weights"] = {}
    for space in ("minatar", "gridworld"):
        group = [n for n in nets if n.space == space]
        R["weights"][space] = [weight_distance(a, b, rng, args.null_reps, args.restarts)
                               for i, a in enumerate(group) for b in group[i + 1:]]

    print("4. first-layer units", file=sys.stderr)
    pairs = [(a, b) for space in ("minatar", "low_tide", "gridworld")
             for i, a in enumerate([n for n in nets if n.space == space])
             for b in [n for n in nets if n.space == space][i + 1:]]
    R["units"] = [unit_match(a, b, rng, args.null_reps) for a, b in pairs]

    print("5-6. states, CKA, agreement", file=sys.stderr)
    lt = get_game("low_tide")
    heuristic = lt.baselines["heuristic"]
    S_lt = {name: collect_states("low_tide", pol, args.state_episodes, 0)
            for name, pol in (("heuristic", heuristic), ("claude", by["claude/low_tide"].policy()),
                              ("gpt", by["gpt/low_tide"].policy()))}
    X_lt = np.concatenate(list(S_lt.values()))
    S_mn = {}
    for g in ("breakout", "seaquest", "asterix"):
        S_mn[g] = np.concatenate([collect_states(g, by[f"astra/{g}"].policy(), args.state_episodes, 0, cap=1500),
                                  collect_states(g, RandomPolicy(6), args.state_episodes, 0, cap=500)])
    X_mn = np.concatenate(list(S_mn.values()))
    R["cka"] = [cka_matrix(by["claude/low_tide"], by["gpt/low_tide"], X_lt, rng)
                | dict(states=len(X_lt), states_from="Low Tide, giocato da euristica, claude e gpt (in parti uguali)")]
    for i, a in enumerate(["breakout", "seaquest", "asterix"]):
        for b in ["breakout", "seaquest", "asterix"][i + 1:]:
            R["cka"].append(cka_matrix(by[f"astra/{a}"], by[f"astra/{b}"], X_mn, rng)
                            | dict(states=len(X_mn), states_from="i tre giochi MinAtar, giocati dalla loro rete e a caso"))

    R["agreement"] = []
    dummy = np.random.default_rng(0)
    P = {"claude/low_tide": probs(by["claude/low_tide"], X_lt, 5), "gpt/low_tide": probs(by["gpt/low_tide"], X_lt, 5),
         "euristica": probs(heuristic, X_lt, 5, dummy)}
    for a, b in (("claude/low_tide", "gpt/low_tide"), ("claude/low_tide", "euristica"), ("gpt/low_tide", "euristica")):
        R["agreement"].append(dict(states="Low Tide", a=a, b=b) | agreement(*P[a], *P[b]))
    for g in ("breakout", "seaquest", "asterix"):
        own = probs(by[f"astra/{g}"], S_mn[g], 6)
        for o in ("breakout", "seaquest", "asterix"):
            if o != g:
                R["agreement"].append(dict(states=g, a=f"astra/{g}", b=f"astra/{o}")
                                      | agreement(*own, *probs(by[f"astra/{o}"], S_mn[g], 6)))

    print("6. cross-play", file=sys.stderr)
    games = ["breakout", "seaquest", "asterix"]
    players = {f"astra/{g}": by[f"astra/{g}"].policy() for g in games} | {"random": RandomPolicy(6)}
    scores = {}
    for p, pol in players.items():
        scores[p] = {}
        for g in games:
            res = evaluate(get_game(g), pol, episodes=args.episodes, workers=args.workers)
            scores[p][g] = {k: res[k] for k in ("mean_return", "stderr", "mean_length")}
            print(f"   {p:16s} on {g:9s} {res['mean_return']:8.2f} ± {res['stderr']:.2f}", file=sys.stderr)
    R["crossplay"] = dict(games=games, players=list(players), episodes=args.episodes, scores=scores)

    print("7. networks of different shapes, in a joint space", file=sys.stderr)
    rng7 = np.random.default_rng(SEED + 7)     # own generator: sections 1-6 do not change
    group7 = [by[n] for n in ("astra/breakout", "astra/seaquest", "astra/asterix", "gpt/low_tide", "claude/low_tide")]
    R["cross_shape"] = [cross_shape(a, b, rng7, args.null_reps, args.restarts)
                        for i, a in enumerate(group7) for b in group7[i + 1:]]
    R["cross_shape_check"] = min(
        min(x[k]["cos"] for k in ("first", "middle", "output"))
        for x in [cross_shape(n, Net(n.name + "*", n.author, n.game, n.space, scrambled(n.layers, rng7)),
                              rng7, 0, args.restarts) for n in group7])
    R["two_sided_check"] = two_sided_check(by["gpt/low_tide"], rng7, args.restarts)

    R["seconds"] = time.time() - t0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(R, indent=1))
    (args.out / "report.md").write_text(report(R) + "\n")
    print(f"wrote {args.out / 'report.md'} and results.json ({R['seconds']:.0f} s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
