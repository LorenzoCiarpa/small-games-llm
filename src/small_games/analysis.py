"""
Tools for asking whether the weights of a policy network can be read as an algorithm.

The same measurements run on hand-written networks and on PPO-trained ones, so the two can be compared:

  weights      sparsity, how many distinct values, how many inputs carry 90% of each first-layer unit's weight
  pruning      score after keeping only the largest |w| of every layer
  ablation     score after resampling one hidden unit (its value in a random other state): importance
  probing      how well game features (features.py) explain each hidden unit: best single correlation, linear R^2
  saturation   for tanh units, how often they sit at +-1 (i.e. work as booleans)
  distillation decision trees on the game features that imitate the network's argmax: agreement on held-out
               states and score when the tree plays the game (is the behaviour a short algorithm?)
  fields       pictures of the first-layer weights of the most important units
  live inputs  the first layer restricted to the inputs that are ever non-zero (on the others a trained network keeps
               its random initial weights), and pruning of that layer alone

Needs the `analysis` dependency group (scikit-learn, matplotlib comes with minatar).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .core import get_game
from .evaluate import evaluate
from .features import FEATURES
from .policy import MLPPolicy, policy_rng, sample_action
from .weights import ReferenceMLP, load_weights


# ------------------------------------------------------------------------------------------------ basics
def hidden_acts(layers, X):
    """Post-activation outputs of every layer for a batch X (n, obs_dim); the last one is the logits."""
    out, x = [], np.asarray(X, dtype=np.float64)
    for W, b, act in layers:
        x = x @ W.T + b
        if act == "relu":
            x = np.maximum(x, 0.0)
        elif act == "tanh":
            x = np.tanh(x)
        out.append(x)
    return out


def collect(game, layers, episodes, seed0, max_states=200_000):
    """Observations visited by the network's own (sampled) play, with the protocol's sticky actions."""
    mlp = ReferenceMLP(layers)
    proto = game.protocol
    env = game.make_env(sticky_prob=proto.sticky_prob, max_steps=proto.max_steps)
    per_ep = max(1, max_states // episodes)
    X, ep_id = [], []
    for k, seed in enumerate(range(seed0, seed0 + episodes)):
        rng = policy_rng(seed)
        obs, _ = env.reset(seed=seed)
        done, n = False, 0
        while not done and n < per_ep:
            X.append(obs)
            ep_id.append(k)
            obs, _, term, trunc, _ = env.step(sample_action(mlp.logits(obs), rng))
            done = term or trunc
            n += 1
    return np.asarray(X, dtype=np.float32), np.asarray(ep_id)


def score(game, layers, episodes, seed0, workers):
    s = evaluate(game, MLPPolicy(ReferenceMLP(layers)), episodes=episodes, seed0=seed0, workers=workers)
    return s["mean_return"], s["stderr"]


# ------------------------------------------------------------------------------------------------ weights
def weight_stats(layers):
    rows = []
    for li, (W, b, act) in enumerate(layers):
        nz = W[np.abs(W) > 1e-9]
        absW = np.abs(W)
        # inputs needed to reach 90% of each unit's total |w|
        k90 = []
        for w in absW:
            s = np.sort(w)[::-1]
            tot = s.sum()
            k90.append(int(np.searchsorted(np.cumsum(s), 0.9 * tot) + 1) if tot > 0 else 0)
        k90 = [k for k in k90 if k > 0]                 # units with all-zero weights do not count
        med = float(np.median(k90)) if k90 else 0.0
        rows.append({
            "layer": li, "shape": list(W.shape), "activation": act,
            "params": int(W.size + b.size),
            "nonzero_frac": float(len(nz) / W.size),
            "distinct_values": int(len(np.unique(np.round(nz, 6)))),
            "max_abs": float(absW.max()),
            "zero_units": int((absW.sum(axis=1) == 0).sum()),
            "inputs_for_90pct_median": med,
            "inputs_for_90pct_frac": med / W.shape[1],
        })
    return rows


def prune(layers, keep):
    """Keep the `keep` fraction of largest |w| in every weight matrix (biases untouched)."""
    out = []
    for W, b, act in layers:
        W2 = W.copy()
        k = int(math.ceil(keep * W.size))
        if k < W.size:
            thr = np.sort(np.abs(W).ravel())[::-1][k - 1]
            W2[np.abs(W2) < thr] = 0.0
        out.append((W2, b.copy(), act))
    return out


def live_inputs(game, nets_layers, episodes=40, seed0=None):
    """Inputs that are non-zero in at least one state visited by any of the networks (`nets_layers`: parsed layers).

    The others never reach a unit. A trained network gets no gradient on their weights, which keep their random
    initial values (small, but non-zero and all different); a hand-written one leaves them at zero. Weight statistics
    restricted to the live inputs compare the two fairly.
    """
    seed0 = game.protocol.seed0 + 5000 if seed0 is None else seed0
    live = np.zeros(game.obs_dim, dtype=bool)
    for layers in nets_layers:
        X, _ = collect(game, layers, episodes, seed0)
        live |= (X != 0).any(axis=0)
    return live


def prune_first_layer(layers, keep, live):
    """Zero the first-layer weights of the dead inputs, then keep the `keep` fraction of the largest |w| among the
    live ones; the other layers stay as they are."""
    W, b, act = layers[0]
    W2 = np.where(live[None, :], W, 0.0)
    k = int(math.ceil(keep * W2[:, live].size))
    if k < W2[:, live].size:
        thr = np.sort(np.abs(W2[:, live]).ravel())[::-1][k - 1]
        W2[np.abs(W2) < thr] = 0.0
    return [(W2, b.copy(), act)] + [(W_.copy(), b_.copy(), a) for W_, b_, a in layers[1:]]


class ResamplePolicy:
    """The network with hidden unit j of layer li replaced, at every step, by its value in a random other state.

    Resample ablation keeps the unit's distribution (a rare-but-large unit stays rare-but-large) and only breaks
    its link with the current state; replacing it with its mean would inject a value it almost never takes.
    """

    def __init__(self, layers, li, j, samples):
        self.layers, self.li, self.j = layers, li, j
        self.samples = np.asarray(samples, dtype=np.float64)

    def __call__(self, obs, rng):
        pick = rng.integers(len(self.samples))      # drawn even when li = -1, so the reference uses the rng alike
        x = np.asarray(obs, dtype=np.float64)
        for k, (W, b, act) in enumerate(self.layers):
            x = W @ x + b
            if act == "relu":
                x = np.maximum(x, 0.0)
            elif act == "tanh":
                x = np.tanh(x)
            if k == self.li:
                x[self.j] = self.samples[pick]
        return sample_action(x, rng)


# ------------------------------------------------------------------------------------------------ probing
def _linear_features(F):
    """Features for linear regression: NONE (99) replaced by 0, plus an 'absent' indicator per such column."""
    cols = []
    for k in range(F.shape[1]):
        f = F[:, k]
        absent = f >= 98
        if absent.any():
            cols.append(np.where(absent, 0.0, f))
            cols.append(absent.astype(float))
        else:
            cols.append(f)
    return np.column_stack(cols)


def probe(H, F, names):
    """Per unit: best |Pearson r| with a single feature, which feature, linear R^2 on all features."""
    Fl = _linear_features(F)
    A = np.column_stack([Fl, np.ones(len(Fl))])
    Fz = np.where(F >= 98, np.nan, F)
    res = []
    for j in range(H.shape[1]):
        h = H[:, j]
        if h.std() < 1e-6:
            res.append({"unit": j, "constant": True, "best_feature": None, "best_r": 0.0, "r2": 0.0})
            continue
        best, best_r = None, 0.0
        for k, name in enumerate(names):
            f = Fz[:, k]
            ok = ~np.isnan(f)
            if ok.sum() < 100 or f[ok].std() < 1e-9:
                continue
            r = abs(np.corrcoef(h[ok], f[ok])[0, 1])
            if r > best_r:
                best, best_r = name, float(r)
        coef, *_ = np.linalg.lstsq(A, h, rcond=None)
        r2 = 1 - ((A @ coef - h) ** 2).sum() / ((h - h.mean()) ** 2).sum()
        res.append({"unit": j, "constant": False, "best_feature": best, "best_r": best_r, "r2": float(r2)})
    return res


# ------------------------------------------------------------------------------------------------ distillation
class TreePolicy:
    """A decision tree on the game features, playing deterministically (picklable, fast traversal)."""

    def __init__(self, tree, game_name):
        t = tree.tree_
        self.left, self.right = t.children_left.copy(), t.children_right.copy()
        self.feature, self.threshold = t.feature.copy(), t.threshold.copy()
        self.leaf_action = tree.classes_[np.argmax(t.value[:, 0, :], axis=1)].astype(int)
        self.game_name = game_name

    def __call__(self, obs, rng):
        f = FEATURES[self.game_name][1](obs)
        n = 0
        while self.left[n] != -1:
            n = self.left[n] if f[self.feature[n]] <= self.threshold[n] else self.right[n]
        return int(self.leaf_action[n])


def distill(F, y, ep_id, depths, seed=0):
    """Trees of each depth trained on 70% of the episodes, agreement measured on the other 30%."""
    from sklearn.tree import DecisionTreeClassifier

    eps = np.unique(ep_id)
    rng = np.random.default_rng(seed)
    test_eps = set(rng.choice(eps, size=max(1, int(0.3 * len(eps))), replace=False).tolist())
    test = np.array([e in test_eps for e in ep_id])
    out = {}
    for d in depths:
        tree = DecisionTreeClassifier(max_depth=d, random_state=seed, min_samples_leaf=20)
        tree.fit(F[~test], y[~test])
        out[d] = (tree, float((tree.predict(F[test]) == y[test]).mean()))
    return out


def tree_text(tree, names, action_names):
    from sklearn.tree import export_text

    txt = export_text(tree, feature_names=list(names), decimals=1)
    for k, a in enumerate(action_names):
        txt = txt.replace(f"class: {k}\n", f"action: {k} {a}\n")
    return txt


# ------------------------------------------------------------------------------------------------ figures
def plot_fields(game, layers, units, path, title):
    """First-layer weights of the given units as images, one panel per input plane."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W = layers[0][0]
    if game.grid is not None and game.name != "low_tide":
        planes = [(game.grid.channels[c], (lambda w, c=c: w[:600].reshape(10, 10, 6)[:, :, c]))
                  for c in range(6) if game.grid.labels[c] is not None]
    else:
        planes = [(n, (lambda w, c=c: w[c * 100:(c + 1) * 100].reshape(10, 10)))
                  for c, n in enumerate(("agent", "water", "shells", "jellyfish"))]
        planes.append(("rising | countd. | stun", lambda w: w[400:403].reshape(1, 3)))
    fig, axes = plt.subplots(len(units), len(planes), figsize=(1.9 * len(planes), 1.9 * len(units)),
                             squeeze=False)
    for i, (u, label) in enumerate(units):
        w = W[u]
        vmax = np.abs(w).max() or 1.0
        for k, (name, get) in enumerate(planes):
            ax = axes[i, k]
            ax.imshow(get(w), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=9)
            if k == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ driver
def analyse(game_name, nets, out_dir: Path, *, score_episodes=200, ablation_episodes=100, collect_episodes=40,
            seed0=None, workers=4, prune_keeps=(1.0, 0.5, 0.25, 0.1, 0.05, 0.02),
            depths=(1, 2, 3, 4, 6, 8, 12), tree_score_depths=(2, 4, 8, 12), log=print):
    """Run every measurement on each network {name: path} and write analysis.json, trees and figures."""
    game = get_game(game_name)
    seed0 = game.protocol.seed0 if seed0 is None else seed0
    names, feat = FEATURES[game_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {"score_episodes": score_episodes, "ablation_episodes": ablation_episodes,
                "collect_episodes": collect_episodes, "seed0": seed0}
    report = {"game": game_name, "features": list(names), "settings": settings, "nets": {}}
    prev = out_dir / "analysis.json"
    if prev.is_file():                       # add to (or replace in) an earlier run with the same settings
        old = json.loads(prev.read_text())
        if old.get("settings") == settings and old.get("features") == list(names):
            report["nets"] = old["nets"]

    for net_name, path in nets.items():
        t0 = time.time()
        layers = load_weights(path, game)
        r = report["nets"][net_name] = {"path": str(path)}
        r["weights"] = weight_stats(layers)
        base = score(game, layers, score_episodes, seed0, workers)
        r["score"] = {"mean": base[0], "stderr": base[1]}
        g = evaluate(game, MLPPolicy(ReferenceMLP(layers), greedy=True), episodes=score_episodes, seed0=seed0,
                     workers=workers)
        r["score_greedy"] = {"mean": g["mean_return"], "stderr": g["stderr"]}
        log(f"[{net_name}] score {base[0]:.2f} ± {base[1]:.2f} (greedy {g['mean_return']:.2f})")

        r["pruning"] = []
        for keep in prune_keeps:
            m, se = score(game, prune(layers, keep), score_episodes, seed0, workers)
            r["pruning"].append({"keep": keep, "mean": m, "stderr": se})
            log(f"[{net_name}] pruning keep={keep}: {m:.2f}")

        X, ep_id = collect(game, layers, collect_episodes, seed0 + 5000)
        acts = hidden_acts(layers, X)
        F = np.stack([feat(o) for o in X])
        y = acts[-1].argmax(axis=1)
        r["states"] = int(len(X))

        r["layers"] = []
        abl_all = []
        for li in range(len(layers) - 1):
            H = acts[li]
            act = layers[li][2]
            info = {"layer": li, "units": int(H.shape[1]), "activation": act}
            if act == "tanh":
                info["saturation"] = float((np.abs(H) > 0.95).mean())
            info["constant_units"] = int((H.std(axis=0) < 1e-6).sum())
            idx = np.random.default_rng(0).choice(len(H), size=min(len(H), 60_000), replace=False)
            info["probe"] = probe(H[idx], F[idx], names)
            abl = []
            pick = np.random.default_rng(1).choice(len(H), size=min(len(H), 5000), replace=False)
            for j in range(H.shape[1]):
                if H[:, j].std() < 1e-9:     # a constant unit: resampling it changes nothing
                    abl.append(None)
                    continue
                s = evaluate(game, ResamplePolicy(layers, li, j, H[pick, j]), episodes=ablation_episodes,
                             seed0=seed0, workers=workers)
                abl.append(s["mean_return"])
            info["ablation"] = abl
            abl_all += [(li, j, m) for j, m in enumerate(abl)]
            r["layers"].append(info)
            log(f"[{net_name}] layer {li}: ablation done")
        ref = evaluate(game, ResamplePolicy(layers, -1, 0, [0.0]), episodes=ablation_episodes, seed0=seed0,
                       workers=workers)
        r["ablation_reference"] = ref["mean_return"]
        r["ablation_reference_stderr"] = ref["stderr"]

        trees = distill(F, y, ep_id, depths)
        r["distill"] = []
        with open(out_dir / f"trees_{net_name}.txt", "w") as f:
            for d, (tree, agree) in trees.items():
                row = {"depth": d, "agreement": agree, "leaves": int(tree.get_n_leaves())}
                if d in tree_score_depths:
                    s = evaluate(game, TreePolicy(tree, game_name), episodes=score_episodes, seed0=seed0,
                                 workers=workers)
                    row["score"], row["stderr"] = s["mean_return"], s["stderr"]
                r["distill"].append(row)
                if d <= 4:
                    f.write(f"=== depth {d}: agreement {agree:.1%}, {row['leaves']} leaves"
                            + (f", score {row['score']:.2f}" if "score" in row else "") + "\n")
                    f.write(tree_text(tree, names, game.action_names) + "\n")
                log(f"[{net_name}] tree d={d}: agreement {agree:.1%}" +
                    (f", score {row['score']:.2f}" if "score" in row else ""))

        top = sorted(((li, j, m) for li, j, m in abl_all if li == 0 and m is not None), key=lambda t: t[2])[:8]
        probe0 = {p["unit"]: p for p in r["layers"][0]["probe"]}
        units = [(j, f"u{j}  abl {m:.1f}\n{probe0[j]['best_feature']} r={probe0[j]['best_r']:.2f}")
                 for _, j, m in top]
        plot_fields(game, layers, units, out_dir / f"fields_{net_name}.png",
                    f"{game_name} / {net_name}: first layer, the 8 most important units (ablation)")
        r["seconds"] = round(time.time() - t0, 1)
        log(f"[{net_name}] done in {r['seconds']:.0f}s")
        with open(out_dir / "analysis.json", "w") as f:
            json.dump(report, f, indent=1)
    return report


# ------------------------------------------------------------------------------------------------ summary
def summarize(report):
    """One row of headline numbers per network (for the READMEs): see the keys below."""
    rows = {}
    for name, r in report["nets"].items():
        base = r["score"]["mean"]
        w0 = r["weights"]
        prune_ok = [p["keep"] for p in r["pruning"] if p["mean"] >= 0.9 * base]
        ref = r["ablation_reference"]
        drops = [0.0 if m is None else ref - m for L in r["layers"] for m in L["ablation"]]
        units = sum(L["units"] for L in r["layers"])
        big = sum(d > 0.1 * abs(ref) for d in drops)
        pos = sorted((d for d in drops if d > 0), reverse=True)
        probes = [p for L in r["layers"] for p in L["probe"] if not p["constant"]]
        trees = {d["depth"]: d for d in r["distill"]}
        rows[name] = {
            "score": base, "score_greedy": r["score_greedy"]["mean"],
            "nonzero_frac": sum(L["nonzero_frac"] * (L["params"] - L["shape"][0]) for L in w0)
                            / sum(L["params"] - L["shape"][0] for L in w0),
            "distinct_values_l0": w0[0]["distinct_values"],
            "inputs_for_90pct_l0": w0[0]["inputs_for_90pct_median"],
            "min_keep_90pct_score": min(prune_ok) if prune_ok else None,
            "saturation": [L.get("saturation") for L in r["layers"]],
            "constant_units": sum(L["constant_units"] for L in r["layers"]),
            "units": units,
            "units_drop_10pct": big,
            "top5_share_of_drop": (sum(pos[:5]) / sum(pos)) if pos else None,
            "probe_r08_frac": sum(p["best_r"] >= 0.8 for p in probes) / max(1, len(probes)),
            "probe_r2_mean": float(np.mean([p["r2"] for p in probes])) if probes else None,
            "tree_agreement": {d: round(t["agreement"], 3) for d, t in trees.items()},
            "tree_score": {d: round(t["score"], 2) for d, t in trees.items() if "score" in t},
            "tree_leaves": {d: t["leaves"] for d, t in trees.items()},
        }
    return rows


# ------------------------------------------------------------------------------------------------ key quantities
def _col(F, names, name):
    return F[:, list(names).index(name)]


def _within(F, names, name, d):
    """1 if the feature (a distance, NONE if absent) is at most d."""
    return (_col(F, names, name) <= d).astype(float)


def _hypothetical_landing(F, names, dx):
    from .features import _landing

    bx, by, pad = _col(F, names, "ball_x"), _col(F, names, "ball_y"), _col(F, names, "paddle_x")
    return np.array([_landing(int(x), int(y), dx, 1) - p for x, y, p in zip(bx, by, pad)], dtype=float)


# The quantities the hand-written algorithms compute (from the decode_* analyses), as functions of the features.
# The names are keys of nets[name]["key_quantities"] in experiments/<game>/ppo/results/analysis/analysis.json.
KEY_QUANTITIES = {
    "low_tide": {
        "margin (dry rows below the agent)": lambda F, n: _col(F, n, "margin"),
        "danger (rising and margin ≤ 2)": lambda F, n: _col(F, n, "rising") * np.maximum(0, 3 - _col(F, n, "margin")),
        "shell to the right in the row": lambda F, n: (_col(F, n, "shells_right") > 0).astype(float),
        "shell to the left in the row": lambda F, n: (_col(F, n, "shells_left") > 0).astype(float),
    },
    "breakout": {
        "landing column − paddle": lambda F, n: _col(F, n, "landing_rel_x"),
        # what Astra's lookup neurons compute: the landing column for one direction, whatever the real one is
        # landing column if the ball came down to the right − paddle
        "landing if the ball came down to the right − paddle": lambda F, n: _hypothetical_landing(F, n, +1),
        # landing column if the ball came down to the left − paddle
        "landing if the ball came down to the left − paddle": lambda F, n: _hypothetical_landing(F, n, -1),
        "ball − paddle (columns)": lambda F, n: _col(F, n, "ball_rel_x"),
        "the ball is coming down": lambda F, n: (_col(F, n, "ball_dy") > 0).astype(float),
    },
    "asterix": {
        "enemy coming in my row (≤ 3)": lambda F, n: _within(F, n, "enemy_eta_same", 3),
        "enemy coming in the row below (≤ 2)": lambda F, n: _within(F, n, "enemy_eta_down1", 2),
        "enemy coming in the row above (≤ 2)": lambda F, n: _within(F, n, "enemy_eta_up1", 2),
        "treasure coming in my row": lambda F, n: _within(F, n, "gold_eta_same", 9),
    },
    "seaquest": {
        "low oxygen (< 60)": lambda F, n: (_col(F, n, "oxygen") < 3).astype(float),
        "at least one diver on board": lambda F, n: (_col(F, n, "divers") >= 1).astype(float),
        "time to surface (oxygen < 60 and at least one diver)":
            lambda F, n: ((_col(F, n, "oxygen") < 3) & (_col(F, n, "divers") >= 1)).astype(float),
        "enemy near in my row (≤ 2)": lambda F, n: _within(F, n, "enemy_near_same", 2),
        "diver to rescue in my row": lambda F, n: _within(F, n, "diver_near_same", 9),
    },
}


def key_fit(acts, y, test):
    """Per hidden layer: best single unit |r| with y, and linear R^2 of y from all units (fit on ~test, scored on test)."""
    rows = []
    for li, H in enumerate(acts[:-1]):
        ok = H.std(axis=0) > 1e-9
        if y.std() < 1e-9 or not ok.any():
            rows.append({"layer": li, "best_unit": None, "best_r": 0.0, "linear_r2": 0.0})
            continue
        r = np.zeros(H.shape[1])
        r[ok] = [abs(np.corrcoef(H[:, j], y)[0, 1]) for j in np.flatnonzero(ok)]
        A = np.column_stack([H[:, ok], np.ones(len(H))])
        coef, *_ = np.linalg.lstsq(A[~test], y[~test], rcond=None)
        pred = A[test] @ coef
        yt = y[test]
        r2 = 1 - ((pred - yt) ** 2).sum() / max(((yt - yt.mean()) ** 2).sum(), 1e-12)
        rows.append({"layer": li, "best_unit": int(r.argmax()), "best_r": float(r.max()), "linear_r2": float(r2)})
    return rows


def key_quantities(game_name, nets, out_dir: Path, collect_episodes=40, seed0=None, log=print):
    """For each network and hidden layer: best single unit |r| and linear (all units) R^2 of each key quantity.

    Answers "does the network compute the intermediate quantities of the hand-written algorithm?": one unit that
    *is* the quantity (r ~ 1), the quantity spread over many units (high R^2, low r), or not there (both low).
    Added to analysis.json under nets[name]["key_quantities"]. The R^2 is measured on held-out episodes.
    """
    game = get_game(game_name)
    seed0 = game.protocol.seed0 if seed0 is None else seed0
    names, feat = FEATURES[game_name]
    path = out_dir / "analysis.json"
    report = json.loads(path.read_text())
    for net_name, wpath in nets.items():
        layers = load_weights(wpath, game)
        if net_name in report["nets"]:
            report["nets"][net_name]["weights"] = weight_stats(layers)
        X, ep = collect(game, layers, collect_episodes, seed0 + 5000)
        F = np.stack([feat(o) for o in X])
        acts = hidden_acts(layers, X)
        test = ep >= int(0.7 * collect_episodes)
        res = {}
        for qname, fn in KEY_QUANTITIES[game_name].items():
            rows = res[qname] = key_fit(acts, fn(F, names), test)
            log(f"[{net_name}] {qname}: " + "  ".join(
                f"layer {x['layer']}: r {x['best_r']:.2f}, R² {x['linear_r2']:.2f}" for x in rows))
        report["nets"].setdefault(net_name, {})["key_quantities"] = res
    path.write_text(json.dumps(report, indent=1))
    return report


# ------------------------------------------------------------------------------------------------ live inputs
def live_input_stats(game_name, nets, out_dir: Path, live_from=None, collect_episodes=40, score_episodes=200,
                     seed0=None, workers=4, keeps=(1.0, 0.5, 0.25, 0.1, 0.05), log=print):
    """The first layer restricted to the live inputs (see `live_inputs`), added to analysis.json as nets[name]["live"]
    (the indices of the live inputs go to "live_inputs").

    The live inputs are those of all the networks in `live_from` (default: `nets`) together. For each network:
    weight statistics on the live inputs, mean |w| on dead and live inputs, and the score after pruning the first
    layer only (dead inputs zeroed, then the `keep` fraction of the largest live weights kept).
    """
    game = get_game(game_name)
    seed0 = game.protocol.seed0 if seed0 is None else seed0
    live = live_inputs(game, [load_weights(p, game) for p in (live_from or nets).values()], collect_episodes)
    log(f"[{game_name}] live inputs: {live.sum()} of {game.obs_dim}")
    path = out_dir / "analysis.json"
    report = json.loads(path.read_text())
    report["live_inputs"] = np.flatnonzero(live).tolist()
    for net_name, wpath in nets.items():
        layers = load_weights(wpath, game)
        W, b, act = layers[0]
        st = weight_stats([(W[:, live], b, act)])[0]
        res = {"inputs": int(live.sum()), "of": int(game.obs_dim),
               "nonzero_frac": st["nonzero_frac"], "distinct_values": st["distinct_values"],
               "inputs_for_90pct_median": st["inputs_for_90pct_median"],
               "inputs_for_90pct_frac": st["inputs_for_90pct_frac"],
               "mean_abs_live": float(np.abs(W[:, live]).mean()),
               "mean_abs_dead": float(np.abs(W[:, ~live]).mean()) if (~live).any() else None,
               "pruning_first_layer": []}
        for keep in keeps:
            m, se = score(game, prune_first_layer(layers, keep, live), score_episodes, seed0, workers)
            res["pruning_first_layer"].append({"keep": keep, "mean": m, "stderr": se})
        log(f"[{net_name}] first layer on the live inputs: " + ", ".join(
            f"{100 * p['keep']:g}% {p['mean']:.1f}" for p in res["pruning_first_layer"]))
        report["nets"][net_name]["live"] = res
    path.write_text(json.dumps(report, indent=1))
    return report


# ------------------------------------------------------------------------------------------------ markdown tables
def _f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def markdown_tables(report, order=None):
    """The comparison tables of the READMEs, generated from analysis.json."""
    nets = order or list(report["nets"])
    R = report["nets"]
    out = {}

    rows = ["| | " + " | ".join(f"`{n}`" for n in nets) + " |", "|---|" + "---|" * len(nets)]
    rows.append("| score (200 episodes, sampled) | " + " | ".join(
        f"{R[n]['score']['mean']:.1f} ± {R[n]['score']['stderr']:.1f}" for n in nets) + " |")
    rows.append("| greedy score | " + " | ".join(_f(R[n]["score_greedy"]["mean"]) for n in nets) + " |")
    out["score"] = "\n".join(rows)

    rows = ["| | " + " | ".join(f"`{n}`" for n in nets) + " |", "|---|" + "---|" * len(nets)]
    def wsum(n, key):
        return R[n]["weights"][0][key]
    tot = {n: sum(L["nonzero_frac"] * L["shape"][0] * L["shape"][1] for L in R[n]["weights"])
           / sum(L["shape"][0] * L["shape"][1] for L in R[n]["weights"]) for n in nets}
    rows.append("| nonzero weights | " + " | ".join(f"{100 * tot[n]:.1f}%" for n in nets) + " |")
    rows.append("| distinct values in the first layer | " + " | ".join(f"{wsum(n, 'distinct_values'):,}"
                                                                    for n in nets) + " |")
    rows.append("| first-layer neurons with all weights at zero | " + " | ".join(
        str(wsum(n, "zero_units")) if "zero_units" in R[n]["weights"][0] else "—" for n in nets) + " |")
    rows.append(f"| inputs that make up 90% of a neuron's weight (median, out of {R[nets[0]]['weights'][0]['shape'][1]})"
                " | " + " | ".join(_f(wsum(n, "inputs_for_90pct_median"), 0) for n in nets) + " |")
    rows.append("| largest weight (in absolute value) | " + " | ".join(_f(max(L["max_abs"] for L in R[n]["weights"]), 2)
                                                                     for n in nets) + " |")
    out["weights"] = "\n".join(rows)

    if all("live" in R[n] for n in nets):
        lv = {n: R[n]["live"] for n in nets}
        k = lv[nets[0]]
        rows = ["| first layer, only the " + f"{k['inputs']} live inputs out of {k['of']}" + " | "
                + " | ".join(f"`{n}`" for n in nets) + " |", "|---|" + "---|" * len(nets)]
        rows.append("| nonzero weights | " + " | ".join(f"{100 * lv[n]['nonzero_frac']:.1f}%" for n in nets) + " |")
        rows.append("| distinct values | " + " | ".join(f"{lv[n]['distinct_values']:,}"
                                                        for n in nets) + " |")
        rows.append(f"| inputs that make up 90% of a neuron's weight (median, out of {k['inputs']}) | "
                    + " | ".join(_f(lv[n]["inputs_for_90pct_median"], 0) for n in nets) + " |")
        rows.append("| mean \\|w\\| on dead / live inputs | " + " | ".join(
            f"{_f(lv[n]['mean_abs_dead'], 3)} / {_f(lv[n]['mean_abs_live'], 3)}" for n in nets) + " |")
        out["live"] = "\n".join(rows)
        keeps = [p["keep"] for p in k["pruning_first_layer"]]
        rows = ["| live first-layer weights kept | " + " | ".join(f"{100 * x:g}%" for x in keeps) + " |",
                "|---|" + "---|" * len(keeps)]
        for n in nets:
            rows.append(f"| `{n}` | " + " | ".join(_f(p["mean"]) for p in lv[n]["pruning_first_layer"]) + " |")
        out["pruning_first_layer"] = "\n".join(rows)

    keeps = [p["keep"] for p in R[nets[0]]["pruning"]]
    rows = ["| weights kept | " + " | ".join(f"{100 * k:g}%" for k in keeps) + " |", "|---|" + "---|" * len(keeps)]
    for n in nets:
        rows.append(f"| `{n}` | " + " | ".join(_f(p["mean"]) for p in R[n]["pruning"]) + " |")
    out["pruning"] = "\n".join(rows)

    rows = ["| | neurons (constant) | neurons that lose more than 10% | maximum loss | reference | "
            "share of the 5 worst neurons in the total loss |", "|---|---|---|---|---|---|"]
    for n in nets:
        ref = R[n]["ablation_reference"]
        se = R[n].get("ablation_reference_stderr")
        abl = [m for L in R[n]["layers"] for m in L["ablation"]]
        drops = [ref - m for m in abl if m is not None]
        units = sum(L["units"] for L in R[n]["layers"])
        const = sum(L["constant_units"] for L in R[n]["layers"])
        pos = sorted((d for d in drops if d > 0), reverse=True)
        share = sum(pos[:5]) / sum(pos) if pos else None
        rows.append(f"| `{n}` | {units} ({const}) | {sum(d > 0.1 * abs(ref) for d in drops)} | {max(drops):.1f} | "
                    f"{ref:.1f} ± {_f(se)} | {_f(100 * share if share is not None else None, 0)}% |")
    out["ablation"] = "\n".join(rows)

    rows = ["| | " + " | ".join(f"`{n}` layer {L['layer']}" for n in nets for L in R[n]["layers"]) + " |",
            "|---|" + "---|" * sum(len(R[n]["layers"]) for n in nets)]
    rows.append("| saturated tanh neurons (\\|a\\| > 0.95) | " + " | ".join(
        "relu" if L.get("saturation") is None else f"{100 * L['saturation']:.0f}%"
        for n in nets for L in R[n]["layers"]) + " |")
    rows.append("| neurons with \\|r\\| ≥ 0.8 with a game quantity | " + " | ".join(
        f"{sum(p['best_r'] >= 0.8 for p in L['probe'] if not p['constant'])}/"
        f"{sum(not p['constant'] for p in L['probe'])}" for n in nets for L in R[n]["layers"]) + " |")
    rows.append("| mean R² of a neuron on the game quantities | " + " | ".join(
        _f(float(np.mean([p['r2'] for p in L['probe'] if not p['constant']])), 2)
        for n in nets for L in R[n]["layers"]) + " |")
    out["probe"] = "\n".join(rows)

    qs = list(next(iter(R[n].get("key_quantities", {}) for n in nets if "key_quantities" in R[n])))
    if qs:
        rows = ["| quantity | " + " | ".join(f"`{n}` r / R²" for n in nets) + " |", "|---|" + "---|" * len(nets)]
        for q in qs:
            cells = []
            for n in nets:
                kq = R[n].get("key_quantities", {}).get(q)
                if not kq:
                    cells.append("—")
                    continue
                best = max(kq, key=lambda x: (x["best_r"] + max(x["linear_r2"], 0)))
                cells.append(f"{best['best_r']:.2f} / {max(best['linear_r2'], 0):.2f}")
            rows.append(f"| {q} | " + " | ".join(cells) + " |")
        out["key"] = "\n".join(rows)

    depths = [d["depth"] for d in R[nets[0]]["distill"]]
    rows = ["| depth | " + " | ".join(f"`{n}` agreement / leaves / points" for n in nets) + " |",
            "|---|" + "---|" * len(nets)]
    for i, d in enumerate(depths):
        cells = []
        for n in nets:
            t = R[n]["distill"][i]
            cells.append(f"{100 * t['agreement']:.1f}% / {t['leaves']}" + (f" / {t['score']:.1f}" if "score" in t else ""))
        rows.append(f"| {d} | " + " | ".join(cells) + " |")
    rows.append("| the network itself, greedy | " + " | ".join(_f(R[n]["score_greedy"]["mean"]) for n in nets) + " |")
    out["distill"] = "\n".join(rows)
    return out
