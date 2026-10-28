"""
Search spaces (protocol section 3; the same ranges as experiments/bench/tune.py).
One definition serves Optuna, random draws, clipping and the LLM prompt.
"""

import math

import optuna

# name: (low, high, kind, log)
SPACES = {
    "xgb": {
        "learning_rate": (0.01, 0.3, "float", True),
        "max_depth": (3, 12, "int", False),
        "min_child_weight": (1.0, 100.0, "float", True),
        "subsample": (0.5, 1.0, "float", False),
        "colsample_bytree": (0.5, 1.0, "float", False),
        "reg_lambda": (1e-3, 10.0, "float", True),
        "reg_alpha": (1e-3, 10.0, "float", True),
        "gamma": (1e-3, 5.0, "float", True),
    },
    "lgbm": {
        "learning_rate": (0.01, 0.3, "float", True),
        "num_leaves": (15, 255, "int", True),
        "min_child_samples": (5, 300, "int", True),
        "subsample": (0.5, 1.0, "float", False),
        "colsample_bytree": (0.5, 1.0, "float", False),
        "reg_lambda": (1e-3, 10.0, "float", True),
        "reg_alpha": (1e-3, 10.0, "float", True),
    },
}
FIXED = {"xgb": {}, "lgbm": {"subsample_freq": 1}}


def distributions(model):
    out = {}
    for k, (lo, hi, kind, log) in SPACES[model].items():
        out[k] = (optuna.distributions.IntDistribution(lo, hi, log=log) if kind == "int"
                  else optuna.distributions.FloatDistribution(lo, hi, log=log))
    return out


def suggest(trial, model):
    p = {}
    for k, (lo, hi, kind, log) in SPACES[model].items():
        p[k] = (trial.suggest_int(k, lo, hi, log=log) if kind == "int"
                else trial.suggest_float(k, lo, hi, log=log))
    return p


def clip(model, params):
    """Round integer hyperparameters and clip to the ranges.
    Returns (params, flags) with flags from {"rounded", "clipped"}."""
    out, flags = {}, set()
    for k, (lo, hi, kind, _) in SPACES[model].items():
        if isinstance(params[k], bool):
            raise ValueError(f"{k} is boolean")
        v = float(params[k])
        if not math.isfinite(v):
            raise ValueError(f"{k} is not finite")
        if kind == "int" and v != round(v):
            v = float(round(v))
            flags.add("rounded")
        w = min(max(v, lo), hi)
        if w != v:
            flags.add("clipped")
        out[k] = int(w) if kind == "int" else float(w)
    return out, sorted(flags)


def in_space(model, params):
    if set(params) != set(SPACES[model]):
        return False
    for k, (lo, hi, kind, _) in SPACES[model].items():
        v = params[k]
        if kind == "int" and int(v) != v:
            return False
        if not (lo <= v <= hi):
            return False
    return True


def describe(model):
    """Plain-text search space for the prompt."""
    lines = []
    for k, (lo, hi, kind, log) in SPACES[model].items():
        lines.append(f"- {k}: {kind}, {lo} to {hi}{', log scale' if log else ''}")
    return "\n".join(lines)
