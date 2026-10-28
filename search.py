"""
One search job (protocol sections 4 and 7). Sees the training period only; writes the trials and
the incumbents at k = 5, 10, 20, 40 to results/search/<job>.json. Test scoring is score.py.

    python -I experiments/hpo_llm/search.py --dataset ulb --model xgb --optimiser O2 \
        --objective pr_auc --fold time --condition anon --rep 0 [--budget 40] [--llm scripted]

Must not import testdata (check C1).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D      # noqa: E402
import llm as L       # noqa: E402
import models as M    # noqa: E402
import space as S     # noqa: E402

OPTIMISERS = ("O0", "O0LLM", "O1", "O2", "O3", "O4")
CHECKPOINTS = (5, 10, 20, 40)
WARM = 5                                 # O4: LLM trials before TPE
TPE_STARTUP = 10                         # D1 (28 Oct): Optuna default, the same for O2 and O4
TIMEOUT_S = 4 * 3600
OUT = HERE / "results"


def job_id(a):
    tag = f"_{a.llm_tag}" if getattr(a, "llm_tag", "") else ""   # S3 arm: separate result files and cache
    return f"{a.dataset}_{a.model}_{a.optimiser}_{a.objective}_{a.fold}_{a.condition}_r{a.rep}_b{a.budget}{tag}"


def seeds(rep):
    return dict(optimiser=1000 + rep, model=rep, llm=2000 + rep)


def make_llm(name, out_dir, rep):
    if name == "scripted":                   # test double for the checks: valid random settings
        rng = np.random.default_rng(3000 + rep)
        return L.ScriptedLLM(lambda prompt, n, rng=rng, m=None: json.dumps(_scripted(prompt, rng)))
    if name == "anthropic":                  # V4a: Claude Opus 5.5 through the Claude API
        return L.AnthropicLLM()
    if name == "anthropic-haiku":            # V4b: Claude Haiku 5.5, S3 sensitivity arm
        return L.AnthropicLLM(model_id=L.HAIKU_ID)
    raise ValueError(f"unknown LLM adapter: {name}")


def _scripted(prompt, rng):
    model = "lgbm" if "LightGBM" in prompt else "xgb"
    return L.random_params(model, rng)


def incumbents(trials, budget):
    """Best trial among the first k, for each checkpoint the search reached (single-fit arms: k = 1)."""
    out = {}
    for k in [k for k in CHECKPOINTS if k <= budget] or [budget]:
        if len(trials) < k:
            continue
        done = trials[:k]
        best = max(done, key=lambda t: (t["value"], -t["trial"]))   # ties go to the earlier trial
        out[str(k)] = dict(trial=best["trial"], params=best["params"], best_iteration=best["trees"],
                           value=best["value"], n_trees_final=M.final_trees(best["trees"]))
    return out


def run(a, llm_name="none", logdir=OUT / "llm_logs"):
    sd = seeds(a.rep)
    tp = D.training_period(a.dataset)
    Xa, ya, Xb, yb, names = tp.inner(a.fold)
    desc = tp.describe(a.fold)
    rng = np.random.default_rng(sd["optimiser"])
    trials, t0 = [], time.time()
    llm = None
    if a.optimiser in ("O0LLM", "O3", "O4"):
        logdir.mkdir(parents=True, exist_ok=True)
        llm = L.LoggedLLM(make_llm(llm_name, logdir, a.rep), logdir / f"{job_id(a)}.jsonl",
                          logdir / "cache.jsonl", scope=job_id(a))
    seen = set()
    budget = 1 if a.optimiser in ("O0", "O0LLM") else a.budget

    def evaluate(params, flags, source):
        p, trees = M.fit_early(a.model, params, Xa, ya, Xb, yb, a.objective, sd["model"])
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            flags = flags + ["duplicate"]
        seen.add(key)
        t = dict(trial=len(trials) + 1, params=params, value=M.val_score(a.objective, yb, p),
                 trees=trees, flags=flags, source=source, t_s=round(time.time() - t0, 2))
        trials.append(t)
        return t["value"]

    def llm_trial():
        prompt = L.build_prompt(a.condition, a.dataset, a.model, a.objective, desc, names,
                                trials, len(trials) + 1, budget)
        params, flags, _ = L.propose(llm, prompt, a.model, sd["llm"] * 1000 + len(trials), rng)
        return params, flags

    if a.optimiser == "O0":
        evaluate({}, [], "defaults")
    elif a.optimiser == "O0LLM":
        evaluate(*llm_trial(), "llm")
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = (optuna.samplers.RandomSampler(seed=sd["optimiser"]) if a.optimiser == "O1"
                   else optuna.samplers.TPESampler(seed=sd["optimiser"], n_startup_trials=TPE_STARTUP))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        dists = S.distributions(a.model)
        while len(trials) < budget and time.time() - t0 < TIMEOUT_S:
            if a.optimiser == "O3" or (a.optimiser == "O4" and len(trials) < WARM):
                params, flags = llm_trial()
                v = evaluate(params, flags, "llm")
                if a.optimiser == "O4":          # TPE learns from the LLM trials (protocol section 4)
                    study.add_trial(optuna.trial.create_trial(params=params, distributions=dists, value=v))
            else:
                tr = study.ask()
                params = S.suggest(tr, a.model)
                study.tell(tr, evaluate(params, [], "optuna"))
    res = dict(job=job_id(a), args=vars(a), seeds=sd, llm=getattr(llm.client, "model_id", None) if llm else None,
               describe=desc, n_trials=len(trials), time_s=round(time.time() - t0, 2),
               timed_out=len(trials) < budget, trials=trials, incumbents=incumbents(trials, budget))
    return res


def write_atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)                     # a killed job leaves no partial result file


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=D.DATASETS, required=True)
    ap.add_argument("--model", choices=list(S.SPACES), required=True)
    ap.add_argument("--optimiser", choices=OPTIMISERS, required=True)
    ap.add_argument("--objective", choices=M.OBJECTIVES, default="pr_auc")
    ap.add_argument("--fold", choices=("time", "random"), default="time")
    ap.add_argument("--condition", choices=("anon", "named"), default="anon")
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--llm", default="none")
    ap.add_argument("--out", default=str(OUT / "search"))
    return ap


def main():
    a = parser().parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm_name = a.llm
    a.llm_tag = "haiku" if llm_name == "anthropic-haiku" else ""
    del a.llm, a.out
    path = out / f"{job_id(a)}.json"
    if path.exists():
        print("skip", path.name)
        return
    write_atomic(path, run(a, llm_name, out.parent / "llm_logs"))
    print("done", path.name, flush=True)


if __name__ == "__main__":
    main()
