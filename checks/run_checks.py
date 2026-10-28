"""
Checks C1-C7 before the freeze (MAKE_SI_FD/VS_students/hpo_llm_protocol.md, section 10).

    python -I experiments/hpo_llm/checks/run_checks.py [C1 C2 ...]

Writes checks/report.json and prints one line per check. C7 uses the scripted LLM test double;
its test scores are pipeline checks only and are not part of any analysis.
"""

import ast
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parents[1]
sys.path.insert(0, str(PKG))
WORK = HERE / "work"
PY = sys.executable

import data as D        # noqa: E402
import llm as L         # noqa: E402
import space as S       # noqa: E402

REPORT = {}


def record(name, ok, **info):
    REPORT[name] = dict(ok=bool(ok), **info)
    print(f"{name}: {'PASS' if ok else 'FAIL'}  {json.dumps(info, default=str)[:400]}", flush=True)


# ------------------------------------------------------------------ C1 test-period guard
def imports(path):
    tree = ast.parse(path.read_text())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
    return out


def c1():
    search_side = ["search.py", "data.py", "llm.py", "models.py", "space.py"]
    static = {f: sorted(imports(PKG / f) & {"testdata", "score"}) for f in search_side}
    static_ok = not any(static.values())
    # runtime: importing and running the search code must not load testdata
    code = ("import sys; sys.path.insert(0, %r); import search; "
            "print('testdata' in sys.modules, 'score' in sys.modules)") % str(PKG)
    p = subprocess.run([PY, "-I", "-c", code], capture_output=True, text=True)
    runtime_ok = p.stdout.strip() == "False False"
    # row-level: the training period holds no test row and ends before the test period starts
    rows = {}
    for name in D.DATASETS:
        d, spec = D._load(name)
        tr, te = D._period_masks(d, spec, name)
        tp = D.training_period(name)
        overlap = len(np.intersect1d(tp.row_index, np.flatnonzero(te)))
        t_tr, t_te = d.loc[tr, spec["time"]].max(), d.loc[te, spec["time"]].min()
        rows[name] = dict(n_train=len(tp.frame), overlap=overlap, last_train_t=float(t_tr), first_test_t=float(t_te),
                          ok=overlap == 0 and t_tr <= t_te and len(tp.frame) == tr.sum())
    rows_ok = all(r["ok"] for r in rows.values())
    record("C1", static_ok and runtime_ok and rows_ok, static_imports=static,
           runtime=p.stdout.strip() or p.stderr[-200:], rows=rows)


# ------------------------------------------------------------------ C2 loader equivalence
def c2():
    sys.path.insert(0, str(ROOT / "experiments" / "bench"))
    import datasets as OLD
    import testdata as T
    out = {}
    for name in D.DATASETS:
        Xtr, Xte, ytr, yte, names, _ = OLD.get(name, "time", 1)
        Ntr, nytr, Nte, nyte, nnames, _ = T.full_fit_and_test(name)
        diff = max(float(np.nanmax(np.abs(Xtr - Ntr))) if Xtr.size else 0.0,
                   float(np.nanmax(np.abs(Xte - Nte))) if Xte.size else 0.0)
        nan_same = bool((np.isnan(Xtr) == np.isnan(Ntr)).all() and (np.isnan(Xte) == np.isnan(Nte)).all())
        out[name] = dict(shape=list(Ntr.shape) + list(Nte.shape), max_abs_diff=diff, nan_pattern_same=nan_same,
                         names_same=list(names) == list(nnames),
                         labels_same=bool((ytr == nytr).all() and (yte == nyte).all()))
        out[name]["ok"] = diff == 0 and nan_same and out[name]["names_same"] and out[name]["labels_same"]
    record("C2", all(v["ok"] for v in out.values()), datasets=out)


# ------------------------------------------------------------------ C3 re-encoding inside the training period
def c3():
    # (a) synthetic: a category that occurs only in the validation fold gets the inner-train prior and frequency 0
    n = 1000
    rng = np.random.default_rng(0)
    f = pd.DataFrame(dict(t=np.arange(n), x=rng.normal(size=n), y=(rng.random(n) < 0.05).astype(int),
                          c=rng.choice(["A", "B"], n)))
    f.loc[f.t >= 900, "c"] = "Z"                       # Z only in the last 10% (inside the validation fold)
    f.loc[(f.c == "Z") & (f.index % 2 == 0), "y"] = 1   # Z is very fraud-heavy, so leakage would show
    spec = dict(time="t", label="y", cats=["c"], dropna=True, num=["x"])
    tp = D.TrainPeriod("synthetic", f, spec, np.arange(n))
    Xa, ya, Xb, yb, names = tp.inner("time")
    a, b = tp.fold_masks("time")
    prior = f.loc[a, "y"].mean()
    z = (f.loc[b, "c"] == "Z").values
    rate_col, freq_col = names.index("c_rate"), names.index("c_freq")
    synth_ok = bool(np.allclose(Xb[z, rate_col], prior) and (Xb[z, freq_col] == 0).all())
    # (b) real data: validation-fold features must not depend on validation labels
    real = {}
    for name in ("ieee", "baf"):
        tp = D.training_period(name)
        Xa1, _, Xb1, _, _ = tp.inner("time")
        g = tp.frame.copy()
        _, bm = tp.fold_masks("time")
        g.loc[bm, tp.spec["label"]] = 1 - g.loc[bm, tp.spec["label"]]      # flip every validation label
        Xa2, _, Xb2, _, _ = D.TrainPeriod(name, g, tp.spec, tp.row_index).inner("time")
        same = bool(np.array_equal(Xa1, Xa2, equal_nan=True) and np.array_equal(Xb1, Xb2, equal_nan=True))
        # and the old full-period encoding would have differed on the validation rows
        Xfull, _, _ = D.encode(tp.frame, [], tp.spec)
        changed = float(np.mean(np.any(np.abs(np.nan_to_num(Xfull[bm]) - np.nan_to_num(Xb1)) > 0, axis=1)))
        real[name] = dict(invariant_to_val_labels=same, share_val_rows_encoded_differently_from_datasets_py=round(changed, 4))
    record("C3", synth_ok and all(r["invariant_to_val_labels"] for r in real.values()),
           synthetic_planted_category=synth_ok, real=real)


# ------------------------------------------------------------------ C4 search-space conformity
def c4():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = {}
    for model in S.SPACES:
        res = {}
        for name, sampler in (("O1", optuna.samplers.RandomSampler(seed=1)), ("O2", optuna.samplers.TPESampler(seed=1))):
            st = optuna.create_study(direction="maximize", sampler=sampler)
            bad = 0
            for _ in range(1000):
                tr = st.ask()
                p = S.suggest(tr, model)
                bad += not S.in_space(model, p)
                st.tell(tr, float(np.random.default_rng(len(st.trials)).random()))
            res[name] = bad
        rng = np.random.default_rng(7)
        res["fallback_random"] = sum(not S.in_space(model, L.random_params(model, rng)) for _ in range(1000))
        # LLM answers drawn far outside the ranges, with floats for ints: must be clipped into the space
        bad, clipped = 0, 0
        for _ in range(1000):
            raw = {k: float(rng.uniform(lo - 2 * (hi - lo), hi + 2 * (hi - lo))) for k, (lo, hi, _, _) in S.SPACES[model].items()}
            p, flags = L.parse_answer(json.dumps(raw), model)
            bad += not S.in_space(model, p)
            clipped += "clipped" in flags
        res["llm_wild_answers_out_of_space"] = bad
        res["llm_wild_answers_flagged_clipped"] = clipped
        out[model] = res
    ok = all(v["O1"] == 0 and v["O2"] == 0 and v["fallback_random"] == 0 and v["llm_wild_answers_out_of_space"] == 0
             for v in out.values())
    record("C4", ok, draws_per_arm=1000, out_of_space=out)


# ------------------------------------------------------------------ C5 LLM answer parser
def c5():
    good = {"learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.01, "gamma": 0.1}
    J = json.dumps
    cases = [
        ("plain JSON", J(good), "ok", []),
        ("JSON in prose and code fence", "Sure.\n```json\n" + J(good) + "\n```\nGood luck", "ok", []),
        ("out of range", J({**good, "max_depth": 40, "subsample": 1.7}), "ok", ["clipped"]),
        ("float for int", J({**good, "max_depth": 6.6}), "ok", ["rounded"]),
        ("extra key", J({**good, "n_estimators": 500}), "ok", ["extra_keys"]),
        ("missing key", J({k: v for k, v in good.items() if k != "gamma"}), "error", None),
        ("no JSON", "I would use a learning rate of 0.05.", "error", None),
        ("broken JSON", '{"learning_rate": 0.05, "max_depth": }', "error", None),
        ("string value", J({**good, "subsample": "high"}), "error", None),
        ("NaN value", J(good).replace("0.8,", "NaN,", 1), "error", None),
        ("empty", "", "error", None),
        ("boolean value", J({**good, "subsample": True}), "error", None),
    ]
    res = []
    for label, text, want, flags in cases:
        try:
            p, f = L.parse_answer(text, "xgb")
            got = "ok" if S.in_space("xgb", p) else "out_of_space"
            ok = want == "ok" and got == "ok" and all(x in f for x in flags)
        except L.ParseError as e:
            got, f, ok = "error", [str(e)], want == "error"
        res.append(dict(case=label, want=want, got=got, flags=f, ok=ok))
    # retry and fallback rule
    rng = np.random.default_rng(0)
    p1, f1, n1 = L.propose(L.ScriptedLLM(["not json", J(good)]), "prompt", "xgb", 0, rng)
    p2, f2, n2 = L.propose(L.ScriptedLLM(["not json", "still not"]), "prompt", "xgb", 0, rng)
    rule_ok = (f1 == ["retried"] and n1 == 2 and p1 == S.clip("xgb", good)[0]
               and f2 == ["fallback"] and n2 == 2 and S.in_space("xgb", p2))
    # log and cache: a repeated request is served from the cache, and both calls are logged
    WORK.mkdir(parents=True, exist_ok=True)
    lg, ca = WORK / "c5_log.jsonl", WORK / "c5_cache.jsonl"
    for x in (lg, ca):
        x.unlink(missing_ok=True)
    inner = L.ScriptedLLM([J(good)])
    w = L.LoggedLLM(inner, lg, ca)
    w.complete("same prompt", 1)
    w.complete("same prompt", 1)
    w2 = L.LoggedLLM(inner, lg, ca)            # a fresh process reloads the cache from disk
    w2.complete("same prompt", 1)
    logged = [json.loads(x) for x in lg.read_text().splitlines()]
    cache_ok = inner.calls == 1 and len(logged) == 3 and [r["cache_hit"] for r in logged] == [False, True, True]
    record("C5", all(r["ok"] for r in res) and rule_ok and cache_ok,
           cases=[(r["case"], r["got"], r["ok"]) for r in res], retry_fallback_rule=rule_ok, log_and_cache=cache_ok)


# ------------------------------------------------------------------ C6 resume after a killed queue
def c6():
    out = WORK / "c6_search"
    shutil.rmtree(out, ignore_errors=True)
    cmd = [PY, "-I", str(PKG / "queue_runs.py"), "--blocks", "B1", "--datasets", "ulb", "--models", "xgb",
           "--reps", "0", "1", "--budget", "3", "--out", str(out)]
    expected = 2 * 2 * 2                                  # O1, O2 x two objectives x two repeats
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600 and len(list(out.glob("*.json"))) < 2:
        time.sleep(0.2)
    time.sleep(1.5)                                       # the third job is now loading data or fitting
    os.killpg(p.pid, signal.SIGKILL)                      # kill the queue and its running job together
    p.wait()
    time.sleep(1)
    killed_mid_job = len(list(out.glob("*.json"))) < expected
    before = {f.name: f.stat().st_mtime for f in out.glob("*.json")}
    subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    after = {f.name: f.stat().st_mtime for f in out.glob("*.json")}
    valid = all(json.loads((out / n).read_text())["n_trials"] == 3 for n in after)
    untouched = all(after.get(n) == m for n, m in before.items())
    tmp_left = [f.name for f in out.glob("*.tmp")]
    log = [json.loads(x) for x in (out / "queue_log.jsonl").read_text().splitlines()]
    reruns = [r for r in log if r["rc"] == 0 and "skip" in r["stdout"]]
    record("C6", killed_mid_job and len(after) == expected and valid and untouched,
           killed_mid_job=killed_mid_job, results_before_kill=len(before), results_after_resume=len(after), expected=expected,
           finished_results_not_rerun=untouched, all_valid=valid, tmp_files_left=tmp_left,
           skipped_on_resume=len(reruns))


# ------------------------------------------------------------------ C7 smoke test
def c7(llm="scripted", tag="C7"):
    base = WORK if tag == "C7" else WORK / tag.lower()
    out, tout = base / "c7_search", base / "c7_test"
    for d in (out, tout, base / "llm_logs"):
        shutil.rmtree(d, ignore_errors=True)
    arms = [("O0", "pr_auc", "time", "anon"), ("O0", "log_loss", "time", "anon"),
            ("O0LLM", "pr_auc", "time", "anon"), ("O0LLM", "pr_auc", "time", "named"),
            ("O1", "pr_auc", "time", "anon"), ("O2", "pr_auc", "time", "anon"),
            ("O2", "log_loss", "time", "anon"), ("O2", "pr_auc", "random", "anon"),
            ("O3", "pr_auc", "time", "anon"), ("O3", "log_loss", "time", "anon"),
            ("O3", "pr_auc", "time", "named"), ("O4", "pr_auc", "time", "anon")]
    runs = []
    for model in ("xgb", "lgbm"):
        for opt, obj, fold, cond in arms:
            cmd = [PY, "-I", str(PKG / "search.py"), "--dataset", "ulb", "--model", model, "--optimiser", opt,
                   "--objective", obj, "--fold", fold, "--condition", cond, "--rep", "0", "--budget", "7",
                   "--llm", llm, "--out", str(out)]
            t0 = time.time()
            p = subprocess.run(cmd, capture_output=True, text=True)
            runs.append(dict(arm=f"{model}/{opt}/{obj}/{fold}/{cond}", rc=p.returncode, s=round(time.time() - t0, 1),
                             err=p.stderr[-300:] if p.returncode else ""))
    s = subprocess.run([PY, "-I", str(PKG / "score.py"), "--search", str(out), "--out", str(tout)],
                       capture_output=True, text=True)
    checks = []
    for f in sorted(out.glob("*.json")):
        r = json.loads(f.read_text())
        t = json.loads((tout / f.name).read_text()) if (tout / f.name).exists() else None
        single = r["args"]["optimiser"] in ("O0", "O0LLM")
        want_trials = 1 if single else 7
        finite = t is not None and all(np.isfinite(v) for c in t["checkpoints"].values()
                                       for k, v in c.items() if k != "trial")
        no_test_in_search = not any(k in json.dumps(r) for k in ("recall_at_fpr", "sum_p_over_frauds", "brier"))
        src = [x["source"] for x in r["trials"]]
        o4_ok = r["args"]["optimiser"] != "O4" or src == ["llm"] * 5 + ["optuna"] * 2
        checks.append(dict(job=f.stem, n_trials=r["n_trials"], ok=r["n_trials"] == want_trials and finite and no_test_in_search and o4_ok,
                           checkpoints=list(r["incumbents"])))
    extra = {}
    if llm != "scripted":                       # live run: tokens, flags and model from the logs
        logs = [json.loads(x) for f in (base / "llm_logs").glob("*.jsonl") if f.name != "cache.jsonl"
                for x in f.read_text().splitlines()]
        u = [r["usage"] for r in logs if r["usage"]]
        flags = [fl for f in out.glob("*.json") for t in json.loads(f.read_text())["trials"] for fl in t["flags"]]
        extra = dict(llm_calls=len(logs), models=sorted({x.get("model") for x in u}),
                     input_tokens=sum(x.get("input_tokens", 0) for x in u),
                     output_tokens=sum(x.get("output_tokens", 0) for x in u),
                     stop_reasons={k: sum(x.get("stop_reason") == k for x in u) for k in {x.get("stop_reason") for x in u}},
                     with_thinking_summary=sum(bool(x.get("thinking_summary")) for x in u),
                     flag_counts={k: flags.count(k) for k in set(flags)})
        anon = [json.loads(x)["response"] for f in (base / "llm_logs").glob("*_anon_*.jsonl")
                for x in f.read_text().splitlines()]
        named = [json.loads(x)["response"] for f in (base / "llm_logs").glob("*_named_*.jsonl")
                 for x in f.read_text().splitlines()]
        extra.update(anon_answers=len(anon), anon_identifying=sum(L.mentions_dataset(t) for t in anon),
                     named_answers=len(named), named_identifying=sum(L.mentions_dataset(t) for t in named),
                     answers_with_prose=sum(not t.strip().startswith("{") for t in anon + named))
    record(tag, all(x["rc"] == 0 for x in runs) and s.returncode == 0 and len(checks) == len(runs) and all(c["ok"] for c in checks),
           arms=len(runs), failed=[x for x in runs if x["rc"]], score_rc=s.returncode,
           bad=[c for c in checks if not c["ok"]], wall_s=round(sum(x["s"] for x in runs), 1),
           per_arm_s={x["arm"]: x["s"] for x in runs}, **extra)


# ------------------------------------------------------------------ C5L / C7L: the same checks against the real API (V4a)
def c5l():
    import llm as LL
    WORK.mkdir(parents=True, exist_ok=True)
    lg, ca = WORK / "c5l_log.jsonl", WORK / "c5l_cache.jsonl"
    for x in (lg, ca):
        x.unlink(missing_ok=True)
    tp = D.training_period("ulb")
    _, _, _, _, names = tp.inner("time")
    desc = tp.describe("time")
    hist = [dict(trial=1, params=L.random_params("xgb", np.random.default_rng(1)), value=0.70, trees=250)]
    res = {}
    for cond in ("anon", "named"):
        prompt = L.build_prompt(cond, "ulb", "xgb", "pr_auc", desc, names, hist, 2, 40)
        live = L.LoggedLLM(LL.AnthropicLLM(), lg, ca, scope="c5l_" + cond)
        p, flags, n = L.propose(live, prompt, "xgb", 1, np.random.default_rng(0))
        again, usage2 = live.complete(prompt, 1)                    # same request: from the cache
        res[cond] = dict(params_in_space=S.in_space("xgb", p), flags=flags, calls=n,
                         cache_hit_on_repeat=usage2.get("cache_hit") is True)
    logged = [json.loads(x) for x in lg.read_text().splitlines()]
    first = [r for r in logged if not r["cache_hit"]]
    u = [r["usage"] for r in first]
    usage_ok = all(x.get("input_tokens", 0) > 0 and x.get("output_tokens", 0) > 0 and x.get("model") == L.MODEL_ID
                   and x.get("stop_reason") == "end_turn" for x in u)

    class Refuser:                                                  # refusal path (A5), no network
        model_id = "refusal-test-double"

        def complete(self, prompt, seed):
            return "", dict(stop_reason="refusal")
    pr, fr, nr = L.propose(Refuser(), "prompt", "xgb", 0, np.random.default_rng(0))
    refusal_ok = fr == ["refusal", "fallback"] and nr == 2 and S.in_space("xgb", pr)
    ok = all(v["params_in_space"] and v["cache_hit_on_repeat"] and "fallback" not in v["flags"] for v in res.values()) \
        and usage_ok and refusal_ok
    record("C5L", ok, conditions=res, api_calls=len(first),
           tokens=dict(input=sum(x["input_tokens"] for x in u), output=sum(x["output_tokens"] for x in u)),
           thinking_summaries=sum(bool(x.get("thinking_summary")) for x in u), usage_logged=usage_ok,
           refusal_rule=refusal_ok)


CHECKS = dict(C1=c1, C2=c2, C3=c3, C4=c4, C5=c5, C6=c6, C7=c7, C5L=c5l, C7L=lambda: c7("anthropic", "C7L"))

if __name__ == "__main__":
    todo = sys.argv[1:] or list(CHECKS)
    rp = HERE / "report.json"
    if rp.exists():
        REPORT.update(json.loads(rp.read_text()))
    for c in todo:
        try:
            CHECKS[c]()
        except Exception as e:  # a crash is a failure, recorded with its message
            import traceback
            record(c, False, error=repr(e), trace=traceback.format_exc()[-1500:])
        rp.write_text(json.dumps(REPORT, indent=1, default=str))
