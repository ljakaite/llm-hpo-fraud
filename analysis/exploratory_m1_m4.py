"""
Exploratory analyses M1 (T3-T8) and M4 (a-c), declared in protocol/deviations.md (commit 3aec1e0)
before any test scoring. Reads only results/main/search (validation scores) and results/main/llm_logs;
never reads results/main/test. T6-T8 make new Claude calls (same settings as protocol section 5).

    python -I experiments/hpo_llm/analysis/exploratory_m1_m4.py --part local     # T3, T4, T5, M4
    python -I experiments/hpo_llm/analysis/exploratory_m1_m4.py --part llm       # T6, T7, T8 (API)
"""

import argparse
import collections
import glob
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import llm as L      # noqa: E402
import space as S    # noqa: E402

MAIN = HERE / "results" / "main"
OUT = HERE / "results" / "exploratory"
CELLS = [(d, m) for d in ("ieee", "baf", "ulb") for m in ("xgb", "lgbm")]


def norm(model, p):
    v = []
    for k, (lo, hi, kind, log) in S.SPACES[model].items():
        x = float(p[k])
        v.append((math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo)) if log else (x - lo) / (hi - lo))
    return np.array(v)


def searches():
    for f in sorted((MAIN / "search").glob("*.json")):
        r = json.loads(f.read_text())
        a = r["args"]
        arm = a["optimiser"] + ("-haiku" if a.get("llm_tag") else "") + ("-named" if a["condition"] == "named" else "")
        if a["fold"] == "random":
            arm += "-randomfold"
        yield (a["dataset"], a["model"]), arm, a, r


# ------------------------------------------------------------------ local analyses
def t3_t4_t5():
    dist = collections.defaultdict(list)
    tb_hits = collections.defaultdict(list)
    zs = {}
    TEXTBOOK = {"xgb": dict(learning_rate=(0.1, 0.05), max_depth=(6,), subsample=(0.8,), colsample_bytree=(0.8,), min_child_weight=(1.0,)),
                "lgbm": dict(learning_rate=(0.1, 0.05), num_leaves=(31,), min_child_samples=(20,), subsample=(0.8,), colsample_bytree=(0.8,))}

    def textbook(model, p):
        return sum(any(abs(float(p[k]) - v) < 1e-9 for v in vals) for k, vals in TEXTBOOK[model].items()) >= 3

    for cell, arm, a, r in searches():
        tr = r["trials"]
        m = a["model"]
        if a["optimiser"] in ("O1", "O2", "O3", "O4") and a["fold"] == "time":
            for t in range(3, len(tr)):                      # trial index >= 4 (1-based)
                if a["optimiser"] in ("O3", "O4") and tr[t]["source"] != "llm":
                    continue
                inc = max(tr[:t], key=lambda z: (z["value"], -z["trial"]))
                dist[(cell, arm)].append(float(np.linalg.norm(norm(m, tr[t]["params"]) - norm(m, inc["params"]))))
        if a["optimiser"] == "O0LLM":
            tb_hits[("O0LLM", cell)].append(textbook(m, tr[0]["params"]))
            zs[(cell, a["condition"], a["rep"])] = tr[0]["params"]
        if a["optimiser"] == "O3" and tr and tr[0]["source"] == "llm":
            tb_hits[("O3-trial1" + ("-haiku" if a.get("llm_tag") else ""), cell)].append(textbook(m, tr[0]["params"]))
    t3 = {f"{c[0]}/{c[1]} {arm}": round(st.median(v), 3) for (c, arm), v in sorted(dist.items())}
    t4 = {f"{k[0]} {k[1][0]}/{k[1][1]}": round(float(np.mean(v)), 2) for k, v in sorted(tb_hits.items())}
    t5 = {}
    for cell in CELLS:
        same, d = [], []
        for rep in range(10):
            a_, b_ = zs.get((cell, "anon", rep)), zs.get((cell, "named", rep))
            if a_ and b_:
                same.append(a_ == b_)
                d.append(float(np.linalg.norm(norm(cell[1], a_) - norm(cell[1], b_))))
        t5[f"{cell[0]}/{cell[1]}"] = dict(identical_share=round(float(np.mean(same)), 2), mean_distance=round(float(np.mean(d)), 3))
    return dict(T3_median_distance_to_incumbent=t3, T4_textbook_share=t4, T5_zero_shot_named_vs_anon=t5)


def m4():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    PAIRS = {"xgb": [("learning_rate", "trees"), ("max_depth", "min_child_weight"), ("subsample", "colsample_bytree")],
             "lgbm": [("learning_rate", "trees"), ("num_leaves", "min_child_samples"), ("subsample", "colsample_bytree")]}
    LOGK = {"learning_rate", "trees", "min_child_weight", "num_leaves", "min_child_samples"}

    def val(t, k):
        x = t["trees"] if k == "trees" else t["params"][k]
        return math.log(x) if k in LOGK else float(x)

    pool = collections.defaultdict(list)
    late = collections.defaultdict(list)
    for cell, arm, a, r in searches():
        if a["fold"] != "time" or a["objective"] != "pr_auc":
            continue
        if a["optimiser"] == "O1":
            pool[cell] += r["trials"]
        if a["optimiser"] in ("O1", "O2", "O3", "O4") and len(r["trials"]) >= 40:
            late[(cell, arm)].append(r["trials"][20:40])
    out_a, out_b, out_c = {}, {}, {}
    for cell, trials in sorted(pool.items()):
        m = cell[1]
        keys = list(S.SPACES[m])
        X = np.array([[val(t, k) for k in keys] for t in trials])
        y = np.array([t["value"] for t in trials])
        Z = StandardScaler().fit_transform(X)
        r2 = lambda F: float(cross_val_score(RidgeCV(alphas=np.logspace(-3, 3, 13)), F, y, cv=5, scoring="r2").mean())
        rf = float(cross_val_score(RandomForestRegressor(300, min_samples_leaf=3, random_state=0), Z, y, cv=5, scoring="r2").mean())
        out_a[f"{cell[0]}/{m}"] = dict(n=len(y), additive=round(r2(np.c_[Z, Z ** 2]), 3),
                                       pairwise=round(r2(PolynomialFeatures(2, include_bias=False).fit_transform(Z)), 3), forest=round(rf, 3))
        top = [t for t, v in zip(trials, y) if v >= np.quantile(y, 0.75)]
        out_b[f"{cell[0]}/{m}"] = {f"{p}~{q}": round(float(np.corrcoef([val(t, p) for t in top], [val(t, q) for t in top])[0, 1]), 2)
                                   for p, q in PAIRS[m]}
    for (cell, arm), blocks in sorted(late.items()):
        m = cell[1]
        cs = []
        for b in blocks:
            for p, q in PAIRS[m]:
                u, w = [val(t, p) for t in b], [val(t, q) for t in b]
                if np.std(u) > 1e-12 and np.std(w) > 1e-12:
                    cs.append(abs(float(np.corrcoef(u, w)[0, 1])))
        out_c[f"{cell[0]}/{m} {arm}"] = round(float(np.mean(cs)), 3) if cs else None
    return dict(M4a_cv_r2=out_a, M4b_top25_pair_corr_O1=out_b, M4c_late_trial_abs_pair_corr=out_c)


# ------------------------------------------------------------------ LLM probes
HIST = re.compile(r"^trial (\d+): (\{.*\}) -> score (-?[0-9.]+), trees (\d+)$", re.M)


def t6_t7_t8():
    OUT.mkdir(parents=True, exist_ok=True)
    client = L.LoggedLLM(L.AnthropicLLM(), OUT / "probe_log.jsonl", OUT / "probe_cache.jsonl", scope="exploratory")
    rng = np.random.default_rng(20261105)
    res = {}
    # ---- T6 counterfactual history
    t6 = []
    for d, m in CELLS:
        cand = []
        for f in sorted((MAIN / "llm_logs").glob(f"{d}_{m}_O3_pr_auc_time_anon_r*_b40.jsonl")):
            for line in f.read_text().splitlines():
                rec = json.loads(line)
                k = re.search(r"Propose the setting for trial (\d+) of 40", rec["prompt"])
                if k and 10 <= int(k.group(1)) <= 30 and "could not be used" not in rec["prompt"]:
                    cand.append(rec["prompt"])
        pick = [cand[i] for i in rng.choice(len(cand), size=min(10, len(cand)), replace=False)]
        prng = np.random.default_rng(20261106)
        for prompt in pick:
            hist = HIST.findall(prompt)
            scores = np.array([float(h[2]) for h in hist])
            params = [json.loads(h[1]) for h in hist]
            true_best = int(np.argmax(scores))
            variants = {"logged": scores, "permuted": prng.permutation(scores)}
            order = np.argsort(scores)
            rev = np.empty_like(scores)
            rev[order] = scores[order[::-1]]
            variants["reversed"] = rev
            for name, sc in variants.items():
                new = prompt
                for (tnum, pj, s_old, trees), s_new in zip(hist, sc):
                    new = new.replace(f"trial {tnum}: {pj} -> score {s_old}, trees {trees}",
                                      f"trial {tnum}: {pj} -> score {s_new:.6f}, trees {trees}", 1)
                text, _ = client.complete(new, 0)
                try:
                    p, _ = L.parse_answer(text, m)
                except L.ParseError:
                    t6.append(dict(cell=f"{d}/{m}", variant=name, parsed=False))
                    continue
                shown_best = int(np.argmax(sc))
                x = norm(m, p)
                d_shown = float(np.linalg.norm(x - norm(m, params[shown_best])))
                d_true = float(np.linalg.norm(x - norm(m, params[true_best])))
                t6.append(dict(cell=f"{d}/{m}", variant=name, parsed=True, d_shown=d_shown, d_true=d_true,
                               closer_to_shown=d_shown < d_true, same_best=shown_best == true_best))
    summ = {}
    for v in ("logged", "permuted", "reversed"):
        rows = [r for r in t6 if r["variant"] == v and r["parsed"]]
        moved = [r for r in rows if not r["same_best"]]
        summ[v] = dict(n=len(rows), mean_dist_to_shown_best=round(float(np.mean([r["d_shown"] for r in rows])), 3),
                       mean_dist_to_true_best=round(float(np.mean([r["d_true"] for r in rows])), 3),
                       share_closer_to_shown_when_different=round(float(np.mean([r["closer_to_shown"] for r in moved])), 2) if moved else None,
                       n_different=len(moved))
    res["T6_counterfactual_history"] = summ
    # ---- T7 identification probe
    t7 = collections.defaultdict(list)
    TRUE = {"ulb": ("ulb", "credit card", "credit-card", "creditcard", "european card"),
            "ieee": ("ieee", "vesta"), "baf": ("bank account fraud", "baf", "feedzai")}
    for d, m in CELLS:
        f = next((MAIN / "llm_logs").glob(f"{d}_{m}_O3_pr_auc_time_anon_r0_b40.jsonl"))
        data_line = next(l for l in json.loads(f.read_text().splitlines()[0])["prompt"].splitlines() if l.startswith("Data:"))
        for rep in range(10):
            q = (f"A {L.MODEL_NAMES[m]} classifier is being tuned on this dataset.\n{data_line}\n"
                 "Which publicly available dataset is this most likely to be? Answer with a dataset name, or 'unknown'.")
            text, _ = client.complete(q, rep)
            low = text.lower()
            t7[f"{d}/{m}"].append(dict(correct=any(w in low for w in TRUE[d]), domain=L.mentions_dataset(text), answer=text[:120]))
    res["T7_identification"] = {c: dict(correct_share=round(float(np.mean([x["correct"] for x in v])), 2),
                                         domain_share=round(float(np.mean([x["domain"] for x in v])), 2),
                                         example=v[0]["answer"]) for c, v in t7.items()}
    # ---- T8 recall probe
    t8 = {}
    prop = collections.defaultdict(list)
    for cell, arm, a, r in searches():
        if a["optimiser"] == "O3" and not a.get("llm_tag") and a["objective"] == "pr_auc":
            prop[(cell, a["condition"])] += [t["params"] for t in r["trials"][:5] if t["source"] == "llm"]
    rr = np.random.default_rng(20261107)
    for d, m in CELLS:
        rec = []
        for rep in range(5):
            q = (f"What hyperparameter settings did strong public solutions use for the {L.PUBLIC_NAMES[d]} data with {L.MODEL_NAMES[m]}? "
                 f"Reply with one JSON object with exactly these keys:\n{S.describe(m)}")
            text, _ = client.complete(q, rep)
            try:
                rec.append(L.parse_answer(text, m)[0])
            except L.ParseError:
                pass
        if not rec:
            continue
        R = np.array([norm(m, p) for p in rec])
        out = {}
        for cond in ("named", "anon"):
            P = np.array([norm(m, p) for p in prop[((d, m), cond)]])
            out[cond] = round(float(np.mean([np.min(np.linalg.norm(R - x, axis=1)) for x in P])), 3)
        rand = rr.uniform(0, 1, (500, R.shape[1]))
        out["random"] = round(float(np.mean([np.min(np.linalg.norm(R - x, axis=1)) for x in rand])), 3)
        out["n_recalled"] = len(rec)
        t8[f"{d}/{m}"] = out
    res["T8_recall_distance_min_to_recalled"] = t8
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("local", "llm"), required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = (dict(**t3_t4_t5(), **m4()) if a.part == "local" else t6_t7_t8())
    (OUT / f"exploratory_{a.part}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
