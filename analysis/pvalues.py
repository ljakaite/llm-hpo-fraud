"""
Supplementary p-values for the manuscript (declared in protocol/deviations.md on 1 Nov 2026, after the
results; not pre-registered). The pre-registered inference remains the bootstrap intervals and count
rules of section 6. Two-sided permutation tests of the difference in medians between two arms
(10 repeats each; 10,000 random relabellings; seed 20261108), per cell, no pooling across datasets.
Also exact two-sided binomial tests for the counterfactual-history probe (T6).

    python -I experiments/hpo_llm/analysis/pvalues.py
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "analysis"))
import h1_h5 as H   # noqa: E402

RNG = np.random.default_rng(20261108)
NP = 10000


def perm_p(a, b):
    a, b = np.asarray(a), np.asarray(b)
    obs = np.median(a) - np.median(b)
    pool = np.r_[a, b]
    n = len(a)
    cnt = 0
    for _ in range(NP):
        RNG.shuffle(pool)
        cnt += abs(np.median(pool[:n]) - np.median(pool[n:])) >= abs(obs) - 1e-12
    return round(float(obs), 4), round((cnt + 1) / (NP + 1), 4)


T = H.load()
V, A = H.vals, H.arm
out = {}
for d, m in H.CELLS:
    c = f"{d}/{m}"
    o2_40 = V(A(T, d, m, "O2"), 40)
    out[c] = {
        "H1_O3@10_vs_O2@40": perm_p(V(A(T, d, m, "O3"), 10), o2_40),
        "H1_O4@10_vs_O2@40": perm_p(V(A(T, d, m, "O4"), 10), o2_40),
        "S3_Haiku_vs_Opus@40": perm_p(V(A(T, d, m, "O3", haiku=True), 40), V(A(T, d, m, "O3"), 40)),
        "H3_PR-AUC_logloss_vs_prauc_objective": perm_p(V(A(T, d, m, "O2", obj="log_loss"), 40), o2_40),
        "H3_calib_err_logloss_vs_prauc_objective": perm_p(H.calib_err(A(T, d, m, "O2", obj="log_loss"), 40), H.calib_err(A(T, d, m, "O2"), 40)),
        "H4_named_vs_anon@40": perm_p(V(A(T, d, m, "O3", cond="named"), 40), V(A(T, d, m, "O3"), 40)),
        "H5_time_vs_random_fold@40": perm_p(o2_40, V(A(T, d, m, "O2", fold="random"), 40)),
        "zero_shot_vs_O2@40": perm_p(V(A(T, d, m, "O0LLM"), 1), o2_40),
        "O2@40_vs_defaults": perm_p(o2_40, V(A(T, d, m, "O0"), 1)),
    }
ex = json.loads((HERE / "results" / "exploratory" / "exploratory_llm.json").read_text())["T6_counterfactual_history"]
t6 = {}
for v in ("permuted", "reversed"):
    n = ex[v]["n_different"]
    k = round(ex[v]["share_closer_to_shown_when_different"] * n)
    t6[v] = dict(k=k, n=n, share=round(k / n, 3), p_binomial_vs_0_5=float(f"{binomtest(k, n, 0.5).pvalue:.2e}"))
out["T6_counterfactual"] = t6
(HERE / "results" / "analysis" / "pvalues.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
