"""
POST-HOC exploratory analysis of the H5 result (declared in protocol/deviations.md on 1 Nov 2026,
after the test results were seen). Descriptive only; no hypothesis test.
1. Validation folds: size, frauds, fraud rate, time span; training and test fraud rates.
2. Settings selected by O2 (PR-AUC objective, k = 40) under each fold: learning rate, final trees,
   validation score.
3. Spearman(log final trees, test PR-AUC) over the O1/O2 k = 40 incumbents of both folds.

    python -I experiments/hpo_llm/analysis/h5_mechanism.py
"""
import glob, json, statistics as st, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import data as D, testdata as T   # noqa: E402
MAIN = HERE / "results" / "main"
out = {"folds": {}, "selected": {}, "trees_vs_test": {}}
for name in D.DATASETS:
    tp = D.training_period(name)
    yte = T.full_fit_and_test(name)[3]
    row = {}
    for fold in ("time", "random"):
        a, b = tp.fold_masks(fold)
        row[fold] = dict(n=int(b.sum()), frauds=int(tp.y[b].sum()), rate=round(float(tp.y[b].mean()), 5),
                         t_min=float(tp.t[b].min()), t_max=float(tp.t[b].max()))
    row["train_rate"], row["test_rate"] = round(float(tp.y.mean()), 5), round(float(yte.mean()), 5)
    out["folds"][name] = row
for d in ("ieee", "baf", "ulb"):
    for m in ("xgb", "lgbm"):
        sel = {}
        for fold in ("time", "random"):
            inc = [json.loads(Path(f).read_text())["incumbents"]["40"] for f in glob.glob(str(MAIN / "search" / f"{d}_{m}_O2_pr_auc_{fold}_anon_r*_b40.json"))]
            sel[fold] = dict(lr=round(st.median(i["params"]["learning_rate"] for i in inc), 4),
                             trees=int(st.median(i["n_trees_final"] for i in inc)), val=round(st.median(i["value"] for i in inc), 4))
        out["selected"][f"{d}/{m}"] = sel
        tr, te = [], []
        for f in glob.glob(str(MAIN / "search" / f"{d}_{m}_O[12]_pr_auc_*_anon_r*_b40.json")):
            tr.append(json.loads(Path(f).read_text())["incumbents"]["40"]["n_trees_final"])
            te.append(json.loads(Path(f.replace("/search/", "/test/")).read_text())["checkpoints"]["40"]["pr_auc"])
        r = spearmanr(np.log(tr), te)
        out["trees_vs_test"][f"{d}/{m}"] = dict(n=len(te), rho=round(float(r.statistic), 3), p=round(float(r.pvalue), 4))
(HERE / "results" / "analysis" / "h5_mechanism.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
