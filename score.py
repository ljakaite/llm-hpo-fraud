"""
Test scoring, a separate step after the search (protocol sections 4 and 10, check C1).
Reads results/search/*.json, refits each incumbent on the whole training period and scores it once
on the test period. Writes results/test/<job>.json. Nothing here is read by the search code.

    python -I experiments/hpo_llm/score.py [--search DIR] [--out DIR] [--only SUBSTRING]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import models as M     # noqa: E402
import testdata as T   # noqa: E402

_full = {}


def recall_at_fpr(y, p, fpr_max=0.01):
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[fpr <= fpr_max].max())


def metrics(y, p):
    return dict(pr_auc=float(average_precision_score(y, p)),
                recall_at_fpr_1pct=recall_at_fpr(y, p),
                log_loss=float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
                brier=float(brier_score_loss(y, p)),
                sum_p_over_frauds=float(p.sum() / y.sum()))


def score_file(path, out_dir):
    r = json.loads(path.read_text())
    out = out_dir / path.name
    if out.exists():
        return "skip"
    a = r["args"]
    if a["dataset"] not in _full:
        _full[a["dataset"]] = T.full_fit_and_test(a["dataset"])
    Xtr, ytr, Xte, yte, _, _ = _full[a["dataset"]]
    res, done = dict(job=r["job"], args=a, checkpoints={}), {}
    for k, inc in r["incumbents"].items():
        key = inc["trial"]
        if key not in done:              # the same incumbent at several k is refitted once
            p = M.refit_predict(a["model"], inc["params"], inc["n_trees_final"], Xtr, ytr, Xte, r["seeds"]["model"])
            done[key] = metrics(yte, p)
        res["checkpoints"][k] = dict(trial=key, **done[key])
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, out)
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", default=str(HERE / "results" / "search"))
    ap.add_argument("--out", default=str(HERE / "results" / "test"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.search).glob("*.json")):
        if a.only in f.name:
            print(score_file(f, out), f.name, flush=True)


if __name__ == "__main__":
    main()
