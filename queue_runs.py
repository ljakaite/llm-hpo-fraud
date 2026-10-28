"""
Run list and queue (protocol section 7). Blocks run in the order B5, B1, B4, B2, B3, B6; the order
within a block is shuffled with seed 20261104. Each job runs as its own process, is skipped if its
result file exists (resume after a crash, check C6) and is retried once if it fails.

    python -I experiments/hpo_llm/queue_runs.py --list                 # print the full run list
    python -I experiments/hpo_llm/queue_runs.py --blocks B5 B1 --llm <adapter>
"""

import argparse
import itertools
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER_SEED = 20261104
BLOCK_ORDER = ("B5", "B1", "B4", "B2", "B3", "B6", "B7")


def jobs(datasets=("ieee", "baf", "ulb"), models=("xgb", "lgbm"), reps=range(10), budget=40):
    cells = list(itertools.product(datasets, models))
    J = {b: [] for b in BLOCK_ORDER}
    for (d, m), r in itertools.product(cells, reps):
        base = dict(dataset=d, model=m, rep=r, budget=budget, fold="time", condition="anon")
        for obj in ("pr_auc", "log_loss"):
            J["B5"].append({**base, "optimiser": "O0", "objective": obj})
            for o in ("O1", "O2"):
                J["B1"].append({**base, "optimiser": o, "objective": obj})
            for o in ("O3", "O4"):
                J["B2"].append({**base, "optimiser": o, "objective": obj})
        J["B4"].append({**base, "optimiser": "O2", "objective": "pr_auc", "fold": "random"})
        J["B3"].append({**base, "optimiser": "O3", "objective": "pr_auc", "condition": "named"})
        for c in ("named", "anon"):
            J["B6"].append({**base, "optimiser": "O0LLM", "objective": "pr_auc", "condition": c})
        # B7 (V4b, S3): O3 anonymised, PR-AUC, with Claude Haiku 5.5
        J["B7"].append({**base, "optimiser": "O3", "objective": "pr_auc", "llm": "anthropic-haiku"})
    rng = random.Random(ORDER_SEED)
    for b in BLOCK_ORDER:
        rng.shuffle(J[b])
    return J


def run_job(j, llm, out, log):
    cmd = [sys.executable, "-I", str(HERE / "search.py"), "--llm", j.get("llm", llm), "--out", str(out)]
    for k, v in j.items():
        if k == "llm":
            continue
        cmd += [f"--{k}", str(v)]
    for attempt in (1, 2):
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        log.write(json.dumps(dict(job=j, attempt=attempt, rc=p.returncode, s=round(time.time() - t0, 1),
                                  stdout=p.stdout[-300:], stderr=p.stderr[-800:])) + "\n")
        log.flush()
        if p.returncode == 0:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="*", default=list(BLOCK_ORDER))
    ap.add_argument("--datasets", nargs="*", default=["ieee", "baf", "ulb"])
    ap.add_argument("--models", nargs="*", default=["xgb", "lgbm"])
    ap.add_argument("--reps", nargs="*", type=int, default=list(range(10)))
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--llm", default="none")
    ap.add_argument("--out", default=str(HERE / "results" / "search"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    J = jobs(a.datasets, a.models, a.reps, a.budget)
    if a.list:
        for b in BLOCK_ORDER:
            print(b, len(J[b]))
        print("total", sum(len(v) for v in J.values()))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    with (out / "queue_log.jsonl").open("a") as log:
        for b in [b for b in BLOCK_ORDER if b in a.blocks]:
            for j in J[b]:
                if not run_job(j, a.llm, out, log):
                    failed.append(j)
    print(json.dumps(dict(failed=len(failed))), flush=True)


if __name__ == "__main__":
    main()
