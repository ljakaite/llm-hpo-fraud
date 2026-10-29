"""
Pilot report (protocol section 11): time per search, tokens per call, parse and fallback rates,
the projected LLM cost of Plans F and R, the stop-rule decision (section 7) and the H4
manipulation check. Reads only search files and LLM logs; no test score is computed or read.

    python -I experiments/hpo_llm/pilot_report.py [--dir results/pilot] [--threshold 230]
"""

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm as L   # noqa: E402

PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-haiku-5-5": (0.10, 0.50)}   # USD per MTok (c_vs_7)
# LLM units in the main design (protocol section 7): (optimiser, condition, model) -> searches or calls
PLAN_F = {("O3", "anon", "opus"): 120, ("O3", "named", "opus"): 60, ("O4", "anon", "opus"): 120,
          ("O0LLM", "any", "opus"): 120, ("O3", "anon", "haiku"): 60}
PLAN_R = {**PLAN_F, ("O3", "anon", "opus"): 60, ("O4", "anon", "opus"): 60}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "results" / "pilot"))
    ap.add_argument("--threshold", type=float, default=230.0)
    a = ap.parse_args()
    d = Path(a.dir)
    runs = [json.loads(f.read_text()) for f in sorted((d / "search").glob("*.json"))]

    # time per search, by dataset and arm
    times = collections.defaultdict(list)
    flags = collections.Counter()
    n_llm_trials = 0
    for r in runs:
        g = r["args"]
        arm = f'{g["optimiser"]}/{g["objective"]}/{g["fold"]}/{g["condition"]}{"/haiku" if g.get("llm_tag") else ""}'
        times[(g["dataset"], arm)].append(r["time_s"])
        for t in r["trials"]:
            if t["source"] == "llm":
                n_llm_trials += 1
                flags.update(t["flags"])

    # tokens per search (or per call for O0LLM), by unit
    per_unit = collections.defaultdict(list)
    ident = collections.Counter()
    calls = 0
    for f in sorted((d / "llm_logs").glob("*.jsonl")):
        if f.name == "cache.jsonl":
            continue
        recs = [json.loads(x) for x in f.read_text().splitlines()]
        recs = [x for x in recs if not x["cache_hit"]]
        calls += len(recs)
        name = f.stem
        opt = next(o for o in ("O0LLM", "O3", "O4") if f"_{o}_" in name)
        cond = "named" if "_named_" in name else "anon"
        model = "haiku" if name.endswith("_haiku") else "opus"
        key = (opt, "any" if opt == "O0LLM" else cond, model)
        inp = sum(x["usage"].get("input_tokens", 0) for x in recs)
        out = sum(x["usage"].get("output_tokens", 0) for x in recs)
        per_unit[key].append((inp, out))
        if cond == "anon":
            for x in recs:
                hit = L.mentions_dataset(x["response"]) or L.mentions_dataset(x["usage"].get("thinking_summary", ""))
                ident[("anon", model, "identifying" if hit else "not")] += 1
        else:
            for x in recs:
                ident[("named", model, "identifying" if L.mentions_dataset(x["response"]) else "not")] += 1

    def unit_cost(key):
        model_id = L.HAIKU_ID if key[2] == "haiku" else L.MODEL_ID
        pin, pout = PRICES[model_id]
        v = per_unit.get(key, [])
        if not v:
            return None
        return st.mean(i for i, _ in v) * pin / 1e6 + st.mean(o for _, o in v) * pout / 1e6

    def plan_cost(plan):
        tot, missing = 0.0, []
        for key, n in plan.items():
            c = unit_cost(key)
            if c is None:
                missing.append(key)
            else:
                tot += c * n
        return tot, missing

    cf, mf = plan_cost(PLAN_F)
    cr, mr = plan_cost(PLAN_R)
    rep = dict(
        searches=len(runs), llm_calls=calls, llm_trials=n_llm_trials,
        flag_rates={k: round(v / max(n_llm_trials, 1), 4) for k, v in flags.items()},
        time_per_search_s={f"{ds} {arm}": round(st.mean(v), 1) for (ds, arm), v in sorted(times.items())},
        tokens_per_unit={"/".join(k): dict(n=len(v), input=round(st.mean(i for i, _ in v)), output=round(st.mean(o for _, o in v)),
                                           usd=round(unit_cost(k), 4)) for k, v in sorted(per_unit.items())},
        projected_cost_usd=dict(plan_F=round(cf, 2), plan_R=round(cr, 2), missing_units=[list(m) for m in mf]),
        stop_rule=dict(threshold_usd=a.threshold, decision="Plan F" if cf <= a.threshold else "Plan R"),
        manipulation_check={"/".join(k): v for k, v in sorted(ident.items())},
    )
    (d / "pilot_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
