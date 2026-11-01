"""
Pre-registered analysis (protocol v1.0 sections 6 and 8): hypothesis table H1-H5, sensitivity
analyses S1-S3, secondary outcomes and the four figures.

Bootstrap: 10,000 resamples of repeats within each arm, seed 20261103 (section 6), for the difference
in medians (H1, H3, H4, H5) and for the SD ratio (H2, percentile interval).

    python -I experiments/hpo_llm/analysis/h1_h5.py
"""

import collections
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import llm as L   # noqa: E402

MAIN = HERE / "results" / "main"
OUT = HERE / "results" / "analysis"
CELLS = [(d, m) for d in ("ieee", "baf", "ulb") for m in ("xgb", "lgbm")]
NB = 10000
RNG = np.random.default_rng(20261103)


# ------------------------------------------------------------------ data
def load():
    T = {}
    for f in (MAIN / "test").glob("*.json"):
        r = json.loads(f.read_text())
        s = json.loads((MAIN / "search" / f.name).read_text())
        a = r["args"]
        T[f.stem] = dict(args=a, ck=r["checkpoints"], search=s)
    return T


def arm(T, d, m, opt, obj="pr_auc", fold="time", cond="anon", haiku=False):
    rows = [v for v in T.values() if v["args"]["dataset"] == d and v["args"]["model"] == m and v["args"]["optimiser"] == opt
            and v["args"]["objective"] == obj and v["args"]["fold"] == fold and v["args"]["condition"] == cond
            and bool(v["args"].get("llm_tag")) == haiku]
    return sorted(rows, key=lambda v: v["args"]["rep"])


def vals(rows, k, metric="pr_auc"):
    out = []
    for v in rows:
        ck = v["ck"].get(str(k)) or v["ck"].get("1")          # single-fit arms have k = 1 only
        out.append(ck[metric])
    return np.array(out)


def calib_err(rows, k):
    return np.array([abs((v["ck"].get(str(k)) or v["ck"]["1"])["sum_p_over_frauds"] - 1) for v in rows])


# ------------------------------------------------------------------ bootstrap helpers
def boot_median_diff(a, b):
    ia = RNG.integers(0, len(a), (NB, len(a)))
    ib = RNG.integers(0, len(b), (NB, len(b)))
    d = np.median(a[ia], axis=1) - np.median(b[ib], axis=1)
    return float(np.median(a) - np.median(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_sd_ratio(a, b):
    ia = RNG.integers(0, len(a), (NB, len(a)))
    ib = RNG.integers(0, len(b), (NB, len(b)))
    sb = np.std(b[ib], axis=1, ddof=1)
    rr = np.std(a[ia], axis=1, ddof=1) / np.where(sb > 0, sb, np.nan)
    return float(np.std(a, ddof=1) / np.std(b, ddof=1)), float(np.nanpercentile(rr, 2.5)), float(np.nanpercentile(rr, 97.5))


def verdict(n_point, n_interval, need):
    return "clearly supported" if n_interval >= need else ("supported" if n_point >= need else "not supported")


R = lambda x: round(x, 4)


# ------------------------------------------------------------------ hypotheses
def h1(T, k_llm=10):
    out = {}
    for opt in ("O3", "O4"):
        cells, npt, nint = {}, 0, 0
        for d, m in CELLS:
            a = vals(arm(T, d, m, opt), k_llm)
            b = vals(arm(T, d, m, "O2"), 40)
            est, lo, hi = boot_median_diff(a, b)
            ok, okc = est >= -0.005, lo >= -0.005
            npt += ok
            nint += ok and okc
            cells[f"{d}/{m}"] = dict(median_llm=R(float(np.median(a))), median_O2_k40=R(float(np.median(b))),
                                     diff=R(est), ci=[R(lo), R(hi)], meets=bool(ok), meets_ci=bool(okc))
        out[opt] = dict(cells=cells, count=int(npt), count_ci=int(nint), verdict=verdict(npt, nint, 4))
    return out


def h2(T):
    out = {}
    for opt in ("O3", "O4"):
        cells, npt, nint = {}, 0, 0
        for d, m in CELLS:
            a, b = vals(arm(T, d, m, opt), 40), vals(arm(T, d, m, "O2"), 40)
            est, lo, hi = boot_sd_ratio(a, b)
            ok, okc = est >= 0.8, lo >= 0.8
            npt += ok
            nint += ok and okc
            cells[f"{d}/{m}"] = dict(sd_llm=R(float(np.std(a, ddof=1))), sd_O2=R(float(np.std(b, ddof=1))),
                                     ratio=R(est), ci=[R(lo), R(hi)], meets=bool(ok), meets_ci=bool(okc))
        out[opt] = dict(cells=cells, count=int(npt), count_ci=int(nint), verdict=verdict(npt, nint, 4))
    return out


def h3(T, opt):
    cells, npt, nint = {}, 0, 0
    for d, m in CELLS:
        ll, pr = arm(T, d, m, opt, obj="log_loss"), arm(T, d, m, opt, obj="pr_auc")
        dl = boot_median_diff(vals(ll, 40, "log_loss"), vals(pr, 40, "log_loss"))       # want < 0
        dc = boot_median_diff(calib_err(ll, 40), calib_err(pr, 40))                       # want < 0
        dp = boot_median_diff(vals(ll, 40), vals(pr, 40))                                 # want >= -0.01
        ok = dl[0] < 0 and dc[0] < 0 and dp[0] >= -0.01
        okc = dl[2] < 0 and dc[2] < 0 and dp[1] >= -0.01
        npt += ok
        nint += ok and okc
        cells[f"{d}/{m}"] = dict(logloss_diff=[R(x) for x in dl], calib_err_diff=[R(x) for x in dc],
                                 prauc_diff=[R(x) for x in dp], meets=bool(ok), meets_ci=bool(okc))
    return dict(cells=cells, count=int(npt), count_ci=int(nint), verdict=verdict(npt, nint, 4))


def h4(T, opt="O3", k=40):
    cells, gaps = {}, {}
    for d, m in CELLS:
        a, b = vals(arm(T, d, m, opt, cond="named"), k), vals(arm(T, d, m, opt, cond="anon"), k)
        est, lo, hi = boot_median_diff(a, b)
        gaps[(d, m)] = est
        cells[f"{d}/{m}"] = dict(median_named=R(float(np.median(a))), median_anon=R(float(np.median(b))),
                                 gap=R(est), ci=[R(lo), R(hi)], positive=bool(est > 0), positive_ci=bool(lo > 0))
    ui = [(d, m) for d, m in CELLS if d in ("ulb", "ieee")]
    npt = sum(gaps[c] > 0 for c in ui)
    nint = sum(cells[f"{c[0]}/{c[1]}"]["positive_ci"] for c in ui)
    baf_gap = float(np.mean([gaps[("baf", m)] for m in ("xgb", "lgbm")]))
    ui_gap = float(np.mean([gaps[c] for c in ui]))
    cond2 = baf_gap < ui_gap
    v = "clearly supported" if (nint >= 3 and cond2) else ("supported" if (npt >= 3 and cond2) else "not supported")
    return dict(cells=cells, count_ulb_ieee=int(npt), count_ci=int(nint), mean_gap_baf=R(baf_gap),
                mean_gap_ulb_ieee=R(ui_gap), baf_gap_smaller=bool(cond2), verdict=v)


def h5(T):
    ds, n_ok, n_okc = {}, 0, 0
    for d in ("ieee", "baf", "ulb"):
        per, both, bothc = {}, True, True
        for m in ("xgb", "lgbm"):
            a, b = vals(arm(T, d, m, "O2", fold="time"), 40), vals(arm(T, d, m, "O2", fold="random"), 40)
            est, lo, hi = boot_median_diff(a, b)
            per[m] = dict(median_time=R(float(np.median(a))), median_random=R(float(np.median(b))), diff=R(est), ci=[R(lo), R(hi)])
            both &= est > 0
            bothc &= lo > 0
        ds[d] = dict(**per, counts=bool(both), counts_ci=bool(both and bothc))
        n_ok += both
        n_okc += both and bothc
    return dict(datasets=ds, count=int(n_ok), count_ci=int(n_okc), verdict=verdict(n_ok, n_okc, 2))


# ------------------------------------------------------------------ secondary and sensitivity
def secondary(T):
    out = {}
    arms = [("O0", {}), ("O0LLM", {}), ("O1", {}), ("O2", {}), ("O3", {}), ("O4", {}), ("O3-haiku", dict(haiku=True))]
    for d, m in CELLS:
        cell = {}
        for name, kw in arms:
            opt = name.split("-")[0]
            rows = arm(T, d, m, opt, **kw)
            if not rows:
                continue
            k = 40 if opt not in ("O0", "O0LLM") else 1
            cell[name] = dict(pr_auc=R(float(np.median(vals(rows, k)))), recall_1pct=R(float(np.median(vals(rows, k, "recall_at_fpr_1pct")))),
                              brier=R(float(np.median(vals(rows, k, "brier")))), log_loss=R(float(np.median(vals(rows, k, "log_loss")))),
                              sum_p_over_frauds=R(float(np.median(vals(rows, k, "sum_p_over_frauds")))),
                              time_s=round(float(np.median([v["search"]["time_s"] for v in rows])), 1))
        # validation-test association (deviation D1): Spearman across the k = 40 incumbents of all PR-AUC
        # time-fold searches in the cell (O1-O4, Haiku), instead of across the 40 trials of each search.
        pts = []
        for v in T.values():
            a = v["args"]
            if a["dataset"] == d and a["model"] == m and a["objective"] == "pr_auc" and a["fold"] == "time" and a["optimiser"] in ("O1", "O2", "O3", "O4"):
                inc = v["search"]["incumbents"]["40"]
                pts.append((inc["value"], v["ck"]["40"]["pr_auc"]))
        rho = spearmanr([p[0] for p in pts], [p[1] for p in pts])
        cell["val_test_spearman_k40_incumbents"] = dict(n=len(pts), rho=R(float(rho.statistic)), p=R(float(rho.pvalue)))
        out[f"{d}/{m}"] = cell
    return out


def s3(T):
    out = {}
    cost = collections.defaultdict(list)
    for f in (MAIN / "llm_logs").glob("*_O3_pr_auc_time_anon_*.jsonl"):
        recs = [json.loads(x) for x in f.read_text().splitlines()]
        u = [r["usage"] for r in recs if not r["cache_hit"]]
        hk = f.stem.endswith("_haiku")
        pin, pout = (0.10, 0.50) if hk else (4.0, 20.0)
        cost[(f.stem.split("_")[0], f.stem.split("_")[1], hk)].append(
            sum(x.get("input_tokens", 0) for x in u) * pin / 1e6 + sum(x.get("output_tokens", 0) for x in u) * pout / 1e6)
    for d, m in CELLS:
        o, h = vals(arm(T, d, m, "O3"), 40), vals(arm(T, d, m, "O3", haiku=True), 40)
        est, lo, hi = boot_median_diff(h, o)
        out[f"{d}/{m}"] = dict(median_opus=R(float(np.median(o))), median_haiku=R(float(np.median(h))), diff_haiku_minus_opus=[R(est), R(lo), R(hi)],
                               sd_opus=R(float(np.std(o, ddof=1))), sd_haiku=R(float(np.std(h, ddof=1))),
                               usd_per_search_opus=round(st.mean(cost[(d, m, False)]), 3), usd_per_search_haiku=round(st.mean(cost[(d, m, True)]), 4))
    return out


def manipulation(T):
    out = {}
    for d, m in CELLS:
        row = {}
        for opt in ("O0LLM", "O3", "O4"):
            for cond in ("anon", "named"):
                n = hit = 0
                for f in (MAIN / "llm_logs").glob(f"{d}_{m}_{opt}_*_{cond}_r*_b40.jsonl"):
                    for x in f.read_text().splitlines():
                        r = json.loads(x)
                        n += 1
                        hit += L.mentions_dataset(r["response"]) or L.mentions_dataset(r["usage"].get("thinking_summary", ""))
                if n:
                    row[f"{opt} {cond}"] = dict(n=n, share=round(hit / n, 4))
        out[f"{d}/{m}"] = row
    return out


def fallbacks(T):
    return sum("fallback" in t["flags"] for v in T.values() for t in v["search"]["trials"])


# ------------------------------------------------------------------ figures
def figures(T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    FIG = OUT / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
    SER = {"O1 random": ("#2a78d6", "o"), "O2 TPE": ("#eb6834", "s"), "O3 Opus": ("#1baf7a", "^"),
           "O4 Opus+TPE": ("#eda100", "D"), "O3 Haiku": ("#e87ba4", "v")}
    ARMS = {"O1 random": ("O1", {}), "O2 TPE": ("O2", {}), "O3 Opus": ("O3", {}), "O4 Opus+TPE": ("O4", {}), "O3 Haiku": ("O3", dict(haiku=True))}
    NAMES = {"ieee": "IEEE-CIS", "baf": "BAF", "ulb": "ULB", "xgb": "XGBoost", "lgbm": "LightGBM"}
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlesize": 8.5, "axes.titlecolor": INK, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF})

    def grid(ax):
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    # 1. anytime curves
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.4))
    ks = [5, 10, 20, 40]
    for ax, (d, m) in zip(axes.flat, CELLS):
        for lab, (opt, kw) in ARMS.items():
            rows = arm(T, d, m, opt, **kw)
            Y = np.array([vals(rows, k) for k in ks])
            med, q1, q3 = np.median(Y, 1), np.percentile(Y, 25, 1), np.percentile(Y, 75, 1)
            c, mk = SER[lab]
            ax.fill_between(ks, q1, q3, color=c, alpha=0.12, linewidth=0)
            ax.plot(ks, med, color=c, marker=mk, markersize=4, linewidth=2, label=lab)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels(ks)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(f"{NAMES[d]} · {NAMES[m]}", loc="left")
        ax.set_xlabel("trials (k)")
        ax.set_ylabel("test PR-AUC")
        grid(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=5, fontsize=7.5)
    fig.suptitle("Figure 1. Anytime curves: median and interquartile range of test PR-AUC (10 repeats)", x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "fig1_anytime.png", dpi=200)
    fig.savefig(FIG / "fig1_anytime.pdf")
    plt.close(fig)

    # 2. spread across repeats at k = 40
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.4))
    for ax, (d, m) in zip(axes.flat, CELLS):
        for i, (lab, (opt, kw)) in enumerate(ARMS.items()):
            y = vals(arm(T, d, m, opt, **kw), 40)
            c, mk = SER[lab]
            x = i + np.linspace(-0.18, 0.18, len(y))
            ax.scatter(x, y, s=16, color=c, marker=mk, edgecolors=SURF, linewidths=0.8, zorder=3)
            ax.hlines(np.median(y), i - 0.3, i + 0.3, color=INK, linewidth=1.2, zorder=4)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([l.replace(" ", "\n", 1) for l in ARMS], fontsize=7)
        ax.set_title(f"{NAMES[d]} · {NAMES[m]}", loc="left")
        ax.set_ylabel("test PR-AUC (k = 40)")
        grid(ax)
    fig.suptitle("Figure 2. Spread across 10 repeats at k = 40 (dots: repeats; bar: median)", x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "fig2_spread.png", dpi=200)
    fig.savefig(FIG / "fig2_spread.pdf")
    plt.close(fig)

    # 3. validation against test (incumbents at k = 5, 10, 20, 40; deviation D1)
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.4))
    for ax, (d, m) in zip(axes.flat, CELLS):
        for lab, (opt, kw) in ARMS.items():
            c, mk = SER[lab]
            xs, ys = [], []
            for v in arm(T, d, m, opt, **kw):
                for k in ("5", "10", "20", "40"):
                    xs.append(v["search"]["incumbents"][k]["value"])
                    ys.append(v["ck"][k]["pr_auc"])
            ax.scatter(xs, ys, s=10, color=c, marker=mk, alpha=0.75, edgecolors="none", label=lab)
        ax.set_title(f"{NAMES[d]} · {NAMES[m]}", loc="left")
        ax.set_xlabel("validation PR-AUC (incumbent)")
        ax.set_ylabel("test PR-AUC")
        grid(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=5, fontsize=7.5, markerscale=1.4)
    fig.suptitle("Figure 3. Validation against test PR-AUC of the incumbents at k = 5, 10, 20, 40", x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "fig3_val_test.png", dpi=200)
    fig.savefig(FIG / "fig3_val_test.pdf")
    plt.close(fig)

    # 4. calibration-in-the-large by objective (O2 and O3, k = 40)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    OBJ = {"PR-AUC objective": ("pr_auc", "#2a78d6", "o"), "log-loss objective": ("log_loss", "#eb6834", "s")}
    for ax, opt in zip(axes, ("O2", "O3")):
        for j, (lab, (obj, c, mk)) in enumerate(OBJ.items()):
            for i, (d, m) in enumerate(CELLS):
                y = vals(arm(T, d, m, opt, obj=obj), 40, "sum_p_over_frauds")
                x = i + (-0.17 if j == 0 else 0.17) + np.linspace(-0.07, 0.07, len(y))
                ax.scatter(x, y, s=12, color=c, marker=mk, edgecolors=SURF, linewidths=0.6, label=lab if i == 0 else None, zorder=3)
        ax.axhline(1.0, color=INK2, linewidth=1, linestyle="--", zorder=2)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(range(len(CELLS)))
        ax.set_xticklabels([f"{NAMES[d]}\n{NAMES[m]}" for d, m in CELLS], fontsize=7)
        ax.set_title({"O2": "O2 TPE", "O3": "O3 Opus"}[opt], loc="left")
        grid(ax)
    axes[0].set_ylabel("Σp / frauds (test, k = 40; 1 = calibrated)")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Figure 4. Calibration-in-the-large by tuning objective", x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "fig4_calibration.png", dpi=200)
    fig.savefig(FIG / "fig4_calibration.pdf")
    plt.close(fig)
    return sorted(p.name for p in FIG.iterdir())


def main():
    T = load()
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(n_scored=len(T), fallback_trials=fallbacks(T),
               H1=h1(T), H2=h2(T), H3_O2_primary=h3(T, "O2"), H3_O3_secondary=h3(T, "O3"),
               H4=h4(T), H4_O0LLM_probe=h4(T, "O0LLM", 1),
               H5=h5(T), S2_H1_k20=h1(T, 20), S3_haiku_vs_opus=s3(T),
               secondary=secondary(T), manipulation_check=manipulation(T))
    res["S1"] = "identical to H1-H3: no trial was flagged 'fallback'" if res["fallback_trials"] == 0 else "see code"
    res["figures"] = figures(T)
    (OUT / "h1_h5.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) and "verdict" in v else
                          {kk: vv.get("verdict") for kk, vv in v.items() if isinstance(vv, dict) and "verdict" in vv} if isinstance(v, dict) else v)
                      for k, v in res.items() if k not in ("secondary", "manipulation_check", "S3_haiku_vs_opus", "figures")}, indent=1))


if __name__ == "__main__":
    main()
