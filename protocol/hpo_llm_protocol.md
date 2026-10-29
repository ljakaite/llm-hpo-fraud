Oct 29, 2026. **v1.0, FROZEN on 29 Oct 2026** after the pilot (§11, §14). From now on, every change is a dated deviation (§12). v0.2 (27 Oct): P1–P3 and V3 decided; code written and checks C1–C7 passed (c_vs_2); §4, §5 and §10 updated to match the code. v0.3 (28 Oct): D2 = A, scope statement in §1, limitation in §8, scope references in §15; no new arm (c_vs_3, c_vs_4). v0.4 (28 Oct): D1 decided (10 TPE start-up trials in O2 and O4); R3 and R4 verified and the §1 wording narrowed to what R3 shows; LLM-call count corrected in §7 (c_vs_5, c_vs_6). v0.5 (28 Oct): V4a decided, primary LLM Claude Opus 5.5 at effort `low`, with the settings it requires (§5; P2 revised: no temperature); cost stop rule with Plans F and R (§7, §11) (c_vs_7, c_vs_7a, c_vs_7b). v0.6 (28 Oct): V5 (c), banded data description in both prompts and a manipulation check for H4 (§5, §6, §8); Claude adapter and live checks C5L, C7L (§10) (c_vs_8). v0.7 (28 Oct): V4b decided, Claude Haiku 5.5 as the S3 arm, block B7 (§5, §7, §8). v0.8 (28 Oct): authorship and roles revised, with LJ as first and corresponding author and VS acknowledged; repository on LJ's GitHub account (People, §9, §12–§14; c_vs_9). v0.9 (28 Oct): VS restored as co-author, last author after the students; repository unchanged (People; c_vs_10). v0.10 (28 Oct): signed off by LJ (reported by VS) and VS; V4d settled (no prompt caching); committed to the repository before the pilot under the tag `protocol-signed` (c_vs_12). v1.0 (29 Oct): pilot run (60 searches, no failures), no amendments needed, stop rule: Plan F; frozen and tagged `protocol-freeze` (c_vs_12).

# Protocol: LLM-guided hyperparameter optimisation for fraud detectors on out-of-time benchmarks

**Paper.** *Can large language models tune fraud detectors? LLM-guided hyperparameter optimisation against Bayesian optimisation on out-of-time fraud benchmarks* (working title; plan in c_vs_1).

**What this document is.** The pre-specified design, hypotheses and analysis for the study. It is written before any LLM optimiser has been run and before any test-period score has been looked at for this study.

**How it comes into force:**
1. LJ and VS settle the open items in §13.
2. The code passes the checks in §10.
3. The pilot (§11) is run and any engineering changes are recorded as amendments.
4. The file is frozen: its sha256 and the code commit hash are written into §14.

From then on, every change is a dated deviation (§12) and is reported in the paper.

**People.** Authors are named by role, because the student team may change before submission. The current holders are listed in §13 (item P1) and are updated there only.

| Role | Responsibility in this protocol |
|---|---|
| LJ | First and corresponding author (University of Bedfordshire; livija.jakaite@lj-analytics.co.uk). Protocol owner: signs off the freeze, the pilot amendments and any deviation, jointly with VS. Repository owner (§9); Discussion subsection on validation practice; independent re-run of one cell |
| VS | **Co-author, last author** (University of Bedfordshire; restored 28 Oct, c_vs_10). Study design, supervision of Students A–C and the day-to-day work; co-signs every sign-off; iACE funding |
| Student A | WP1: harness, baselines O0–O2, run queue, test-period guard |
| Student B | WP2: LLM optimisers O0-LLM, O3, O4; prompts; logging and cache |
| Student C | WP3: analysis, figures, reproducibility package. If there is no Student C, WP3 is split between A and B by VS |

**Independence rules** (c_vs_1 §3.2):
- No JBDT model in any run.
- No stored result from `experiments/results*` is used as data; existing code may be reused and is declared.
- Calibration is reported as test-period outcomes only, with no fitting/drift decomposition.

## 1. Objectives

The study compares LLM-guided hyperparameter optimisation (HPO) with standard HPO for gradient-boosted fraud detectors. The detectors are evaluated on a later time period than they are tuned on. It asks five questions (§6):
- **H1:** does an LLM reach a good setting with fewer trials?
- **H2:** does it reduce run-to-run selection noise?
- **H3:** does tuning on log loss give better probabilities than tuning on PR-AUC?
- **H4:** does the LLM rely on memorised knowledge of public benchmarks?
- **H5:** does the design of the validation fold matter more than the optimiser?

**Scope (D2 = A, 28 Oct).** The object of comparison is the hyperparameter optimiser, not the detector, so the study makes no claim about which detector is best. Gradient-boosted trees are used for three reasons:
1. They are the standard strong baselines on tabular fraud data, including in the fraud-transformer literature, where tuned LightGBM and XGBoost are the reference models [R1].
2. On standard tabular benchmarks, tuned tree ensembles match or outperform deep models [R3, R4]. R3's main benchmark uses medium-sized data (about 10,000 training samples); our training sets are larger (199,000–795,000 rows), so this is supporting evidence rather than a result for data of this size.
3. Published fraud transformers model per-account event sequences on proprietary data [R1, R2]. None of the three public benchmarks provides account histories: ULB has no customer identifier, BAF has one row per account application, and IEEE-CIS has no official customer identifier. These models therefore cannot be fitted as published.

No deep or transformer model is included in any arm. Whether the findings carry over to such models is left to future work and stated as a limitation (§8).

## 2. Data

| Dataset | Source and licence | Rows | Fraud rate | Time variable |
|---|---|---|---|---|
| IEEE-CIS | Kaggle competition training file `train_transaction.csv` + `train_identity.csv` (competition rules; data not redistributed) | 590,540 | 3.50% | `TransactionDT` (seconds offset) |
| BAF Base | Jesus et al. 2022, Kaggle `sgpjesus/bank-account-fraud-dataset-neurips-2022`, CC BY-NC-SA 4.0 (research use) | 1,000,000 | 1.10% | `month` (0–7) |
| ULB | Dal Pozzolo et al., Kaggle `mlg-ulb/creditcardfraud` | 284,807 | 0.17% | `Time` (seconds, 2 days) |

**Features.** The features are those of `experiments/bench/datasets.py` (functions `ieee`, `baf`, `ulb`): 53, 35 and 31 features. For IEEE-CIS and BAF, the categorical variables are encoded as a smoothed target rate (m = 20) and a frequency.

**Re-encoding inside the training period (change from `datasets.py`).** `datasets.py` fits the target and frequency encodings on the whole training period. The validation fold lies inside that period, so its rows would be encoded with their own labels and validation scores would be optimistic. For this study:
- during the search, the encodings are refitted on the inner-training part only and applied to the validation fold;
- for the final refit, they are refitted on the whole training period and applied to the test period.

ULB has no target encoding and is unaffected. The new loader keeps the `datasets.py` columns and is tested to give identical outputs when the inner split is switched off (§10, check C2).

**Out-of-time split.**
- IEEE-CIS and ULB: training = rows with time at or below the 70% quantile; test = the rest (as in `datasets.py`).
- BAF: the authors' split, months 0–5 for training and months 6–7 for test.
- The split is deterministic, so all repeats share it.

**Validation fold** (inside the training period):
- *Time-ordered fold (default):* rows with time above the 80% quantile of the training period. For BAF this is month 5.
- *Stratified random fold (H5 arm only):* a stratified random 20% of the training period, drawn with seed 20261101.

Sizes, checked on 27 Oct with the existing loaders (frauds in brackets):

| Dataset | Training period | Time-ordered validation fold | Test period |
|---|---|---|---|
| IEEE-CIS | 413,378 (14,538) | 82,676 (3,358) | 177,162 (6,125) |
| BAF | 794,989 (8,151) | 119,323 (1,411) | 205,011 (2,878) |
| ULB | 199,368 (384) | 39,874 (28) | 85,439 (108) |

ULB's validation fold has only 28 frauds. It is kept on purpose as the high-noise case, and this is stated in the paper.

**PaySim is out** (V3, decided 27 Oct).

## 3. Models and search spaces

Two models: XGBoost (3.2, CUDA, `tree_method="hist"`) and LightGBM (4.7, CPU). CatBoost is out (V3, decided 27 Oct). Library versions are pinned in an environment file.

The search spaces are those of `experiments/bench/tune.py`. They are identical for every optimiser and are shown to the LLM verbatim.

| Model | Hyperparameter | Range | Scale |
|---|---|---|---|
| Both | `learning_rate` | 0.01–0.3 | log |
| XGBoost | `max_depth` | 3–12 | integer |
| | `min_child_weight` | 1–100 | log |
| | `subsample` | 0.5–1.0 | linear |
| | `colsample_bytree` | 0.5–1.0 | linear |
| | `reg_lambda`, `reg_alpha` | 0.001–10 | log |
| | `gamma` | 0.001–5 | log |
| LightGBM | `num_leaves` | 15–255 | integer, log |
| | `min_child_samples` | 5–300 | integer, log |
| | `subsample` | 0.5–1.0, with `subsample_freq = 1` | linear |
| | `colsample_bytree` | 0.5–1.0 | linear |
| | `reg_lambda`, `reg_alpha` | 0.001–10 | log |

**Number of trees.** The number of trees is not searched. Each trial uses early stopping on the validation fold: patience 100 rounds, at most 3,000 rounds. The early-stopping metric matches the HPO objective (PR-AUC or log loss).

**Final refit.** The selected setting is refitted on the whole training period with `n_estimators = round(best_iteration / 0.8)`, as in c_jbdt_4, and is scored once on the test period.

## 4. Optimisers

| Code | Optimiser | Definition |
|---|---|---|
| O0 | Library defaults | Default hyperparameters, early stopping as in §3. One fit per repeat (the model seed varies) |
| O0-LLM | LLM zero-shot | One call: the task description (§5) and the search space; the LLM returns one setting. No history |
| O1 | Random search | Uniform on the scales in §3 (`optuna.samplers.RandomSampler`) |
| O2 | TPE | `optuna.samplers.TPESampler` with default settings: 10 random start-up trials, set explicitly in the code (`TPE_STARTUP = 10`) |
| O3 | LLM proposer | At each trial, the LLM receives the task description, the search space and the full history (setting, validation score, trees used) and returns the next setting |
| O4 | LLM warm start + TPE | O3 for trials 1–5; trials 6–40 by TPE. Each LLM trial is added to the TPE study as a completed trial (`study.add_trial`), so TPE models them. TPE keeps 10 start-up trials, the same as O2 (D1, decided 28 Oct), so trials 6–10 are TPE's random start-up draws and TPE's model starts at trial 11. O4 therefore differs from O2 only in its first five trials |

**Budget.** 40 trials per search, plus a 4-hour wall-clock limit per search, whichever comes first. A search stopped by the time limit is reported with the number of trials it reached.

**Incumbent.** After trial k, the incumbent is the trial with the best validation score so far; ties go to the earlier trial. Incumbents at k = 5, 10, 20 and 40 are each refitted (§3) and scored on the test period. This gives the anytime curves. Test scores are written to a separate file that the optimiser code cannot import (§10, check C1).

**Handling LLM answers:**
- The answer must be a JSON object with exactly the keys of the search space.
- Non-integer values for integer hyperparameters are rounded and flagged `rounded`.
- Values outside a range are clipped and the trial is flagged `clipped`.
- Booleans, strings and non-finite values count as unparsable.
- A missing key or a non-parsable answer gets one retry, with the parser's error message appended. If the retry fails too, the trial uses a random draw and is flagged `fallback`.
- Exact repeats of an earlier setting are allowed and flagged `duplicate`.
- All flags are reported.

## 5. LLM settings and prompts

**Model.**
- **Primary model (V4a, decided 28 Oct): Claude Opus 5.5 (Anthropic), model ID `claude-opus-5-5`**, through the Claude API with the official `anthropic` Python SDK. It is fixed for the whole study. The SDK version, the API version and the access dates are recorded in §14.
- Settings, fixed: `output_config.effort = "low"`; `thinking = {"type": "adaptive", "display": "summarized"}` (thinking cannot be disabled on this model; the reasoning summaries are logged and may be quoted descriptively in the paper); `max_tokens = 16000`. No server-side model fallback (it would switch models mid-arm) and no Batch API. Prompt caching is not used unless it is decided before the freeze (V4d; c_vs_7a §3 expects little saving).
- **Secondary model for S3 (V4b, decided 28 Oct): Claude Haiku 5.5, model ID `claude-haiku-5-5`**, replacing the local open-weights model. It uses the same adapter, prompts and settings as the primary model (effort `low`, adaptive thinking with summaries, `max_tokens` 16,000), so the model is the only difference between the arms. It runs only in block B7 (§7). Its results files and cache entries carry the suffix `_haiku`.
- No model is changed mid-study. If the provider withdraws the model, the deviation rule (§12) applies.

**Sampling.**
- The provider's default sampling (P2 revised 28 Oct). Claude Opus 5.5 does not accept sampling parameters (`temperature`, `top_p`, `top_k`), so none is sent. The earlier choice of temperature 0.7 cannot be implemented.
- The Claude API takes no sampling seed, so repeats differ through sampling alone. The repeat's LLM seed (2000 + r, §7) is kept in the cache key, so that each repeat and trial is a separate request and never reuses another's cached answer.

**Logging.**
- Every request and response is stored in JSONL: timestamp, model identifier, parameters, full prompt, full response, token counts and latency.
- A cache keyed on the job identifier and the full request allows an exact re-run of a job without new calls. The job identifier is part of the key, so arms that send the same prompt (O0-LLM, O3 and O4 at trial 1) still sample independently.
- The logs are released with the code.

**What is sent.** Only the task description, the search space, hyperparameter settings and validation scores. No data rows, no test-period information, and no feature values.

**Two prompt conditions.** The anonymised prompt is the primary condition for O3 and O4 in H1–H3. The named prompt is used only in the H4 arm of O3 and in O0-LLM.

| Element | Anonymised prompt (primary) | Named prompt (H4 arm) |
|---|---|---|
| Task | "Binary classification on tabular data" | "Fraud detection on the [public name and source] data" (for example "IEEE-CIS Fraud Detection, Kaggle competition, 2019") |
| Data description (V5 c, 28 Oct) | Positive class rate as a band ("below 1%", "between 1% and 5%", "above 5%"); training and validation row counts to two significant figures ("about 160,000") | The same |
| Features | "between 30 and 60 numeric features" (all three datasets fall inside: 31, 35 and 53), with no feature list | The exact count and the original feature names, in the original order |
| Model and objective | Model name, search space, objective, early-stopping rule | The same |
| History (O3, O4) | Trial number, setting, validation score, trees used | The same |

The prompt templates are frozen with the protocol as `prompts/anon.txt` and `prompts/named.txt`. Their wording is fixed in the pilot (§11) and not tuned on test scores.

*Why the bands (V5 c).* In the live smoke test with the exact rate, row counts and an f1…fp feature list, 4 of 40 anonymised answers said the data looked like the credit-card fraud dataset. The rate (0.19%) and the feature count (31) were enough to recognise ULB. With the bands, the same test gave 0 of 40 (c_vs_8). Both prompts carry the same data description, so the H4 contrast is the dataset name and the feature names only. Answers stay free text with a JSON object, not structured outputs, so that the manipulation check (§6) remains possible.

## 6. Hypotheses

**Shared definitions:**
- A *cell* is a dataset × model pair: 3 × 2 = 6 cells.
- "Test PR-AUC" is the average precision of the refitted incumbent on the test period.
- The median and the spread are taken over the 10 repeats of a cell.
- For every comparison, a 95% bootstrap interval (10,000 resamples of repeats, seed 20261103) is computed for the difference in medians.
- *Supported:* a hypothesis's count condition is met using point estimates.
- *Clearly supported:* the condition is also met using the interval bounds.
- *Not supported:* otherwise.

| # | Hypothesis | Arms (objective) | Condition |
|---|---|---|---|
| H1 | Efficiency | O3 and O4 at k = 10 against O2 at k = 40 (PR-AUC objective) | For each of O3 and O4 separately: median test PR-AUC (k = 10) ≥ O2's median (k = 40) − 0.005 in at least 4 of 6 cells. Clearly supported if the lower bound of the difference is ≥ −0.005 in those cells |
| H2 | Selection noise is not reduced | O3, O4 and O2 at k = 40 (PR-AUC objective) | For each of O3 and O4: SD(test PR-AUC) / SD(O2) ≥ 0.8 in at least 4 of 6 cells. The interval for the ratio is a bootstrap percentile interval |
| H3 | Log loss gives better probabilities | O2 at k = 40, log-loss against PR-AUC objective (primary); O3 the same (secondary) | Test log loss is lower and \|Σp / frauds − 1\| is smaller with the log-loss objective, while median test PR-AUC falls by no more than 0.01, all three in at least 4 of 6 cells |
| H4 | Memorised prior | O3 named against O3 anonymised, k = 40 (PR-AUC objective); O0-LLM named against anonymised as a supporting probe | Named > anonymised in median test PR-AUC in at least 3 of the 4 ULB and IEEE-CIS cells, **and** the mean BAF gap is smaller than the mean ULB/IEEE-CIS gap |
| H5 | The validation fold matters | O2 at k = 40 (PR-AUC objective), time-ordered against random fold | A dataset counts if the time-ordered fold gives the higher median test PR-AUC for both models; supported if at least 2 of 3 datasets count |

**Expected direction.** H1 and H3–H5 are expected to hold. H2 is expected to hold because LLM-driven HPO inherits selection noise (LJ's review, §7.4). A failure of any of them is reported with the same prominence.

**Secondary outcomes** (descriptive, no hypothesis):
- recall at 1% false-positive rate;
- Brier score;
- calibration-in-the-large Σp / frauds;
- Spearman correlation between validation and test score across the 40 trials of each search;
- wall time;
- tokens, calls and flagged trials (LLM arms);
- **manipulation check for H4 (V5 c):** the share of answers in the anonymised arms whose text, or thinking summary where the model returns one, names the domain or a benchmark. The frozen word list is in `llm.IDENTIFY_WORDS` (fraud, credit card, Kaggle, ULB, IEEE, Vesta, bank account, BAF, Feedzai, PaySim, cardholder, transaction). It is reported per cell for O0-LLM, O3 and O4, with the same share for the named arm as a reference. It is a lower bound, because answers that contain only JSON cannot show identification. If the share in the anonymised arms is above 10% in any cell, H4 is reported for that cell with this caveat.
- all of the above at k = 5, 10, 20, 40.

## 7. Design and run list

**Repeats.** r = 0…9 for every arm.
- Optimiser seed: 1000 + r (Optuna samplers).
- Model seed: r.
- LLM seed: 2000 + r. It is not sent to the API (no seed parameter), but it is part of the cache key (§5).

The same r gives the same model seed across arms, so arms are paired on r. Pairing is used only in the descriptive figures. The tests in §6 compare medians.

| Block | Arms | Objectives | Runs |
|---|---|---|---|
| B1 | O1, O2 (time-ordered fold) | PR-AUC, log loss | 2 × 6 cells × 2 × 10 = 240 searches |
| B2 | O3 anonymised, O4 | PR-AUC, log loss | 2 × 6 × 2 × 10 = 240 searches |
| B3 | O3 named (H4) | PR-AUC | 6 × 10 = 60 searches |
| B4 | O2, random fold (H5) | PR-AUC | 6 × 10 = 60 searches |
| B5 | O0 | PR-AUC, log loss (early-stopping metric) | 6 × 2 × 10 = 120 single fits |
| B6 | O0-LLM, named and anonymised | PR-AUC | 6 × 2 × 10 = 120 single calls and fits |
| B7 | O3 anonymised with Claude Haiku 5.5 (S3, V4b) | PR-AUC | 6 × 10 = 60 searches |
| **Total** | | | **660 searches of 40 trials (about 26,400 fits), 240 single fits; about 7,900 LLM calls with Opus 5.5 (about 12.6 million tokens; c_vs_5 §3) and 2,400 with Haiku 5.5** |

This revises c_vs_1 §5.6 (660 searches). The B3 log-loss arm is dropped, because H4 is stated for PR-AUC only.

**LLM budget and cost stop rule** (28 Oct; c_vs_7a). The LLM budget is set by the iACE award (requested £400, c_vs_5), and the rule below is written for a budget of £200. Two plans are pre-specified:
- **Plan F (default):** the full design above.
- **Plan R (reduced):** in B2, only the PR-AUC objective is run for O3 and O4 (60 + 60 searches instead of 240). B3, B6 and all non-LLM blocks are unchanged. H1–H5 are still tested in full; only the secondary H3 analysis with O3 (log-loss objective) is lost. B7 (Haiku, about $1) runs under both plans.

*Stop rule:* after the pilot (§11), the LLM cost of Plan F is projected from the tokens measured in the pilot, at the list prices on that day. If the projection exceeds USD 230, or exceeds 90% of the awarded budget if that is lower, Plan R is run instead of Plan F. The decision uses cost only, never pilot scores, and is recorded in §14 before any main LLM run starts. A spending limit equal to the award is set on the API account. If the limit is reached during the main runs, the remaining LLM searches are reported as missing (§8, item 4) and no other change is made.

**Run order.**
- Blocks run in the order B5, B1, B4, B2, B3, B6, B7.
- Within a block, the order is shuffled with seed 20261104, so that machine load does not line up with one arm.
- Each search writes one JSON result file and is skipped if that file already exists, so the queue can resume after a failure.
- Timing is recorded, but timing comparisons are reported only for runs on the same machine and under the same load.

## 8. Analysis

1. **Hypothesis table.** For each of H1–H5: cell-level estimates, intervals, the count, and the verdict (supported, clearly supported or not supported).
2. **Figures:**
   - anytime curves (median and interquartile range of test PR-AUC against k);
   - spread across repeats per cell;
   - validation-against-test scatter (all trials, one panel per cell);
   - calibration-in-the-large by objective.
3. **Sensitivity analyses** (pre-specified):
   - S1: H1–H3 with the trials flagged `fallback` treated as failures (scored at the previous incumbent);
   - S2: H1 at k = 20;
   - S3: block B7, O3 anonymised with Claude Haiku 5.5 against O3 anonymised with Opus 5.5 (from B2), PR-AUC objective, all six cells. Reported descriptively, separately from H1–H5: median test PR-AUC and the spread across repeats for each model, tokens and cost per search. No hypothesis is tested.
4. **No pooling.** No significance test is pooled across datasets, and no cell or arm is dropped after results are seen. A run that fails twice is reported as missing, with the reason.
5. **Limitation stated in the paper** (D2 = A): "The study covers gradient-boosted trees on three public benchmarks without account histories. Whether its conclusions about LLM-guided tuning hold for deep tabular models, or for sequence models trained on account histories such as fraud transformers, is outside its scope."

## 9. Reproducibility

- **Code.** The folder `experiments/hpo_llm/` is kept under git from the first commit. Repository: `github.com/ljakaite/llm-hpo-fraud` (P3, revised 28 Oct), private until submission. VS and the students are collaborators, with secret scanning and push protection on. MIT licence, copyright "© 2026 the authors (University of Bedfordshire)". No commercial content in the repository. At submission it is made public with a GitHub release archived on Zenodo, and the DOI is cited in the data availability statement.
- **Environment.** An environment file pinning Python, XGBoost, LightGBM, Optuna and the LLM client versions.
- **Released at submission:**
  - code;
  - prompt templates;
  - LLM logs and cache;
  - all result JSON files;
  - the analysis notebook or script that regenerates every table and figure from the JSON files.
- **Not released:** the data. Download instructions and checksums are given instead.

## 10. Checks before the freeze

| Check | What it verifies |
|---|---|
| C1 test-period guard | The search code receives only the training period; a test asserts that no test-period index can reach the optimiser, and the test scorer runs as a separate step that refits from the saved settings |
| C2 loader equivalence | With the inner split switched off, the new loader reproduces `datasets.py` exactly (max absolute difference 0) |
| C3 re-encoding | During the search, encodings are fitted on inner-training rows only (unit test with a planted category that occurs only in the validation fold) |
| C4 search-space conformity | 1,000 random draws per optimiser fall inside the §3 ranges; clipping works |
| C5 LLM parser | Malformed, partial and out-of-range answers are handled as in §4 |
| C6 resume | A queue killed mid-run resumes without repeating or losing searches |
| C7 smoke test | One 5-trial search per arm on ULB runs end to end |

**Status (27 Oct, c_vs_2):** C1–C7 all pass (`experiments/hpo_llm/checks/report.json`). C5 and C7 used the scripted test double, because the LLM is not chosen yet (V4). With the real adapter, C5 (parser and logging) and C7 (smoke test) are repeated before the pilot. **Done 28 Oct:** the Claude adapter (`llm.AnthropicLLM`, refusal treated as an unusable answer and flagged `refusal`) passed C5L (2 live calls: answers in the search space, cache hit on repeat, usage logged, refusal rule) and C7L (24 ULB arms, 56 live calls, all `end_turn`, no retries or fallbacks, model `claude-opus-5-5`). C7L found that in 4 of 40 anonymised answers the model said the data looked like credit-card fraud data. **V5 (c), decided 28 Oct:** banded data description in both prompts and a manipulation check (§5, §6). C1, C5, C7, C5L and C7L were rerun with the new prompts: all PASS; 0 of 40 anonymised answers identify the data (c_vs_8).

## 11. Pilot

- **Scope.** ULB and BAF, XGBoost only, all arms, 2 repeats, with seeds r = 90, 91. These are outside the main range, so pilot runs never enter the analysis.
- **What the pilot may change** (each recorded as an amendment before the freeze):
  - prompt wording, for parse failures only;
  - the time limit per search;
  - the retry rule;
  - the run budget, by the cost stop rule in §7 only (Plan F or Plan R). If compute time, not cost, makes §7 unaffordable, repeats may go down to 5 for IEEE-CIS only, with the reason stated.
- **What it may not change:** the hypotheses, the thresholds, the arms (apart from the pre-specified switch from Plan F to Plan R), the datasets, the outcomes, the model or its settings (§5).
- **Pilot report:** time per search per cell, tokens per call (input, output and thinking, from `usage`), parse and fallback rates, the projected LLM cost of Plan F and Plan R, and the stop-rule decision. Test scores of pilot runs are not computed, so that they cannot steer the protocol.

**Pilot results (28–29 Oct 2026; `protocol/pilot_report.json`, script `pilot_report.py`).**
- **Runs:** 60 searches (ULB and BAF, XGBoost, r = 90, 91, all arms including B7), 2.3 hours, no failed jobs. 688 LLM calls (LLM access 28 Oct 23:17 to 29 Oct 00:54), all with `stop_reason` `end_turn`.
- **Flags:** no refusals, retries or fallbacks; `clipped` 0.15% and `duplicate` 2.9% of LLM trials.
- **Tokens per 40-trial O3 search:** Opus 5.5 about 105,000 input and 5,900 output (anonymised), about $0.54; Haiku 5.5 about 104,000 input and 15,000 output, about $0.02.
- **Projected LLM cost:** Plan F USD 103, Plan R USD 69. **Stop rule: Plan F** (threshold USD 230).
- **Time per search:** up to 540 s (BAF, O3, log loss), far below the 4-hour limit. Projected main-run compute is about 2–3 days sequential; it is not a reason to change §7.
- **Manipulation check:** 0 of 524 anonymised answers (364 Opus, 160 Haiku, text and thinking summaries) identify the data.
- **Amendments arising from the pilot: none.** The prompt wording, time limit, retry rule and run budget are unchanged.

## 12. Deviations

After the freeze, any change is entered in `deviations.md` with:
- the date;
- what changed and why;
- who approved it (LJ and VS);
- whether any result had been seen.

The paper reports every deviation. If the hosted LLM becomes unavailable before B2 and B3 are complete, the remaining LLM runs move to a successor model; results are reported by model, and H1–H4 are assessed only on the model that completed all six cells.

## 13. Open items before the freeze

| # | Item | Proposal |
|---|---|---|
| P1 | Current holders of roles A, B and C | **Decided 27 Oct:** Student A = T.F.; Student B = P.K.; Student C vacant (WP3 split between A and B by VS). Names may change; edit this row only |
| P2 | LLM temperature | Decided 27 Oct: 0.7. **Revised 28 Oct:** provider default; no sampling parameters, because Claude Opus 5.5 does not accept them (§5) |
| P3 | Code repository | Decided 27 Oct: private group repository. **Revised 28 Oct:** `github.com/ljakaite/llm-hpo-fraud` on LJ's personal account (c_vs_9), private until submission, MIT licence. First commit `9b53c61` (9b53c61b83bd421f9144a242bc2a090449af54b7) pushed to `main` on 28 Oct by VS (`vitsch`, collaborator); it contains code, prompts, checks, environment, LICENSE (MIT) and README, with no data, results or keys |
| V3 | PaySim and CatBoost | **Decided 27 Oct:** both out |
| V4 | Hosted model and budget; local model arm | **V4a decided 28 Oct:** Claude Opus 5.5, effort `low`. Budget: iACE award, with the §7 stop rule. A5 applied in the adapter (refusal = unusable answer, flagged). A4 (structured outputs) not adopted: it would remove the free text the H4 manipulation check needs (V5 c). **V4b decided 28 Oct:** Claude Haiku 5.5, reduced design (B7, 60 searches) |
| V5 | Leaky anonymised prompt | **Decided 28 Oct, option (c):** bands for rate, rows and features in both prompts, and the manipulation check (§5, §6) |
| H-thr | Thresholds in §6 (0.005, 0.8, 0.01, 4 of 6, 3 of 4, 2 of 3) | As proposed in c_vs_1 §5.1. **Signed off 28 Oct** |
| V4d | Prompt caching | **Settled 28 Oct:** not used (§5) |
| Sign-off | All items above | **LJ signed off 28 Oct (reported by VS); VS 28 Oct.** Nothing left open. Pilot next (§11) |

## 14. Freeze record

| Field | Value |
|---|---|
| Frozen on | 29 Oct 2026 |
| sha256 of this file | Recorded in `protocol/FROZEN.sha256` and in the message of the tag `protocol-freeze`; a file cannot contain its own hash |
| Code commit | The commit tagged `protocol-freeze` on `main` (first commit 9b53c61; signed protocol 25c3fb0, tag `protocol-signed`) |
| Hosted LLM identifier and provider | `claude-opus-5-5`, Anthropic (Claude API); `anthropic` SDK version 2.3.0; API version 2026-10-21; access dates: pilot 28–29 Oct 2026, main runs [recorded at completion] |
| Stop-rule decision (§7) | **Plan F**; projected cost USD 103.12 (Plan R USD 69.35); threshold USD 230 (the rule for a £200 award; the iACE award is still pending, and if it is below £200 the 90% rule is re-applied before the LLM blocks start) |
| Approved by | LJ [x] and VS [x] (sign-off of v0.10, 28 Oct); freeze approval after the pilot: LJ [x] VS [x] (VS's instruction of 28 Oct to freeze after LJ's sign-off; no amendments arose) |

## 15. References for the scope statement (§1)

All four references verified (R1, R2 in LJ's review; R3, R4 on 28 Oct).

| # | Reference | Status |
|---|---|---|
| R1 | Aminian, G.; Elliott, A.; Li, T.; Wong, T.C.H.; Dehon, V.C.; Szpruch, L.; Maple, C.; Read, C.; Brown, M.; Reinert, G.; Mamouei, M. FraudTransformer: Time-aware GPT for digital payment fraud detection. *2026 IEEE Conference on Artificial Intelligence (CAI)*, 2026, pp. 850–856. https://doi.org/10.1109/CAI68641.2026.11536152 (preprint arXiv:2509.23712) | Verified in LJ's review (ref. 23); data and baselines read in the local PDF (c_vs_3) |
| R2 | Zhao, Z.Y.; Zhu, Z.; Li, G.; Wang, W.; Wang, B. Generative pretraining at scale: Transformer-based encoding of transactional behavior for fraud detection. Preprint arXiv:2312.14406, 2023 | Verified in LJ's review (ref. 22); local PDF |
| R3 | Grinsztajn, L.; Oyallon, E.; Varoquaux, G. Why do tree-based models still outperform deep learning on typical tabular data? In *Advances in Neural Information Processing Systems 35* (NeurIPS 2022, Datasets and Benchmarks Track); Curran Associates, 2022; pp. 507–520. https://doi.org/10.52202/068431-0037 (preprint arXiv:2207.08815) | **Verified 28 Oct** against the official NeurIPS BibTeX (c_vs_6) |
| R4 | Shwartz-Ziv, R.; Armon, A. Tabular data: Deep learning is not all you need. *Information Fusion* 2022, 81, 84–90. https://doi.org/10.1016/j.inffus.2021.11.011 (preprint arXiv:2106.03253) | **Verified 28 Oct** against Crossref and DBLP (c_vs_6) |
