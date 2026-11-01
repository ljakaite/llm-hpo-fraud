# Deviations from the frozen protocol (v1.0, tag protocol-freeze)

Every change after the freeze is entered here with its date, what changed and why, who approved it, and whether any result had been seen (protocol §12). The paper reports every entry.

## Deviations

- **D1 (1 Nov 2026), validation-test secondary outcome and Figure 3.**
  - *What changed.* §6 lists "Spearman correlation between validation and test score across the 40 trials of each search", and §8 a "validation-against-test scatter (all trials)". `score.py` (frozen code, §4) scores only the incumbents at k = 5, 10, 20 and 40, so trial-level test scores do not exist.
  - *Replacement.* Spearman correlation across the k = 40 incumbents of all PR-AUC, time-fold searches of O1–O4 in each cell (n = 60 per cell), and a scatter of the incumbents at k = 5, 10, 20 and 40.
  - *Why.* Scoring all 26,400 trials would take an estimated 130 hours of refits.
  - *Results seen.* Test scores of the incumbents existed when the replacement was chosen; no hypothesis or threshold is affected.
  - *Approved by:* VS, 1 Nov 2026 (confirmed).

## Notes (clarifications, not deviations)

- **29 Oct 2026 — funding timing.**
  - The iACE award is not yet decided: VS will submit the spending form after all experiments are complete, with the receipts.
  - VS pre-funds the Claude API account.
  - The stop-rule decision in §14 (Plan F, projected USD 103, threshold USD 230) stands: the threshold rule was applied as written, and the protocol's budget is the funding actually available on the account.
  - No design element changes.
  - Results seen: none. Pilot validation scores were summarised descriptively in c_vs_13; no test score has been computed.
- **31 Oct 2026, main runs completed** (completion record for §14; no deviation).
  - Runs: all 900 jobs (Plan F), 29 Oct 09:23 to 31 Oct, one queue in the protocol's block order. 0 failed attempts, 0 missing results; 61.8 hours of compute on the Dell.
  - LLM: 10,322 calls, all `end_turn`. Opus 5.5: 19.0 M input and 1.06 M output tokens, about USD 97.3. Haiku 5.5: 6.0 M input and 0.89 M output tokens, about USD 1.0.
  - Flags on the 10,320 LLM trials: duplicate 308, clipped 36, retried 2 (both recovered); no fallbacks or refusals.
  - Main-run access dates for §14: 29–31 Oct 2026.
  - Results seen: none. No test score has been computed.

## Additional exploratory analyses (declared 31 Oct 2026, after the main runs and BEFORE any test scoring)

These change no hypothesis, threshold, arm or outcome of the frozen protocol. They are reported in the paper as exploratory. They use only the search files (validation scores) and the LLM logs, plus new LLM calls for T6–T8. None reads a test score. Approved by VS (and LJ, through the protocol sign-off of exploratory reporting).

**M3: clarification of O2.** "TPE with default settings" is Optuna 5.0 `TPESampler(seed, n_startup_trials=10)`. With `multivariate=None`, this resolves to the **multivariate** TPE for single-objective studies (Optuna source, `_is_multivariate`). The Methods state this. Consequence: O1 and O2 share their first 10 trials by construction, because the start-up draws use the same seed.

**M1: memorised or inferred LLM proposals** (c_vs_LLM_idea_2). Seeds 20261105–20261107.
- **T3, history dependence.** For every LLM-proposed trial with index ≥ 4 in O3 and O4 (Opus, and Haiku in B7), compute the Euclidean distance in the normalised search space (log scale where §3 says log) between the proposal and the incumbent at that point. Compare it with O1 and O2 at the same trial indices. Report medians per cell.
- **T4, textbook values.** The share of O0-LLM settings and O3 first-trial settings that match at least 3 of these defaults: XGBoost (learning rate 0.1 or 0.05; `max_depth` 6; `subsample` 0.8; `colsample_bytree` 0.8; `min_child_weight` 1); LightGBM (learning rate 0.1 or 0.05; `num_leaves` 31; `min_child_samples` 20; `subsample` 0.8; `colsample_bytree` 0.8).
- **T5, zero-shot sensitivity.** Per cell, the share of repeats where the O0-LLM named and anonymised settings are identical, and their mean normalised distance.
- **T6, counterfactual history** (new Opus calls, same settings as §5).
  - Sample: 60 logged O3 anonymised PR-AUC prompts at trial index 10–30, 10 per cell (seed 20261105).
  - Three variants of each: (a) as logged; (b) scores permuted across trials (seed 20261106); (c) scores reversed in rank order, so that the worst trial shows the best score.
  - Outcome: the share of proposals that lie closer to the trial *presented* as best than to the trial that truly scored best, and the mean distance to the presented-best trial, per variant.
  - About 180 calls.
- **T7, identification probe** (Opus). Each dataset × model's anonymised data description, and the question: "Which publicly available dataset is this most likely to be? Answer with a dataset name, or 'unknown'." 10 calls per cell (60 in total). Outcome: the share naming the correct dataset or its domain (`llm.IDENTIFY_WORDS`).
- **T8, recall probe** (Opus). "What hyperparameter settings did strong public solutions use for [dataset public name] with [model]? Reply with one JSON object with exactly these keys: [search space]." 5 calls per cell (30 in total). Outcome: the normalised distance between the recalled settings and the O3 named and anonymised proposals (first 5 trials), compared with the distance to random draws.

**M4: interaction structure and ridge coherence** (c_vs_LLM_idea_3, c_vs_LLM_idea_4 Part A).
- **(a) Interactions.** Per cell, pooling the O1 (random search) PR-AUC trials of all 10 repeats (400 trials), report the 5-fold cross-validated R² of validation PR-AUC on the hyperparameters: main effects plus squares, versus also all pairwise interactions (ridge regression on standardised log/linear parameters), and a random forest.
- **(b) Ridge strength in the data.** Within-pair correlation among the top 25% of O1 trials for the pairs (log learning rate, log trees chosen), (depth or log `num_leaves`, log `min_child_weight` or log `min_child_samples`) and (`subsample`, `colsample_bytree`).
- **(c) Ridge coherence of each optimiser.** The mean |within-pair correlation| of trials 21–40 for the same pairs, for O1, O2, O3 (Opus), O3 (Haiku) and O4. The simulation's reference values: 0.2–0.3 for the classical optimisers, against a true ridge of 0.83.
- **1 Nov 2026, test scoring completed** (`score.py`, 31 Oct 23:39 to about 14:00 on 1 Nov): 900/900 files. The pre-registered analysis was run with `analysis/h1_h5.py` (bootstrap seed 20261103, 10,000 resamples). S1 is identical to the primary analysis, because no trial was flagged `fallback`.

## Post-hoc exploratory analyses (declared AFTER the test results were seen; reported as such)

- **1 Nov 2026, the mechanism behind H5** (`analysis/h5_mechanism.py`). Descriptive only, prompted by the reversed H5 result:
  - fold sizes, frauds, fraud rates and time spans;
  - settings selected by O2 under each fold;
  - Spearman correlation between log final trees and test PR-AUC.
  
  It does not change the H5 verdict, which is reported as pre-registered ("not supported", reversed on IEEE-CIS).
- **1 Nov 2026, reporting decisions for H5** (VS). H5-a: H5 stays in the hypothesis table with its pre-registered verdict ("not supported; reversed on IEEE-CIS"). H5-b: the post-hoc mechanism analysis is reported in the exploratory section, labelled post hoc. H5-c (an extra figure): not adopted.
- **1 Nov 2026, supplementary p-values** (`analysis/pvalues.py`, added after the results; not pre-registered). Two-sided permutation tests of the difference in medians per cell (10,000 relabellings, seed 20261108), and exact binomial tests for T6. They are reported in the manuscript beside the pre-registered bootstrap intervals and count rules, which remain the basis of the verdicts.
- **1 Nov 2026, author change (no design change).** Pawanpreet Kaur is replaced by Courage Mabani as an author (VS). Protocol §13 P1 named roles only; the holder of role B changes accordingly.
