# Deviations from the frozen protocol (v1.0, tag protocol-freeze)

Every change after the freeze is entered here with its date, what changed and why, who approved it, and whether any result had been seen (protocol §12). The paper reports every entry.

## Deviations

None.

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
