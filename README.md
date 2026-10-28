# llm-hpo-fraud

Code, prompts and logs for a pre-registered study of LLM-guided hyperparameter optimisation (HPO) for gradient-boosted fraud detectors, evaluated out of time on three public benchmarks (IEEE-CIS, Bank Account Fraud, ULB credit card). Claude Opus 5.5 and Claude Haiku 5.5 propose hyperparameters, and they are compared with random search and TPE (Optuna).

**Status:** private, before the protocol freeze. The pre-registered protocol and the prompt templates are committed at the freeze and tagged `protocol-freeze`; main runs start only after that tag. The repository is made public, with an archived release and DOI, when the paper is submitted.

**Authors:** see the paper (University of Bedfordshire). **Licence:** MIT (see `LICENSE`).

**Data are not included.** The datasets are subject to their own licences and terms (IEEE-CIS competition rules; BAF: CC BY-NC-SA 4.0; ULB: Kaggle). Download them yourself and place them as:

    <project>/data/ieee/train_transaction.csv, train_identity.csv
    <project>/data/baf/Base.csv
    <project>/data/ulb/creditcard.csv

where this repository is cloned to `<project>/experiments/hpo_llm`, or set `HPO_FRAUD_DATA` to the data folder.

**LLM access:** the Claude API, through the `anthropic` Python SDK. The key is read from the `ANTHROPIC_API_KEY` environment variable and must never be committed. Check C2 compares the loader with the group's earlier benchmark loader (`experiments/bench/datasets.py`), which is not part of this repository.

| File | Role | Sees the test period? |
|---|---|---|
| `data.py` | Loaders, out-of-time split, validation folds, encodings refitted on inner-training rows | No (training period only) |
| `space.py` | Search spaces, Optuna suggestions, clipping and rounding | No |
| `models.py` | Early-stopped fits and the final refit | No |
| `llm.py` | Prompts, answer parser, retry/fallback, JSONL log and cache; `ScriptedLLM` test double | No |
| `search.py` | One search job (O0, O0LLM, O1–O4) -> `results/search/<job>.json` | No |
| `queue_runs.py` | Run list (blocks B1–B6), shuffled order, resume and one retry | No |
| `testdata.py` | Test-period access | Yes |
| `score.py` | Refits incumbents at k = 5, 10, 20, 40 and scores them on the test period -> `results/test/` | Yes |
| `checks/run_checks.py` | Checks C1–C7 | C2 and C7 only |
| `prompts/anon.txt`, `prompts/named.txt` | Prompt templates (frozen with the protocol) | |

Run from the project root with the project `.venv`:

    .venv/bin/python -I experiments/hpo_llm/checks/run_checks.py          # all checks
    .venv/bin/python -I experiments/hpo_llm/queue_runs.py --list           # run-list counts
    .venv/bin/python -I experiments/hpo_llm/queue_runs.py --blocks B5 B1   # non-LLM blocks
    .venv/bin/python -I experiments/hpo_llm/score.py                       # after the searches

The LLM blocks (B2, B3, B6) need a provider adapter in `search.make_llm`, which is not written yet (decision V4).
