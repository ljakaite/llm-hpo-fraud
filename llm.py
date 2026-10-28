"""
LLM side of the study (protocol sections 4 and 5): prompts, answer parsing, logging and cache.

The hosted model is not yet chosen (decision V4), so only a provider-neutral interface is here.
A provider adapter implements `complete(prompt, seed) -> (text, usage)`. ScriptedLLM is the test
double used by the checks; it never contacts a service.
"""

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

import space as S

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
TEMPERATURE = None           # P2 revised 28 Oct: provider default; Claude Opus 5.5 accepts no sampling parameters
MODEL_ID = "claude-opus-5-5"  # V4a (28 Oct)
HAIKU_ID = "claude-haiku-5-5"  # V4b (28 Oct): S3 sensitivity arm, same settings as the primary model
EFFORT = "low"
PUBLIC_NAMES = {
    "ieee": "IEEE-CIS Fraud Detection (Kaggle competition, 2019; card-not-present e-commerce transactions)",
    "baf": "Bank Account Fraud suite, Base variant (Jesus et al., NeurIPS Datasets and Benchmarks 2022; bank account opening applications)",
    "ulb": "Credit Card Fraud Detection (ULB Machine Learning Group, Kaggle; European cardholders, September 2013)",
}
MODEL_NAMES = {"xgb": "XGBoost", "lgbm": "LightGBM"}
OBJECTIVE_TEXT = {"pr_auc": "maximise validation PR-AUC (average precision)",
                  "log_loss": "minimise validation log loss (reported as its negative, so higher is better)"}


# ------------------------------------------------------------------ prompts
# V5 (c), 28 Oct: the anonymised prompt gave ULB away through its exact positive rate and feature
# count (C7L: 4 of 40 answers named credit-card fraud data). Both prompts now give the rate as a band
# and the row counts to two significant figures; the anonymised prompt gives the feature count as a
# band and no feature list. The named prompt adds only the dataset name and the feature names.
RATE_BANDS = ((0.01, "below 1%"), (0.05, "between 1% and 5%"), (1.01, "above 5%"))
FEATURE_BAND = (30, 60)                      # IEEE-CIS 53, BAF 35, ULB 31 features: all inside


def rate_band(rate):
    return next(text for upper, text in RATE_BANDS if rate < upper)


def about(n):
    """Two significant figures, e.g. 159494 -> 'about 160,000'."""
    from math import floor, log10
    r = int(round(n, -int(floor(log10(abs(n)))) + 1)) if n else 0
    return f"about {r:,}"


def feature_text(names, condition):
    if condition == "named":
        return f"{len(names)} numeric features ({', '.join(names)})"
    lo, hi = FEATURE_BAND
    assert lo <= len(names) <= hi, "feature count outside the anonymised band"
    return f"between {lo} and {hi} numeric features"


# Manipulation check for H4 (V5 c): an anonymised answer "identifies" the data if its text names the
# domain or a benchmark. Frozen word list; a lower bound, since answers with JSON only cannot show it.
IDENTIFY_WORDS = ("fraud", "credit card", "credit-card", "creditcard", "kaggle", "ulb", "ieee",
                  "vesta", "bank account", "baf", "feedzai", "paysim", "cardholder", "transaction")


def mentions_dataset(text):
    t = (text or "").lower()
    return any(w in t for w in IDENTIFY_WORDS)


def build_prompt(condition, dataset, model, objective, desc, feature_names, history, k_next, budget):
    tpl = (PROMPTS / f"{condition}.txt").read_text()
    task = (f"fraud detection on the {PUBLIC_NAMES[dataset]} data" if condition == "named"
            else "binary classification on tabular data")
    hist = "\n".join(
        f"trial {h['trial']}: {json.dumps(h['params'], sort_keys=True)} -> score {h['value']:.6f}, trees {h['trees']}"
        for h in history) or "(no trials yet)"
    return tpl.format(task=task, positive_rate=rate_band(desc["positive_rate"]), n_train=about(desc["n_train"]),
                      n_val=about(desc["n_val"]), features=feature_text(feature_names, condition),
                      model=MODEL_NAMES[model], objective=OBJECTIVE_TEXT[objective], space=S.describe(model),
                      history=hist, k_next=k_next, budget=budget)


# ------------------------------------------------------------------ parsing (check C5)
class ParseError(ValueError):
    pass


def parse_answer(text, model):
    """Extract one JSON object with exactly the search-space keys; clip to the ranges.
    Returns (params, flags). Raises ParseError on missing keys or unparsable text."""
    blocks = re.findall(r"\{[^{}]*\}", text or "", flags=re.S)
    if not blocks:
        raise ParseError("no JSON object found")
    try:
        obj = json.loads(blocks[-1])
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e.msg}") from None
    keys = set(S.SPACES[model])
    missing, extra = keys - set(obj), set(obj) - keys
    if missing:
        raise ParseError(f"missing keys: {sorted(missing)}")
    try:
        params, flags = S.clip(model, {k: obj[k] for k in keys})
    except (TypeError, ValueError) as e:
        raise ParseError(f"bad value: {e}") from None
    if extra:
        flags.append("extra_keys")
    return params, flags


# ------------------------------------------------------------------ clients, log and cache
class AnthropicLLM:
    """Claude API adapter (V4a; protocol section 5). Settings are fixed for the study: effort `low`,
    adaptive thinking with summaries, no sampling parameters, no seed (the API has none), no
    server-side fallback. The key is read from ANTHROPIC_API_KEY by the SDK."""

    def __init__(self, model_id=MODEL_ID, effort=EFFORT, max_tokens=16000, max_retries=5):
        import anthropic                      # imported here, so the checks run without the SDK
        self.anthropic = anthropic
        self.model_id, self.effort, self.max_tokens = model_id, effort, max_tokens
        self.client = anthropic.Anthropic(max_retries=max_retries)   # retries 408/409/429/5xx

    def complete(self, prompt, seed):
        r = self.client.messages.create(
            model=self.model_id, max_tokens=self.max_tokens,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in r.content if b.type == "text")
        thinking = "".join(b.thinking or "" for b in r.content if b.type == "thinking")
        usage = dict(stop_reason=r.stop_reason, model=r.model, input_tokens=r.usage.input_tokens,
                     output_tokens=r.usage.output_tokens, thinking_summary=thinking,
                     sdk=self.anthropic.__version__)
        if r.stop_reason == "refusal":       # A5: treated as an unusable answer by propose()
            sd = getattr(r, "stop_details", None)
            usage["refusal_category"] = getattr(sd, "category", None)
            text = ""
        return text, usage


class ScriptedLLM:
    """Test double: returns answers from a list or a function of the prompt. No network."""
    model_id = "scripted-test-double"

    def __init__(self, answers):
        self.answers, self.calls = answers, 0

    def complete(self, prompt, seed):
        self.calls += 1
        a = self.answers(prompt, self.calls) if callable(self.answers) else self.answers[(self.calls - 1) % len(self.answers)]
        return a, dict(input_tokens=len(prompt) // 4, output_tokens=len(a) // 4)


class LoggedLLM:
    """Wraps a client: JSONL log of every call and a cache keyed on the full request."""

    def __init__(self, client, log_path, cache_path, scope=""):
        # scope = the job id: the cache serves exact re-runs of the same job only, so arms that
        # happen to send the same prompt (O0-LLM, O3 and O4 at trial 1) still sample independently
        self.client, self.log_path, self.cache_path = client, Path(log_path), Path(cache_path)
        self.scope = scope
        self.cache, self.stops = {}, {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                r = json.loads(line)
                self.cache[r["key"]] = r["response"]
                self.stops[r["key"]] = r.get("stop_reason")

    def key(self, prompt, seed):
        req = dict(scope=self.scope, model=self.client.model_id, temperature=TEMPERATURE, seed=seed, prompt=prompt)
        return hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()

    def complete(self, prompt, seed):
        k = self.key(prompt, seed)
        hit = k in self.cache
        t0 = time.time()
        if hit:
            text, usage = self.cache[k], dict(stop_reason=self.stops.get(k), cache_hit=True)
        else:
            text, usage = self.client.complete(prompt, seed)
            self.cache[k], self.stops[k] = text, usage.get("stop_reason")
            with self.cache_path.open("a") as f:
                f.write(json.dumps(dict(key=k, response=text, stop_reason=usage.get("stop_reason"))) + "\n")
        with self.log_path.open("a") as f:
            f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), model=self.client.model_id,
                                    temperature=TEMPERATURE, seed=seed, key=k, cache_hit=hit,
                                    prompt=prompt, response=text, usage=usage,
                                    latency_s=round(time.time() - t0, 3))) + "\n")
        return text, usage


def propose(llm, prompt, model, seed, rng):
    """One proposal with the protocol's retry and fallback rule (section 4).
    Returns (params, flags, n_calls)."""
    text, usage = llm.complete(prompt, seed)
    refused = ["refusal"] if (usage or {}).get("stop_reason") == "refusal" else []
    try:
        p, flags = parse_answer(text, model)
        return p, flags, 1
    except ParseError as e:
        retry = prompt + f"\n\nYour previous answer could not be used ({e}). Reply with one JSON object only."
        text, usage = llm.complete(retry, seed)
        if (usage or {}).get("stop_reason") == "refusal" and "refusal" not in refused:
            refused.append("refusal")
        try:
            p, flags = parse_answer(text, model)
            return p, flags + refused + ["retried"], 2
        except ParseError:
            return random_params(model, rng), refused + ["fallback"], 2


def random_params(model, rng):
    out = {}
    for k, (lo, hi, kind, log) in S.SPACES[model].items():
        if log:
            v = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            v = float(rng.uniform(lo, hi))
        out[k] = int(min(max(round(v), lo), hi)) if kind == "int" else v
    return out
