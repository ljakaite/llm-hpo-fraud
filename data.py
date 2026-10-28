"""
Data for the LLM-HPO study (MAKE_SI_FD/VS_students/hpo_llm_protocol.md, section 2).

Same rows, features and out-of-time split as experiments/bench/datasets.py, with one change:
target/frequency encodings are refitted on the inner-training rows during the search, so that
validation rows are never encoded with their own labels.

The search side gets only a TrainPeriod. The test period is reachable only through
testdata.py, which the search code must not import (check C1).
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
# Data folder: <project>/data by default; set HPO_FRAUD_DATA to use another location (README).
DATA = Path(os.environ.get("HPO_FRAUD_DATA", ROOT / "data"))
DATASETS = ("ieee", "baf", "ulb")
VAL_Q = 0.80                 # time-ordered fold: last 20% of the training period
RANDOM_FOLD_SEED = 20261101  # stratified random fold (H5 arm)
SMOOTH_M = 20.0

_raw = {}


# ------------------------------------------------------------------ raw frames (as datasets.py)
IEEE_NUM = (["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1"]
            + [f"C{i}" for i in range(1, 15)] + ["D1", "D2", "D3", "D4", "D10", "D11", "D15"])
IEEE_CAT = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "M4", "M5", "M6", "DeviceType"]
BAF_CAT = ["payment_type", "employment_status", "housing_status", "source", "device_os"]


def _load(name):
    if name in _raw:
        return _raw[name]
    if name == "ulb":
        d = pd.read_csv(DATA / "ulb" / "creditcard.csv")
        d["hour"] = (d["Time"] // 3600) % 24
        d["log_amount"] = np.log1p(d["Amount"])
        spec = dict(time="Time", label="Class", cats=[], dropna=False,
                    num=["hour"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "log_amount"])
    elif name == "ieee":
        t = pd.read_csv(DATA / "ieee" / "train_transaction.csv",
                        usecols=["TransactionID", "TransactionDT", "isFraud"] + IEEE_NUM + IEEE_CAT[:-1])
        i = pd.read_csv(DATA / "ieee" / "train_identity.csv", usecols=["TransactionID", "DeviceType"])
        d = t.merge(i, on="TransactionID", how="left")
        d["log_amt"] = np.log1p(d["TransactionAmt"])
        d["hour"] = (d["TransactionDT"] // 3600) % 24
        spec = dict(time="TransactionDT", label="isFraud", cats=IEEE_CAT + ["card1", "addr1"],
                    dropna=False, num=IEEE_NUM + ["log_amt", "hour"])
    elif name == "baf":
        d = pd.read_csv(DATA / "baf" / "Base.csv")
        spec = dict(time="month", label="fraud_bool", cats=BAF_CAT, dropna=True,
                    num=[c for c in d.columns if c not in BAF_CAT + ["fraud_bool", "month"]])
    else:
        raise ValueError(name)
    _raw[name] = (d, spec)
    return _raw[name]


def _period_masks(d, spec, name):
    t = d[spec["time"]]
    if name == "baf":                       # the authors' split: months 0-5 train, 6-7 test
        return (t <= 5).values, (t >= 6).values
    cut = np.quantile(t, 0.70)
    return (t <= cut).values, (t > cut).values


# ------------------------------------------------------------------ encodings
def encode(fit, others, spec):
    """Smoothed target rate (m = 20) and frequency, fitted on `fit` only.
    Returns the feature matrices for fit and each frame in `others`, and the column names."""
    fit, others = fit.copy(), [o.copy() for o in others]
    lab = spec["label"]
    prior = fit[lab].mean()
    names = list(spec["num"])
    for c in spec["cats"]:
        g = fit.groupby(c, dropna=spec["dropna"])[lab].agg(["sum", "count"])
        rate = (g["sum"] + SMOOTH_M * prior) / (g["count"] + SMOOTH_M)
        for df in [fit] + others:
            df[c + "_rate"] = df[c].map(rate).astype(float).fillna(prior).values
            df[c + "_freq"] = df[c].map(g["count"]).astype(float).fillna(0).values
        names += [c + "_rate", c + "_freq"]
    fill = -999 if spec["cats"] and not spec["dropna"] else None   # IEEE-CIS fills NaN as datasets.py

    def mat(df):
        x = df[names].astype(float)
        return (x.fillna(fill) if fill is not None else x).values

    return mat(fit), [mat(o) for o in others], names


# ------------------------------------------------------------------ what the search sees
@dataclass(frozen=True)
class TrainPeriod:
    name: str
    frame: pd.DataFrame          # training-period rows only
    spec: dict
    row_index: np.ndarray        # positions in the raw frame (for the guard in check C1)

    @property
    def y(self):
        return self.frame[self.spec["label"]].values

    @property
    def t(self):
        return self.frame[self.spec["time"]].values

    def fold_masks(self, fold):
        """Boolean masks (inner-train, validation) inside the training period."""
        if fold == "time":
            cut = np.quantile(self.t, VAL_Q)
            b = self.t > cut
            return ~b, b
        if fold == "random":
            idx = np.arange(len(self.frame))
            _, vb = train_test_split(idx, test_size=0.20, stratify=self.y, random_state=RANDOM_FOLD_SEED)
            b = np.zeros(len(idx), bool)
            b[vb] = True
            return ~b, b
        raise ValueError(fold)

    def inner(self, fold="time"):
        """Xa, ya, Xb, yb with encodings fitted on the inner-training rows only (protocol section 2)."""
        a, b = self.fold_masks(fold)
        fa, fb = self.frame[a], self.frame[b]
        Xa, (Xb,), names = encode(fa, [fb], self.spec)
        lab = self.spec["label"]
        return Xa, fa[lab].values, Xb, fb[lab].values, names

    def describe(self, fold="time"):
        a, b = self.fold_masks(fold)
        return dict(n_train=int(a.sum()), n_val=int(b.sum()), n_features=len(self.spec["num"]) + 2 * len(self.spec["cats"]),
                    positive_rate=float(self.y.mean()), n_val_pos=int(self.y[b].sum()))


def training_period(name):
    d, spec = _load(name)
    tr, _ = _period_masks(d, spec, name)
    pos = np.flatnonzero(tr)
    return TrainPeriod(name, d.iloc[pos].reset_index(drop=True), spec, pos)
