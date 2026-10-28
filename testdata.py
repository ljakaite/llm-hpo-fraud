"""
Test-period access for the scoring step only (protocol sections 4 and 10, check C1).
Imported by score.py and the checks; never by the search code.
"""

import numpy as np

import data as D


def full_fit_and_test(name):
    """Encodings fitted on the whole training period; returns Xtr, ytr, Xte, yte, names, test row positions."""
    d, spec = D._load(name)
    tr, te = D._period_masks(d, spec, name)
    ftr, fte = d[tr], d[te]
    Xtr, (Xte,), names = D.encode(ftr, [fte], spec)
    lab = spec["label"]
    return Xtr, ftr[lab].values, Xte, fte[lab].values, names, np.flatnonzero(te)
