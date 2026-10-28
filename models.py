"""
Model fitting (protocol section 3): early stopping on the validation fold with the metric that
matches the HPO objective; final refit with n_estimators = round(best_iteration / 0.8).
"""

import numpy as np
from sklearn.metrics import average_precision_score, log_loss

from space import FIXED

MAX_ROUNDS, PATIENCE = 3000, 100
OBJECTIVES = ("pr_auc", "log_loss")


def val_score(objective, y, p):
    """Higher is better for both objectives."""
    if objective == "pr_auc":
        return float(average_precision_score(y, p))
    return -float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))


def fit_early(model, params, Xa, ya, Xb, yb, objective, seed):
    """Fit on inner-train with early stopping on the validation fold.
    Returns validation probabilities and the number of trees used."""
    prm = {**params, **FIXED[model]}
    if model == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(**prm, n_estimators=MAX_ROUNDS, early_stopping_rounds=PATIENCE,
                              tree_method="hist", device="cuda",
                              eval_metric="aucpr" if objective == "pr_auc" else "logloss",
                              random_state=seed, verbosity=0)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        return m.predict_proba(Xb)[:, 1], int(m.best_iteration) + 1
    if model == "lgbm":
        import lightgbm as lgb
        # metric set in the constructor, so early stopping watches this metric only
        m = lgb.LGBMClassifier(**prm, n_estimators=MAX_ROUNDS, random_state=seed, n_jobs=6, verbose=-1,
                               metric="average_precision" if objective == "pr_auc" else "binary_logloss")
        m.fit(Xa, ya, eval_set=[(Xb, yb)],
              callbacks=[lgb.early_stopping(PATIENCE, first_metric_only=True, verbose=False)])
        return m.predict_proba(Xb)[:, 1], int(m.best_iteration_ or MAX_ROUNDS)
    raise ValueError(model)


def refit_predict(model, params, n_trees, Xtr, ytr, Xte, seed):
    prm = {**params, **FIXED[model]}
    if model == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(**prm, n_estimators=n_trees, tree_method="hist", device="cuda",
                              random_state=seed, verbosity=0)
    else:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**prm, n_estimators=n_trees, random_state=seed, n_jobs=6, verbose=-1)
    return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def final_trees(best_iteration):
    return max(1, int(round(best_iteration / 0.8)))
