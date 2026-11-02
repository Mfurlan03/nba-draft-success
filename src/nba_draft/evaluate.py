"""Evaluation for a small, imbalanced, ordinal problem.

With ~100 labeled players a single train/test split is mostly noise, so every model is
scored with repeated stratified K-fold CV and reported as mean ± std across repeats.
A label-permutation test checks whether the best model beats chance at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold


def within_one_tier(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


METRICS = {
    "macro_f1": lambda t, p: f1_score(t, p, average="macro", labels=range(5), zero_division=0),
    "balanced_acc": balanced_accuracy_score,
    "qwk": lambda t, p: cohen_kappa_score(t, p, weights="quadratic"),
    "within_1_tier": within_one_tier,
}


@dataclass
class CVResult:
    per_repeat: pd.DataFrame  # one row per repeat, pooled out-of-fold predictions
    oof_predictions: np.ndarray  # (n_repeats, n_samples)

    def summary(self) -> dict[str, float]:
        out = {}
        for m in METRICS:
            out[f"{m}_mean"] = float(self.per_repeat[m].mean())
            out[f"{m}_std"] = float(self.per_repeat[m].std(ddof=1)) if len(self.per_repeat) > 1 else 0.0
        return out


def repeated_cv(
    model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, n_repeats: int = 10, seed: int = 0
) -> CVResult:
    """Out-of-fold predictions for each repeat; metrics computed on pooled OOF predictions.

    Pooling per repeat (rather than averaging per-fold scores) keeps macro-F1 meaningful
    when a fold holds only one Superstar.
    """
    y = np.asarray(y)
    rows, oofs = [], []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        oof = np.empty_like(y)
        for train_idx, test_idx in skf.split(X, y):
            m = clone(model).fit(X.iloc[train_idx], y[train_idx])
            oof[test_idx] = m.predict(X.iloc[test_idx])
        rows.append({name: fn(y, oof) for name, fn in METRICS.items()})
        oofs.append(oof)
    return CVResult(pd.DataFrame(rows), np.vstack(oofs))


def permutation_pvalue(
    model, X: pd.DataFrame, y: pd.Series, metric: str = "qwk", n_permutations: int = 100, seed: int = 0
) -> tuple[float, float, np.ndarray]:
    """P-value that the model's CV score could arise with labels shuffled (no real signal)."""
    rng = np.random.default_rng(seed)
    observed = repeated_cv(model, X, y, n_repeats=1, seed=seed).per_repeat[metric].iloc[0]
    null = np.empty(n_permutations)
    y_arr = np.asarray(y)
    for i in range(n_permutations):
        y_perm = pd.Series(rng.permutation(y_arr))
        null[i] = repeated_cv(model, X, y_perm, n_repeats=1, seed=seed).per_repeat[metric].iloc[0]
    p = (1 + np.sum(null >= observed)) / (1 + n_permutations)
    return float(observed), float(p), null
