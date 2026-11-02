"""Candidate models. Every model is a full sklearn Pipeline, so imputation and scaling are
fit inside each CV fold and never see the validation rows."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

N_CLASSES = 5


def _preprocess() -> list:
    # add_indicator: whether a player skipped a combine drill is itself informative
    return [SimpleImputer(strategy="median", add_indicator=True), StandardScaler()]


class OrdinalRegressionClassifier(ClassifierMixin, BaseEstimator):
    """Treats tiers 0..4 as an ordinal score: fit a regressor, round to the nearest tier.

    Exploits the ordering that a plain multiclass model ignores (mistaking a Star for a
    Superstar is less wrong than mistaking a Star for a Bust), and pools signal across
    tiers, which matters when the rarest tier has 6 examples.
    """

    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor if self.regressor is not None else Ridge(alpha=10.0))
        self.regressor_.fit(X, y)
        self.classes_ = np.arange(N_CLASSES)
        return self

    def predict(self, X):
        return np.clip(np.rint(self.regressor_.predict(X)), 0, N_CLASSES - 1).astype(int)

    def predict_score(self, X):
        """Continuous tier score, useful for ranking prospects."""
        return self.regressor_.predict(X)


def build_models(seed: int = 0) -> dict[str, Pipeline]:
    return {
        "majority_baseline": make_pipeline(*_preprocess(), DummyClassifier(strategy="most_frequent")),
        "stratified_random": make_pipeline(*_preprocess(), DummyClassifier(strategy="stratified", random_state=seed)),
        "logreg_balanced": make_pipeline(
            *_preprocess(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)
        ),
        "ordinal_ridge": make_pipeline(*_preprocess(), OrdinalRegressionClassifier(Ridge(alpha=30.0))),
        "random_forest": make_pipeline(
            *_preprocess(),
            RandomForestClassifier(
                n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample", random_state=seed, n_jobs=-1
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            *_preprocess(),
            HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, class_weight="balanced",
                random_state=seed,
            ),
        ),
    }  # fmt: skip
