import numpy as np
import pandas as pd
import pytest

from nba_draft.data import (
    COMPACT_FEATURES,
    TIERS,
    assert_no_leakage,
    load_prospects,
    load_training_data,
    pre_draft_feature_columns,
)
from nba_draft.evaluate import permutation_pvalue, repeated_cv, within_one_tier
from nba_draft.models import OrdinalRegressionClassifier, build_models


@pytest.fixture(scope="module")
def data():
    return load_training_data()


def test_no_post_draft_columns_in_features(data):
    X, _, _ = data
    for col in X.columns:
        assert not col.endswith("_nba")
        assert col not in {"VORP", "WS/48", "GP", "GS", "MP", "Label", "Player"}


def test_leakage_guard_raises():
    with pytest.raises(ValueError, match="post-draft"):
        assert_no_leakage(["PER_college", "PER_nba"])
    with pytest.raises(ValueError):
        assert_no_leakage(["VORP"])


def test_feature_allow_list_ignores_unknown_columns():
    df = pd.DataFrame(columns=["PER_college", "PER_nba", "mystery", "HGT", "Label"])
    assert pre_draft_feature_columns(df) == ["PER_college", "HGT"]


def test_labels_are_ordinal_codes(data):
    _, y, ids = data
    assert set(y.unique()) <= set(range(len(TIERS)))
    assert len(y) == len(ids) == 109


def test_compact_features_exist(data):
    X, _, _ = data
    assert set(COMPACT_FEATURES) <= set(X.columns)


def test_prospects_align_with_training_features(data):
    X, _, _ = data
    Xp, ids = load_prospects(list(X.columns))
    assert list(Xp.columns) == list(X.columns)
    assert len(Xp) == len(ids) > 0


def test_within_one_tier():
    assert within_one_tier([0, 1, 2, 4], [1, 1, 4, 3]) == 0.75


def test_ordinal_classifier_clips_to_valid_tiers():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.clip(np.arange(20) // 4, 0, 4)
    clf = OrdinalRegressionClassifier().fit(X, y)
    pred = clf.predict(np.array([[-100.0], [100.0]]))
    assert pred.min() >= 0 and pred.max() <= 4


def test_every_model_runs_in_cv(data):
    X, y, _ = data
    for name, model in build_models().items():
        res = repeated_cv(model, X[COMPACT_FEATURES], y, n_splits=3, n_repeats=1)
        assert res.oof_predictions.shape == (1, len(y)), name
        assert -1 <= res.summary()["qwk_mean"] <= 1


def test_cv_preprocessing_is_fit_inside_folds(data):
    """Imputer/scaler live inside the Pipeline, so CV must not need pre-imputed data."""
    X, y, _ = data
    assert X.isna().any().any()  # raw data has gaps
    repeated_cv(build_models()["logreg_balanced"], X, y, n_splits=3, n_repeats=1)


def test_permutation_test_detects_strong_signal():
    rng = np.random.default_rng(0)
    y = pd.Series(np.repeat(np.arange(5), 20))
    X = pd.DataFrame({"signal": y + rng.normal(0, 0.3, len(y)), "noise": rng.normal(size=len(y))})
    obs, p, _ = permutation_pvalue(build_models()["ordinal_ridge"], X, y, n_permutations=19)
    assert obs > 0.8 and p <= 0.05
