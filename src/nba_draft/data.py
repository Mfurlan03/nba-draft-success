"""Data loading and the pre-draft feature contract.

The labeled table mixes pre-draft inputs (college box score / advanced stats, combine
measurements) with post-draft NBA career stats, which were used to *derive* the career-tier
label. Any NBA column used as a feature would leak the answer, so the feature set is an
explicit allow-list, and `assert_no_leakage` guards it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "final_master"
TRAIN_CSV = DATA_DIR / "final_training_dataset.csv"
PROSPECTS_CSV = DATA_DIR / "college_combine_new_players.csv"

ID_COL = "join_key"
TARGET = "Label"

# Career tiers are ordinal: order matters for "within one tier" accuracy and kappa.
TIERS = ["Bust", "Role Player", "Starter", "Star", "Superstar"]

COMBINE_FEATURES = [
    "HGT", "WGT", "BMI", "BF", "WNGSPN", "STNDRCH", "HANDL", "HANDW", "STNDVERT",
    "LPVERT", "LANE", "SHUTTLE", "SPRINT", "BENCH", "PAN", "PBHGT", "PDHGT",
]  # fmt: skip

# Columns that are entirely or almost entirely empty in the source data.
DROPPED_EMPTY = ["POS_college", "BAR"]


def _is_post_draft(col: str) -> bool:
    nba_markers = ("_nba",)
    nba_exact = {"WS/48", "VORP", "GP", "GS", "MP", "Player", TARGET}
    return col.endswith(nba_markers) or col in nba_exact


def pre_draft_feature_columns(df: pd.DataFrame) -> list[str]:
    """Allow-list: per-game usage + every college stat + combine measurements."""
    college = [c for c in df.columns if c.endswith("_college") or c in {"PProd", "WS/40"}]
    usage = ["G_per_game", "GS_per_game", "MP_per_game"]
    cols = [c for c in usage + college + COMBINE_FEATURES if c in df.columns and c not in DROPPED_EMPTY]
    assert_no_leakage(cols)
    return cols


def assert_no_leakage(cols: list[str]) -> None:
    leaked = [c for c in cols if _is_post_draft(c)]
    if leaked:
        raise ValueError(f"post-draft columns in feature set: {leaked}")


def load_training_data(path: Path = TRAIN_CSV) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, ids). y is the ordinal tier encoded 0..4."""
    df = pd.read_csv(path)
    unknown = set(df[TARGET].dropna()) - set(TIERS)
    if unknown:
        raise ValueError(f"unexpected labels: {unknown}")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    X = df[pre_draft_feature_columns(df)].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].map({t: i for i, t in enumerate(TIERS)}).astype(int)
    return X, y, df[ID_COL]


def load_prospects(feature_cols: list[str], path: Path = PROSPECTS_CSV) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    # The prospect export names college stats without the "_college" suffix used in training.
    df = df.rename(columns={c[: -len("_college")]: c for c in feature_cols if c.endswith("_college")})
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"prospect file is missing feature columns: {missing}")
    return df[feature_cols].apply(pd.to_numeric, errors="coerce"), df[ID_COL]


# A compact feature set chosen *a priori* from draft-analytics practice (box plus-minus,
# win shares per minute, efficiency, rate stats, size/length), not by searching this data.
# With ~100 labeled players, fewer features means less variance.
COMPACT_FEATURES = [
    "BPM_college", "OBPM_college", "DBPM_college", "WS/40", "PER_college", "TS%_college",
    "USG%_college", "AST%_college", "STL%_college", "BLK%_college", "TRB%_college",
    "MP_per_game", "HGT", "WNGSPN",
]  # fmt: skip
