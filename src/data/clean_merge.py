import pandas as pd
from src.utils.io import data_path, read_csv, write_csv


def normalize_names(s: pd.Series) -> pd.Series:
    return (s.str.lower()
             .str.normalize("NFKD")
             .str.replace(r"[^a-z\s]", "", regex=True)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())


def merge_ncaa_to_nba():
    ncaa = read_csv(data_path("raw", "ncaa_stats.csv"))
    draft = read_csv(data_path("raw", "draft_history.csv"))
    nba = read_csv(data_path("raw", "nba_career.csv"))

    for df in (ncaa, draft, nba):
        if "player" in df.columns:
            df["player_norm"] = normalize_names(df["player"])
        if "draft_year" not in df.columns and "year" in df.columns:
            df["draft_year"] = df["year"]

    merged = (draft
              .merge(ncaa, on=["player_norm", "draft_year"], how="left", suffixes=("", "_ncaa"))
              .merge(nba, on=["player_norm"], how="left", suffixes=("", "_nba")))

    write_csv(merged, data_path("interim", "merged.csv"))
    return merged


if __name__ == "__main__":
    merge_ncaa_to_nba()
