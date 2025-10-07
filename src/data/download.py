from pathlib import Path
from src.utils.io import data_path

def expected_raw_files():
    return {
        "ncaa_stats": data_path("raw", "ncaa_stats.csv"),
        "draft_history": data_path("raw", "draft_history.csv"),
        "nba_career": data_path("raw", "nba_career.csv"),
    }

if __name__ == "__main__":
    # Placeholder: manually place CSVs in data/raw/ per expected_raw_files
    for name, p in expected_raw_files().items():
        print(f"[INFO] Expecting raw file: {name} -> {p}")
