from src.data.clean_merge import merge_ncaa_to_nba
from src.data.label_outcomes import apply_labels
from src.utils.io import data_path, read_csv, write_csv, load_yaml
from pathlib import Path

if __name__ == "__main__":
    merged = merge_ncaa_to_nba()
    cfg = load_yaml(Path("config/labels.yaml"))
    labeled = apply_labels(merged, cfg)
    write_csv(labeled, data_path("processed", "labeled.csv"))
    print("[OK] Processed dataset at data/processed/labeled.csv")
