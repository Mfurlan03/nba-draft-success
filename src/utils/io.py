from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

def data_path(*parts) -> Path:
    return ROOT / "data" / Path(*parts)

def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def write_csv(df: pd.DataFrame, path: Path, index=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
