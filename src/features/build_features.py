import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.utils.io import data_path, read_csv, write_csv, load_yaml
from pathlib import Path


def build_features(df: pd.DataFrame, cfg: dict):
    X = df[cfg["numeric_features"]].copy()
    y = df[cfg["target_col"]].copy()

    imputer = SimpleImputer(strategy=cfg.get("impute_strategy", "median"))
    X_imp = imputer.fit_transform(X)

    if cfg.get("standardize", True):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imp)
    else:
        X_scaled = X_imp

    X_df = pd.DataFrame(X_scaled, columns=cfg["numeric_features"], index=df.index)
    return X_df, y, imputer, (scaler if cfg.get("standardize", True) else None)


if __name__ == "__main__":
    df = read_csv(data_path("processed", "labeled.csv"))
    cfg = load_yaml(Path("config/features.yaml"))
    X, y, imputer, scaler = build_features(df, cfg)
    out = X.join(y)
    write_csv(out, data_path("processed", "features.csv"))
