import pandas as pd
from src.utils.io import data_path, read_csv, write_csv, load_yaml
from pathlib import Path


def apply_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    def tier_for_row(r):
        # Extract commonly used metrics; default to 0 if missing
        ws = r.get("win_shares", 0) or 0
        bpm = r.get("bpm", 0) or 0
        g = r.get("games_played", 0) or 0
        mpg = r.get("minutes_per_game", 0) or 0
        all_star = r.get("all_star_selections", 0) or 0
        all_nba = r.get("all_nba_teams", 0) or 0

        # Evaluate tiers in priority order
        for tier in cfg["priority_order"]:
            rules = cfg["tiers"].get(tier, {})
            if rules.get("fallback"):
                return tier
            any_of = rules.get("any_of", [])
            all_of = rules.get("all_of", [])

            checks = {
                "win_shares": ws,
                "bpm": bpm,
                "games_played": g,
                "minutes_per_game": mpg,
                "all_star_selections": all_star,
                "all_nba_teams": all_nba,
            }

            def cond_pass(cond):
                for k, expr in cond.items():
                    val = checks.get(k, 0)
                    op, thr = expr.split()
                    thr = float(thr)
                    if op == ">=" and not (val >= thr):
                        return False
                    if op == ">" and not (val > thr):
                        return False
                    if op == "<=" and not (val <= thr):
                        return False
                    if op == "<" and not (val < thr):
                        return False
                    if op == "==" and not (val == thr):
                        return False
                    if op == "!=" and not (val != thr):
                        return False
                return True

            any_ok = any(cond_pass(c) for c in any_of) if any_of else False
            all_ok = all(cond_pass(c) for c in all_of) if all_of else False
            if (any_of and any_ok) or (all_of and all_ok) or (not any_of and not all_of):
                return tier
        return "Bust"

    df["outcome_tier"] = df.apply(tier_for_row, axis=1)
    return df


if __name__ == "__main__":
    merged = read_csv(data_path("interim", "merged.csv"))
    cfg = load_yaml(Path("config/labels.yaml"))
    labeled = apply_labels(merged, cfg)
    write_csv(labeled, data_path("processed", "labeled.csv"))
