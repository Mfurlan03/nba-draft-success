"""`python -m nba_draft train` - full reproducible run.

1. Repeated stratified CV for every candidate model on two feature sets
2. Pick the model with the best mean quadratic-weighted kappa (QWK)
3. Label-permutation test for that model: is there any real signal?
4. Fit on all labeled players, compute permutation feature importance
5. Rank the unlabeled prospect pool and write everything to reports/
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score, confusion_matrix, make_scorer

from .data import COMPACT_FEATURES, ROOT, TIERS, load_prospects, load_training_data
from .evaluate import permutation_pvalue, repeated_cv
from .models import build_models

REPORTS = ROOT / "reports"


def _fmt(s: dict, m: str) -> str:
    return f"{s[m + '_mean']:.3f} ± {s[m + '_std']:.3f}"


def run(n_repeats: int, n_permutations: int, out_dir: Path, seed: int = 0) -> dict:
    warnings.filterwarnings("ignore")
    out_dir.mkdir(parents=True, exist_ok=True)
    X_full, y, ids = load_training_data()
    feature_sets = {"all_pre_draft": list(X_full.columns), "compact_a_priori": COMPACT_FEATURES}

    rows, best = [], None
    for fs_name, cols in feature_sets.items():
        for name, model in build_models(seed).items():
            res = repeated_cv(model, X_full[cols], y, n_repeats=n_repeats, seed=seed)
            s = res.summary()
            rows.append({"features": fs_name, "n_features": len(cols), "model": name, **s})
            is_baseline = name in {"majority_baseline", "stratified_random"}
            if not is_baseline and (best is None or s["qwk_mean"] > best["qwk_mean"]):
                best = {"features": fs_name, "model": name, "cols": cols, "oof": res.oof_predictions, **s}
            print(f"{fs_name:18s} {name:24s} QWK {_fmt(s, 'qwk')}  macroF1 {_fmt(s, 'macro_f1')}")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "cv_results.csv", index=False)

    model = build_models(seed)[best["model"]]
    X = X_full[best["cols"]]
    observed, p_value, _ = permutation_pvalue(model, X, y, "qwk", n_permutations, seed)
    print(f"\nBest: {best['model']} on {best['features']} | permutation test QWK={observed:.3f}, p={p_value:.3f}")

    # Confusion matrix pooled over all CV repeats
    oof = best["oof"]
    cm = confusion_matrix(np.tile(y.to_numpy(), oof.shape[0]), oof.ravel(), labels=range(5))
    cm_df = pd.DataFrame(cm, index=[f"true {t}" for t in TIERS], columns=[f"pred {t}" for t in TIERS])

    final = model.fit(X, y)
    qwk = make_scorer(cohen_kappa_score, weights="quadratic")
    imp = permutation_importance(final, X, y, scoring=qwk, n_repeats=20, random_state=seed)
    importance = (
        pd.DataFrame(
            {"feature": X.columns, "importance_mean": imp.importances_mean, "importance_std": imp.importances_std}
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    joblib.dump({"model": final, "features": best["cols"], "tiers": TIERS}, out_dir / "model.joblib")

    Xp, pid = load_prospects(best["cols"])
    pred = final.predict(Xp)
    est = final[-1]
    score = est.predict_score(final[:-1].transform(Xp)) if hasattr(est, "predict_score") else None
    if score is None and hasattr(final, "predict_proba"):
        score = final.predict_proba(Xp) @ np.arange(len(TIERS))  # expected tier
    prospects = pd.DataFrame({"player": pid, "predicted_tier": [TIERS[i] for i in pred], "tier_score": score})
    prospects.sort_values("tier_score", ascending=False).to_csv(out_dir / "prospect_rankings.csv", index=False)

    summary = {
        "n_labeled": int(len(y)),
        "label_counts": {t: int((y == i).sum()) for i, t in enumerate(TIERS)},
        "cv": f"{n_repeats}x repeated stratified 5-fold",
        "best_model": best["model"],
        "best_features": best["features"],
        "best_cv": {k: best[k] for k in best if k.endswith(("_mean", "_std"))},
        "permutation_test": {
            "metric": "qwk",
            "observed": observed,
            "p_value": p_value,
            "n_permutations": n_permutations,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_markdown(out_dir / "results.md", table, summary, cm_df, importance, prospects)
    return summary


def _write_markdown(path, table, summary, cm_df, importance, prospects) -> None:
    t = table.copy()
    for m in ["qwk", "macro_f1", "balanced_acc", "within_1_tier"]:
        t[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(t[m + "_mean"], t[m + "_std"], strict=True)]
    t = t[["features", "model", "qwk", "macro_f1", "balanced_acc", "within_1_tier"]]
    pt = summary["permutation_test"]
    lines = [
        "# Results",
        "",
        f"Labeled players: **{summary['n_labeled']}** {summary['label_counts']}. CV: {summary['cv']}.",
        "",
        "## Cross-validated performance (mean ± std over repeats)",
        "",
        t.to_markdown(index=False),
        "",
        f"## Best model: `{summary['best_model']}` on `{summary['best_features']}`",
        "",
        f"Label-permutation test ({pt['n_permutations']} shuffles): QWK = {pt['observed']:.3f}, **p = {pt['p_value']:.3f}**",
        "",
        "### Confusion matrix (pooled out-of-fold predictions)",
        "",
        cm_df.to_markdown(),
        "",
        "### Permutation feature importance (drop in QWK, top 10)",
        "",
        importance.head(10).to_markdown(index=False, floatfmt=".3f"),
        "",
        "### Top 15 prospects by predicted tier score",
        "",
        f"> Only meaningful if the permutation test above is significant (p = {pt['p_value']:.3f}). "
        "Treat this list as a demonstration of the scoring pipeline, not as scouting advice.",
        "",
        prospects.sort_values("tier_score", ascending=False).head(15).to_markdown(index=False, floatfmt=".2f"),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(prog="nba_draft")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="CV all models, permutation-test the best, fit, rank prospects")
    t.add_argument("--repeats", type=int, default=10)
    t.add_argument("--permutations", type=int, default=200)
    t.add_argument("--out", type=Path, default=REPORTS)
    t.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.cmd == "train":
        run(a.repeats, a.permutations, a.out, a.seed)


if __name__ == "__main__":
    main()
