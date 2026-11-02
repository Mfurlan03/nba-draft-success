# Predicting NBA Career Tiers from Pre-Draft Data

[![CI](https://github.com/Mfurlan03/nba-draft-success/actions/workflows/ci.yml/badge.svg)](https://github.com/Mfurlan03/nba-draft-success/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Can college performance and NBA Draft Combine measurements, using **only information
available before the draft**, predict whether a player becomes a Bust, Role Player,
Starter, Star or Superstar?

This repo contains the data collection and cleaning work (notebooks) and a reproducible
modeling pipeline (`src/nba_draft`) built for a **small, imbalanced, ordinal** problem.
That means repeated cross-validation, ordinal metrics, an explicit leakage guard and a
label-permutation test, so the results say what the data actually supports.

**Headline finding:** on the 109 players with complete, matched pre-draft and NBA data, no
model beats chance with statistical significance. The best model (random forest on 14
pre-chosen features) reaches a quadratic-weighted kappa of 0.14 ± 0.03, but a label-permutation
test gives **p = 0.27**. The pipeline is ready to measure signal properly once more labeled
players are matched.

---

## Data

| Source | Contents |
|---|---|
| College Basketball Reference | NCAA box score and advanced metrics (PER, BPM, WS/40, usage, rate stats) |
| Basketball Reference | NBA career statistics, used **only to derive labels** |
| NBA Draft Combine (Kaggle) | Height, wingspan, reach, hand size, vertical, agility, sprint, bench |
| Custom player-ID mapping | Resolves naming inconsistencies across sources |

Scraping and cleaning live in `stat_scraper/`, `data_cleaning/` and `eda_analysis/`.
Career tiers are assigned in `data_cleaning/Label_NBA.ipynb` from position-normalized
z-scores of NBA impact metrics.

| | Players |
|---|---:|
| Raw college player rows | 2,559 |
| Labeled NBA players | 332 |
| **Labeled, with matched college + combine data (training set)** | **109** |
| Unlabeled prospects with the same features (`college_combine_new_players.csv`) | 408 |

Label balance in the training set: Starter 34 · Role Player 31 · Star 20 · Bust 18 · Superstar 6.

## Method

```bash
pip install -e ".[dev]"
python -m nba_draft train --repeats 5 --permutations 50     # writes reports/
pytest -q
```

1. **Leakage guard.** The training table also holds the post-draft NBA stats the labels were
   computed from. Features come from an explicit allow-list (college + combine + usage), and
   `assert_no_leakage` raises if any `*_nba`, VORP, WS/48, GP/GS/MP or label column gets
   through. Tests enforce this.
2. **Preprocessing inside the CV fold.** Median imputation (with missing-indicator columns,
   since skipping a combine drill is informative) and scaling are pipeline steps, refit on
   each training fold.
3. **Repeated stratified 5-fold CV.** With 109 players a single split is mostly noise, so
   every model is scored on pooled out-of-fold predictions for each repeat and reported as
   mean ± std across repeats.
4. **Ordinal-aware metrics.** Tiers are ordered, so the primary metric is **quadratic-weighted
   kappa** (calling a Superstar a Star costs less than calling them a Bust). Macro-F1,
   balanced accuracy and within-one-tier accuracy are reported alongside.
5. **Baselines first.** Majority-class and stratified-random baselines sit in the same table.
6. **Two feature sets:** all 62 pre-draft features, and 14 features **chosen before looking
   at results** from draft-analytics practice (BPM, WS/40, PER, TS%, usage and rate stats,
   minutes, height, wingspan).
7. **Label-permutation test** on the best model: shuffle the labels, rerun CV, and see how
   often chance does as well.

Models: logistic regression (class-balanced), an ordinal ridge regressor rounded to tiers,
random forest and histogram gradient boosting. Hyperparameters were set a priori, not tuned,
to avoid overfitting model selection on 109 rows.

## Results

5× repeated stratified 5-fold CV (full table: [`reports/results.md`](reports/results.md)):

| Features | Model | QWK | Macro-F1 | Within-1-tier |
|---|---|---|---|---|
| — | majority class | 0.000 | 0.095 | 0.780 |
| — | stratified random | 0.053 ± 0.103 | 0.210 ± 0.027 | 0.635 ± 0.016 |
| compact (14) | **random forest** | **0.142 ± 0.034** | 0.204 ± 0.022 | 0.716 ± 0.017 |
| all (62) | ordinal ridge | 0.133 ± 0.059 | 0.195 ± 0.037 | 0.761 ± 0.022 |
| compact (14) | ordinal ridge | 0.131 ± 0.014 | 0.180 ± 0.004 | **0.806 ± 0.010** |
| all (62) | random forest | 0.121 ± 0.080 | **0.227 ± 0.017** | 0.734 ± 0.028 |
| all (62) | gradient boosting | 0.025 ± 0.080 | 0.210 ± 0.025 | 0.686 ± 0.040 |

**Permutation test** (best model, 50 label shuffles): QWK 0.090 on the test repeat, **p = 0.275**.

### Interpretation

- The models are consistently a little better than random on the ordinal metric. That's
  consistent with weak signal, but the permutation test can't rule out chance at this sample size.
- "Within one tier" looks high (72–81%) but is mostly structural: the majority-class
  predictor already scores 78%, because the middle tiers are adjacent to almost everything.
- Gradient boosting performs worst. Flexible models overfit badly at n = 109.
- Permutation importance ranks rebounding rate, usage, steal rate and defensive BPM highest.
  With no significant overall signal, treat that as a hypothesis, not a finding.

### What would make this work

The bottleneck is the **109 / 332 match rate** between labeled NBA players and complete
college + combine records. Fuzzy-matching names through `player_ids.csv`, handling players
who skipped the combine, and adding draft age and conference strength would grow the
labeled set. That's the highest-value next step, ahead of any model change.

> **Note on an earlier version of this README:** it described a two-stage
> LightGBM → CatBoost model with test Macro-F1 ≈ 0.45 on ~1,200 prospects. Neither that
> code nor a dataset of that size is in this repository, so those figures were replaced with
> results the repo reproduces.

## Repository layout

```
src/nba_draft/
  data.py       # loading, pre-draft feature allow-list, leakage guard
  models.py     # candidate pipelines incl. ordinal regression classifier
  evaluate.py   # repeated CV, ordinal metrics, permutation test
  cli.py        # `python -m nba_draft train`
reports/        # generated: results.md, cv_results.csv, feature_importance.csv, prospect_rankings.csv
stat_scraper/   # scraping notebooks + raw CSVs
data_cleaning/  # integration, labeling notebooks
eda_analysis/   # exploratory analysis
final_master/   # model-ready tables
tests/
```

## Authors

- Michael Furlano
- Mikhael Saikaly
- Hryhorii Ovcharenko
