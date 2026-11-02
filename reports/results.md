# Results

Labeled players: **109** {'Bust': 18, 'Role Player': 31, 'Starter': 34, 'Star': 20, 'Superstar': 6}. CV: 5x repeated stratified 5-fold.

## Cross-validated performance (mean ± std over repeats)

| features         | model                  | qwk            | macro_f1      | balanced_acc   | within_1_tier   |
|:-----------------|:-----------------------|:---------------|:--------------|:---------------|:----------------|
| all_pre_draft    | majority_baseline      | 0.000 ± 0.000  | 0.095 ± 0.000 | 0.200 ± 0.000  | 0.780 ± 0.000   |
| all_pre_draft    | stratified_random      | 0.053 ± 0.103  | 0.210 ± 0.027 | 0.217 ± 0.024  | 0.635 ± 0.016   |
| all_pre_draft    | logreg_balanced        | 0.044 ± 0.087  | 0.219 ± 0.032 | 0.226 ± 0.037  | 0.661 ± 0.018   |
| all_pre_draft    | ordinal_ridge          | 0.133 ± 0.059  | 0.195 ± 0.037 | 0.219 ± 0.040  | 0.761 ± 0.022   |
| all_pre_draft    | random_forest          | 0.121 ± 0.080  | 0.227 ± 0.017 | 0.236 ± 0.014  | 0.734 ± 0.028   |
| all_pre_draft    | hist_gradient_boosting | 0.025 ± 0.080  | 0.210 ± 0.025 | 0.213 ± 0.025  | 0.686 ± 0.040   |
| compact_a_priori | majority_baseline      | 0.000 ± 0.000  | 0.095 ± 0.000 | 0.200 ± 0.000  | 0.780 ± 0.000   |
| compact_a_priori | stratified_random      | 0.053 ± 0.103  | 0.210 ± 0.027 | 0.217 ± 0.024  | 0.635 ± 0.016   |
| compact_a_priori | logreg_balanced        | 0.080 ± 0.051  | 0.209 ± 0.021 | 0.203 ± 0.027  | 0.626 ± 0.039   |
| compact_a_priori | ordinal_ridge          | 0.131 ± 0.014  | 0.180 ± 0.004 | 0.243 ± 0.004  | 0.806 ± 0.010   |
| compact_a_priori | random_forest          | 0.142 ± 0.034  | 0.204 ± 0.022 | 0.213 ± 0.021  | 0.716 ± 0.017   |
| compact_a_priori | hist_gradient_boosting | -0.010 ± 0.038 | 0.185 ± 0.013 | 0.185 ± 0.012  | 0.628 ± 0.022   |

## Best model: `random_forest` on `compact_a_priori`

Label-permutation test (50 shuffles): QWK = 0.090, **p = 0.275**

### Confusion matrix (pooled out-of-fold predictions)

|                  |   pred Bust |   pred Role Player |   pred Starter |   pred Star |   pred Superstar |
|:-----------------|------------:|-------------------:|---------------:|------------:|-----------------:|
| true Bust        |           2 |                 35 |             33 |          16 |                4 |
| true Role Player |          19 |                 63 |             44 |          26 |                3 |
| true Starter     |          15 |                 38 |             61 |          43 |               13 |
| true Star        |           7 |                 19 |             45 |          18 |               11 |
| true Superstar   |           3 |                  4 |             12 |           8 |                3 |

### Permutation feature importance (drop in QWK, top 10)

| feature      |   importance_mean |   importance_std |
|:-------------|------------------:|-----------------:|
| TRB%_college |             0.041 |            0.019 |
| USG%_college |             0.020 |            0.011 |
| STL%_college |             0.019 |            0.013 |
| DBPM_college |             0.019 |            0.012 |
| MP_per_game  |             0.014 |            0.009 |
| AST%_college |             0.013 |            0.008 |
| TS%_college  |             0.012 |            0.006 |
| BLK%_college |             0.011 |            0.012 |
| OBPM_college |             0.003 |            0.010 |
| WS/40        |             0.000 |            0.010 |

### Top 15 prospects by predicted tier score

> Only meaningful if the permutation test above is significant (p = 0.275). Treat this list as a demonstration of the scoring pipeline, not as scouting advice.

| player            | predicted_tier   |   tier_score |
|:------------------|:-----------------|-------------:|
| charles jenkins   | Superstar        |         2.59 |
| norris cole       | Superstar        |         2.34 |
| keith benson      | Star             |         2.25 |
| jarnell stokes    | Star             |         2.23 |
| trae young        | Superstar        |         2.21 |
| reed sheppard     | Starter          |         2.21 |
| taylor hendricks  | Starter          |         2.19 |
| jalen smith       | Star             |         2.19 |
| neemias queta     | Star             |         2.18 |
| landry shamet     | Superstar        |         2.17 |
| mfiondu kabengele | Star             |         2.14 |
| andrew nicholson  | Star             |         2.14 |
| zeke nnaji        | Starter          |         2.14 |
| jevon carter      | Superstar        |         2.13 |
| grant williams    | Star             |         2.13 |
