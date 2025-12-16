# Predicting NBA Draft Outcomes Using Pre-Draft Data

## Overview
NBA draft decisions involve significant financial and strategic risk, yet projecting long-term player success remains highly uncertain. This project investigates whether long-term NBA career outcomes can be predicted using only information available before the draft, such as college performance, contextual factors, and NBA Draft Combine measurements.

Using a two-stage machine learning pipeline, we model early-career NBA performance and then classify players into qualitative career tiers:
- Bust
- Role Player
- Starter
- Star
- Superstar

The goal is not to replace traditional scouting, but to determine whether historical pre-draft data contains consistent, objective signals that can support better draft decision-making.

## Data Sources
This project integrates multiple publicly available basketball datasets:
- College Basketball Reference – NCAA box score and advanced efficiency metrics
- Basketball Reference – NBA career statistics and advanced impact metrics
- NBA Draft Combine (Kaggle) – Physical and athletic measurements
- Custom Player ID Mapping – Resolves naming inconsistencies across sources

The final dataset contains ~1,200 prospects from 2000–2025, with 100+ pre-draft features.

Data Cleaning & Integration (`data_cleaning`)

This stage ensures data quality, consistency, and realism:

- Standardized player identities across NCAA, NBA, and Combine sources
- Removed post-draft information to prevent data leakage
- Imputed missing Combine measurements using height/weight/wingspan-grouped medians
- Filtered players to those with sufficient pre-draft data
- Constructed a unified pre-draft feature table

This step produces a clean, model-ready dataset that reflects real draft-time constraints

Exploratory Data Analysis (`eda_analysis`)

EDA is used strictly for validation and insight, not feature leakage:
- Missingness analysis of NBA Draft Combine data
- Distribution of career outcome labels
- Correlation analysis of efficiency metrics (PER, BPM, WS/48, etc.)
- Sanity checks on era effects, positional context, and sample balance

Key findings:
- Efficiency metrics are more predictive than raw box-score totals
- Career outcome labels are highly imbalanced, with Superstars <3% of players
- Most misclassifications occur between adjacent career tiers

Modeling Pipeline (`final_master`)
# Two-Stage Approach

## Stage 1 — Regression
Predicts early-career NBA performance using pre-draft features:
- PER
- BPM
- Win Shares
- VORP
## Models:
Ridge Regression (baseline)
- LightGBM Regressor (nonlinear interactions)
Out-of-fold predictions are passed to Stage 2 to reduce noise.

## Stage 2 — Classification
Classifies players into long-term career tiers using:
- Original pre-draft features
- Predicted early-career metrics from Stage 1
### Model: 
- CatBoost Classifier 
  -  Handles categorical features
  -  Robust to class imbalance
  -  Optimized for Macro-F1

Validation uses grouped splits to ensure no player appears in multiple partitions.

## Model Performance (Summary)

- Macro-F1 (Test): ~0.45
- Strong performance identifying Busts, Role Players, and Starters
- Expected difficulty separating Stars vs. Superstars due to rarity and overlapping profiles

These results demonstrate that pre-draft data contains meaningful predictive structure, while also highlighting the inherent uncertainty of elite player development.

## Key Takeaways

- Pre-draft data alone can meaningfully distinguish broad NBA career trajectories
- Predicting elite outcomes remains challenging due to rarity and nonlinear development
- A structured two-stage model mirrors real front-office evaluation workflows
- Analytics can complement, not replace, traditional scouting

##Authors
- Michael Furlano
- Mikhael Saikaly
- Hryhorii Ovcharenko
