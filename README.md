# Predicting NBA Draft Success with ML

This project classifies NBA draft prospects into qualitative career outcome tiers (Superstar, Star, Starter, Role Player, Bust) using college performance metrics, physical/athletic attributes, and draft metadata. We compare interpretable baselines (e.g., Random Forest) with a neural network to assess gains from nonlinear modeling.

## Data sources
- Sports-Reference NCAA: college stats and advanced metrics
- Basketball-Reference: draft history and NBA career outcomes (WS, BPM, VORP)
- Stathead: bulk downloads and linking NCAA to NBA careers

Note: Follow each site's terms of use. Store raw files under `data/raw/` (not versioned).

## Workflow
1) Data ingest: download/collect raw NCAA, draft, and NBA career data
2) Cleaning & linking: unify player IDs; merge college→NBA records
3) Labeling: apply 8-season outcome thresholds to assign tiers
4) Features: build standardized feature set from college stats + attributes
5) Modeling: baseline models vs. neural network
6) Evaluation: accuracy, macro-F1, per-class metrics, confusion matrix
7) Analysis: feature importance, partial dependence, calibration

## Repo layout
See directory tree in the root of this README.

## Reproducibility
- Python 3.11+ recommended
- `pip install -r requirements.txt`
- Scripts:
  - `python scripts/make_dataset.py`
  - `python scripts/train.py`
  - `python scripts/evaluate.py`

## Team
- Michael Furlano 
- Mikhael Saikaly 
- Hryhorii Ovcharenko 
