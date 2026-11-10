# Simplified reproduction report: source-gated tourism demand forecasting

## What was reproduced
This project implements a simplified version of the paper's expert-model plus learned-context-gate idea, and compares it with early fusion, equal-weight averaging, and seasonal-naive forecasting.

## Data
- Source: previously downloaded Eurostat `tour_occ_nim` workbook (extracted 2026-10-23). No new download was performed.
- Target: EU27 monthly non-resident overnight stays (`FOR`, `NR`, `I551-I553`), January 2015–December 2025.
- Auxiliary feature families: lagged hotel-only nights (`I551`) and other-accommodation nights (`I552_I553`). These are subseries of the same Eurostat table, not independent external data sources.
- 3,564 theoretical target observations; 69 target values are missing in 2015–2025. Missing values are not zero-filled. Rows with missing target labels are excluded from fitting/evaluation. This differs from the paper's target-gap interpolation.
- For each feature row, source values are truncated at the publication cutoff before interpolation/feature construction. Values after that row's cutoff are never used to construct features.

## Model and split
- Forecast horizons: H=1, 3, 6 months.
- Models are refit once per forecast year and horizon, using training labels only up to January of that forecast year minus the horizon and two-month publication lag. This annual expanding refit is a simplification of the paper's monthly updates.
- Gate fitting period: 2022; early stopping/validation period: 2023.
- Final test period: 2024–2025 (648 country-month target observations per horizon; the common comparison set is 647 for H=6 because one seasonal-naive lag is unavailable at the cutoff).
- Three LightGBM feature-family experts: history-only; history + hotel subseries; history + other-accommodation subseries. A small 32/16-unit MLP gate outputs softmax weights.
- LightGBM uses L1 loss in this implementation rather than the paper's Huber objective, to keep the simplified model numerically stable.
- Baselines: seasonal naive, early-fusion LightGBM, and equal-weight expert ensemble.

## Results: median country-level MASE (lower is better)
| Model | H=1 | H=3 | H=6 |
|---|---:|---:|---:|
| Seasonal naive | 0.433 | 0.433 | 0.433 |
| Equal-weight experts | 0.396 | 0.412 | 0.624 |
| Early-fusion LightGBM | 0.406 | 0.443 | 0.514 |
| Context-gated fusion | 0.439 | 0.398 | 0.658 |

- H=1: best median country MASE in this run was **Equal-weight experts** (0.396).
- H=3: best median country MASE in this run was **Context-gated fusion** (0.398).
- H=6: best median country MASE in this run was **Seasonal naive** (0.433).

The gate does not win at every horizon. This is a useful result, not a reason to hide the comparison: with only lagged subseries from one Eurostat table, the learned gate has limited new source information to exploit. See `results/test_metrics.csv` for MAE, RMSE, WAPE, and mean country MASE.

The original paper reports SG-MSF median MASE 0.504/0.566/0.627 for H=1/3/6 and early-fusion LightGBM 0.563/0.539/0.568. Those values are stored separately in `results/paper_reported_reference_metrics.csv`; they are not directly comparable to this simplified run because the feature sources, missing-value treatment, loss, and refit schedule differ.

## Limitations
1. This is not a full replication of the paper's six independent source families. We use only the existing Eurostat workbook and two lagged accommodation subseries as auxiliary feature families.
2. The gate is trained on 2022 expert forecasts and early-stopped on 2023; the paper uses expanding-window out-of-fold expert forecasts from 2017–2021 and a more detailed gate input.
3. We do not reproduce Google Trends, Eurostat aviation, Eurocontrol, HICP, accommodation capacity, ERA5-Land, conformal intervals, destination clusters, or SHAP-based competitiveness diagnostics.
4. Therefore, describe this as a simplified architecture reproduction/pilot, not an exact replication of the paper's reported results.

## Next step toward true multi-source replication
Add at least one independent external source (e.g. monthly air-passenger data or HICP) with documented publication lags, then repeat the comparison with a separate source expert. Add Google Trends/climate only if time and data access permit.
