# Reproducibility, data availability, and limitations

## Evidence status (2026-11-10)
- `paper.pdf`: **authored for AT4**, 10-page PDF, with 12 references.
- `results/*.csv`, `logs/*.json`, `figures/*.png`: **existing AT3 real experiment outputs generated 2026-10-28**.
- `src/prepare_data.py`, `src/run_experiment.py`: **AT3 implementation supplied with the user's original archived project**.
- `src/verify_artifacts.py`: **new AT4 checker**: recalculates median country MASE from stored country-level CSV; checks input panel, missingness, sample counts, and artifacts.
- A **fresh model training run has not been asserted** by the AT4 report. To train afresh, follow README section C.

## Simplification relative to publication
1. The original SG-MSF has 6 experts from independent source families; the pilot has only 3 experts based on a single Eurostat workbook.
2. Original source publication dates are not individually replayed; a 2-month fixed lag is assumed.
3. LightGBM L1 instead of Huber.
4. Annual expanding refits instead of monthly.
5. Gate uses 2022 training + 2023 early-stopping, instead of original long-period OOF stacking.
6. No SHAP, source competition diagnostics, conformal intervals, destination clustering, statistical significance tests, or cross-region transfer.

## Why these limitations matter
The gate's 3-month median country MASE of 0.398 is a **descriptive winner within this pilot under this evaluation measure**, not universal or statistically proven superiority. Seasonal naive performs best by WAPE at all three horizons. Original paper's MASE values (0.504/0.566/0.627 for SG-MSF) are not on directly comparable experimental conditions.

## Suggested next experiment
Add one independent, versioned data stream (e.g., published aviation passenger counts or HICP), with documented as-of dates. Refit the same 3-horizon protocol, compare strict ablations, report country level metrics, repeated seeds, WAPE, and predictive accuracy tests.
