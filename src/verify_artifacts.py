"""Offline verification of archived AT3 metrics used in DDS001 AT4 paper.

This audits results without claiming a fresh model training run.
"""
from pathlib import Path
import pandas as pd
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
metrics=pd.read_csv(ROOT/'results/test_metrics.csv')
country=pd.read_csv(ROOT/'results/country_level_metrics.csv')
data=pd.read_csv(ROOT/'data/processed/eurostat_eu27_nonresident_2015_2025.csv')
summary=json.loads((ROOT/'results/run_summary.json').read_text(encoding='utf-8'))
assert data[['geo_code','month']].drop_duplicates().shape[0] == 3564
assert data['geo_code'].nunique()==27
assert data['overnight_stays_total'].isna().sum()==69
assert summary['missing_target_values_2015_2025']==69
assert set(metrics['horizon_months'])=={1,3,6}
assert len(metrics)==21
for row in metrics.itertuples():
    subset=country[(country.model==row.model)&(country.horizon_months==row.horizon_months)]
    assert len(subset)==27,(row.model,row.horizon_months,len(subset))
    assert abs(np.median(subset['MASE']) - row.median_country_MASE) < 1e-10
    assert row.n_test==({1:648,3:648,6:647}[row.horizon_months])
for fn in ['mase_model_comparison.png','gate_weights_by_horizon.png','actual_vs_prediction_selected_countries.png']:
    assert (ROOT/'results'/'figures'/fn).is_file()
assert (ROOT/'paper.pdf').is_file()
print('PASS: 27 countries, 3564 target rows, 69 missing values, 21 horizon-model results,')
print('      median-country MASE recalculated from 27-country CSV; figures and PDF present.')
print('Note: audit of archived results, not a fresh model training run.')
