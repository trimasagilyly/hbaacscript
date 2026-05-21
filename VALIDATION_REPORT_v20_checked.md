# Validation report for v20

## v19

- file: submission_v19_LINESEARCH_public_private_g138.csv
- size_mb: 8.317
- shape: 31944 rows x 29 cols
- duplicate_id: 0
- negative_values: 0
- nan_values: 0
- total_forecast_sum: 31008.209480
- nonzero_cells: 132112
- all_zero_rows: 26240

## v20_checked

- file: submission_v20_FINAL_weighted_selective_g_CHECKED.csv
- size_mb: 8.317
- shape: 31944 rows x 29 cols
- duplicate_id: 0
- negative_values: 0
- nan_values: 0
- total_forecast_sum: 31194.668776
- nonzero_cells: 85552
- all_zero_rows: 28274

## ID comparison

- v20 IDs equal v19 IDs: True
- v20 columns equal v19 columns: True
- cells different from v19: 132088
- sum(v20-v19): 186.459296

Reason v20 file may be smaller/larger: v20 prunes more long-tail small forecasts to zero; fixed-format export is provided to remove formatting concerns.