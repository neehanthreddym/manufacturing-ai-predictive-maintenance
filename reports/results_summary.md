Measured quick-mode run.

Validation comparison (each uncalibrated candidate at its validation F2 threshold):

| Model | AP | Recall | Precision | F1 | F2 |
|---|---:|---:|---:|---:|---:|
| dummy | 0.0340 | 1.0000 | 0.0340 | 0.0658 | 0.1496 |
| logistic_regression | 0.3083 | 0.6667 | 0.2048 | 0.3134 | 0.4595 |
| random_forest | 0.7262 | 0.7647 | 0.6290 | 0.6903 | 0.7331 |
| xgboost | 0.7415 | 0.8039 | 0.4713 | 0.5942 | 0.7045 |

Selected **xgboost**, operating threshold **0.088757**. Final held-out test (1500 rows):

| Metric | At 0.5 | Operational |
|---|---:|---:|
| average_precision | 0.8909 | 0.8909 |
| roc_auc | 0.9799 | 0.9799 |
| recall | 0.8627 | 0.9020 |
| precision | 0.7857 | 0.4946 |
| f1 | 0.8224 | 0.6389 |
| f2 | 0.8462 | 0.7744 |
| false_negative_rate | 0.1373 | 0.0980 |
| brier_score | 0.0105 | 0.0105 |

Test false negatives: 7 at 0.5 versus 5 operational; false positives: 12 versus 47.
The validation-selected threshold is not guaranteed to maximize test F2. Inspect both thresholds; do not retune against this test set.

Calibration retained: True. Validation Brier: 0.02493 raw, 0.01721 calibrated.

Split counts (rows/positive/negative): {'train': {'rows': 7000, 'positive': 237, 'negative': 6763}, 'validation': {'rows': 1500, 'positive': 51, 'negative': 1449}, 'test': {'rows': 1500, 'positive': 51, 'negative': 1449}}.

All results are synthetic UCI benchmark measurements, not validated plant performance.
