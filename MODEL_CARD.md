# Model card: Manufacturing AI failure detection POC

**Intended use:** portfolio demonstration and human review of benchmark telemetry risk. Fleet prioritization illustrates a configurable decision policy.

**Non-intended use:** safety control, automatic repair/intervention, certified maintenance, aerospace deployment, remaining useful life or guaranteed future failure forecasting. This does not replace technicians, reliability engineers or manufacturer procedures.

**Dataset:** official UCI AI4I 2020, DOI 10.24432/C5HS5C, CC BY 4.0, 10,000 synthetic examples. Not plant-validated. Raw data is unchanged and fingerprinted in artifact metadata.

**Target/features:** binary `Machine failure`; type, air/process temperatures, speed, torque and wear. Excluded: UDI, Product ID, TWF, HDF, PWF, OSF, RNF. Feature allowlist is tested. Median/mode imputation, standard scaling and fixed L/M/H one-hot encoding reside in the serialized pipeline.

**Candidates:** prior DummyClassifier, balanced LogisticRegression, balanced RandomForestClassifier, training-weighted XGBClassifier. XGBoost uses a small training-only stratified CV grid optimizing average precision. No pre-split resampling.

**Evaluation:** seed 42, stratified 70/15/15 split; validation AP selects model, recall/F2 break ties. Training-only out-of-fold sigmoid calibration is kept if validation Brier score improves. Validation F2 chooses threshold, precision then larger threshold resolve ties. No refit on validation or test. Held-out probabilities are computed once after selections, then evaluated at 0.5 and the frozen operational threshold. `make evaluate` reads the saved results.

<!-- RESULTS_START -->
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
<!-- RESULTS_END -->

**Explainability:** Tree SHAP for tree models, linear SHAP for logistic regression and zero drivers for a constant baseline. SHAP values describe model attribution, not causal relationships. They explain the uncalibrated base estimator and do not add up to calibrated probability. Type indicators are grouped locally. Units are supplied per attribution.

**Operational policy:** demonstration probability bands at .25/.50/.75. Positive decisions promote low/medium priority to high. Boundaries are policy assumptions, not industry standards. Fleet scores multiply probability by caller-supplied impact weight; this is not economic impact measured from the dataset.

**Limitations:** no temporal/group holdout, live sensors, failure forecast horizon, real maintenance costs or external plant validation. Random splits may share operating regimes. Calibration, thresholds and feature effects can drift. Small search/single holdout have uncertainty; no confidence interval or production savings is claimed. Broad input sanity bounds are not manufacturer operating limits.

**Ethical/operational considerations:** false negatives can hide failures and false positives can consume inspection capacity. Human reviewers must use approved maintenance and safety processes. Groq is the primary live summary provider, with OpenAI as an optional alternative and an offline template default. LLM prose is optional and fallible; it cannot modify structured model decisions. Trusted local joblib artifacts only. API needs access controls and operational validation before real integration.

**Reproducibility:** Python 3.12; `make setup`, `make data`, `make prepare`, `make train ARGS=--quick`, `make evaluate`, `make test`, `make lint`. Seed, config, split counts, source checksum, dependency versions and timestamps are persisted. Quick and standard searches differ; inspect the quick flag. The dependency snapshot in `requirements-lock.txt` records the verified session versions.
