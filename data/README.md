# Data provenance and acquisition
Use the official [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i), DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C), published 2020. It is a synthetic benchmark reflecting industrial telemetry, not measurements from a deployed plant. Attribution: AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); project source licensing does not replace the dataset license.

Run `make data` or `python scripts/download_data.py` with the installed package/PYTHONPATH. The downloader extracts `ai4i2020.csv` from the official UCI ZIP into `data/raw/ai4i2020.csv`, preserves original bytes and skips existing files unless `--force` is given. Training metadata records its SHA-256.

If network access fails, open the linked UCI page, download the ZIP and extract the CSV to that path. Change `data.raw_path` in `configs/model.yaml` or pass `--config` to use another filename. No fake dataset is substituted.

`make prepare` produces `data/processed/telemetry.csv` with six normalized feature columns and `target`, plus `reports/data_validation.json`. Imputation/encoding/scaling are fitted only after splitting, inside model pipelines. IDs and five outcome columns are excluded through an explicit allowlist.

Raw/processed data and binary model artifacts are Git-ignored for a small reproducible repository. `tests/fixtures/sample_ai4i.csv` is explicitly synthetic test-only data. It is never an alternate training source and its metrics must not be used in project results.
