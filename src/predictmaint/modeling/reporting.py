"""Publish measured portfolio summaries; never infer unmeasured business impact."""

import json
from pathlib import Path


def publish_results(metadata: dict) -> None:
    comparison = json.loads(Path("reports/model_comparison.json").read_text())
    lines = [
        "Measured quick-mode run." if metadata["quick"] else "Measured standard-mode run.",
        "",
        "Validation comparison (each uncalibrated candidate at its validation F2 threshold):",
        "",
        "| Model | AP | Recall | Precision | F1 | F2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['model']} | {row['average_precision']:.4f} | {row['recall']:.4f} | "
            f"{row['precision']:.4f} | {row['f1']:.4f} | {row['f2']:.4f} |"
        )
    result = metadata["final_test_metrics"]
    lines += [
        "",
        f"Selected **{metadata['model_name']}**, operating threshold "
        f"**{metadata['decision_threshold']:.6f}**. Final held-out test "
        f"({metadata['splits']['test']['rows']} rows):",
        "",
        "| Metric | At 0.5 | Operational |",
        "|---|---:|---:|",
    ]
    for key in [
        "average_precision",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "f2",
        "false_negative_rate",
        "brier_score",
    ]:
        lines.append(f"| {key} | {result['at_0_5'][key]:.4f} | {result['operational'][key]:.4f} |")
    default = result["at_0_5"]
    operational = result["operational"]
    cal = metadata["calibration"]
    lines += [
        "",
        f"Test false negatives: {default['confusion_matrix'][1][0]} at 0.5 versus "
        f"{operational['confusion_matrix'][1][0]} operational; false positives: "
        f"{default['confusion_matrix'][0][1]} versus {operational['confusion_matrix'][0][1]}.",
        "The validation-selected threshold is not guaranteed to maximize test F2. "
        "Inspect both thresholds; do not retune against this test set.",
        "",
        f"Calibration retained: {cal['used']}. Validation Brier: "
        f"{cal['validation_brier_raw']:.5f} raw, "
        f"{cal['validation_brier_calibrated']:.5f} calibrated.",
        "",
        f"Split counts (rows/positive/negative): {metadata['splits']}.",
        "",
        "All results are synthetic UCI benchmark measurements, not validated plant performance.",
    ]
    summary = "\n".join(lines)
    Path("reports/results_summary.md").write_text(summary + "\n")
    for name in ["README.md", "MODEL_CARD.md"]:
        path = Path(name)
        if not path.exists():
            continue
        source = path.read_text()
        start = "<!-- RESULTS_START -->"
        end = "<!-- RESULTS_END -->"
        if start in source and end in source:
            before = source.split(start)[0]
            after = source.split(end)[1]
            path.write_text(before + start + "\n" + summary + "\n" + end + after)
