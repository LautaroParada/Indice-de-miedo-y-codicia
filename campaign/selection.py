"""Separar la elección descriptiva de una ablación y su comprobación posterior."""

import json
from pathlib import Path

import pandas as pd

from .engine import save_json


def selection_audit(predictions_path, output, protocol_path=Path("campaign/selection-check.json")):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    data = pd.read_csv(predictions_path, parse_dates=["quarter"])
    results = {}
    for family in protocol["optional_families"]:
        periods = {}
        for phase in ["selection", "evaluation"]:
            selected = data.loc[
                data.quarter.between(protocol[phase + "_start"], protocol[phase + "_end"])
            ]
            metrics = {}
            for target, frame in selected.groupby("target"):
                errors = frame.pivot(index="quarter", columns="model", values="error").abs()
                parent = errors.panel6.mean()
                child = errors["panel6_without_" + family].mean()
                metrics[target] = {"n": len(errors), "relative_mae_loss": float(child / parent - 1)}
            periods[phase] = {
                "targets": metrics,
                "passes_all": bool(
                    len(metrics) == 6
                    and all(
                        m["relative_mae_loss"] <= protocol["maximum_relative_mae_loss_each_target"]
                        for m in metrics.values()
                    )
                ),
            }
        results[family] = periods
    eligible = [k for k, v in results.items() if v["selection"]["passes_all"]]
    candidate = eligible[0] if len(eligible) == 1 else None
    retained = candidate if candidate and results[candidate]["evaluation"]["passes_all"] else None
    report = {
        "protocol": protocol,
        "results": results,
        "selected_on2021_2022": candidate,
        "retained_after2023_2026": retained,
        "reference_method": "panel6_without_" + retained if retained else "panel6",
        "interpretation": (
            "Retrospective separation check; no formal noninferiority or fear/greed validation"
        ),
    }
    save_json(output, report)
    return report


if __name__ == "__main__":
    report = selection_audit(
        Path("outputs/campaign/economics_predictions.csv"),
        Path("outputs/campaign/selection_audit.json"),
    )
    print("Selection:", report["selected_on2021_2022"], "Reference:", report["reference_method"])
