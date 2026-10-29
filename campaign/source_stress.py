"""Fixed source perturbations for v2 and PCA; revised-data sensitivity, not real time."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from imc.config import SERIES, WEIGHTS, Methodology
from imc.engine import calculate_index

from . import engine

RAW_ALIASES = {
    "dolar": "usd",
    "cobre": "copper",
    "tasa_10y": "nominal10",
    "incertidumbre": "idie",
}


def _raw_frames(snapshot):
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    records = {r["alias"]: r for r in manifest["series"]}
    frames = {}
    for name, alias in RAW_ALIASES.items():
        record, path = records[alias], snapshot / (alias + ".csv")
        if record.get("status") != "ok" or record["code"] != SERIES[name].code:
            raise ValueError("Required v2 source does not match the BCCh snapshot.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Required v2 source hash mismatch.")
        frames[name] = pd.read_csv(path, parse_dates=["date"], index_col="date").loc[
            manifest["start"] : manifest["end"]
        ]
    return frames


def _weighted(scores, omit=None):
    weights = pd.Series({"score_" + k: v for k, v in WEIGHTS.items() if k != omit})
    return scores[weights.index].mul(weights / weights.sum()).sum(axis=1, min_count=len(weights))


def _v2_scores(frames, index, methodology=None):
    result = calculate_index(frames, methodology)
    columns = ["score_" + k for k in WEIGHTS]
    # Every selected daily row is complete; last() cannot mix partially missing rows.
    return result.loc[result.complete, columns].resample("ME").last().reindex(index)


def _development(series, train_end):
    observed = series.loc[:train_end].dropna()
    if observed.empty:
        raise ValueError("No development observations.")
    median = observed.median()
    mad = (observed - median).abs().median()
    if not np.isfinite(mad) or mad <= 0:
        raise ValueError("Zero or invalid development MAD; no substitute scale invented.")
    event = observed.loc["2020-01-01":"2020-12-31"]
    if event.empty:
        raise ValueError("No valid 2020 development observation for the fixed shock.")
    return float(median), float(mad), event.index[-1]


def _revision(series, source, direction, mad):
    changed = series.copy()
    mask = (changed.index >= "2021-01-01") & (changed.index <= "2021-12-31")
    if source in {"dolar", "cobre"}:
        changed.loc[mask] *= np.exp(direction * 0.025)
    else:
        changed.loc[mask] += direction * (0.25 if source == "tasa_10y" else 0.25 * mad)
    if source in {"dolar", "cobre"} and changed.dropna().le(0).any():
        raise ValueError("Hypothetical price revision violates the positive domain.")
    if source == "incertidumbre" and changed.dropna().lt(0).any():
        raise ValueError("Hypothetical IDIE revision would be negative; scenario unavailable.")
    return changed


def evaluate_source_stress(snapshot: Path, features, levels, baseline, reps, config, output: Path):
    """Return 54 fixed cases and export source_stress.csv; never alter supplied frames."""
    records, missingness = [], []
    test_start, train_end = config["data"]["test_start"], config["data"]["train_end"]
    raw = _raw_frames(Path(snapshot))
    native = _weighted(_v2_scores(raw, baseline.index))
    base_pca, pca_meta = engine.fit_pca(features, levels, config)

    def record(label, kind, method, reference, operation, **details):
        row = {"scenario": label, "kind": kind, "method": method, **details}
        try:
            changed = operation()
            metrics = engine.metric_change(reference.loc[test_start:], changed.loc[test_start:])
            row.update(metrics, status="ok" if metrics["n"] else "no_common_dates")
        except (ValueError, np.linalg.LinAlgError) as exc:
            row.update(n=0, status="unavailable", reason=str(exc))
        records.append(row)

    full60 = _weighted(
        _v2_scores(
            raw, baseline.index, Methodology(normalization_window=60 * 21, min_history=60 * 21)
        )
    )
    for months in [36, 120]:
        record(
            f"v2_window_{months}_vs60",
            "reference_horizon",
            "v2",
            full60,
            lambda months=months: _weighted(
                _v2_scores(
                    raw,
                    baseline.index,
                    Methodology(normalization_window=months * 21, min_history=months * 21),
                )
            ),
            reference_definition="60x21 observations, full history; differs from native 504/min252",
        )
    for factor in WEIGHTS:
        record(
            "v2_without_" + factor,
            "component_ablation",
            "v2",
            reps["v2"].iloc[:, 0],
            lambda factor=factor: _weighted(baseline, omit=factor),
            component=factor,
        )
    for name, frame in raw.items():
        observed = frame.value.where(frame.status_code.eq("OK"))

        def raw_outlier(name=name, observed=observed):
            median, mad, event = _development(observed, train_end)
            changed = {k: v.copy(deep=True) for k, v in raw.items()}
            changed[name].loc[event, "value"] = median + 8 * mad
            return _weighted(_v2_scores(changed, baseline.index))

        record(
            "v2_extreme_" + name,
            "historical_outlier",
            "v2",
            native,
            raw_outlier,
            source=name,
            refit="Native rolling normalization recomputed",
        )
        for direction in [-1, 1]:

            def raw_revision(name=name, observed=observed, direction=direction):
                _, mad, _ = _development(observed, train_end)
                changed = {k: v.copy(deep=True) for k, v in raw.items()}
                changed[name]["value"] = _revision(observed, name, direction, mad)
                return _weighted(_v2_scores(changed, baseline.index))

            record(
                f"v2_revision_{name}_{direction}",
                "hypothetical_revision",
                "v2",
                native,
                raw_revision,
                source=name,
                direction=direction,
            )

    pca_columns = config["pca"]["features"]
    if len(pca_columns) != 7 or len(set(pca_columns)) != 7:
        raise ValueError("This finite campaign requires the seven registered PCA inputs.")
    effective = engine.pca_inputs(features, levels, config)
    for column in pca_columns:

        def ablate(column=column):
            altered_config = copy.deepcopy(config)
            altered_config["pca"]["features"] = [c for c in pca_columns if c != column]
            selected = engine.pca_inputs(features, levels, altered_config)
            if set(selected.columns) != set(altered_config["pca"]["features"]):
                raise ValueError("PCA input selection did not respect the registered ablation.")
            return engine.fit_pca(features, levels, altered_config)[0]

        record(
            "pca_without_" + column,
            "component_ablation",
            "dt905_adapted",
            base_pca,
            ablate,
            component=column,
            refit="Development only",
        )

        def pca_change(column=column, direction=None):
            median, mad, event = _development(effective[column], train_end)
            f, level = features.copy(deep=True), levels.copy(deep=True)
            target = level if column == "real10" else f
            if direction is None:
                target.loc[event, column] = median + 8 * mad
                return engine.fit_pca(f, level, config)[0]
            target.loc["2021-01":"2021-12", column] += direction * 0.25 * mad
            return engine.pca_project(engine.pca_inputs(f, level, config), pca_meta)

        record(
            "pca_extreme_" + column,
            "historical_outlier",
            "dt905_adapted",
            base_pca,
            pca_change,
            component=column,
            refit="Development only",
        )
        for direction in [-1, 1]:
            record(
                f"pca_revision_{column}_{direction}",
                "hypothetical_revision",
                "dt905_adapted",
                base_pca,
                lambda direction=direction: pca_change(direction=direction),
                component=column,
                direction=direction,
                refit="Frozen original development fit",
            )
        f, level = features.copy(deep=True), levels.copy(deep=True)
        target = level if column == "real10" else f
        target.loc["2021-01":"2021-12", column] = np.nan
        projected = engine.pca_project(engine.pca_inputs(f, level, config), pca_meta)
        check_dates = base_pca.loc["2021-01":"2021-12"].dropna().index
        retained = column in pca_meta["retained"]
        missingness.append(
            {
                "component": column,
                "retained": retained,
                "evaluated_months": len(check_dates),
                "all_abstained": bool(projected.loc[check_dates].isna().all())
                if retained and len(check_dates)
                else None,
                "status": "retained_input_missing" if retained else "excluded_in_development",
            }
        )

    scenarios = len(records) + len(missingness)
    if scenarios != 54 or scenarios > config["resource_limits"]["sensitivity_scenarios_max"]:
        raise ValueError("Source-stress campaign exceeded or changed its 54 fixed cases.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "source_stress.csv", index=False)
    return {
        "scenario_count": scenarios,
        "metric_cases": len(records),
        "missingness_cases": len(missingness),
        "records": records,
        "missingness": missingness,
        "v2_native_reproduction": engine.metric_change(reps["v2"].iloc[:, 0], native),
        "pca_native_reproduction": engine.metric_change(reps["dt905_adapted"].iloc[:, 0], base_pca),
        "interpretation": "Hypothetical revised-data sensitivity; not point-in-time validation",
        "window_note": "36/60/120 months approximated by 21 observations each; full minimum. "
        "60-month reference is separate from native v2's 504-observation window and 252 minimum. "
        "Daily v2 normalization includes the evaluated observation; the monthly panel excludes it.",
        "revision_note": "2021 block: v2 prices multiplied by exp(+/-0.025), nominal rate shifted "
        "+/-0.25 pp, IDIE shifted +/-0.25 development MAD; invalid domains yield NA. "
        "PCA inputs shifted +/-0.25 development MAD, real10 in levels. These perturbations are "
        "not economically equivalent across representations; no new model fitted on 2021 data.",
    }
