"""Quarterly ECB association tests with fixed ridge and paired block intervals.

``evaluate_economics(representations, levels, quarterly, config, output)`` expects
one dated row per month in each representation and in raw ``levels``. Each
column of panel6/minimal3 is one complete family, so deleting a column deletes
that family. ECB inputs have at most one observed value per target and quarter.
The output directory receives economics_metrics.json and economics_predictions.csv.
No fitting or filesystem access occurs on import; no external data is fetched.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ARCHITECTURES = ("v2", "equal_family", "panel6", "dt905_adapted", "minimal3")


def ridge_predict(x_train, x_test, y_train, alpha=10.0):
    """Return predictions; population z-scales and unpenalized intercept use train only."""
    x, xt, y = (np.asarray(a, dtype=float) for a in (x_train, x_test, y_train))
    if x.ndim != 2 or xt.ndim != 2 or y.ndim != 1:
        raise ValueError("Ridge requires two matrices and one target vector.")
    if len(x) != len(y) or not len(y) or x.shape[1] != xt.shape[1] or alpha <= 0:
        raise ValueError("Incompatible ridge dimensions or nonpositive penalty.")
    if not all(np.isfinite(a).all() for a in (x, xt, y)):
        raise ValueError("Ridge cannot fit missing or infinite observations.")
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    constant = scale <= np.finfo(float).eps * np.maximum(1.0, np.abs(mean))
    scale = np.where(constant, 1.0, scale)
    z, zt = (x - mean) / scale, (xt - mean) / scale
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - y.mean()))
    return zt @ coef + y.mean()


def _dated(frame, offset):
    result = frame.astype(float).replace([np.inf, -np.inf], np.nan).copy()
    index = result.index
    index = (
        index.to_timestamp(how="end")
        if isinstance(index, pd.PeriodIndex)
        else pd.DatetimeIndex(index)
    )
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = (index + offset).normalize()
    return result.sort_index()


def _monthly(frame, calendar):
    frame = _dated(frame, pd.offsets.MonthEnd(0))
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("Monthly inputs require unique months and columns.")
    return frame.reindex(calendar)


def _quarter_mean(frame):
    complete = frame.notna().all(axis=1)
    counts = complete.resample("QE").sum().astype(int)
    means = frame.where(complete, axis=0).resample("QE").mean()
    return means.where(counts.eq(3), axis=0), counts


def _span(index):
    return {
        "n": len(index),
        "start": str(index.min().date()) if len(index) else None,
        "end": str(index.max().date()) if len(index) else None,
        "quarters": [str(date.date()) for date in index],
    }


def _block_indices(index, draws, block, rng):
    """Moving blocks never cross a missing quarter; abstain if some rows cannot enter."""
    n = len(index)
    serial = index.to_period("Q").asi8
    starts = [i for i in range(n - block + 1) if np.all(np.diff(serial[i : i + block]) == 1)]
    covered = {i + j for i in starts for j in range(block)}
    if len(covered) != n or not starts:
        return None
    chosen = rng.choice(starts, size=(draws, (n + block - 1) // block))
    return (chosen[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :n]


def _gain(baseline_errors, model_errors, indices):
    denominator = float(baseline_errors.mean())
    estimate = float(1 - model_errors.mean() / denominator) if denominator > 0 else None
    result = {"relative_mae_gain": estimate, "ci95": None, "bootstrap_draws": 0}
    if denominator <= 0:
        return result | {"status": "undefined_zero_baseline_mae"}
    if indices is None:
        return result | {"status": "unavailable_noncontiguous_block_coverage"}
    means = baseline_errors[indices].mean(axis=1)
    if np.any(means <= 0):
        return result | {"status": "undefined_zero_bootstrap_baseline_mae"}
    gains = 1 - model_errors[indices].mean(axis=1) / means
    return result | {
        "status": "exploratory",
        "ci95": np.quantile(gains, [0.025, 0.975]).tolist(),
        "bootstrap_draws": len(indices),
    }


def evaluate_economics(
    representations: dict[str, pd.DataFrame],
    levels: pd.DataFrame,
    quarterly: pd.DataFrame,
    config: dict,
    output: Path,
) -> dict:
    """Evaluate common-quarter contemporaneous ECB associations, including family ablations.

    Quarterly predictors require three complete monthly rows. All five architectures,
    controls, and the current target determine one common sample used by every fit.
    Ablations retain that sample. Returns the same JSON-safe dict written to disk;
    insufficient samples are reported without fitting. Bootstrap uses a shared draw
    matrix for paired comparisons; positive gain means lower MAE for the tested model.
    """
    missing = set(ARCHITECTURES) - representations.keys()
    if missing:
        raise ValueError(f"Missing registered architectures: {sorted(missing)}")
    if any(representations[name].shape[1] == 0 for name in ARCHITECTURES):
        raise ValueError("An architecture has no observed coordinates.")
    settings, limits = config["data"], config["resource_limits"]
    train_end = pd.Timestamp(settings["train_end"])
    test_start = pd.Timestamp(settings["test_start"])
    if train_end >= test_start:
        raise ValueError("Training and evaluation periods must not overlap.")
    calendar = pd.date_range(settings["start"], settings["end"], freq="ME")
    if not len(calendar):
        raise ValueError("The configured period has no complete month.")
    levels = _monthly(levels[["imacec", "cpi", "tpm"]], calendar)
    controls = pd.DataFrame(
        {
            "imacec_yoy": np.log(levels.imacec.where(levels.imacec > 0)).diff(12),
            "cpi_yoy": np.log(levels.cpi.where(levels.cpi > 0)).diff(12),
            "tpm_level": levels.tpm,
        }
    )
    q_controls, counts = _quarter_mean(controls)
    coverage = {"controls": {str(k.date()): int(v) for k, v in counts.items()}}
    all_predictors = q_controls.add_prefix("controls::")
    control_cols = list(all_predictors.columns)
    model_cols, parents = {"controls": control_cols}, {}
    for name in ARCHITECTURES:
        q_rep, counts = _quarter_mean(_monthly(representations[name], calendar))
        q_rep = q_rep.add_prefix(name + "::")
        coverage[name] = {str(k.date()): int(v) for k, v in counts.items()}
        all_predictors = all_predictors.join(q_rep)
        columns = list(q_rep.columns)
        model_cols[name] = control_cols + columns
        if name in ("panel6", "minimal3"):
            for column in columns:
                ablation = name + "_without_" + column.split("::", 1)[1]
                model_cols[ablation] = control_cols + [c for c in columns if c != column]
                parents[ablation] = name
    targets = _dated(quarterly, pd.offsets.QuarterEnd(0))
    draws = int(limits.get("bootstrap_draws", 400))
    block = int(limits.get("bootstrap_block_quarters", 4))
    if draws < 1 or block < 1:
        raise ValueError("Bootstrap draws and block length must be positive.")
    report = {
        "status": "completed",
        "contract_id": config.get("id"),
        "ridge_alpha": 10.0,
        "association": "same-quarter revised-data association; not a real-time forecast",
        "controls": ["log(imacec).diff(12)", "log(cpi).diff(12)", "tpm level"],
        "quarter_coverage_complete_months": coverage,
        "targets": {},
        "bootstrap": {
            "draws": draws,
            "block_quarters": block,
            "seed": config["seed"],
            "interval": "95% exploratory; no multiplicity adjustment or formal claim",
        },
        "limitations": [
            "Known previous exploration; no pristine holdout.",
            "ECB is not criterion truth for psychological fear or greed.",
            "No classification thresholds were registered; no classification metrics.",
            "Failure of this fixed ridge does not refute all economic information.",
            "Intervals abstain when full blocks cannot cover all paired observations.",
        ],
    }
    predictions = []
    for target_number, target in enumerate(config["external_economic_test"]["targets"]):
        if target not in targets:
            report["targets"][target] = {"status": "missing_target"}
            continue
        observed = targets[target].dropna()
        if observed.index.has_duplicates:
            raise ValueError(f"ECB target {target} is repeated within a quarter.")
        common = all_predictors.join(observed.rename("__target"), how="inner").dropna()
        train, test = common.loc[:train_end], common.loc[test_start:]
        entry = {
            "train": _span(train.index),
            "test": _span(test.index),
            "target_observed_quarters": int(len(observed)),
            "predictor_complete_quarters": int(all_predictors.notna().all(axis=1).sum()),
            "models": {},
        }
        report["targets"][target] = entry
        if len(train) < settings.get("fit_min_quarters", 20) or len(test) < settings.get(
            "test_min_quarters", 16
        ):
            entry["status"] = "insufficient_common_sample"
            continue
        entry["status"] = "evaluated"
        errors, squared = {}, {}
        for name, columns in model_cols.items():
            fitted = ridge_predict(train[columns], test[columns], train["__target"])
            residual = test["__target"].to_numpy() - fitted
            errors[name], squared[name] = np.abs(residual), residual**2
            predictions.extend(
                {
                    "target": target,
                    "model": name,
                    "quarter": str(date.date()),
                    "observed": float(actual),
                    "predicted": float(predicted),
                    "error": float(error),
                }
                for date, actual, predicted, error in zip(
                    test.index, test["__target"], fitted, residual
                )
            )
        rng = np.random.default_rng(np.random.SeedSequence([int(config["seed"]), target_number]))
        indices = _block_indices(test.index, draws, block, rng)
        for name in model_cols:
            entry["models"][name] = {
                "mae": float(errors[name].mean()),
                "mse": float(squared[name].mean()),
                "gain_vs_controls": _gain(errors["controls"], errors[name], indices),
                "gain_vs_minimal3": _gain(errors["minimal3"], errors[name], indices),
            }
            if name in parents:
                entry["models"][name]["ablated_from"] = parents[name]
                entry["models"][name]["gain_vs_full_representation"] = _gain(
                    errors[parents[name]], errors[name], indices
                )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = ["target", "model", "quarter", "observed", "predicted", "error"]
    pd.DataFrame(predictions, columns=fields).to_csv(
        output / "economics_predictions.csv", index=False
    )
    (output / "economics_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report
