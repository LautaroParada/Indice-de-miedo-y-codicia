"""Experimentos registrados: representaciones, estabilidad y límites dinámicos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from imc.config import WEIGHTS


def rank_past(frame, window=60, minimum=36):
    """Rango frente a meses anteriores; ninguna interpolación ni futuro."""
    if not 1 <= minimum <= window:
        raise ValueError("Historia mínima fuera de ventana")
    values = frame.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    for i in range(1, len(values)):
        prior = values[max(0, i - window) : i]
        counts = np.isfinite(prior).sum(axis=0)
        valid = (counts >= minimum) & np.isfinite(values[i])
        finite = np.isfinite(prior)
        ranks = ((prior < values[i]) & finite).sum(axis=0) + 0.5 * (
            (prior == values[i]) & finite
        ).sum(axis=0)
        out[i, valid] = 100 * ranks[valid] / counts[valid]
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def family_profiles(scores, groups, weights=None):
    """Media con todos los componentes obligatorios. Pesos solo para variantes registradas."""
    out = {}
    for name, cols in groups.items():
        if not cols:
            continue
        w = np.array([1.0 if weights is None else weights.get(c, 1.0) for c in cols])
        if not np.isfinite(w).all() or (w <= 0).any():
            raise ValueError("Pesos deben ser positivos")
        out[name] = (
            scores[cols].mul(w / w.sum()).sum(axis=1).where(scores[cols].notna().all(axis=1))
        )
    return pd.DataFrame(out, index=scores.index)


def pca_inputs(features, levels, config):
    x = features[config["pca"]["features"]].copy()
    # DT905 usa nivel de tasa real: no sustituirlo silenciosamente por cambio mensual.
    if "real10" in x:
        x["real10"] = levels.real10
    return x


def pca_project(x, metadata, multipliers=None):
    columns = metadata["retained"]
    z = (x[columns] - pd.Series(metadata["mean"])) / pd.Series(metadata["std"])
    loadings = np.array(metadata["loadings"])
    if multipliers is not None:
        loadings = loadings * np.array([multipliers.get(c, 1) for c in columns])
        loadings /= np.linalg.norm(loadings)
    value = z.mul(loadings).sum(axis=1).where(z.notna().all(axis=1))
    reference = value.loc[: metadata["train_end"]].dropna().to_numpy()
    result = value.apply(
        lambda v: (
            100 * ((reference < v).sum() + 0.5 * (reference == v).sum()) / len(reference)
            if pd.notna(v)
            else np.nan
        )
    )
    return result.rename("financial_tightness")


def fit_pca(features, levels, config):
    x = pca_inputs(features, levels, config)
    train_end = config["data"]["train_end"]
    train = x.loc[:train_end].dropna()
    if len(train) < config["pca"]["minimum_train_months"]:
        raise ValueError("Historia insuficiente para PCA registrado")
    retained, excluded = [], []
    corr = train.corr()
    for col in x:
        if train[col].std() <= 1e-12 or any(abs(corr.loc[col, k]) > 0.8 for k in retained):
            excluded.append(col)
        else:
            retained.append(col)
    if not retained:
        raise ValueError("PCA sin componentes variables")
    mean, std = train[retained].mean(), train[retained].std()
    z = (train[retained] - mean) / std
    eig, vectors = np.linalg.eigh(np.atleast_2d(np.cov(z.to_numpy().T)))
    loadings = vectors[:, -1]
    direction = np.array([config["pca"]["orientation"][c] for c in retained])
    if np.dot(loadings, direction) < 0:
        loadings *= -1
    meta = {
        "retained": retained,
        "excluded_development_correlation_or_constant": excluded,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "loadings": loadings.tolist(),
        "train_end": train_end,
        "train_n": len(train),
        "explained_variance": float(eig[-1] / eig.sum()),
        "interpretation": "PC1 oriented toward tighter conditions; adaptation of DT905",
    }
    return pca_project(x, meta), meta


def equal_family(panel, weights=None):
    x = panel[["market", "confidence", "friction"]].copy()
    x.friction = 100 - x.friction
    w = np.array([1 if weights is None else weights.get(c, 1) for c in x])
    return x.mul(w / w.sum()).sum(axis=1).where(x.notna().all(axis=1)).rename("aggregate")


def build_representations(features, levels, baseline, config, window=60, minimum=36):
    scores = rank_past(features, window, minimum)
    panel = family_profiles(scores, config["families"])
    v2 = baseline.mul({"score_" + k: v for k, v in WEIGHTS.items()}).sum(axis=1)
    v2 = v2.where(baseline.notna().all(axis=1))
    pc, meta = fit_pca(features, levels, config)
    reps = {
        "v2": v2.rename("aggregate").to_frame(),
        "equal_family": equal_family(panel).to_frame(),
        "panel6": panel,
        "dt905_adapted": pc.to_frame(),
        "minimal3": family_profiles(scores, config["minimal_families"]),
    }
    return reps, scores, meta


def categories(x):
    """0=bajo,1=central,2=alto; jamás convertir NaN en central."""
    return (1 + (x > 75).astype(float) - (x < 25).astype(float)).where(x.notna())


def metric_change(base, changed):
    pair = pd.concat([base.rename("base"), changed.rename("changed")], axis=1)
    valid = pair.dropna()
    if valid.empty:
        return {"n": 0}
    delta = (valid.base - valid.changed).abs()
    cb, cc = categories(valid.base), categories(valid.changed)
    d = pair.diff().dropna()
    meaningful = d[(d.base.abs() > 5) & (d.changed.abs() > 5)]
    # Pearson of average ranks is Spearman, including ties; no optional SciPy dependency.
    rho = (
        valid.base.rank().corr(valid.changed.rank())
        if valid.base.nunique() > 1 and valid.changed.nunique() > 1
        else np.nan
    )
    return {
        "n": len(valid),
        "median_abs": float(delta.median()),
        "p95_abs": float(delta.quantile(0.95)),
        "max_abs": float(delta.max()),
        "spearman": float(rho) if pd.notna(rho) else None,
        "category_change_rate": float((cb != cc).mean()),
        "opposite_tail_rate": float((cb - cc).abs().eq(2).mean()),
        "direction_inversion_n": int(((meaningful.base * meaningful.changed) < 0).sum()),
        "direction_comparable_n": len(meaningful),
        "max_date": str(delta.idxmax().date()),
    }


def ridge_predict(train_x, train_y, test_x):
    mean, std = train_x.mean(), train_x.std().replace(0, 1)
    x, xt = ((train_x - mean) / std).to_numpy(), ((test_x - mean) / std).to_numpy()
    yc = train_y.to_numpy() - train_y.mean().to_numpy()
    coef = np.linalg.solve(x.T @ x + 10 * np.eye(x.shape[1]), x.T @ yc)
    return pd.DataFrame(
        xt @ coef + train_y.mean().to_numpy(), index=test_x.index, columns=train_y.columns
    )


def diagnostic_tasks(reps, config):
    core = config["diagnostic_tasks"]["essential_families"]
    truth = reps["panel6"]
    common = truth.notna().all(axis=1)
    for frame in reps.values():
        common &= frame.notna().all(axis=1)
    target = truth.loc[common]
    train = target.index <= config["data"]["train_end"]
    test = target.index >= config["data"]["test_start"]
    answer = {}
    predictions = {}
    for name, frame in reps.items():
        x = frame.loc[target.index]
        pred = ridge_predict(x.loc[train], target.loc[train], x.loc[test])
        for col in sorted(set(frame.columns) & set(target.columns)):
            pred[col] = frame.loc[pred.index, col]
        predictions[name] = pred
        observed = target.loc[test]
        core_accuracy = (categories(pred[core]) == categories(observed[core])).mean()

        def opposition(p):
            return ((p.market < 25) & (p.confidence > 75)) | ((p.market > 75) & (p.confidence < 25))

        expected_opp, got_opp = opposition(observed), opposition(pred)
        change = observed.reindex(truth.index).diff().loc[observed.index]
        pred_change = pred.reindex(truth.index).diff().loc[pred.index]
        ok = change[core].notna().all(axis=1) & pred_change[core].notna().all(axis=1)

        def direction(d):
            return np.sign(d).where(d.abs() > 5, 0)

        big_change = change[core].abs().max(axis=1) > 5
        answer[name] = {
            "train_n": int(train.sum()),
            "test_n": int(test.sum()),
            "core_category_accuracy": core_accuracy.to_dict(),
            "optional_category_accuracy": (
                categories(pred[config["diagnostic_tasks"]["optional_families"]])
                == categories(observed[config["diagnostic_tasks"]["optional_families"]])
            )
            .mean()
            .to_dict(),
            "opposition_accuracy": float((expected_opp == got_opp).mean()),
            "opposition_positive_n": int(expected_opp.sum()),
            "opposition_recall": float(got_opp[expected_opp].mean())
            if expected_opp.any()
            else None,
            "opposition_false_positive_n": int((got_opp & ~expected_opp).sum()),
            "change_direction_accuracy": float(
                (direction(change.loc[ok, core]) == direction(pred_change.loc[ok, core]))
                .to_numpy()
                .mean()
            ),
            "dominant_change_accuracy": float(
                (
                    change.loc[ok & big_change, core].abs().idxmax(axis=1)
                    == pred_change.loc[ok & big_change, core].abs().idxmax(axis=1)
                ).mean()
            ),
            "interpretation": (
                "Information preservation only; displayed axes are not independently validated"
            ),
        }
    # Same-number / different-composition examples, no claim of new economic information.
    agg = reps["equal_family"]["aggregate"].loc[target.loc[test].index]
    collisions = []
    idx = list(agg.dropna().index)
    for i, a in enumerate(idx):
        for b in idx[i + 1 :]:
            if abs(agg.loc[a] - agg.loc[b]) <= 2:
                distance = (truth.loc[a, core] - truth.loc[b, core]).abs().max()
                if distance >= 40:
                    collisions.append(
                        {
                            "a": str(a.date()),
                            "b": str(b.date()),
                            "aggregate_a": float(agg.loc[a]),
                            "aggregate_b": float(agg.loc[b]),
                            "max_profile_difference": float(distance),
                        }
                    )
    return {
        "methods": answer,
        "collisions": sorted(collisions, key=lambda x: -x["max_profile_difference"])[:10],
        "collision_pairs_total": len(collisions),
        "core": core,
        "scalar_plus_components_control": (
            "Identical displayed components give identical information to panel; scalar discards it"
        ),
    }, predictions


def stable_envelopes(features, config):
    frames = {
        w: family_profiles(rank_past(features, w, w), config["minimal_families"])
        for w in config["normalization"]["windows"]
    }
    test_index = features.loc[config["data"]["test_start"] :].index
    records, metrics = [], {}
    for family in config["minimal_families"]:
        x = pd.DataFrame({str(w): f[family] for w, f in frames.items()}).reindex(test_index)
        complete = x.notna().all(axis=1)
        low, high = x.min(axis=1).where(complete), x.max(axis=1).where(complete)
        same = categories(x).nunique(axis=1).eq(1) & complete
        stable = same & (high - low <= 20)
        metrics[family] = {
            "denominator_all_test_months": len(x),
            "complete_windows_n": int(complete.sum()),
            "stable_n": int(stable.sum()),
            "stable_coverage": float(stable.mean()),
            "median_width_common": float((high - low).median()) if complete.any() else None,
            "p95_width_common": float((high - low).quantile(0.95)) if complete.any() else None,
            "pass_coverage": bool(
                stable.mean() >= config["stability"]["stable_label_coverage_minimum"]
            ),
        }
        for date in test_index:
            records.append(
                {
                    "date": date,
                    "family": family,
                    "rank36": x.loc[date, "36"],
                    "rank60": x.loc[date, "60"],
                    "rank120": x.loc[date, "120"],
                    "low": low.loc[date],
                    "high": high.loc[date],
                    "label": ["bajo", "central", "alto"][int(categories(x["60"]).loc[date])]
                    if stable.loc[date]
                    else "abstencion_sensible_o_historia_insuficiente",
                }
            )
    return metrics, pd.DataFrame(records)


def sensitivity(features, levels, baseline, reps, scores, pca_meta, config):
    rng = np.random.default_rng(config["seed"])
    test_start = config["data"]["test_start"]
    columns = sorted({c for g in config["families"].values() for c in g})
    records, episode_records, scenarios = [], [], 0

    def compare(label, kind, altered, base_reps=None):
        nonlocal scenarios
        scenarios += 1
        for method, candidate in altered.items():
            reference = (reps if base_reps is None else base_reps)[method]
            for col in sorted(set(reference.columns) & set(candidate.columns)):
                m = metric_change(reference[col].loc[test_start:], candidate[col].loc[test_start:])
                records.append(
                    {"scenario": label, "kind": kind, "method": method, "profile": col, **m}
                )
                for begin, end in config["stability"]["episode_windows"]:
                    begin_at = max(pd.Timestamp(begin), pd.Timestamp(test_start))
                    end_at = pd.Timestamp(end) + pd.offsets.MonthEnd(0)
                    if begin_at > end_at:
                        continue
                    episode_records.append(
                        {
                            "scenario": label,
                            "kind": kind,
                            "method": method,
                            "profile": col,
                            "episode": f"{begin}/{end}",
                            **metric_change(
                                reference[col].loc[begin_at:end_at],
                                candidate[col].loc[begin_at:end_at],
                            ),
                        }
                    )

    def panel_alternatives(s, groups=None, component_weights=None, family_weights=None):
        p = family_profiles(s, groups or config["families"], component_weights)
        methods = {"panel6": p, "minimal3": p[[c for c in config["minimal_families"] if c in p]]}
        if {"market", "confidence", "friction"}.issubset(p):
            methods["equal_family"] = equal_family(p, family_weights).to_frame()
        return methods

    for i in range(config["stability"]["weight_draws"]):
        cw = dict(zip(columns, rng.uniform(0.8, 1.2, len(columns)), strict=True))
        fw = dict(zip(config["minimal_families"], rng.uniform(0.8, 1.2, 3), strict=True))
        altered = panel_alternatives(scores, component_weights=cw, family_weights=fw)
        bw = pd.Series({"score_" + k: w * rng.uniform(0.8, 1.2) for k, w in WEIGHTS.items()})
        altered["v2"] = (
            baseline.mul(bw / bw.sum())
            .sum(axis=1)
            .where(baseline.notna().all(axis=1))
            .rename("aggregate")
            .to_frame()
        )
        pw = {c: rng.uniform(0.8, 1.2) for c in pca_meta["retained"]}
        altered["dt905_adapted"] = pca_project(
            pca_inputs(features, levels, config), pca_meta, pw
        ).to_frame()
        compare(f"weights_{i:02}", "minor_weights", altered)

    # Full-history-window contrasts: all methods use identical dates for their own pair.
    full60 = panel_alternatives(rank_past(features, 60, 60))
    for w in [36, 120]:
        compare(
            f"window_{w}_vs60",
            "reference_horizon",
            panel_alternatives(rank_past(features, w, w)),
            full60,
        )
    for col in columns:
        groups = {k: [c for c in v if c != col] for k, v in config["families"].items()}
        compare(f"without_{col}", "component_ablation", panel_alternatives(scores, groups))
    # Omitting an entire scalar family is an intentional alternative, not missing-data policy.
    oriented = reps["minimal3"].copy()
    oriented.friction = 100 - oriented.friction
    for family in oriented:
        new = (
            oriented.drop(columns=family)
            .mean(axis=1)
            .where(oriented.drop(columns=family).notna().all(axis=1))
        )
        compare(
            f"without_family_{family}",
            "family_ablation",
            {"equal_family": new.rename("aggregate").to_frame()},
        )
    for lag in [1, 2]:
        compare(
            f"all_profiles_delay{lag}",
            "publication_delay",
            {k: v.shift(lag) for k, v in reps.items()},
        )
        for family, cols in config["families"].items():
            lagged = scores.copy()
            lagged[cols] = lagged[cols].shift(lag)
            compare(f"delay{lag}_{family}", "publication_delay", panel_alternatives(lagged))
    train = features.loc[: config["data"]["train_end"]]
    for col in columns:
        observed = train[col].dropna()
        if observed.empty:
            continue
        med = observed.median()
        mad = (observed - med).abs().median()
        modified = features.copy()
        event = observed.index[-1]
        modified.loc[event, col] = med + 8 * max(mad, 1e-12)
        compare(
            "historical_extreme_" + col,
            "historical_outlier",
            panel_alternatives(rank_past(modified)),
        )
        for direction in [-1, 1]:
            modified = features.copy()
            modified.loc["2021-01":"2021-12", col] += direction * 0.25 * mad
            compare(
                f"revision_{col}_{direction}",
                "hypothetical_revision",
                panel_alternatives(rank_past(modified)),
            )
    # Data losses do not silently remove a mandatory component.
    missing_checks = []
    for col in columns:
        altered = scores.copy()
        altered.loc["2021-01":"2021-12", col] = np.nan
        p = family_profiles(altered, config["families"])
        affected = [k for k, v in config["families"].items() if col in v]
        missing_checks.append(
            {
                "component": col,
                "affected": affected,
                "all_abstained": bool(p.loc["2021-01":"2021-12", affected].isna().all().all()),
            }
        )
        scenarios += 1
    if scenarios > config["resource_limits"]["sensitivity_scenarios_max"]:
        raise RuntimeError("Se excedió el contrato experimental")
    table = pd.DataFrame(records)
    weight_rows = table.loc[table.kind.eq("minor_weights")]
    passes = (
        (weight_rows.n >= config["stability"]["minimum_common_months"])
        & (weight_rows.median_abs <= 5)
        & (weight_rows.p95_abs <= 10)
        & (weight_rows.category_change_rate <= 0.1)
        & (weight_rows.opposite_tail_rate <= 0.01)
    )
    summary = {
        "scenario_count": scenarios,
        "episode_rows": episode_records,
        "missingness": missing_checks,
        "weight_cases_passed": int(passes.sum()),
        "weight_cases_total": len(passes),
        "weight_by_method": {
            method: {
                "cases": len(g),
                "median_abs_worst": float(g.median_abs.max()),
                "p95_abs_worst": float(g.p95_abs.max()),
                "category_change_worst": float(g.category_change_rate.max()),
                "opposite_tail_worst": float(g.opposite_tail_rate.max()),
            }
            for method, g in weight_rows.groupby("method")
        },
    }
    return summary, table


def dynamics(reps, config):
    x = reps["minimal3"]
    train = x.loc[: config["data"]["train_end"]].dropna()
    rho = x.loc[: config["data"]["train_end"]].apply(lambda s: s.autocorr(1))
    neff = (len(train) * (1 - rho) / (1 + rho)).clip(1, len(train))
    features = x.shift(1)
    valid = x.notna().all(axis=1) & features.notna().all(axis=1)
    tr = valid & (x.index <= config["data"]["train_end"])
    te = valid & (x.index >= config["data"]["test_start"])
    gate_config = config["dynamic_gate"]
    gate = (
        tr.sum() >= 10 * gate_config["minimum_training_rows_per_library_term"]
        and neff.min() >= gate_config["minimum_effective_training_rows"]
        and te.sum() >= gate_config["minimum_validation_months"]
    )
    pred = ridge_predict(features.loc[tr], x.loc[tr], features.loc[te])
    mse_persistence = ((x.loc[te] - features.loc[te]) ** 2).mean()
    mse_var = ((x.loc[te] - pred) ** 2).mean()
    return {
        "train_months": len(train),
        "training_transitions": int(tr.sum()),
        "test_months": int(te.sum()),
        "lag1_autocorrelation": rho.to_dict(),
        "effective_n": neff.to_dict(),
        "library_terms_per_equation": 10,
        "sindy_identifiable_gate": bool(gate),
        "sindy_fitted": False,
        "sindy_reason": "Identification gate failed; no model fitted"
        if not gate
        else "Gate passed; model would require explicit registered recovery check",
        "persistence_mse": mse_persistence.to_dict(),
        "ridge_var1_mse": mse_var.to_dict(),
        "var1_mse_gain_vs_persistence": (1 - mse_var / mse_persistence).to_dict(),
        "interpretation": (
            "Profile reconstruction, revised data; not return forecasts or causal dynamics"
        ),
    }


def save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run(
    base_dir,
    output,
    contract_path=Path("campaign/contract.json"),
    snapshot_dir=Path("campaign/snapshot-reference"),
):
    config = json.loads(contract_path.read_text(encoding="utf-8"))

    def read(name):
        return pd.read_csv(base_dir / (name + ".csv"), index_col=0, parse_dates=True)

    features, levels, baseline, quarterly = [
        read(n) for n in ["features", "monthly_levels", "baseline_v2", "quarterly"]
    ]
    output.mkdir(parents=True, exist_ok=True)
    reps, scores, pca_meta = build_representations(features, levels, baseline, config)
    for name, frame in reps.items():
        frame.to_csv(output / (name + ".csv"), date_format="%Y-%m-%d")
    save_json(output / "pca_fit.json", pca_meta)
    tasks, preds = diagnostic_tasks(reps, config)
    for name, pred in preds.items():
        pred.to_csv(output / ("decoded_" + name + ".csv"), date_format="%Y-%m-%d")
    sensitivity_summary, sensitivity_rows = sensitivity(
        features, levels, baseline, reps, scores, pca_meta, config
    )
    sensitivity_rows.to_csv(output / "sensitivity.csv", index=False)
    pd.DataFrame(sensitivity_summary.pop("episode_rows")).to_csv(
        output / "sensitivity_episodes.csv", index=False
    )
    coverage, envelopes = stable_envelopes(features, config)
    envelopes.to_csv(output / "envelopes.csv", index=False, date_format="%Y-%m-%d")
    from .economics import evaluate_economics

    economic = evaluate_economics(reps, levels, quarterly, config, output)
    from .source_stress import evaluate_source_stress

    source_stress = evaluate_source_stress(
        snapshot_dir, features, levels, baseline, reps, config, output
    )
    total_scenarios = sensitivity_summary["scenario_count"] + source_stress["scenario_count"]
    if total_scenarios > config["resource_limits"]["sensitivity_scenarios_max"]:
        raise RuntimeError("Se excedió el presupuesto total de escenarios")
    results = {
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "diagnostic_tasks": tasks,
        "sensitivity": sensitivity_summary,
        "stable_labels": coverage,
        "dynamics": dynamics(reps, config),
        "economics": economic,
        "source_stress": source_stress,
        "total_sensitivity_scenarios": total_scenarios,
        "coverage": {
            k: {
                "available_months": int(v.notna().all(axis=1).sum()),
                "test_complete_months": int(
                    v.loc[config["data"]["test_start"] :].notna().all(axis=1).sum()
                ),
            }
            for k, v in reps.items()
        },
    }
    save_json(output / "results.json", results)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("outputs/campaign-base"))
    parser.add_argument("--output", type=Path, default=Path("outputs/campaign"))
    parser.add_argument("--snapshot", type=Path, default=Path("campaign/snapshot-reference"))
    args = parser.parse_args()
    results = run(args.base, args.output, snapshot_dir=args.snapshot)
    print("Campaign completed; results:", args.output / "results.json")
