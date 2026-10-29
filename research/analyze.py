"""Auditoría retrospectiva de información adicional y panel interpretable BDE."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from imc.engine import calculate_index


def percentile_past(series, window=60, minimum=36):
    """Referente estrictamente anterior; no confundir observación con publicación."""
    result = pd.Series(np.nan, index=series.index)
    for i, value in enumerate(series):
        prior = series.iloc[max(0, i - window) : i].dropna()
        if pd.notna(value) and len(prior) >= minimum:
            result.iloc[i] = (
                100 * ((prior < value).sum() + 0.5 * (prior == value).sum()) / len(prior)
            )
    return result


def transform(series, kind):
    if kind.startswith("logdiff"):
        return np.log(series.where(series > 0)).diff(int(kind[7:]))
    if kind == "diff":
        return series.diff()
    if kind == "level":
        return series.copy()
    raise ValueError("Transformación no admitida")


def ridge_reconstruction(base, candidate, train_end="2018-12-31", test_start="2019-01-01"):
    if pd.Timestamp(test_start) <= pd.Timestamp(train_end):
        raise ValueError("Los periodos de ajuste y contraste no pueden superponerse.")
    pair = base.join(candidate.rename("target")).dropna()
    train = pair.loc[:train_end]
    test = pair.loc[test_start:]
    if len(train) < 36 or len(test) < 24:
        return {"status": "insufficient", "train_n": len(train), "test_n": len(test)}
    cols = list(base.columns)
    mean, std = train[cols].mean(), train[cols].std().replace(0, 1)
    x, xt = ((train[cols] - mean) / std).to_numpy(), ((test[cols] - mean) / std).to_numpy()
    y = train.target.to_numpy()
    coef = np.linalg.solve(x.T @ x + 10 * np.eye(x.shape[1]), x.T @ (y - y.mean()))
    predicted = xt @ coef + y.mean()
    truth = test.target.to_numpy()
    error = np.sum((truth - predicted) ** 2)
    denominator = np.sum((truth - truth.mean()) ** 2)
    constant_error = np.sum((truth - y.mean()) ** 2)
    return {
        "status": "ok",
        "train_n": len(train),
        "test_n": len(test),
        "r2_test": float(1 - error / denominator) if denominator > 0 else None,
        "mse_vs_train_constant": float(error / constant_error) if constant_error > 0 else None,
    }


def complete_mean(frame, columns):
    return frame[columns].mean(axis=1).where(frame[columns].notna().all(axis=1))


def credit_distribution(quarterly):
    cols = [
        "ecb_supply_sme_" + s
        for s in ["less", "moderately_less", "unchanged", "moderately_more", "more"]
    ]
    if not set(cols).issubset(quarterly):
        return pd.DataFrame()
    if "ecb_supply_sme" not in quarterly:
        raise ValueError("Falta el saldo ECB publicado para contrastar la distribución.")
    q = quarterly[cols].dropna()
    if not q.ge(-0.05).all().all() or not q.le(100.05).all().all():
        raise ValueError("Proporciones ECB fuera de rango.")
    if not (q.sum(axis=1) - 100).abs().le(0.5).all():
        raise ValueError("Las cinco proporciones ECB no suman 100.")
    easing = q[cols[:2]].sum(axis=1)
    tightening = q[cols[3:]].sum(axis=1)
    out = pd.DataFrame(
        {
            "flexibilizan": easing,
            "restringen": tightening,
            "sin_cambios": q[cols[2]],
            "saldo_reconstruido": easing - tightening,
            "amplitud_cambio": easing + tightening,
            "coexistencia_direcciones": 2 * np.minimum(easing, tightening),
        }
    )
    out["saldo_publicado"] = quarterly.ecb_supply_sme
    if not (out.saldo_reconstruido - out.saldo_publicado).dropna().abs().le(0.5).all():
        raise ValueError("El saldo reconstruido no coincide con el publicado por BCCh.")
    return out


def geometry(scores, base):
    names = [
        "ipsa",
        "price_book",
        "turnover",
        "ipec",
        "imce_nonmining",
        "bank_spread",
        "funding_deviation",
        "idie",
        "rate_volatility",
        "liquid_share",
        "fx_expectation_gap",
        "spot_position",
    ]
    common = scores[names].join(base).dropna()
    train, test = common.loc[:"2018-12-31"], common.loc["2019-01-01":]
    if len(train) < len(names) + 2 or len(test) < 24:
        return {"status": "insufficient"}
    mean, std = train[names].mean(), train[names].std()
    z = ((train[names] - mean) / std).to_numpy()
    zt = ((test[names] - mean) / std).to_numpy()
    eig, basis = np.linalg.eigh(np.cov(z.T))
    pc_train, pc_test = z @ basis, zt @ basis

    def offdiag(data):
        c = np.corrcoef(data.T)
        return float(np.max(np.abs(c - np.eye(c.shape[0]))))

    def dimension(data):
        c = data.corr(method="spearman").to_numpy()
        return float(np.trace(c) ** 2 / np.trace(c @ c))

    return {
        "status": "ok",
        "features": names,
        "train_n": len(train),
        "test_n": len(test),
        "pca_components_95_train": int(np.searchsorted(np.cumsum(eig[::-1]) / eig.sum(), 0.95) + 1),
        "pca_max_abs_correlation_train": offdiag(pc_train),
        "pca_max_abs_correlation_test": offdiag(pc_test),
        "effective_dimension_train": dimension(train[names]),
        "effective_dimension_test": dimension(test[names]),
        "baseline_effective_dimension_same_train": dimension(train[list(base.columns)]),
        "baseline_effective_dimension_same_test": dimension(test[list(base.columns)]),
    }


def analyze(candidates, raw_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    cutoff = pd.Timestamp(manifest["end"])
    calendar = pd.date_range(manifest["start"], cutoff, freq="ME")
    available = {item["alias"]: item for item in manifest["series"] if item["status"] == "ok"}
    monthly, frames, quarterly = {}, {}, {}
    features, definitions = {}, {}
    for spec in candidates:
        alias = spec["alias"]
        path = raw_dir / (alias + ".csv")
        if alias not in available:
            continue
        if available[alias]["code"] != spec["code"]:
            raise ValueError("El código del manifest no coincide con el candidato.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != available[alias]["sha256"]:
            raise ValueError("El CSV no corresponde al snapshot del manifest.")
        frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
        frames[alias] = frame
        values = frame.value.where(frame.status_code.eq("OK")).dropna().sort_index()
        if spec["frequency"] == "Trimestral":
            q = values.groupby(values.index.to_period("Q")).last()
            q.index = q.index.to_timestamp(how="end").normalize()
            quarterly[alias] = q.loc[:cutoff]
            continue
        # Daily to monthly mean; monthly observations preserve the recorded value.
        agg = values.resample("ME").mean().reindex(calendar)
        if spec["frequency"] == "Diaria" and alias != "eof_financial_conditions":
            agg = agg.where(values.resample("ME").count().reindex(calendar) >= 3)
        monthly[alias] = agg
        # Raw inputs with sign 0 stay visible as context, not votes in a headline index.
        features[alias] = transform(agg, spec["transform"])
        definitions[alias] = {
            "block": spec["block"],
            "transform": spec["transform"],
            "source_codes": [spec["code"]],
            "sign": spec["sign"],
        }
    levels = pd.DataFrame(monthly, index=calendar)
    required = {
        "turnover_value",
        "market_cap",
        "interbank",
        "tpm",
        "nominal10",
        "nominal2",
        "nominal5",
        "real5",
        "m1",
        "m2",
        "eee_fx12",
        "usd",
        "ipsa",
        "price_book",
        "ipec",
        "imce_nonmining",
        "bank_spread",
        "idie",
        "copper",
        "spot_position",
    }
    if not required.issubset(levels):
        raise ValueError("Faltan series necesarias para comparar el panel; revisa el manifest.")

    def derived(name, series, inputs, formula, block):
        features[name] = series
        definitions[name] = {"block": block, "transform": formula, "inputs": inputs, "sign": 0}

    derived(
        "turnover",
        levels.turnover_value / levels.market_cap.where(levels.market_cap > 0),
        ["turnover_value", "market_cap"],
        "monto/capitalizacion",
        "participacion",
    )
    derived(
        "funding_gap",
        levels.interbank - levels.tpm,
        ["interbank", "tpm"],
        "interbancaria-TPM",
        "fondeo",
    )
    derived(
        "funding_deviation",
        (levels.interbank - levels.tpm).abs(),
        ["interbank", "tpm"],
        "abs(interbancaria-TPM)",
        "fondeo",
    )
    derived(
        "term_slope",
        levels.nominal10 - levels.nominal2,
        ["nominal10", "nominal2"],
        "nominal10-nominal2",
        "curva",
    )
    derived(
        "inflation_compensation5",
        levels.nominal5 - levels.real5,
        ["nominal5", "real5"],
        "nominal5-real5",
        "inflacion_mercado",
    )
    derived(
        "liquid_share",
        levels.m1 / levels.m2.where(levels.m2 > 0),
        ["m1", "m2"],
        "M1/M2",
        "composicion_dinero",
    )
    derived(
        "fx_expectation_gap",
        np.log(levels.eee_fx12 / levels.usd),
        ["eee_fx12", "usd"],
        "ln(EEE11meses/USDmediaMes)",
        "expectativas_cambio",
    )
    # Same-date yield changes first; no substitution of a monthly rate change for volatility.
    rates = frames["nominal10"].value.dropna()
    changes = rates.diff().where(rates.index.to_series().diff().dt.days.le(7))
    vol = changes.resample("ME").std(ddof=0).reindex(calendar)
    vol = vol.where(changes.resample("ME").count().reindex(calendar) >= 10)
    derived("rate_volatility", vol, ["nominal10"], "std(cambios diarios pp), min10", "curva")
    if "cpi" in levels:
        derived(
            "real_credit_growth",
            np.log(levels.commercial_credit / levels.cpi).diff(12),
            ["commercial_credit", "cpi"],
            "d12 ln(credito/IPC)",
            "credito_real",
        )
    if "real2" in levels:
        derived(
            "inflation_compensation2",
            levels.nominal2 - levels.real2,
            ["nominal2", "real2"],
            "nominal2-real2",
            "inflacion_mercado",
        )
    raw_features = pd.DataFrame(features)
    scores = raw_features.apply(percentile_past)
    baseline_frames = {
        k: frames[v]
        for k, v in {
            "dolar": "usd",
            "cobre": "copper",
            "tasa_10y": "nominal10",
            "incertidumbre": "idie",
        }.items()
    }
    v2 = calculate_index(baseline_frames)
    base = v2.loc[v2.complete].filter(like="score_").resample("ME").last().reindex(calendar)
    # Retrospective descriptive panel: market and confidence remain separate.
    panel = pd.DataFrame(index=calendar)
    panel["mercado"] = complete_mean(scores, ["ipsa", "price_book"])
    panel["confianza"] = complete_mean(scores, ["ipec", "imce_nonmining"])
    panel["friccion"] = complete_mean(
        scores, ["bank_spread", "funding_deviation", "rate_volatility", "idie"]
    )
    panel["participacion"] = scores.turnover
    panel["credito_real"] = scores.get("real_credit_growth", np.nan)
    panel["composicion_liquida"] = scores.liquid_share
    panel["brecha_mercado_confianza"] = panel.mercado - panel.confianza
    panel["promedio_que_oculta_brecha"] = complete_mean(panel, ["mercado", "confianza"])
    panel["componentes_friccion"] = (
        scores[["bank_spread", "funding_deviation", "rate_volatility", "idie"]].notna().sum(axis=1)
    )
    correlations = {}
    for label, start, end in [
        ("development", "2006-01-01", "2018-12-31"),
        ("temporal_test", "2019-01-01", str(cutoff.date())),
        ("recent", "2023-01-01", str(cutoff.date())),
    ]:
        subset = scores.loc[start:end]
        corr = subset.corr(method="spearman", min_periods=24)
        corr.to_csv(output / ("correlations_" + label + ".csv"))
        pairs = []
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1 :]:
                r = corr.loc[a, b]
                if pd.notna(r) and abs(r) >= 0.8:
                    pairs.append(
                        {
                            "a": a,
                            "b": b,
                            "rho": float(r),
                            "n": int(subset[[a, b]].dropna().shape[0]),
                        }
                    )
        correlations[label] = sorted(pairs, key=lambda p: abs(p["rho"]), reverse=True)
    reconstruction = {col: ridge_reconstruction(base, scores[col]) for col in scores}
    panel_corr = (
        panel[["mercado", "confianza", "friccion", "participacion"]]
        .loc["2019":]
        .corr(method="spearman")
    )
    summary = {
        "candidates": len(candidates),
        "live_ok": sum(r["status"] == "ok" for r in manifest["series"]),
        "monthly_features": len(scores.columns),
        "quarterly_series": len(quarterly),
        "cutoff": str(cutoff.date()),
        "last_complete_month": str(calendar[-1].date()),
        "redundant_pairs": correlations,
        "reconstruction_from_v2": reconstruction,
        "panel_correlations_since_2019": panel_corr.to_dict(),
        "panel_complete_rows": int(
            panel[["mercado", "confianza", "friccion", "participacion"]].dropna().shape[0]
        ),
        "geometry": geometry(scores, base),
        "limitations": [
            "Historia revisada; sin vintages ni publicación reconstruida.",
            "Baja correlación y R2 bajo no demuestran utilidad ni sentimiento.",
            "Panel experimental, pesos convencionales no optimizados.",
        ],
    }
    for name, frame in [
        ("monthly_levels", levels),
        ("features", raw_features),
        ("scores", scores),
        ("baseline_v2", base),
        ("panel", panel),
        ("quarterly", pd.DataFrame(quarterly)),
    ]:
        frame.index.name = "period_end"
        frame.to_csv(output / (name + ".csv"), date_format="%Y-%m-%d")
    credit_distribution(pd.DataFrame(quarterly)).to_csv(
        output / "credit_distribution.csv", date_format="%Y-%m-%d"
    )
    (output / "definitions.json").write_text(
        json.dumps(definitions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidatos", type=Path, default=Path("research/candidates.json"))
    parser.add_argument("--datos", type=Path, default=Path("outputs/research/raw"))
    parser.add_argument("--salida", type=Path, default=Path("outputs/research/analysis"))
    args = parser.parse_args()
    result = analyze(
        json.loads(args.candidatos.read_text(encoding="utf-8")), args.datos, args.salida
    )
    print("Candidatos:", result["candidates"], "Series OK:", result["live_ok"])
