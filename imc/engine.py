"""Cálculo puro: fechas de observación, ventanas retrospectivas y cobertura explícita."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FACTOR_SOURCE, SERIES, WEIGHTS, Methodology
from .data import DataError


def rolling_percentile(values: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    """Percentil con rango medio para empates; solo se usa el pasado y el día actual."""
    if not 2 <= min_periods <= window:
        raise ValueError("Se requiere 2 <= min_periods <= window.")

    def percentile(sample):
        current = sample[-1]
        if not np.isfinite(current):
            return np.nan
        sample = sample[np.isfinite(sample)]
        return 100 * (np.sum(sample < current) + 0.5 * np.sum(sample == current)) / len(sample)

    return values.rolling(window, min_periods=min_periods).apply(percentile, raw=True)


def _valid_series(frame: pd.DataFrame, alias: str) -> pd.Series:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is not None:
        raise DataError(f"{alias}: se requiere DatetimeIndex sin zona horaria.")
    if frame.index.has_duplicates or frame.index.hasnans:
        raise DataError(f"{alias}: fechas duplicadas o inválidas.")
    values = pd.to_numeric(frame["value"], errors="coerce")
    if "status_code" in frame:
        values = values.where(frame["status_code"].eq("OK"))
    values = values.where(np.isfinite(values)).dropna().sort_index()
    if values.empty:
        raise DataError(f"{alias}: no hay observaciones válidas.")
    if alias in {"dolar", "cobre"} and values.le(0).any():
        raise DataError(f"{alias}: los precios deben ser positivos.")
    if alias == "incertidumbre" and values.lt(0).any():
        raise DataError("incertidumbre: se requieren valores no negativos.")
    return values.astype(float)


def _raw_factors(values: dict[str, pd.Series], config: Methodology):
    h = config.horizon
    result = {}
    for alias, series in values.items():
        # Un corte largo reinicia las ventanas, evitando comprimir meses sin datos
        # en una variación supuestamente diaria o de 20 observaciones.
        segments = series.index.to_series().diff().dt.days.gt(config.max_source_gap_days).cumsum()
        pieces = []
        for _, part in series.groupby(segments):
            if alias == "dolar":
                returns = np.log(part).diff()
                piece = pd.DataFrame(
                    {
                        "presion_cambiaria": np.log(part / part.shift(h)),
                        "volatilidad_cambiaria": returns.rolling(h, min_periods=h).std(ddof=0),
                    }
                )
            elif alias == "cobre":
                piece = pd.DataFrame({"momentum_cobre": np.log(part / part.shift(h))})
            elif alias == "tasa_10y":
                piece = pd.DataFrame(
                    {"variabilidad_tasas": part.diff().rolling(h, min_periods=h).std(ddof=0)}
                )
            else:
                piece = pd.DataFrame({"incertidumbre": part.rolling(h, min_periods=h).mean()})
            pieces.append(piece)
        result[alias] = pd.concat(pieces).sort_index()
    return result


def calculate_index(frames: dict[str, pd.DataFrame], config: Methodology | None = None):
    config = config or Methodology()
    if set(frames) != set(SERIES):
        raise DataError("Se requieren exactamente las cuatro series de la metodología v2.")
    values = {alias: _valid_series(frame, alias) for alias, frame in frames.items()}
    raw = _raw_factors(values, config)
    # El calendario lo fija el dólar realmente observado, no días artificiales rellenados.
    result = pd.DataFrame(index=values["dolar"].index)
    result.index.name = "date"
    for alias, source in raw.items():
        source = source.copy()
        segments = source.index.to_series().diff().dt.days.gt(config.max_source_gap_days).cumsum()
        for factor in source.columns:
            percentile = (
                source[factor]
                .groupby(segments)
                .transform(
                    lambda values: rolling_percentile(
                        values, window=config.normalization_window, min_periods=config.min_history
                    )
                )
            )
            source[f"score_{factor}"] = (
                percentile if factor == "momentum_cobre" else 100 - percentile
            )
        source[f"value_{alias}"] = values[alias]
        source[f"date_{alias}"] = source.index
        # Se conserva un factor NaN en una fecha observada. No buscar uno anterior válido.
        aligned = pd.merge_asof(
            pd.DataFrame(index=result.index),
            source,
            left_index=True,
            right_index=True,
            direction="backward",
            tolerance=pd.Timedelta(days=SERIES[alias].max_age_days),
        )
        result = result.join(aligned)
        result[f"age_{alias}_days"] = (result.index.to_series() - result[f"date_{alias}"]).dt.days

    factors = list(WEIGHTS)
    scores = [f"score_{factor}" for factor in factors]
    result["available_components"] = result[scores].notna().sum(axis=1)
    # La suma no repondera componentes ausentes ni los convierte en cero.
    for factor, weight in WEIGHTS.items():
        result[f"contribution_{factor}"] = result[f"score_{factor}"] * weight
    contributions = [f"contribution_{factor}" for factor in factors]
    result["index"] = result[contributions].sum(axis=1, min_count=len(factors))
    result["complete"] = result["index"].notna()
    result["oldest_input_date"] = result[[f"date_{x}" for x in SERIES]].min(axis=1)
    return result


def latest_complete(result: pd.DataFrame, *, as_of: str, max_age_days: int = 7):
    cutoff = pd.Timestamp(as_of)
    result = result.loc[:cutoff]
    if result.empty or not result["complete"].any():
        raise DataError("No hay un índice completo: falta historia o cobertura de alguna serie.")
    latest = result.loc[result["complete"]].iloc[-1]
    # Verifica también la fuente más antigua, no solo la etiqueta temporal del índice.
    if (cutoff - latest.name).days > max_age_days or (
        cutoff - latest["oldest_input_date"]
    ).days > max_age_days:
        raise DataError("Los datos disponibles están desactualizados para la fecha de corte.")
    missing = [factor for factor in WEIGHTS if pd.isna(result.iloc[-1][f"score_{factor}"])]
    return latest, sorted({FACTOR_SOURCE[factor] for factor in missing})
