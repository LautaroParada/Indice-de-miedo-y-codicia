"""Sensibilidad descriptiva de las ventanas y bloques; no es un backtest financiero."""

import argparse
import json
from pathlib import Path

import pandas as pd

from imc.config import SERIES, Methodology
from imc.engine import calculate_index


def evaluate(source: Path):
    frames = {
        alias: pd.read_csv(source / f"{alias}.csv", parse_dates=["date"], index_col="date")
        for alias in SERIES
    }
    baseline = calculate_index(frames)
    output = {
        "method": "Sensibilidad descriptiva; no mide rentabilidad ni capacidad predictiva.",
        "windows": {},
        "leave_one_block_out": {},
    }
    for window in [252, 504, 756]:
        alternative = calculate_index(frames, Methodology(normalization_window=window))
        pair = pd.concat(
            [baseline["index"], alternative["index"]], axis=1, keys=["base", "alternative"]
        ).dropna()
        diff = pair.alternative - pair.base
        output["windows"][str(window)] = {
            "common_rows": len(pair),
            "mean_absolute_difference": float(diff.abs().mean()),
            "maximum_absolute_difference": float(diff.abs().max()),
            "correlation": float(pair.base.corr(pair.alternative)),
            "latest_value": float(pair.alternative.iloc[-1]),
        }
    blocks = {
        "cambiario": ["presion_cambiaria", "volatilidad_cambiaria"],
        "cobre": ["momentum_cobre"],
        "tasas": ["variabilidad_tasas"],
        "incertidumbre": ["incertidumbre"],
    }
    complete = baseline.loc[baseline.complete]
    for block, factors in blocks.items():
        contribution = sum(complete[f"contribution_{factor}"] for factor in factors)
        alternative = (complete["index"] - contribution) / 0.75
        diff = alternative - complete["index"]
        output["leave_one_block_out"][block] = {
            "mean_absolute_difference": float(diff.abs().mean()),
            "maximum_absolute_difference": float(diff.abs().max()),
            "latest_value": float(alternative.iloc[-1]),
        }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series_dir", type=Path)
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(
        json.dumps(evaluate(args.series_dir), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
