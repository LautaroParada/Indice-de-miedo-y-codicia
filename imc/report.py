"""Salidas reproducibles con los componentes, antigüedad y fuentes del cálculo."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import __version__
from .config import LABELS, WEIGHTS

NAVY = "#10313b"
ORANGE = "#f2a367"
TEAL = "#237d85"


def plot_index(result: pd.DataFrame, destination: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none"}):
        figure, axes = plt.subplots(
            2, 1, figsize=(12, 7), height_ratios=[2, 1], layout="constrained"
        )
        ax, detail = axes
        ax.plot(result.index, result["index"], color=NAVY, linewidth=1.4)
        ax.axhline(50, color="#91a5ab", linewidth=0.8, linestyle="--")
        ax.set(ylim=(0, 100), ylabel="Puntaje descriptivo (0–100)")
        ax.set_title(
            "Condiciones de mercado de Chile",
            loc="left",
            fontsize=19,
            color=NAVY,
            fontweight="bold",
            pad=17,
        )
        locator = AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        ax.grid(axis="y", color="#dce4e7", linewidth=0.6)
        latest = result.loc[result["complete"]].iloc[-1]
        factors = list(WEIGHTS)
        detail.barh(
            [LABELS[factor] for factor in factors],
            [latest[f"score_{factor}"] for factor in factors],
            color=TEAL,
            height=0.6,
        )
        detail.set(xlim=(0, 100), xlabel="Componentes normalizados: mayor puntaje, menor presión")
        detail.invert_yaxis()
        detail.set_title(
            f"Último índice completo: {latest.name.date()} · {latest['index']:.1f}/100",
            loc="left",
            color=NAVY,
            fontsize=11,
        )
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
        figure.supxlabel(
            "MyM Ingenierías · Fuente: Banco Central de Chile (BDE) · Metodología v2\n"
            "Datos históricos revisables. Sin pronósticos ni recomendaciones de inversión.",
            fontsize=9,
            color="#536c73",
        )
        for extension in ("svg", "png"):
            figure.savefig(destination.with_suffix(f".{extension}"), dpi=160)
        plt.close(figure)


def export_run(result, frames, sources, config, *, output: Path, start: str, end: str, plot=True):
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / "series"
    raw_dir.mkdir(exist_ok=True)
    hashes = {}
    for alias, frame in frames.items():
        path = raw_dir / f"{alias}.csv"
        frame.to_csv(path, date_format="%Y-%m-%d", float_format="%.15g")
        hashes[f"series/{alias}.csv"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = output / "indice.csv"
    result.to_csv(path, date_format="%Y-%m-%d", float_format="%.15g")
    hashes["indice.csv"] = hashlib.sha256(path.read_bytes()).hexdigest()
    valid = result.loc[result["complete"]]
    latest = valid.iloc[-1]
    metadata = {
        "methodology_version": __version__,
        "methodology": asdict(config),
        "weights": WEIGHTS,
        "requested_start": start,
        "requested_end": end,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "sources": sources,
        "complete_rows": len(valid),
        "total_rows": len(result),
        "latest_complete_date": latest.name.date().isoformat(),
        "latest_value": float(latest["index"]),
        "latest_source_dates": {
            alias: latest[f"date_{alias}"].date().isoformat() for alias in frames
        },
        "sha256": hashes,
        "interpretation": "Mayor puntaje: condiciones relativamente más distendidas en la muestra.",
        "limitations": [
            "No mide directamente emociones, valor justo ni rendimientos futuros.",
            "Las fechas son de observación, no de publicación; no es una base point-in-time.",
            "Los pesos, ventanas y orientaciones son decisiones de diseño no calibradas.",
        ],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if plot:
        plot_index(result, output / "indice")
