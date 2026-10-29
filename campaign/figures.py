"""Render two MyM figures from saved campaign results; never refit or edit inputs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

NAVY, ORANGE, TEAL = "#10313b", "#f2a367", "#237d85"
MUTED, GRID, MISSING = "#536970", "#dde5e7", "#e8edef"
METHODS = {
    "v2": "Índice v2",
    "equal_family": "Promedio\npor familias",
    "dt905_adapted": "DT905\nadaptado",
    "minimal3": "Panel de\n3 familias",
    "panel6": "Panel de\n6 familias",
}
TARGETS = {
    "ecb_supply_sme": "Oferta · pymes",
    "ecb_supply_large": "Oferta · empresas grandes",
    "ecb_supply_households": "Oferta · hogares (consumo)",
    "ecb_demand_sme": "Demanda · pymes",
    "ecb_demand_large": "Demanda · empresas grandes",
    "ecb_demand_households": "Demanda · hogares (consumo)",
}
FAMILIES = {"market": "Mercado", "confidence": "Confianza", "friction": "Presiones"}


def _number(value, signed=False):
    return (f"{value:+.1f}" if signed else f"{value:.1f}").replace(".", ",")


def _quarter(value):
    stamp = pd.Timestamp(value)
    return f"T{stamp.quarter} {stamp.year}"


def _clean(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=MUTED, labelsize=10, pad=8)


def _header(fig, title, subtitle):
    fig.text(
        0.055, 0.95, "MyM  /  INVESTIGACIÓN ABIERTA", color=TEAL, fontsize=11, fontweight="bold"
    )
    fig.text(0.055, 0.90, title, color=NAVY, fontsize=22, fontweight="bold")
    fig.text(0.055, 0.855, subtitle, color=MUTED, fontsize=11)


def _save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        path = output / f"{name}.{extension}"
        fig.savefig(
            path,
            dpi=180,
            facecolor="white",
            metadata={"Date": None} if extension == "svg" else None,
        )
        if extension == "svg":
            clean = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(clean + "\n", encoding="utf-8", newline="\n")
        print(path)
    plt.close(fig)


def comparison(metrics, output):
    """Each cell contains relative MAE reduction and its paired exploratory interval."""
    values = np.full((len(TARGETS), len(METHODS)), np.nan)
    row_labels, intervals = [], {}
    starts, ends = [], []
    for row, (target, label) in enumerate(TARGETS.items()):
        result = metrics["targets"].get(target, {})
        train, test = result.get("train", {}), result.get("test", {})
        row_labels.append(f"{label}\nn={train.get('n', 0)} ajuste / {test.get('n', 0)} contraste")
        if test.get("start") and test.get("end"):
            starts.append(test["start"])
            ends.append(test["end"])
        for col, method in enumerate(METHODS):
            gain = result.get("models", {}).get(method, {}).get("gain_vs_controls", {})
            value = gain.get("relative_mae_gain")
            values[row, col] = np.nan if value is None else 100 * value
            intervals[(row, col)] = gain.get("ci95")
    finite = np.abs(values[np.isfinite(values)])
    limit = max(10, float(np.ceil(finite.max() / 10) * 10)) if len(finite) else 10
    cmap = LinearSegmentedColormap.from_list("mym_gain", [ORANGE, "white", TEAL])
    cmap.set_bad(MISSING)
    fig, ax = plt.subplots(figsize=(15, 9.4))
    fig.subplots_adjust(left=0.26, right=0.96, top=0.76, bottom=0.25)
    mesh = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(METHODS)), list(METHODS.values()), fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(TARGETS)), row_labels, fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(METHODS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(TARGETS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    _clean(ax)
    for (row, col), interval in intervals.items():
        value = values[row, col]
        color = "white" if np.isfinite(value) and value > 0.64 * limit else NAVY
        estimate = "Sin estimación" if not np.isfinite(value) else _number(value, True) + " %"
        ci = (
            f"IC [{_number(100 * interval[0], True)}; {_number(100 * interval[1], True)}]"
            if interval
            else "IC no disponible"
        )
        ax.text(
            col,
            row - 0.12,
            estimate,
            ha="center",
            va="center",
            color=color,
            fontsize=14,
            fontweight="bold",
        )
        ax.text(col, row + 0.20, ci, ha="center", va="center", color=color, fontsize=9)
    cax = fig.add_axes([0.39, 0.17, 0.46, 0.018])
    bar = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[-limit, 0, limit])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=9)
    bar.set_label("Reducción relativa del MAE (%) · positivo = menor error", fontsize=10)
    period = f"{_quarter(min(starts))}–{_quarter(max(ends))}" if starts else "sin muestra evaluable"
    _header(
        fig,
        "Qué aporta cada representación frente a controles",
        f"Encuesta de Crédito Bancario · Contraste {period} · Datos: BDE del BCCh",
    )
    bootstrap = metrics["bootstrap"]
    fig.text(
        0.055,
        0.105,
        f"Celdas: estimación e IC 95 % exploratorio, en %. "
        f"Bootstrap pareado: {bootstrap['draws']} remuestreos, "
        f"bloques de {bootstrap['block_quarters']} trimestres. Sin ajuste por multiplicidad.",
        color=MUTED,
        fontsize=9.5,
    )
    fig.text(
        0.055,
        0.07,
        "Controles: actividad (Imacec), inflación (IPC) y TPM. "
        "Misma muestra por objetivo para las cinco representaciones; no acredita dominancia.",
        color=MUTED,
        fontsize=9.5,
    )
    fig.text(
        0.055,
        0.035,
        "Análisis retrospectivo con historia revisada. "
        "Los resultados describen asociaciones; no son recomendaciones de inversión.",
        color=MUTED,
        fontsize=9.5,
    )
    _save(fig, output, "closure-comparison")


def stability(results, envelopes, output):
    """Separate sensitivity from unavailable history using the full calendar denominator."""
    stats = results["stable_labels"]
    envelopes = envelopes.copy()
    envelopes["date"] = pd.to_datetime(envelopes["date"])
    rank_columns = [column for column in envelopes if column.startswith("rank")]
    if not rank_columns or envelopes.empty:
        raise ValueError("No normalization envelopes are available.")
    first, last = envelopes.date.min(), envelopes.date.max()
    calendar = pd.date_range(first, last, freq="ME")
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_axes([0.18, 0.51, 0.65, 0.27])
    keys = list(FAMILIES)
    stable, sensitive, absent = [], [], []
    for family in keys:
        item = stats[family]
        n, complete, kept = (
            item[k] for k in ("denominator_all_test_months", "complete_windows_n", "stable_n")
        )
        if n != len(calendar) or not 0 <= kept <= complete <= n:
            raise ValueError("Stable-label counts do not match the envelope calendar.")
        stable.append(100 * kept / n)
        sensitive.append(100 * (complete - kept) / n)
        absent.append(100 * (n - complete) / n)
    y = np.arange(len(keys))
    ax.barh(y, stable, color=TEAL, height=0.53, label="Etiqueta estable")
    ax.barh(y, sensitive, left=stable, color=ORANGE, height=0.53, label="Etiqueta sensible")
    ax.barh(
        y,
        absent,
        left=np.array(stable) + sensitive,
        color=MISSING,
        height=0.53,
        label="Historia incompleta",
    )
    for row, family in enumerate(keys):
        item = stats[family]
        ax.text(
            103,
            row - 0.06,
            _number(stable[row]) + " %",
            color=NAVY,
            fontsize=17,
            fontweight="bold",
            va="center",
        )
        ax.text(
            103,
            row + 0.24,
            f"{item['stable_n']} de {item['denominator_all_test_months']} meses",
            color=MUTED,
            fontsize=9,
            va="center",
        )
        left = 0
        counts = [
            item["stable_n"],
            item["complete_windows_n"] - item["stable_n"],
            item["denominator_all_test_months"] - item["complete_windows_n"],
        ]
        for width, count, color in zip(
            [stable[row], sensitive[row], absent[row]], counts, ["white", NAVY, NAVY]
        ):
            if width >= 6:
                ax.text(left + width / 2, row, str(count), color=color, ha="center", va="center")
            left += width
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_yticks(y, list(FAMILIES.values()))
    ax.invert_yaxis()
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=3,
        frameon=False,
        fontsize=10,
        borderaxespad=0,
    )
    _clean(ax)
    widths = envelopes["high"] - envelopes["low"]
    limit = max(10, float(np.ceil(widths.max() / 10) * 10)) if widths.notna().any() else 10
    for column, (family, label) in enumerate(FAMILIES.items()):
        chart = fig.add_axes([0.075 + column * 0.31, 0.20, 0.27, 0.19])
        frame = envelopes.loc[envelopes.family.eq(family)].set_index("date").reindex(calendar)
        width = (frame.high - frame.low).where(frame[rank_columns].notna().all(axis=1))
        chart.plot(calendar, width, color=NAVY, linewidth=1.5)
        chart.fill_between(calendar, 0, width, color=TEAL, alpha=0.12)
        chart.set_title(label, loc="left", color=NAVY, fontsize=11, fontweight="bold")
        chart.set_xlim(first, last)
        chart.set_ylim(0, limit)
        chart.xaxis.set_major_locator(mdates.YearLocator(2))
        chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        chart.grid(axis="y", color=GRID, linewidth=0.6)
        if column == 0:
            chart.set_ylabel("Amplitud · puntos", color=MUTED, fontsize=10)
        _clean(chart)
    period = f"{first:%m/%Y}–{last:%m/%Y}"
    _header(
        fig,
        "Cuándo la etiqueta resiste cambiar la referencia",
        f"Cobertura sobre todos los meses de {period} · Datos: BDE del BCCh",
    )
    horizons = ", ".join(str(int(c[4:])) for c in rank_columns)
    fig.text(
        0.075,
        0.435,
        f"Amplitud entre los puntajes con referencias de {horizons} meses",
        color=NAVY,
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.115,
        "Las cifras dentro de las barras son meses. "
        "La historia incompleta cuenta como abstención en el denominador; "
        "los tramos sin comparación permanecen vacíos.",
        color=MUTED,
        fontsize=9.5,
    )
    fig.text(
        0.055,
        0.075,
        "Estabilidad de una descripción, no intervalo de confianza ni prueba de "
        "exactitud económica. La regla de etiquetado es la registrada en el contrato.",
        color=MUTED,
        fontsize=9.5,
    )
    fig.text(
        0.055,
        0.035,
        "Análisis retrospectivo con historia revisada. "
        "Este panel no anticipa retornos ni ofrece recomendaciones de inversión.",
        color=MUTED,
        fontsize=9.5,
    )
    _save(fig, output, "closure-stability")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/campaign"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    results = json.loads((args.results / "results.json").read_text(encoding="utf-8"))
    economics = json.loads((args.results / "economics_metrics.json").read_text(encoding="utf-8"))
    envelopes = pd.read_csv(args.results / "envelopes.csv")
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "text.color": NAVY,
            "svg.fonttype": "none",
            "svg.hashsalt": "mym-bcch-closure",
        }
    ):
        comparison(economics, args.output)
        stability(results, envelopes, args.output)


if __name__ == "__main__":
    main()
