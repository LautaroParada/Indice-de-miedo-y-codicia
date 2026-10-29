"""Figuras descriptivas reproducibles del panel BDE: python -m research.figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "research" / "analysis"
OUTPUT = ROOT / "docs" / "assets"
NAVY = "#10313b"
ORANGE = "#f2a367"
TEAL = "#237d85"
MUTED = "#536970"
GRID = "#dde5e7"
MONTHS = (
    "ene.",
    "feb.",
    "mar.",
    "abr.",
    "may.",
    "jun.",
    "jul.",
    "ago.",
    "sep.",
    "oct.",
    "nov.",
    "dic.",
)
FACTORS = {
    "ipsa": "IPSA · variación",
    "price_book": "Precio / libro",
    "turnover": "Rotación bursátil",
    "ipec": "Confianza hogares",
    "imce_nonmining": "Confianza empresas",
    "bank_spread": "Spread bancario",
    "funding_deviation": "Desvío de fondeo",
    "idie": "Incertidumbre IDIE",
    "rate_volatility": "Variabilidad de tasas",
    "liquid_share": "M1 / M2",
    "fx_expectation_gap": "Brecha cambiaria EEE",
    "spot_position": "Posición spot",
}


def month_label(date):
    """Mes y año en español sin depender de la configuración regional."""
    return f"{MONTHS[date.month - 1]} {date.year}"


def save_figure(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        path = OUTPUT / f"{name}.{extension}"
        fig.savefig(path, dpi=180, facecolor="white")
        print(path.relative_to(ROOT))
    plt.close(fig)


def correlations():
    scores = pd.read_csv(INPUT / "scores.csv", index_col=0, parse_dates=True)
    sample = scores.loc["2019-01-01":"2026-12-31", list(FACTORS)]
    if sample.empty:
        raise ValueError("No hay puntajes en el periodo 2019–2026.")
    present = sample.notna().astype(int)
    counts = present.T @ present
    corr = sample.corr(method="spearman", min_periods=24)
    lower = np.tril(np.ones(corr.shape, dtype=bool))
    valid = counts.ge(24).to_numpy() & corr.notna().to_numpy()
    mask = ~lower | ~valid
    values = np.ma.array(corr.to_numpy(), mask=mask)
    cmap = LinearSegmentedColormap.from_list("mym_diverging", [NAVY, "#ffffff", ORANGE])
    cmap.set_bad("#f2f5f5")
    fig, ax = plt.subplots(figsize=(12.5, 11.0))
    fig.subplots_adjust(left=0.205, right=0.88, top=0.865, bottom=0.235)
    mesh = ax.imshow(values, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    labels = list(FACTORS.values())
    ax.set_xticks(range(len(labels)), labels, rotation=48, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.tick_params(length=0, pad=8)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(len(labels)):
        for col in range(row + 1):
            n = int(counts.iloc[row, col])
            value = corr.iloc[row, col]
            color = "white" if valid[row, col] and value < -0.52 else NAVY
            text = f"{0 if abs(value) < 0.005 else value:.2f}" if valid[row, col] else "—"
            ax.text(col, row - 0.10, text, color=color, ha="center", va="center", fontsize=8)
            ax.text(col, row + 0.19, f"n={n}", color=color, ha="center", va="center", fontsize=6.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([0.91, 0.36, 0.017, 0.32])
    bar = fig.colorbar(mesh, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    bar.set_label("Correlación de Spearman (ρ)", color=NAVY, fontsize=10, labelpad=10)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=9, length=0)
    unique = np.tril(np.ones(corr.shape, dtype=bool), -1)
    pair_counts = counts.to_numpy()[unique & valid]
    valid_pairs = len(pair_counts)
    possible_pairs = int(unique.sum())
    n_text = f"{pair_counts.min()}–{pair_counts.max()}" if valid_pairs else "sin pares válidos"
    ax.text(
        0.53,
        0.88,
        "12 dimensiones observables",
        transform=ax.transAxes,
        color=NAVY,
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.53,
        0.73,
        f"{valid_pairs} / {possible_pairs} pares estimables\n"
        f"Observaciones por par: {n_text}\n"
        "Cada celda: ρ y meses compartidos\n"
        "Se omiten pares con n < 24",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
        linespacing=1.8,
    )
    fig.text(
        0.06, 0.95, "MyM  /  INVESTIGACIÓN ABIERTA", color=TEAL, fontsize=10, fontweight="bold"
    )
    fig.text(
        0.06,
        0.91,
        "Cuánta información comparten las dimensiones",
        color=NAVY,
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.875,
        f"Puntajes mensuales · {month_label(sample.index.min())}–"
        f"{month_label(sample.index.max())} · Fuente: BDE del Banco Central de Chile",
        color=MUTED,
        fontsize=10,
    )
    fig.text(
        0.06,
        0.055,
        "Una correlación baja no demuestra independencia ni utilidad adicional. "
        "Los puntajes comparan cada serie con su propia historia.",
        color=MUTED,
        fontsize=9,
    )
    fig.text(
        0.06,
        0.033,
        "Historia revisable. Esta matriz describe asociaciones; "
        "no identifica causalidad ni anticipa retornos.",
        color=MUTED,
        fontsize=9,
    )
    save_figure(fig, "research-correlations")


def panel_history():
    panel = pd.read_csv(INPUT / "panel.csv", index_col=0, parse_dates=True)
    columns = ["mercado", "confianza", "friccion", "brecha_mercado_confianza"]
    available = panel[columns].notna().any(axis=1)
    if not available.any():
        raise ValueError("No hay datos válidos para la historia del panel.")
    valid_dates = panel.index[available]
    history = panel.loc[valid_dates.min() : valid_dates.max(), columns]
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8.4), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]}
    )
    fig.subplots_adjust(left=0.085, right=0.965, top=0.79, bottom=0.18, hspace=0.30)
    colors = {"mercado": NAVY, "confianza": TEAL, "friccion": ORANGE}
    labels = {"mercado": "Mercado", "confianza": "Confianza", "friccion": "Fricción financiera"}
    for col, color in colors.items():
        axes[0].plot(history.index, history[col], color=color, linewidth=1.7, label=labels[col])
    axes[0].set_ylim(0, 100)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].set_ylabel("Puntaje · 0 a 100", color=NAVY, labelpad=11)
    axes[0].legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.035),
        ncol=3,
        frameon=False,
        borderaxespad=0,
        columnspacing=2.0,
    )
    axes[0].axhline(50, color=GRID, linewidth=0.8, zorder=0)
    gap = history.brecha_mercado_confianza
    axes[1].fill_between(
        history.index, 0, gap, where=gap.ge(0), interpolate=True, color=ORANGE, alpha=0.38
    )
    axes[1].fill_between(
        history.index, 0, gap, where=gap.lt(0), interpolate=True, color=TEAL, alpha=0.22
    )
    axes[1].plot(history.index, gap, color=NAVY, linewidth=1.35)
    axes[1].axhline(0, color=MUTED, linewidth=0.8)
    axes[1].set_ylim(-100, 100)
    axes[1].set_yticks([-100, -50, 0, 50, 100])
    axes[1].set_ylabel("Brecha · puntos", color=NAVY, labelpad=10)
    axes[1].set_title(
        "Mercado − confianza", loc="left", fontsize=11, fontweight="bold", color=NAVY, pad=12
    )
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlim(history.index.min(), history.index.max())
    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0, pad=8, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(
        0.06, 0.95, "MyM  /  INVESTIGACIÓN ABIERTA", color=TEAL, fontsize=10, fontweight="bold"
    )
    fig.text(
        0.06,
        0.905,
        "Un panel para leer coincidencias y divergencias",
        color=NAVY,
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.865,
        f"Historia mensual · {month_label(history.index.min())}–"
        f"{month_label(history.index.max())} · Fuente: BDE del Banco Central de Chile",
        color=MUTED,
        fontsize=10,
    )
    fig.text(
        0.06,
        0.10,
        "Cada puntaje resume su dimensión: un valor alto de fricción indica "
        "más fricción, no mayor apetito por riesgo.",
        color=MUTED,
        fontsize=9.5,
    )
    fig.text(
        0.06,
        0.075,
        "Brecha positiva: mercado por encima de confianza. "
        "Brecha negativa: confianza por encima de mercado. "
        "No son señales de compra o venta.",
        color=MUTED,
        fontsize=9,
    )
    fig.text(
        0.06,
        0.045,
        "Historia revisable · Sin reconstrucción de disponibilidad en tiempo real. "
        "Los tramos sin datos completos se mantienen vacíos.",
        color=MUTED,
        fontsize=9,
    )
    save_figure(fig, "research-panel")


def main():
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "svg.fonttype": "none",
            "svg.hashsalt": "mym-bcch-research",
            "font.size": 10,
        }
    ):
        correlations()
        panel_history()


if __name__ == "__main__":
    main()
