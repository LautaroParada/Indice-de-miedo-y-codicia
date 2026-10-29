"""Metodología explícita, sin credenciales ni acceso a la red."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesSpec:
    code: str
    label: str
    unit: str
    max_age_days: int


SERIES = {
    "dolar": SeriesSpec("F073.TCO.PRE.Z.D", "Dólar observado", "CLP/USD", 0),
    "cobre": SeriesSpec("F019.PPB.PRE.100.D", "Cobre refinado BML", "USD/libra", 3),
    "tasa_10y": SeriesSpec(
        "F022.BCLP.TIS.AN10.NO.Z.D", "Bonos en pesos a 10 años (BCP/BTP)", "% anual", 3
    ),
    "incertidumbre": SeriesSpec(
        "F029.IDIE.IND.TOT.D", "Índice diario de incertidumbre económica", "índice", 7
    ),
}

# Cuatro bloques de 25%. El bloque cambiario se divide entre dos medidas.
WEIGHTS = {
    "presion_cambiaria": 0.125,
    "volatilidad_cambiaria": 0.125,
    "momentum_cobre": 0.25,
    "variabilidad_tasas": 0.25,
    "incertidumbre": 0.25,
}

LABELS = {
    "presion_cambiaria": "Presión cambiaria",
    "volatilidad_cambiaria": "Volatilidad cambiaria",
    "momentum_cobre": "Variación del cobre",
    "variabilidad_tasas": "Variabilidad de tasas",
    "incertidumbre": "Incertidumbre económica",
}

FACTOR_SOURCE = {
    "presion_cambiaria": "dolar",
    "volatilidad_cambiaria": "dolar",
    "momentum_cobre": "cobre",
    "variabilidad_tasas": "tasa_10y",
    "incertidumbre": "incertidumbre",
}


@dataclass(frozen=True)
class Methodology:
    horizon: int = 20
    normalization_window: int = 504
    min_history: int = 252
    max_source_gap_days: int = 31

    def __post_init__(self):
        for name in ("horizon", "normalization_window", "min_history", "max_source_gap_days"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} debe ser un entero de al menos 2.")
        if self.min_history > self.normalization_window:
            raise ValueError("min_history no puede superar normalization_window.")
