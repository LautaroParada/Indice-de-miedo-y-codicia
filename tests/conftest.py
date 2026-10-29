import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def frames():
    dates = pd.bdate_range("2020-01-02", periods=800)
    t = np.arange(len(dates))
    rng = np.random.default_rng(2026)
    values = {
        "dolar": 750 * np.exp(np.cumsum(rng.normal(0, 0.004, len(t)))),
        "cobre": 3 * np.exp(np.cumsum(rng.normal(0.0001, 0.006, len(t)))),
        "tasa_10y": 4 + np.sin(t / 50) + rng.normal(0, 0.08, len(t)),
        "incertidumbre": 100 + 20 * np.sin(t / 60) + rng.uniform(0, 20, len(t)),
    }
    return {
        name: pd.DataFrame({"value": value, "status_code": "OK"}, index=dates)
        for name, value in values.items()
    }
