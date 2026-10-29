"""Acceso mediante bcch-sdk, cliente no oficial, y validación de observaciones BDE."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, quote_plus

import numpy as np
import pandas as pd
import requests
from bcch import BancoCentralDeChile
from bcch.exceptions import BCCHAPIError, BCCHError
from dotenv import load_dotenv

from .config import SERIES


class DataError(ValueError):
    """Datos insuficientes o incompatibles con la metodología."""


class SecretFilter(logging.Filter):
    """Elimina el token de mensajes HTTP, incluso con logging DEBUG habilitado."""

    def __init__(self, secret: str):
        super().__init__()
        self._variants = {secret, quote(secret, safe=""), quote_plus(secret)}

    def filter(self, record):
        message = record.getMessage()
        for value in sorted(self._variants, key=len, reverse=True):
            if value:
                message = message.replace(value, "<redacted>")
        record.msg, record.args = message, ()
        # HTTP exceptions can carry authenticated URLs in their traceback.
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


class NoRedirectSession(requests.Session):
    """El endpoint es fijo: no reenviar consultas autenticadas por redirecciones."""

    def get(self, url, **kwargs):
        kwargs["allow_redirects"] = False
        return super().get(url, **kwargs)


@contextmanager
def configured_client(env_file: Path = Path(".env")):
    # Solo el archivo indicado; no buscar .env en carpetas superiores.
    load_dotenv(env_file, override=False, interpolate=False)
    key = os.environ.get("BCCH_API_KEY", "").strip()
    if not key:
        raise DataError("Falta BCCH_API_KEY. Configúrala en el entorno o en .env.")
    guard = SecretFilter(key)
    loggers = [
        logging.getLogger(name)
        for name in (
            "urllib3.connectionpool",
            "requests.packages.urllib3.connectionpool",
            "requests",
        )
    ]
    for logger in loggers:
        logger.addFilter(guard)
    try:
        with NoRedirectSession() as session:
            yield BancoCentralDeChile(token=key, session=session, timeout=30, max_retries=2)
    finally:
        for logger in loggers:
            logger.removeFilter(guard)


def observations_frame(result, *, code: str, alias: str) -> pd.DataFrame:
    if result.series_id != code:
        raise DataError(f"La API devolvió un código distinto al solicitado para {alias}.")
    frame = result.to_dataframe().copy()
    if frame.empty:
        raise DataError(f"La serie {alias} no contiene observaciones.")
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    if frame.index.has_duplicates or frame.index.hasnans:
        raise DataError(f"La serie {alias} contiene fechas duplicadas o inválidas.")
    values = pd.to_numeric(frame["value"], errors="coerce")
    # Un valor con estado ND/otro no se considera observado, aunque sea numérico.
    frame["value"] = values.where(frame["status_code"].eq("OK") & np.isfinite(values))
    frame = frame.sort_index()
    if frame["value"].notna().sum() == 0:
        raise DataError(f"La serie {alias} no tiene observaciones válidas con estado OK.")
    if alias in {"dolar", "cobre"} and frame["value"].dropna().le(0).any():
        raise DataError(f"La serie {alias} debe contener precios positivos.")
    if alias == "incertidumbre" and frame["value"].dropna().lt(0).any():
        raise DataError("El índice de incertidumbre no puede ser negativo.")
    return frame


def fetch_series(client, *, start: str, end: str):
    frames, metadata = {}, {}
    for alias, spec in SERIES.items():
        try:
            result = client.get_series(spec.code, start=start, end=end)
        except BCCHAPIError as exc:
            raise DataError(f"BCCh no entregó {alias} ({spec.code}); código {exc.code}.") from None
        except BCCHError:
            raise DataError(
                f"No fue posible consultar {alias}. Revisa conexión y acceso BDE."
            ) from None
        frame = observations_frame(result, code=spec.code, alias=alias)
        frame = frame.loc[start:end]
        valid = frame["value"].dropna()
        if valid.empty:
            raise DataError(f"Sin datos de {alias} en el intervalo solicitado.")
        frames[alias] = frame
        # Lista explícita de campos. Nunca guardar result.raw, URLs ni credenciales.
        metadata[alias] = {
            "series_id": spec.code,
            "label": spec.label,
            "unit": spec.unit,
            "first_observation": valid.index[0].date().isoformat(),
            "last_observation": valid.index[-1].date().isoformat(),
            "valid_observations": int(len(valid)),
            "missing_observations": int(frame["value"].isna().sum()),
            "max_age_days": spec.max_age_days,
        }
    return frames, metadata
