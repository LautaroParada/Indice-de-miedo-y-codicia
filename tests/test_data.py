import logging
from datetime import date
from urllib.parse import quote_plus

import pandas as pd
import pytest
from bcch import Observation, SeriesResult
from bcch.exceptions import BCCHAPIError

from imc.config import SERIES
from imc.data import DataError, SecretFilter, configured_client, fetch_series, observations_frame


def payload(*, code="F073.TCO.PRE.Z.D", observations=None):
    return SeriesResult(
        code,
        "Prueba",
        "Test",
        tuple(
            observations
            or [
                Observation(date(2026, 1, 2), 900.0, "OK"),
                Observation(date(2026, 1, 3), 901.0, "ND"),
                Observation(date(2026, 1, 4), float("inf"), "OK"),
            ]
        ),
        {},
    )


def test_nd_and_nonfinite_are_missing_not_valid_prices():
    frame = observations_frame(payload(), code=SERIES["dolar"].code, alias="dolar")
    assert frame["value"].notna().sum() == 1
    assert frame.iloc[1].status_code == "ND"
    assert pd.isna(frame.iloc[1]["value"])


def test_returned_series_must_match_request():
    with pytest.raises(DataError, match="distinto"):
        observations_frame(payload(code="WRONG"), code=SERIES["dolar"].code, alias="dolar")


def test_missing_key_fails_without_network(monkeypatch, tmp_path):
    monkeypatch.delenv("BCCH_API_KEY", raising=False)
    with pytest.raises(DataError, match="BCCH_API_KEY"):
        with configured_client(tmp_path / "absent.env"):
            pytest.fail("Client should not be created")


def test_env_precedence_and_no_interpolation(monkeypatch, tmp_path):
    from imc import data

    captured = []
    monkeypatch.setattr(data, "BancoCentralDeChile", lambda **kw: captured.append(kw["token"]))
    env = tmp_path / ".env"
    env.write_text("BCCH_API_KEY='local-${MISSING}-placeholder'\n", encoding="utf-8")
    monkeypatch.setenv("BCCH_API_KEY", "environment-placeholder")
    with configured_client(env):
        pass
    assert captured.pop() == "environment-placeholder"
    monkeypatch.delenv("BCCH_API_KEY")
    with configured_client(env):
        pass
    assert captured.pop() == "local-${MISSING}-placeholder"


def test_debug_http_logs_redact_raw_and_encoded_token():
    secret = "example token+/=?"
    record = logging.LogRecord(
        "urllib3.connectionpool",
        logging.DEBUG,
        "",
        1,
        "GET %s %s",
        (secret, quote_plus(secret)),
        None,
    )
    SecretFilter(secret).filter(record)
    assert secret not in record.getMessage()
    assert quote_plus(secret) not in record.getMessage()


def test_api_failure_never_echoes_response_or_credentials():
    class Failure:
        def get_series(self, *args, **kwargs):
            raise BCCHAPIError(-50, "URL?token=example-secret")

    with pytest.raises(DataError) as error:
        fetch_series(Failure(), start="2020-01-01", end="2026-01-01")
    assert "example-secret" not in str(error.value)
    assert "-50" in str(error.value)


def test_four_series_are_each_requested_once_and_metadata_is_allowlisted():
    calls = []

    class Client:
        def get_series(self, code, **kwargs):
            calls.append((code, kwargs))
            return payload(code=code)

    frames, metadata = fetch_series(Client(), start="2026-01-01", end="2026-01-04")
    assert len(calls) == len(set(code for code, _ in calls)) == 4
    assert set(frames) == set(SERIES)
    assert all("raw" not in item and "token" not in item for item in metadata.values())
