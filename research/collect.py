"""Consultar candidatos exclusivamente al BCCh, sin exportar URLs ni secretos."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from bcch.exceptions import BCCHAPIError, BCCHError

from imc.data import configured_client, observations_frame


def collect(candidates, output, env_file, start, end):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with configured_client(env_file) as client:
        for spec in candidates:
            record = {k: spec[k] for k in ("alias", "code", "block", "frequency")}
            path = output / (spec["alias"] + ".csv")
            try:
                response = client.get_series(spec["code"], start=start, end=end)
                frame = observations_frame(response, code=spec["code"], alias="candidate")
                frame = frame.loc[start:end]
                valid = frame.value.dropna()
                if valid.empty:
                    record["status"] = "empty"
                else:
                    frame.to_csv(path, date_format="%Y-%m-%d", float_format="%.15g")
                    record.update(
                        status="ok",
                        observations=len(valid),
                        first=valid.index[0].date().isoformat(),
                        last=valid.index[-1].date().isoformat(),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
            except BCCHAPIError as exc:
                record.update(status="api_error", api_code=exc.code)
            except (BCCHError, ValueError):
                record["status"] = "unavailable_or_invalid"
            records.append(record)
            # Only controlled identifiers/status, never response text or exception URLs.
            print(record["alias"], record["status"], flush=True)
    manifest = {
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "start": start,
        "end": end,
        "provider": "Banco Central de Chile",
        "series": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidatos", type=Path, default=Path("research/candidates.json"))
    parser.add_argument("--salida", type=Path, default=Path("outputs/research/raw"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--inicio", default="2006-01-01")
    parser.add_argument("--fin", default="2026-10-28")
    args = parser.parse_args()
    collect(
        json.loads(args.candidatos.read_text(encoding="utf-8")),
        args.salida,
        args.env_file,
        args.inicio,
        args.fin,
    )
