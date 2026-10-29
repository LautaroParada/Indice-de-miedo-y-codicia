"""Register BCCh snapshot captures locally; these are not official publication dates."""

import argparse
import csv
import hashlib
import io
import json
import math
import sqlite3
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path

EXPECTED = {
    row["alias"]: row["code"]
    for row in json.loads(
        (Path(__file__).resolve().parents[1] / "research/candidates.json").read_text("utf-8")
    )
}
NOTICE = (
    "first_seen_at is the earliest capture in supplied snapshots, not official publication. "
    "Historical replay is not prospective validation."
)


def _utc(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Capture timestamps must include a timezone.")
    return stamp.astimezone(UTC).isoformat(timespec="microseconds")


def validate_snapshot(snapshot):
    """Read only manifest.json and the 52 expected CSVs, validating before any DB write."""
    snapshot = Path(snapshot).resolve()
    manifest_bytes = (snapshot / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    records = manifest["series"]
    if (
        manifest.get("provider") != "Banco Central de Chile"
        or len(EXPECTED) != 52
        or len(records) != 52
        or {r["alias"]: r["code"] for r in records} != EXPECTED
    ):
        raise ValueError("Expected the complete 52-series BCCh snapshot.")
    batch_times = [_utc(manifest["retrieved_at_utc"])]
    if "additional_batch_retrieved_at_utc" in manifest:
        batch_times.append(_utc(manifest["additional_batch_retrieved_at_utc"]))
    batch_times.extend(_utc(r["retrieved_at_utc"]) for r in records if "retrieved_at_utc" in r)
    fallback = max(batch_times)
    observations, keys = [], set()
    for record in records:
        alias = record["alias"]
        path = (snapshot / f"{alias}.csv").resolve()
        if record.get("status") != "ok" or path.parent != snapshot:
            raise ValueError("Each expected series must be available inside the snapshot.")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
            raise ValueError(f"Snapshot SHA256 mismatch: {alias}")
        seen = _utc(record["retrieved_at_utc"]) if "retrieved_at_utc" in record else fallback
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
        if reader.fieldnames != ["date", "value", "status_code"]:
            raise ValueError("Invalid observation columns.")
        count = 0
        for row in reader:
            day = date.fromisoformat(row["date"]).isoformat()
            key = (alias, day)
            if day != row["date"] or key in keys or not row["status_code"]:
                raise ValueError("Invalid or duplicate observation.")
            value = float(row["value"]) if row["value"] else None
            if value is not None and not math.isfinite(value):
                raise ValueError("Non-finite observation value.")
            if row["status_code"] == "OK" and value is None:
                raise ValueError("OK observation has no value.")
            observations.append((*key, value, row["status_code"], seen))
            keys.add(key)
            count += 1
        if not count:
            raise ValueError("Empty series cannot enter the registry.")
    return hashlib.sha256(manifest_bytes).hexdigest(), fallback, observations


def register_snapshot(snapshot, registry):
    snapshot_hash, captured_at, rows = validate_snapshot(snapshot)
    registry = Path(registry)
    registry.parent.mkdir(parents=True, exist_ok=True)
    counts = {"added": 0, "changed": 0, "unchanged": 0}
    with closing(sqlite3.connect(registry)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "CREATE TABLE IF NOT EXISTS snapshots (hash TEXT PRIMARY KEY, captured_at TEXT, "
            "registered_at TEXT, timestamp_policy TEXT, observations INTEGER)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS observations (alias TEXT, date TEXT, value REAL, "
            "status TEXT, first_seen_at TEXT, last_seen_at TEXT, PRIMARY KEY(alias,date))"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS revisions (snapshot_hash TEXT, alias TEXT, date TEXT, "
            "seen_at TEXT, kind TEXT, old_value REAL, old_status TEXT, new_value REAL, "
            "new_status TEXT, PRIMARY KEY(snapshot_hash,alias,date))"
        )
        duplicate = db.execute("SELECT 1 FROM snapshots WHERE hash=?", (snapshot_hash,)).fetchone()
        if duplicate:
            counts["unchanged"] = len(rows)
        else:
            latest = db.execute("SELECT MAX(captured_at) FROM snapshots").fetchone()[0]
            if latest and min(row[4] for row in rows) <= latest:
                raise ValueError("Import snapshots in capture order; earlier replay rejected.")
            db.execute(
                "INSERT INTO snapshots VALUES (?,?,?,?,?)",
                (
                    snapshot_hash,
                    captured_at,
                    datetime.now(UTC).isoformat(timespec="microseconds"),
                    "Series timestamp when mapped; otherwise maximum manifest batch timestamp",
                    len(rows),
                ),
            )
            for alias, day, value, status, seen in rows:
                previous = db.execute(
                    "SELECT value,status FROM observations WHERE alias=? AND date=?", (alias, day)
                ).fetchone()
                kind = (
                    "added"
                    if previous is None
                    else ("unchanged" if previous == (value, status) else "changed")
                )
                counts[kind] += 1
                if kind != "unchanged":
                    db.execute(
                        "INSERT INTO revisions VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            snapshot_hash,
                            alias,
                            day,
                            seen,
                            kind,
                            *(previous or (None, None)),
                            value,
                            status,
                        ),
                    )
                db.execute(
                    "INSERT INTO observations VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT(alias,date) DO UPDATE SET value=excluded.value, "
                    "status=excluded.status,last_seen_at=excluded.last_seen_at",
                    (alias, day, value, status, seen, seen),
                )
        counts.update(
            duplicate=bool(duplicate),
            snapshot_hash=snapshot_hash,
            snapshots=db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0],
            observations=db.execute("SELECT COUNT(*) FROM observations").fetchone()[0],
            max_first_seen_at=db.execute("SELECT MAX(first_seen_at) FROM observations").fetchone()[
                0
            ],
            observed_revisions=db.execute(
                "SELECT COUNT(*) FROM revisions WHERE kind='changed'"
            ).fetchone()[0],
            notice=NOTICE,
        )
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(register_snapshot(args.snapshot, args.registry), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
