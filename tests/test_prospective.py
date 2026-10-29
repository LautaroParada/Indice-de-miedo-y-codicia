"""Registry integrity, capture chronology and observed revision semantics."""

import hashlib
import json
import sqlite3

import pytest

from campaign.prospective import EXPECTED, main, register_snapshot


@pytest.fixture
def snapshot_factory(tmp_path):
    def make(name, timestamp, *, value="1.5", status="OK", additional=None):
        folder = tmp_path / name
        folder.mkdir()
        records = []
        for alias, code in EXPECTED.items():
            payload = f"date,value,status_code\n2020-01-01,{value},{status}\n".encode()
            (folder / f"{alias}.csv").write_bytes(payload)
            records.append(
                {
                    "alias": alias,
                    "code": code,
                    "status": "ok",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        manifest = {
            "provider": "Banco Central de Chile",
            "retrieved_at_utc": timestamp,
            "series": records,
        }
        if additional:
            manifest["additional_batch_retrieved_at_utc"] = additional
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return folder

    return make


def test_replay_is_idempotent_and_does_not_claim_publication(snapshot_factory, tmp_path, capsys):
    snapshot = snapshot_factory("first", "2026-10-29T18:00:00+00:00")
    registry = tmp_path / "registry.sqlite"
    first = register_snapshot(snapshot, registry)
    assert (first["added"], first["changed"], first["snapshots"]) == (52, 0, 1)
    second = register_snapshot(snapshot, registry)
    assert (second["added"], second["changed"], second["unchanged"]) == (0, 0, 52)
    assert second["duplicate"] and second["snapshots"] == 1
    with sqlite3.connect(registry) as db:
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 52
        assert (
            db.execute("SELECT MIN(first_seen_at) FROM observations")
            .fetchone()[0]
            .startswith("2026-10-29")
        )  # Capture, not the 2020 reference period.
        columns = [row[1] for row in db.execute("PRAGMA table_info(observations)")]
        assert "publication_date" not in columns
    main(["--snapshot", str(snapshot), "--registry", str(registry)])
    output = json.loads(capsys.readouterr().out)
    assert "not official publication" in output["notice"]
    assert "not prospective validation" in output["notice"]


def test_revision_preserves_old_new_and_first_capture(snapshot_factory, tmp_path):
    first = snapshot_factory("first", "2026-10-28T12:00:00Z")
    second = snapshot_factory("second", "2026-10-29T12:00:00Z", value="2.5")
    third = snapshot_factory("third", "2026-10-30T12:00:00Z", value="2.5")
    registry = tmp_path / "registry.sqlite"
    register_snapshot(first, registry)
    summary = register_snapshot(second, registry)
    assert summary["changed"] == summary["observed_revisions"] == 52
    assert summary["added"] == 0 and summary["snapshots"] == 2
    assert register_snapshot(third, registry)["unchanged"] == 52
    with sqlite3.connect(registry) as db:
        assert db.execute(
            "SELECT old_value,new_value,old_status,new_status FROM revisions WHERE kind='changed'"
        ).fetchone() == (1.5, 2.5, "OK", "OK")
        value, first_seen, last_seen = db.execute(
            "SELECT value,first_seen_at,last_seen_at FROM observations"
        ).fetchone()
        assert value == 2.5
        assert first_seen.startswith("2026-10-28") and last_seen.startswith("2026-10-30")
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 104


def test_status_revision_and_missing_are_not_silently_zero(snapshot_factory, tmp_path):
    first = snapshot_factory("first", "2026-10-28T12:00:00Z")
    second = snapshot_factory("second", "2026-10-29T12:00:00Z", value="", status="ND")
    registry = tmp_path / "registry.sqlite"
    register_snapshot(first, registry)
    assert register_snapshot(second, registry)["changed"] == 52
    with sqlite3.connect(registry) as db:
        assert db.execute("SELECT value,status FROM observations").fetchone() == (None, "ND")
        assert db.execute(
            "SELECT old_value,new_value,old_status,new_status FROM revisions WHERE kind='changed'"
        ).fetchone() == (1.5, None, "OK", "ND")


def test_all_hashes_checked_before_creating_registry(snapshot_factory, tmp_path):
    snapshot = snapshot_factory("bad", "2026-10-29T12:00:00Z")
    last_alias = list(EXPECTED)[-1]
    (snapshot / f"{last_alias}.csv").write_text("corrupt", encoding="utf-8")
    registry = tmp_path / "registry.sqlite"
    with pytest.raises(ValueError, match="SHA256"):
        register_snapshot(snapshot, registry)
    assert not registry.exists()


def test_missing_series_rejected_before_write(snapshot_factory, tmp_path):
    snapshot = snapshot_factory("partial", "2026-10-29T12:00:00Z")
    path = snapshot / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["series"].pop()
    path.write_text(json.dumps(manifest))
    registry = tmp_path / "registry.sqlite"
    with pytest.raises(ValueError, match="52-series"):
        register_snapshot(snapshot, registry)
    assert not registry.exists()


def test_unmapped_extra_batch_uses_latest_capture(snapshot_factory, tmp_path):
    snapshot = snapshot_factory(
        "batches", "2026-10-29T18:00:00Z", additional="2026-10-29T19:00:00Z"
    )
    registry = tmp_path / "registry.sqlite"
    summary = register_snapshot(snapshot, registry)
    assert summary["max_first_seen_at"] == "2026-10-29T19:00:00.000000+00:00"
    with sqlite3.connect(registry) as db:
        distinct = db.execute("SELECT COUNT(DISTINCT first_seen_at) FROM observations").fetchone()[
            0
        ]
        assert distinct == 1
        assert (
            "maximum manifest" in db.execute("SELECT timestamp_policy FROM snapshots").fetchone()[0]
        )


def test_earlier_snapshot_cannot_overwrite_history(snapshot_factory, tmp_path):
    recent = snapshot_factory("recent", "2026-10-29T12:00:00Z", value="2.5")
    old = snapshot_factory("old", "2026-10-28T12:00:00Z")
    registry = tmp_path / "registry.sqlite"
    register_snapshot(recent, registry)
    with pytest.raises(ValueError, match="capture order"):
        register_snapshot(old, registry)
    with sqlite3.connect(registry) as db:
        assert db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 1
        assert db.execute("SELECT MIN(value) FROM observations").fetchone()[0] == 2.5


def test_naive_timestamp_rejected(snapshot_factory, tmp_path):
    snapshot = snapshot_factory("naive", "2026-10-29T12:00:00")
    with pytest.raises(ValueError, match="timezone"):
        register_snapshot(snapshot, tmp_path / "registry.sqlite")


def test_mapped_series_capture_preserved_without_guessing_other_batches(snapshot_factory, tmp_path):
    snapshot = snapshot_factory("mapped", "2026-10-29T18:00:00Z", additional="2026-10-29T19:00:00Z")
    path = snapshot / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["series"][0]["retrieved_at_utc"] = "2026-10-29T18:30:00Z"
    alias = manifest["series"][0]["alias"]
    path.write_text(json.dumps(manifest))
    registry = tmp_path / "registry.sqlite"
    register_snapshot(snapshot, registry)
    with sqlite3.connect(registry) as db:
        assert (
            db.execute("SELECT first_seen_at FROM observations WHERE alias=?", (alias,)).fetchone()[
                0
            ]
            == "2026-10-29T18:30:00.000000+00:00"
        )
        assert (
            db.execute(
                "SELECT MIN(first_seen_at) FROM observations WHERE alias<>?", (alias,)
            ).fetchone()[0]
            == "2026-10-29T19:00:00.000000+00:00"
        )


def test_corrupt_new_snapshot_preserves_existing_registry(snapshot_factory, tmp_path):
    good = snapshot_factory("good", "2026-10-28T12:00:00Z")
    bad = snapshot_factory("bad", "2026-10-29T12:00:00Z", value="2.5")
    registry = tmp_path / "registry.sqlite"
    register_snapshot(good, registry)
    before = registry.read_bytes()
    (bad / f"{list(EXPECTED)[-1]}.csv").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        register_snapshot(bad, registry)
    assert registry.read_bytes() == before
