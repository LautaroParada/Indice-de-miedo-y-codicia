"""Reproducir la campaña congelada desde el snapshot BCCh, sin red ni pasos manuales."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd

from research.analyze import analyze

from .engine import run, save_json
from .figures import NAVY, comparison, plt, stability
from .prospective import register_snapshot, validate_snapshot
from .report import make_report

SNAPSHOT_SHA256 = "799ab89183499a6c2a6516fe595a76006f9af63d94e2744259a54422d172ceff"
CONTRACT_SHA256 = "d943db3890f310b6d92cdd2737a8f5eb637a9781d4be91ea9b22a8799a0f17b0"
SELECTION_SHA256 = "6de1ca219d885a3f757faa95229e5879075aa83fa66a49761bf56bc131cfec0f"


def hashes(root):
    """Compare portable results; SQLite/runtime reports are intentionally excluded."""
    return {
        str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and p.suffix in {".json", ".csv", ".md"}
        and p.name not in {"reproduction.json", "prospective_status.json"}
    }


def reproduce(snapshot, output, compare_to=None):
    snapshot_hash, captured_at, rows = validate_snapshot(snapshot)
    contract = Path("campaign/contract.json")
    if snapshot_hash != SNAPSHOT_SHA256:
        raise ValueError(
            "Esta campaña exige el snapshot congelado; no es una evaluación de datos nuevos."
        )
    if hashlib.sha256(contract.read_bytes()).hexdigest() != CONTRACT_SHA256:
        raise ValueError("Contrato distinto al congelado. No reutilizar el veredicto.")
    if (
        hashlib.sha256(Path("campaign/selection-check.json").read_bytes()).hexdigest()
        != SELECTION_SHA256
    ):
        raise ValueError("Protocolo de selección distinto al congelado.")
    output.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(Path("research/candidates.json").read_text(encoding="utf-8"))
    analyze(candidates, snapshot, output / "base")
    results = run(output / "base", output / "experiments", contract, snapshot)
    selected = make_report(output / "experiments", output / "base", output / "delivery", contract)
    registry_status = register_snapshot(snapshot, output / "prospective.sqlite")
    save_json(output / "prospective_status.json", registry_status)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "text.color": NAVY,
            "svg.fonttype": "none",
            "svg.hashsalt": "mym-bcch-closure",
        }
    ):
        comparison(results["economics"], output / "figures")
        stability(results, pd.read_csv(output / "experiments/envelopes.csv"), output / "figures")
    outputs = hashes(output)
    verification = None
    if compare_to:
        reference = hashes(compare_to)
        differences = sorted(
            k for k in set(outputs) | set(reference) if outputs.get(k) != reference.get(k)
        )
        verification = {
            "equal": not differences,
            "files_compared": len(outputs),
            "differences": differences,
        }
        if differences:
            save_json(output / "reproduction_failure.json", verification)
            raise ValueError("La reproducción difiere: revisar reproduction_failure.json")
    report = {
        "snapshot_sha256": snapshot_hash,
        "contract_sha256": CONTRACT_SHA256,
        "selection_sha256": SELECTION_SHA256,
        "observations_validated": len(rows),
        "capture_timestamp_conservative": captured_at,
        "fresh_api_download": False,
        "snapshot_origin": ("Real BCCh API snapshot; fresh campaign call failed connectivity"),
        "scenario_count": results["total_sensitivity_scenarios"],
        "reference_method": selected["reference_method"],
        "comparison": verification,
        "runtime": {
            "python": platform.python_version(),
            **{p: importlib.metadata.version(p) for p in ["numpy", "pandas", "matplotlib", "bcch"]},
        },
        "output_sha256": outputs,
        "source_sha256": {
            str(p).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ["campaign", "research", "imc"]
            for p in sorted(Path(folder).glob("*.py"))
        },
    }
    save_json(output / "reproduction.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("campaign/snapshot-reference"))
    parser.add_argument("--output", type=Path, default=Path("outputs/closure-run"))
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()
    report = reproduce(args.snapshot, args.output, args.compare_to)
    print("Reproducción completa:", args.output / "delivery/verdict.md")
    if report["comparison"]:
        print("Resultados idénticos:", report["comparison"]["equal"])
