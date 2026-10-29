import pandas as pd

from campaign.selection import selection_audit


def predictions(path, liquid_selection_error):
    rows = []
    for quarter in pd.date_range("2021-03-31", "2026-06-30", freq="QE"):
        selection = quarter.year <= 2022
        errors = {
            "panel6": 10,
            "panel6_without_liquid_composition": liquid_selection_error if selection else 14,
            "panel6_without_participation": 12 if selection else 9,
            "panel6_without_real_credit": 12 if selection else 9,
        }
        for target in ["sme_s", "sme_d", "large_s", "large_d", "cons_s", "cons_d"]:
            for method, error in errors.items():
                rows.append({"quarter": quarter, "target": target, "model": method, "error": error})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_failed_selected_removal_cannot_be_replaced_by_later_winner(tmp_path):
    source = tmp_path / "predictions.csv"
    predictions(source, 10.4)
    report = selection_audit(source, tmp_path / "audit.json")
    assert report["selected_on2021_2022"] == "liquid_composition"
    assert report["retained_after2023_2026"] is None
    assert report["reference_method"] == "panel6"
    assert report["results"]["participation"]["evaluation"]["passes_all"]


def test_no_selection_cannot_choose_from_evaluation(tmp_path):
    source = tmp_path / "predictions.csv"
    predictions(source, 11)
    report = selection_audit(source, tmp_path / "audit.json")
    assert report["selected_on2021_2022"] is None
    assert report["reference_method"] == "panel6"
