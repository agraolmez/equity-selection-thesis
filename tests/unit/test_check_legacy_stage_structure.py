from scripts.check_legacy_stage_structure import audit


def test_historical_fifteen_stage_responsibilities_are_structurally_present() -> None:
    result = audit()
    assert result["status"] == "PASS", result["stages"]
