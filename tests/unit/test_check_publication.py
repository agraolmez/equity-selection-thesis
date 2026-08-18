from scripts.check_publication import audit


def test_staged_snapshot_has_no_publication_hazards() -> None:
    result = audit()
    assert result["status"] == "PASS", result["failures"]
