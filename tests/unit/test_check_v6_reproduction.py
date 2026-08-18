from scripts.check_v6_reproduction import audit


def test_repository_contains_complete_v6_reproduction_closure() -> None:
    result = audit(database=None, repository_only=True)  # type: ignore[arg-type]
    assert result["repository_complete"] is True
    assert result["missing_repository_paths"] == []
