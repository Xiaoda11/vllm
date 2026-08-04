import pytest

from scripts.lab_day13_analyze import median_for, relative_change_percent


def test_relative_change_percent() -> None:
    assert relative_change_percent(100.0, 70.0) == pytest.approx(-30.0)
    assert relative_change_percent(100.0, 105.0) == pytest.approx(5.0)


def test_relative_change_rejects_zero_baseline() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        relative_change_percent(0.0, 1.0)


def test_median_for_filters_threshold() -> None:
    rows = [
        {"threshold": 0, "metric": 10.0},
        {"threshold": 2048, "metric": 30.0},
        {"threshold": 0, "metric": 20.0},
    ]
    assert median_for(rows, 0, "metric") == 15.0
