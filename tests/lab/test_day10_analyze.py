from scripts.lab_day10_analyze import percentile, summarize_values


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([10.0, 20.0, 30.0], 0.50) == 20.0
    assert percentile([10.0, 20.0], 0.95) == 19.5
    assert percentile([], 0.95) is None


def test_summarize_values_converts_seconds_to_milliseconds() -> None:
    assert summarize_values([0.01, 0.02, 0.03]) == {
        "count": 3,
        "mean_ms": 20.0,
        "p50_ms": 20.0,
        "p95_ms": 29.0,
        "max_ms": 30.0,
    }
