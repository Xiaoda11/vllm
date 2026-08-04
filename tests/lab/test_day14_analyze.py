import pytest

from scripts.lab_day14_analyze import jain_index, relative_change_percent, summarize


def test_jain_index() -> None:
    assert jain_index([1.0, 1.0]) == 1.0
    assert jain_index([1.0, 0.0]) == 0.5
    with pytest.raises(ValueError, match="non-empty"):
        jain_index([])


def test_relative_change_percent() -> None:
    assert relative_change_percent(100.0, 25.0) == pytest.approx(-75.0)
    with pytest.raises(ValueError, match="non-zero"):
        relative_change_percent(0.0, 1.0)


def test_summarize_builds_repeat_gate() -> None:
    base = {
        "a_ttft_ms": 10.0,
        "b_ttft_ms": 30.0,
        "c_ttft_ms": 31.0,
        "a_e2e_ms": 20.0,
        "b_e2e_ms": 40.0,
        "c_e2e_ms": 41.0,
        "makespan_ms": 50.0,
        "output_tokens_per_s": 10.0,
        "b_c_ttft_jain": 1.0,
        "b_first_scheduled_step": 50,
        "c_first_scheduled_step": 60,
        "waiting_allocation_failures": 40,
        "preemptions": 0,
        "mrv2_input_shape_mismatches": 0,
        "hol_witness": True,
    }
    tuned = {
        **base,
        "c_ttft_ms": 1.0,
        "b_first_scheduled_step": 50,
        "c_first_scheduled_step": 10,
    }
    summary = summarize([base] * 3, [tuned] * 3)

    assert all(summary["repeat_gate"].values())
    assert summary["modified_vs_baseline_percent"]["c_ttft_ms"] == pytest.approx(
        -96.774
    )
