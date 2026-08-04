from scripts.lab_day14_burst_analyze import summarize_pair


def test_summarize_pair_detects_b_delay() -> None:
    baseline = {
        "a_ttft_ms": 1.0,
        "b_ttft_ms": 10.0,
        "b_e2e_ms": 11.0,
        "c_ttft_median_ms": 10.0,
        "c_ttft_max_ms": 11.0,
        "all_ttft_jain": 1.0,
        "makespan_ms": 20.0,
        "output_tokens_per_s": 5.0,
        "b_first_scheduled_step": 50,
        "c_first_scheduled_step_min": 60,
        "c_first_scheduled_step_max": 70,
        "allocation_failures": 10,
        "preemptions": 0,
        "mrv2_input_shape_mismatches": 0,
        "hol_witness": True,
    }
    modified = {
        **baseline,
        "b_first_scheduled_step": 55,
        "c_first_scheduled_step_min": 10,
        "c_first_scheduled_step_max": 20,
    }
    summary = summarize_pair(baseline, modified)

    assert summary["fairness_gate"]["modified_delays_b_first_schedule"]
    assert summary["modified_vs_baseline_percent"]["b_first_scheduled_step"] == 10.0
