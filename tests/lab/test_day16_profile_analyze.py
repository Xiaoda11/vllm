import pytest

from scripts.lab_day16_profile_analyze import (
    classify_execution,
    parse_execution_annotation,
)


@pytest.mark.parametrize(
    ("name", "expected_class", "total_tokens"),
    [
        (
            "execute_1_context_0(sq0sk0sqsq0sqsk0)_generation_1"
            "(sq1sk8248sqsq1sqsk8248)",
            "single_decode",
            1,
        ),
        (
            "execute_1025_context_1(sq1024sk1024sqsq1048576sqsk1048576)"
            "_generation_1(sq1sk8265sqsq1sqsk8265)",
            "prefill_decode_mixed",
            1025,
        ),
        (
            "execute_2_context_0(sq0sk0sqsq0sqsk0)_generation_2"
            "(sq2sk9291sqsq2sqsk9291)",
            "multi_decode",
            2,
        ),
    ],
)
def test_execution_annotation_classification(
    name: str, expected_class: str, total_tokens: int
) -> None:
    values = parse_execution_annotation(name)

    assert values["total_tokens"] == total_tokens
    assert classify_execution(values) == expected_class


def test_execution_annotation_rejects_inconsistent_token_total() -> None:
    name = (
        "execute_3_context_1(sq1sk1sqsq1sqsk1)_generation_1"
        "(sq1sk1sqsq1sqsk1)"
    )

    with pytest.raises(ValueError, match="inconsistent"):
        parse_execution_annotation(name)
