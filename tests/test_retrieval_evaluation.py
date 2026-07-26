from tests.evaluate_retrieval import (
    find_source_rank,
)


def test_find_source_rank_uses_first_matching_position() -> None:
    rank = find_source_rank(
        expected_source="refund.md",
        ranked_sources=[
            "shipping.md",
            "refund.md",
            "refund.md",
        ],
    )

    assert rank == 2


def test_find_source_rank_returns_none_when_missing() -> None:
    rank = find_source_rank(
        expected_source="refund.md",
        ranked_sources=["shipping.md"],
    )

    assert rank is None
