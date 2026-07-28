from tests.load_test import (
    RequestResult,
    percentile,
    summarize_results,
)


def test_percentile_and_load_summary() -> None:
    results = [
        RequestResult(True, 200, 100.0),
        RequestResult(True, 200, 200.0),
        RequestResult(False, 504, 300.0, "HTTP 504"),
    ]

    assert percentile(
        [100.0, 200.0, 300.0],
        0.95,
    ) == 200.0

    summary = summarize_results(
        results=results,
        elapsed_seconds=1.5,
        concurrency=2,
    )

    assert summary["request_count"] == 3
    assert summary["success_rate"] == 66.67
    assert summary["requests_per_second"] == 2.0
    assert summary["status_distribution"] == {
        "200": 2,
        "504": 1,
    }
    assert summary["errors"] == {
        "HTTP 504": 1,
    }
