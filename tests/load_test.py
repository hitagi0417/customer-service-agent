import argparse
import asyncio
import json
import os
import statistics
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx


DEFAULT_QUESTIONS = (
    "客服几点上班？",
    "退款申请期限是多久？",
    "XJ-900标准版保修多久？",
    "帮我转人工客服",
    "你们的办公地址在哪里？",
)


@dataclass(frozen=True)
class RequestResult:
    success: bool
    status_code: int
    duration_ms: float
    error: str | None = None


def percentile(
    values: list[float],
    percent: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    index = int(
        (len(ordered) - 1) * percent
    )
    return round(ordered[index], 2)


def summarize_results(
    results: list[RequestResult],
    elapsed_seconds: float,
    concurrency: int,
) -> dict:
    durations = [
        result.duration_ms
        for result in results
    ]
    succeeded = sum(
        1
        for result in results
        if result.success
    )
    total = len(results)

    return {
        "generated_at": (
            datetime.now(timezone.utc).isoformat()
        ),
        "concurrency": concurrency,
        "request_count": total,
        "success_count": succeeded,
        "failure_count": total - succeeded,
        "success_rate": round(
            succeeded / total * 100
            if total
            else 0.0,
            2,
        ),
        "elapsed_seconds": round(
            elapsed_seconds,
            2,
        ),
        "requests_per_second": round(
            total / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0,
            2,
        ),
        "average_duration_ms": round(
            statistics.mean(durations)
            if durations
            else 0.0,
            2,
        ),
        "p50_duration_ms": percentile(
            durations,
            0.50,
        ),
        "p95_duration_ms": percentile(
            durations,
            0.95,
        ),
        "p99_duration_ms": percentile(
            durations,
            0.99,
        ),
        "status_distribution": dict(
            sorted(
                Counter(
                    str(result.status_code)
                    for result in results
                ).items()
            )
        ),
        "errors": dict(
            sorted(
                Counter(
                    result.error
                    for result in results
                    if result.error
                ).items()
            )
        ),
    }


async def run_load_test(
    base_url: str,
    api_key: str,
    concurrency: int,
    request_count: int,
    timeout_seconds: float,
) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    headers = {
        "X-API-Key": api_key,
    }

    async with httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=timeout_seconds,
    ) as client:
        async def send(index: int) -> RequestResult:
            question = DEFAULT_QUESTIONS[
                index % len(DEFAULT_QUESTIONS)
            ]

            async with semaphore:
                started_at = time.perf_counter()

                try:
                    response = await client.post(
                        "/api/chat",
                        json={
                            "question": question,
                            "conversation_id": (
                                f"load-{uuid.uuid4().hex}"
                            ),
                        },
                    )
                    duration_ms = (
                        time.perf_counter()
                        - started_at
                    ) * 1000
                    return RequestResult(
                        success=(
                            response.status_code == 200
                        ),
                        status_code=(
                            response.status_code
                        ),
                        duration_ms=round(
                            duration_ms,
                            2,
                        ),
                        error=(
                            None
                            if response.status_code == 200
                            else (
                                f"HTTP "
                                f"{response.status_code}"
                            )
                        ),
                    )
                except Exception as error:
                    duration_ms = (
                        time.perf_counter()
                        - started_at
                    ) * 1000
                    return RequestResult(
                        success=False,
                        status_code=0,
                        duration_ms=round(
                            duration_ms,
                            2,
                        ),
                        error=type(error).__name__,
                    )

        started_at = time.perf_counter()
        results = await asyncio.gather(
            *[
                send(index)
                for index in range(request_count)
            ]
        )
        elapsed_seconds = (
            time.perf_counter() - started_at
        )

    return summarize_results(
        results=list(results),
        elapsed_seconds=elapsed_seconds,
        concurrency=concurrency,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "对已部署的智能客服API执行并发压测。"
            "真实请求会调用大模型并产生费用。"
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
    )
    parser.add_argument(
        "--confirm-real-llm-cost",
        action="store_true",
        help="确认本次压测可能产生真实模型费用",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/load_test_report.json"
        ),
    )
    arguments = parser.parse_args()

    if arguments.concurrency <= 0:
        raise ValueError("并发数必须大于0")
    if arguments.requests <= 0:
        raise ValueError("请求数必须大于0")
    if not arguments.confirm_real_llm_cost:
        raise RuntimeError(
            "压测会产生真实LLM调用；"
            "确认后添加--confirm-real-llm-cost"
        )

    api_key = os.getenv("SERVICE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "必须通过SERVICE_API_KEY环境变量提供API密钥"
        )

    report = asyncio.run(
        run_load_test(
            base_url=arguments.base_url,
            api_key=api_key,
            concurrency=arguments.concurrency,
            request_count=arguments.requests,
            timeout_seconds=arguments.timeout,
        )
    )
    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    arguments.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
