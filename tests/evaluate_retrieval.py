import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.config import PROJECT_ROOT, settings
from app.retrieval import (
    RetrievalMode,
    create_knowledge_retriever,
)
from app.schemas import IntentType
from tests.evaluate import (
    calculate_p95,
    calculate_rate,
    load_eval_cases,
)


MODES: tuple[RetrievalMode, ...] = (
    "vector",
    "keyword",
    "hybrid",
    "hybrid_rerank",
)


def find_source_rank(
    expected_source: str,
    ranked_sources: list[str],
) -> int | None:
    """返回标准来源第一次出现的1起始排名。"""
    try:
        return (
            ranked_sources.index(
                expected_source
            )
            + 1
        )
    except ValueError:
        return None


def run_retrieval_comparison(
    top_k: int = 3,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k必须大于0")

    cases = [
        case
        for case in load_eval_cases(
            PROJECT_ROOT / "tests" / "eval_cases.json"
        )
        if (
            case.expected_intent
            == IntentType.KNOWLEDGE_QUERY
            and case.required_sources
        )
    ]
    retriever = create_knowledge_retriever()
    mode_reports: dict[str, Any] = {}

    for mode in MODES:
        recall_at_1 = 0
        recall_at_k = 0
        reciprocal_rank_total = 0.0
        durations: list[float] = []
        results: list[dict[str, Any]] = []
        expected_source_count = sum(
            len(case.required_sources)
            for case in cases
        )

        for case in cases:
            started_at = time.perf_counter()
            matches = retriever.search(
                query=case.question,
                top_k=top_k,
                min_score=-1.0,
                min_keyword_score=0.0,
                mode=mode,
            )
            duration_ms = (
                time.perf_counter()
                - started_at
            ) * 1000
            durations.append(duration_ms)
            ranked_sources = [
                match.source
                for match in matches
            ]
            ranks = {
                source: find_source_rank(
                    expected_source=source,
                    ranked_sources=ranked_sources,
                )
                for source in case.required_sources
            }

            for rank in ranks.values():
                if rank == 1:
                    recall_at_1 += 1

                if rank is not None and rank <= top_k:
                    recall_at_k += 1
                    reciprocal_rank_total += (
                        1.0 / rank
                    )

            results.append(
                {
                    "id": case.id,
                    "category": case.category,
                    "question": case.question,
                    "expected_sources": (
                        case.required_sources
                    ),
                    "ranks": ranks,
                    "duration_ms": round(
                        duration_ms,
                        2,
                    ),
                    "matches": [
                        {
                            "source": match.source,
                            "score": match.score,
                            "vector_score": (
                                match.vector_score
                            ),
                            "keyword_score": (
                                match.keyword_score
                            ),
                            "rerank_score": (
                                match.rerank_score
                            ),
                        }
                        for match in matches
                    ],
                }
            )

        case_count = len(cases)
        mode_reports[mode] = {
            "metrics": {
                "recall_at_1": calculate_rate(
                    recall_at_1,
                    expected_source_count,
                ),
                f"recall_at_{top_k}": calculate_rate(
                    recall_at_k,
                    expected_source_count,
                ),
                "mrr": round(
                    reciprocal_rank_total
                    / expected_source_count
                    if expected_source_count
                    else 0.0,
                    4,
                ),
                "average_duration_ms": round(
                    statistics.mean(durations)
                    if durations
                    else 0.0,
                    2,
                ),
                "p95_duration_ms": calculate_p95(
                    durations
                ),
            },
            "results": results,
            "expected_source_count": (
                expected_source_count
            ),
        }

    return {
        "generated_at": (
            datetime.now(timezone.utc).isoformat()
        ),
        "embedding_model_name": (
            settings.embedding_model_name
        ),
        "case_count": len(cases),
        "top_k": top_k,
        "modes": mode_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "比较纯向量、纯BM25和混合检索"
        )
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "retrieval_comparison.json"
        ),
    )
    arguments = parser.parse_args()
    report = run_retrieval_comparison(
        top_k=arguments.top_k
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

    print("=" * 60)
    print(
        f"检索对照实验，共{report['case_count']}题"
    )

    for mode in MODES:
        metrics = report["modes"][mode][
            "metrics"
        ]
        print(
            f"{mode:>7}："
            f"R@1={metrics['recall_at_1']:.2f}%  "
            f"R@{arguments.top_k}="
            f"{metrics[f'recall_at_{arguments.top_k}']:.2f}%  "
            f"MRR={metrics['mrr']:.4f}  "
            f"平均耗时="
            f"{metrics['average_duration_ms']:.2f}ms"
        )

    print(f"报告位置：{arguments.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
