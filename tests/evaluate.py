import argparse
import json
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

# 直接执行 `python tests/evaluate.py` 时，Python 默认只把 tests
# 目录加入模块搜索路径。这里显式加入项目根目录，使 app 包可被导入。
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.agent import create_customer_service_agent
from app.config import PROJECT_ROOT, settings
from app.schemas import IntentType


class EvalCase(BaseModel):
    """
    一条标准评测题。
    """

    id: str = Field(
        min_length=1,
    )

    question: str = Field(
        min_length=1,
    )

    category: str = Field(
        default="general",
        min_length=1,
    )

    expected_intent: IntentType

    expected_source: str | None = None

    expected_sources: list[str] = Field(
        default_factory=list,
    )

    expected_need_human: bool

    expected_answer_keywords: list[
        str | list[str]
    ] = Field(
        default_factory=list,
    )

    @property
    def required_sources(self) -> list[str]:
        if self.expected_sources:
            return self.expected_sources

        if self.expected_source:
            return [self.expected_source]

        return []


def load_eval_cases(
    path: Path,
) -> list[EvalCase]:
    """
    加载并校验评测集。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"评测集不存在：{path}"
        )

    content = path.read_text(
        encoding="utf-8"
    )

    adapter = TypeAdapter(
        list[EvalCase]
    )

    cases = adapter.validate_json(content)

    if not cases:
        raise ValueError("评测集不能为空")

    case_ids = [
        case.id
        for case in cases
    ]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError(
            "评测集中存在重复的id"
        )

    return cases


def normalize_text(text: str) -> str:
    """
    对答案进行简单标准化。

    删除空格和换行，并转换为小写，
    减少格式差异对关键词检查的影响。
    """
    normalized = "".join(
        text.lower().split()
    )

    # 大模型可能使用“周一至周五”或“周一到周五”等等价表达。
    # 先把常见格式差异归一化，避免把正确回答误判为错误。
    return (
        normalized
        .replace("至", "到")
        .replace("：", ":")
        .replace("—", "-")
        .replace("－", "-")
    )


def find_missing_keyword_requirements(
    answer: str,
    requirements: list[str | list[str]],
) -> list[str]:
    """检查必需概念；嵌套列表中的同义表达满足任意一个即可。"""
    normalized_answer = normalize_text(answer)
    missing: list[str] = []

    for requirement in requirements:
        alternatives = (
            requirement
            if isinstance(requirement, list)
            else [requirement]
        )
        matched = any(
            normalize_text(alternative)
            in normalized_answer
            for alternative in alternatives
        )

        if not matched:
            missing.append(
                "/".join(alternatives)
            )

    return missing


def calculate_rate(
    correct: int,
    total: int,
) -> float:
    """
    将正确数量转换为百分比。
    """
    if total == 0:
        return 0.0

    return round(
        correct / total * 100,
        2,
    )


def run_evaluation(
    cases: list[EvalCase],
) -> dict:
    """
    运行完整Agent评测。
    """

    # 保存正式数据库路径
    original_database_path = (
        settings.database_path
    )

    results: list[dict] = []
    failures: list[dict] = []
    durations: list[float] = []

    intent_correct = 0
    transfer_correct = 0
    keyword_correct = 0
    keyword_total = 0

    retrieval_correct = 0
    retrieval_total = 0

    rejection_correct = 0
    rejection_total = 0

    pipeline_success_count = 0
    auto_resolved_count = 0
    category_totals: dict[str, int] = {}
    category_passed: dict[str, int] = {}

    try:
        # 使用临时数据库，避免评测数据污染正式数据库
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_database = (
                Path(temp_dir)
                / "evaluation.db"
            )

            object.__setattr__(
                settings,
                "database_path",
                temporary_database,
            )

            agent = (
                create_customer_service_agent()
            )

            total_cases = len(cases)

            for index, case in enumerate(
                cases,
                start=1,
            ):
                print(
                    f"[{index:02d}/{total_cases:02d}] "
                    f"正在评测：{case.id} "
                    f"{case.question}"
                )

                case_failures: list[str] = []

                try:
                    # 完整执行一次Agent链路
                    response = agent.run(
                        question=case.question
                    )

                    durations.append(
                        response.total_duration_ms
                    )

                    # 1. 检查意图
                    intent_is_correct = (
                        response.intent
                        == case.expected_intent
                    )

                    if intent_is_correct:
                        intent_correct += 1
                    else:
                        case_failures.append(
                            "意图错误："
                            f"期望={case.expected_intent.value}，"
                            f"实际={response.intent.value}"
                        )

                    # 2. 检查是否正确转人工
                    transfer_is_correct = (
                        response.need_human
                        == case.expected_need_human
                    )

                    if transfer_is_correct:
                        transfer_correct += 1
                    else:
                        case_failures.append(
                            "转人工错误："
                            f"期望={case.expected_need_human}，"
                            f"实际={response.need_human}"
                        )

                    # 3. 检查回答关键词
                    keywords_are_correct = True

                    if case.expected_answer_keywords:
                        keyword_total += 1

                        missing_keywords = (
                            find_missing_keyword_requirements(
                                response.answer,
                                case.expected_answer_keywords,
                            )
                        )

                        if missing_keywords:
                            keywords_are_correct = False

                            case_failures.append(
                                "回答缺少关键词："
                                + ", ".join(
                                    missing_keywords
                                )
                            )
                        else:
                            keyword_correct += 1

                    # 4. 检查知识检索
                    retrieval_is_correct: (
                        bool | None
                    ) = None

                    retrieved_sources: list[str] = []

                    if (
                        case.expected_intent
                        == IntentType.KNOWLEDGE_QUERY
                    ):
                        matches = (
                            agent
                            .tools
                            .retriever
                            .search(case.question)
                        )

                        retrieved_sources = [
                            match.source
                            for match in matches
                        ]

                        required_sources = (
                            case.required_sources
                        )

                        if required_sources:
                            retrieval_total += 1

                            missing_sources = [
                                source
                                for source
                                in required_sources
                                if source
                                not in retrieved_sources
                            ]
                            retrieval_is_correct = (
                                not missing_sources
                            )

                            if retrieval_is_correct:
                                retrieval_correct += 1
                            else:
                                case_failures.append(
                                    "知识召回错误："
                                    f"期望来源="
                                    f"{required_sources}，"
                                    f"实际来源="
                                    f"{retrieved_sources}"
                                )

                        else:
                            # 混合检索负责召回候选知识，不负责最终判定
                            # 问题是否可回答。无答案问题应根据 Agent 的
                            # 最终行为判断是否完成了安全拒绝。
                            rejection_total += 1

                            safe_rejection = (
                                response.need_human
                                and not response.auto_resolved
                                and not response.sources
                            )

                            retrieval_is_correct = safe_rejection

                            if safe_rejection:
                                rejection_correct += 1
                            else:
                                case_failures.append(
                                    "无答案安全拒绝失败："
                                    f"need_human="
                                    f"{response.need_human}，"
                                    f"auto_resolved="
                                    f"{response.auto_resolved}，"
                                    f"sources="
                                    f"{len(response.sources)}"
                                )

                    # 5. 从评测数据库读取链路状态
                    evaluation_record = (
                        agent
                        .evaluation_repository
                        .get(response.request_id)
                    )

                    pipeline_success = bool(
                        evaluation_record
                        and evaluation_record.success
                    )

                    if pipeline_success:
                        pipeline_success_count += 1
                    else:
                        case_failures.append(
                            "技术链路执行失败"
                        )

                    if response.auto_resolved:
                        auto_resolved_count += 1

                    case_passed = (
                        len(case_failures) == 0
                    )
                    category_totals[case.category] = (
                        category_totals.get(
                            case.category,
                            0,
                        )
                        + 1
                    )

                    if case_passed:
                        category_passed[case.category] = (
                            category_passed.get(
                                case.category,
                                0,
                            )
                            + 1
                        )

                    result = {
                        "id": case.id,
                        "category": case.category,
                        "question": case.question,
                        "passed": case_passed,
                        "expected_intent": (
                            case.expected_intent.value
                        ),
                        "actual_intent": (
                            response.intent.value
                        ),
                        "expected_need_human": (
                            case.expected_need_human
                        ),
                        "actual_need_human": (
                            response.need_human
                        ),
                        "auto_resolved": (
                            response.auto_resolved
                        ),
                        "expected_source": (
                            case.expected_source
                        ),
                        "expected_sources": (
                            case.required_sources
                        ),
                        "retrieved_sources": (
                            retrieved_sources
                        ),
                        "answer": response.answer,
                        "duration_ms": (
                            response.total_duration_ms
                        ),
                        "failures": case_failures,
                    }

                    results.append(result)

                    if case_failures:
                        failures.append(result)

                        print(
                            "    结果：FAIL - "
                            + "；".join(case_failures)
                        )
                    else:
                        print("    结果：PASS")

                except Exception as error:
                    category_totals[case.category] = (
                        category_totals.get(
                            case.category,
                            0,
                        )
                        + 1
                    )
                    failure = {
                        "id": case.id,
                        "category": case.category,
                        "question": case.question,
                        "passed": False,
                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    }

                    results.append(failure)
                    failures.append(failure)

                    print(
                        "    结果：ERROR - "
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

    finally:
        # 恢复正式数据库路径
        object.__setattr__(
            settings,
            "database_path",
            original_database_path,
        )

    total_count = len(cases)

    report = {
        "generated_at": (
            datetime.now(timezone.utc)
            .isoformat()
        ),

        "model_name": (
            settings.llm_model_name
        ),

        "embedding_model_name": (
            settings.embedding_model_name
        ),

        "case_count": total_count,

        "metrics": {
            "intent_accuracy": calculate_rate(
                intent_correct,
                total_count,
            ),

            "retrieval_recall_at_k": (
                calculate_rate(
                    retrieval_correct,
                    retrieval_total,
                )
            ),

            "knowledge_abstention_accuracy": (
                calculate_rate(
                    rejection_correct,
                    rejection_total,
                )
            ),

            "human_transfer_accuracy": (
                calculate_rate(
                    transfer_correct,
                    total_count,
                )
            ),

            "answer_keyword_accuracy": (
                calculate_rate(
                    keyword_correct,
                    keyword_total,
                )
            ),

            "pipeline_success_rate": (
                calculate_rate(
                    pipeline_success_count,
                    total_count,
                )
            ),

            "auto_resolution_rate": (
                calculate_rate(
                    auto_resolved_count,
                    total_count,
                )
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

            "category_accuracy": {
                category: calculate_rate(
                    category_passed.get(category, 0),
                    total,
                )
                for category, total
                in sorted(category_totals.items())
            },
        },

        "counts": {
            "intent_correct": intent_correct,
            "retrieval_correct": (
                retrieval_correct
            ),
            "retrieval_total": retrieval_total,
            "rejection_correct": (
                rejection_correct
            ),
            "rejection_total": rejection_total,
            "transfer_correct": (
                transfer_correct
            ),
            "keyword_correct": keyword_correct,
            "keyword_total": keyword_total,
            "failure_count": len(failures),
        },

        "failures": failures,
        "results": results,
    }

    return report


def calculate_p95(
    values: list[float],
) -> float:
    """
    计算P95响应时间。

    P95表示95%的请求响应时间都不超过该值。
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)

    index = int(
        len(sorted_values) * 0.95
    )

    index = min(
        index,
        len(sorted_values) - 1,
    )

    return round(
        sorted_values[index],
        2,
    )


def main() -> None:
    """
    自动评测程序入口。
    """
    parser = argparse.ArgumentParser(
        description="运行智能客服Agent评测"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只运行前N条评测题",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "eval_report.json"
        ),
        help="评测报告输出路径",
    )

    arguments = parser.parse_args()

    eval_path = (
        PROJECT_ROOT
        / "tests"
        / "eval_cases.json"
    )

    cases = load_eval_cases(eval_path)

    if arguments.limit is not None:
        if arguments.limit <= 0:
            raise ValueError(
                "--limit必须大于0"
            )

        cases = cases[:arguments.limit]

    print("=" * 60)
    print("智能客服Agent自动评测")
    print(f"评测题数量：{len(cases)}")
    print("=" * 60)

    report = run_evaluation(cases)

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

    print()
    print("=" * 60)
    print("评测完成")
    print(
        json.dumps(
            report["metrics"],
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"失败题数："
        f"{report['counts']['failure_count']}"
    )
    print(
        f"报告位置："
        f"{arguments.output.resolve()}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
