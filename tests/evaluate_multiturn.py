import argparse
import json
import sys
import tempfile
import uuid
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from app.agent import create_customer_service_agent
from app.config import PROJECT_ROOT, settings
from app.schemas import IntentType
from tests.evaluate import calculate_rate, find_missing_keyword_requirements


class MultiTurn(BaseModel):
    question: str = Field(min_length=1)
    expected_intent: IntentType
    expected_tools: list[str] = Field(default_factory=list)
    expected_answer_keywords: list[str | list[str]] = Field(default_factory=list)
    expected_rewrite_keywords: list[str] = Field(default_factory=list)
    forbidden_answer_keywords: list[str] = Field(default_factory=list)


class MultiTurnCase(BaseModel):
    id: str = Field(min_length=1)
    customer_id: str = "demo_customer"
    turns: list[MultiTurn] = Field(min_length=2)


def run(cases: list[MultiTurnCase]) -> dict:
    original_database_path = settings.database_path
    results: list[dict] = []
    passed = 0
    total = sum(len(case.turns) for case in cases)
    token_total = 0
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            object.__setattr__(
                settings,
                "database_path",
                Path(temp_dir) / "multiturn-eval.db",
            )
            agent = create_customer_service_agent()
            for case in cases:
                conversation_id = f"eval-{case.id}-{uuid.uuid4().hex[:8]}"
                for index, turn in enumerate(case.turns, start=1):
                    failures: list[str] = []
                    response = agent.run(
                        turn.question,
                        conversation_id=conversation_id,
                        customer_id=case.customer_id,
                    )
                    actual_tools = [item.tool_name for item in response.tool_calls]
                    if response.intent != turn.expected_intent:
                        failures.append(
                            f"intent={response.intent.value}, "
                            f"expected={turn.expected_intent.value}"
                        )
                    for tool_name in turn.expected_tools:
                        if tool_name not in actual_tools:
                            failures.append(f"missing_tool={tool_name}")
                    missing_answer = find_missing_keyword_requirements(
                        response.answer,
                        turn.expected_answer_keywords,
                    )
                    if missing_answer:
                        failures.append(f"missing_answer={missing_answer}")
                    rewrite_text = response.rewritten_question or ""
                    for keyword in turn.expected_rewrite_keywords:
                        if keyword.lower() not in rewrite_text.lower():
                            failures.append(f"missing_rewrite={keyword}")
                    for forbidden in turn.forbidden_answer_keywords:
                        if forbidden.lower() in response.answer.lower():
                            failures.append(f"forbidden_answer={forbidden}")
                    if not failures:
                        passed += 1
                    token_total += response.token_usage.total_tokens
                    results.append(
                        {
                            "case_id": case.id,
                            "turn": index,
                            "passed": not failures,
                            "question": turn.question,
                            "rewritten_question": response.rewritten_question,
                            "actual_tools": actual_tools,
                            "planning_steps": response.planning_steps,
                            "token_usage": response.token_usage.model_dump(),
                            "failures": failures,
                        }
                    )
    finally:
        object.__setattr__(settings, "database_path", original_database_path)
    return {
        "conversation_count": len(cases),
        "turn_count": total,
        "turn_pass_rate": calculate_rate(passed, total),
        "average_total_tokens": round(token_total / total, 2) if total else 0.0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行多轮Agent评测")
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "tests" / "eval_multiturn_cases.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval_multiturn_report.json",
    )
    arguments = parser.parse_args()
    cases = TypeAdapter(list[MultiTurnCase]).validate_json(
        arguments.cases.read_text(encoding="utf-8")
    )
    report = run(cases)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
