from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from app.agent import CustomerServiceAgent
from app.config import settings
from app.conversation import ConversationRepository
from app.planning import PlanDecision, PlanningResult
from app.schemas import (
    EvaluationRecord,
    IntentResult,
    IntentType,
    TokenUsage,
    ToolCallRecord,
)


class FakeIntentClassifier:
    def classify(self, question: str) -> IntentResult:
        return IntentResult(
            intent=IntentType.SERVICE_REQUEST,
            confidence=0.99,
            reason="需要查询订单",
            token_usage=TokenUsage(
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
            ),
        )


class FakeRewriteClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"standalone_question": '
                            '"XJ-900 的保修期是多久？"}'
                        )
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )


class FakePlanner:
    def __init__(self) -> None:
        self.index = 0

    def plan(self, **kwargs) -> PlanningResult:
        decisions = [
            PlanDecision(
                action="query_order",
                arguments={
                    "order_id": "DEMO-1001",
                    "customer_id": "customer-attacker",
                },
                reason="先查询订单",
            ),
            PlanDecision(
                action="finish",
                reason="已有订单状态",
                answer="订单 DEMO-1001 已签收。",
            ),
        ]
        decision = decisions[self.index]
        self.index += 1
        return PlanningResult(
            decision=decision,
            token_usage=TokenUsage(total_tokens=7),
        )


class FakeTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, tool_name: str, arguments: dict) -> ToolCallRecord:
        self.calls.append((tool_name, arguments))
        return ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            success=True,
            result={
                "found": True,
                "order_id": arguments.get("order_id"),
                "status": "delivered",
            },
        )


@dataclass
class CapturingEvaluationRepository:
    saved: EvaluationRecord | None = None

    def save(self, record: EvaluationRecord) -> None:
        self.saved = record


def test_planner_uses_server_customer_and_records_trace() -> None:
    tools = FakeTools()
    evaluations = CapturingEvaluationRepository()
    agent = CustomerServiceAgent(
        client=object(),
        intent_classifier=FakeIntentClassifier(),
        tools=tools,
        evaluation_repository=evaluations,
        planner=FakePlanner(),
    )

    response = agent.run(
        "查一下 DEMO-1001",
        conversation_id="conv-test",
        customer_id="customer-real",
    )

    assert response.answer == "订单 DEMO-1001 已签收。"
    assert response.planning_steps == ["1:query_order", "2:finish"]
    assert response.token_usage.total_tokens == 19
    assert tools.calls[0][1]["customer_id"] == "customer-real"
    assert "customer-attacker" not in tools.calls[0][1].values()
    assert evaluations.saved is not None
    assert evaluations.saved.planning_steps == response.planning_steps
    assert evaluations.saved.stage_durations_ms["agent_execution"] >= 0


class FinishPlanner:
    def plan(self, question: str, **kwargs) -> PlanningResult:
        return PlanningResult(
            decision=PlanDecision(
                action="finish",
                reason="测试结束",
                answer=f"已理解：{question}",
            ),
            token_usage=TokenUsage(total_tokens=2),
        )


def test_second_turn_is_rewritten_and_persisted(tmp_path: Path) -> None:
    original_path = settings.database_path
    object.__setattr__(settings, "database_path", tmp_path / "agent.db")
    try:
        agent = CustomerServiceAgent(
            client=FakeRewriteClient(),
            intent_classifier=FakeIntentClassifier(),
            tools=FakeTools(),
            evaluation_repository=CapturingEvaluationRepository(),
            planner=FinishPlanner(),
            conversation_repository=ConversationRepository(),
        )
        agent.run(
            "XJ-900 怎么样？",
            conversation_id="conv-rewrite",
        )
        response = agent.run(
            "它保修多久？",
            conversation_id="conv-rewrite",
        )

        assert response.rewritten_question == "XJ-900 的保修期是多久？"
        assert response.answer == "已理解：XJ-900 的保修期是多久？"
        assert response.token_usage.total_tokens == 22
        assert response.stage_durations_ms["question_rewrite"] >= 0
    finally:
        object.__setattr__(settings, "database_path", original_path)
