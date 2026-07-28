from types import SimpleNamespace

from app.agent import CustomerServiceAgent
from app.schemas import (
    EvaluationRecord,
    IntentResult,
    IntentType,
    KnowledgeMatch,
    TokenUsage,
    ToolCallRecord,
)


class FakeIntentClassifier:
    def classify(
        self,
        question: str,
    ) -> IntentResult:
        return IntentResult(
            intent=IntentType.KNOWLEDGE_QUERY,
            confidence=0.99,
            reason="测试知识查询",
            token_usage=TokenUsage(
                model_calls=1,
                usage_available_calls=1,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )


class FakeTools:
    def __init__(self) -> None:
        self.match = KnowledgeMatch(
            chunk_id="chunk_policy",
            source="policy.md",
            content="退款申请期限为7天。",
            score=0.9,
        )

    def execute(
        self,
        tool_name: str,
        arguments: dict,
    ) -> ToolCallRecord:
        assert tool_name == "search_knowledge_base"
        return ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            success=True,
            result={
                "match_count": 1,
                "matches": [
                    self.match.model_dump(
                        mode="json"
                    )
                ],
            },
            duration_ms=12.5,
        )


class FakeEvaluationRepository:
    def __init__(self) -> None:
        self.saved: EvaluationRecord | None = None

    def save(
        self,
        record: EvaluationRecord,
    ) -> None:
        self.saved = record


class FakeAnswerClient:
    def __init__(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=20,
                completion_tokens=8,
                total_tokens=28,
            ),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"can_answer":true,'
                            '"answer":"退款期限为7天。",'
                            '"cited_chunk_ids":'
                            '["chunk_policy"]}'
                        )
                    )
                )
            ],
        )
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: response
            )
        )


def test_agent_persists_complete_observability() -> None:
    repository = FakeEvaluationRepository()
    agent = CustomerServiceAgent(
        client=FakeAnswerClient(),
        intent_classifier=FakeIntentClassifier(),
        tools=FakeTools(),
        evaluation_repository=repository,
    )

    response = agent.run("退款期限是多久？")

    assert response.auto_resolved is True
    assert response.token_usage.model_calls == 2
    assert response.token_usage.prompt_tokens == 30
    assert response.token_usage.completion_tokens == 13
    assert response.token_usage.total_tokens == 43
    assert len(response.tool_calls) == 1
    assert response.sources[0].chunk_id == "chunk_policy"

    assert repository.saved is not None
    assert repository.saved.tool_calls[0].success is True
    assert repository.saved.cited_chunk_ids == [
        "chunk_policy"
    ]
    assert repository.saved.cited_sources == [
        "policy.md"
    ]
    assert repository.saved.token_usage.total_tokens == 43
