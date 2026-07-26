from pathlib import Path

import pytest

from app.config import settings
from app.database import get_connection
from app.evaluation import EvaluationRepository
from app.schemas import (
    EvaluationRecord,
    IntentType,
    KnowledgeMatch,
)
from app.tools import CustomerServiceTools


class FakeRetriever:
    """
    模拟知识检索器。

    避免单元测试加载真实向量模型。
    """

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[KnowledgeMatch]:
        return [
            KnowledgeMatch(
                chunk_id="chunk_test_001",
                source="company.txt",
                content=(
                    "客服工作时间是周一到周五"
                    "9:00到18:00。"
                ),
                score=0.9,
            )
        ]


@pytest.fixture
def temporary_database(
    tmp_path: Path,
):
    """
    每个测试使用独立的临时数据库。

    测试完成后恢复正式数据库路径。
    """
    original_path = settings.database_path

    test_database_path = (
        tmp_path / "test_customer_service.db"
    )

    object.__setattr__(
        settings,
        "database_path",
        test_database_path,
    )

    try:
        yield test_database_path

    finally:
        object.__setattr__(
            settings,
            "database_path",
            original_path,
        )


@pytest.fixture
def tools(
    temporary_database: Path,
) -> CustomerServiceTools:
    """
    创建使用临时数据库的工具集合。
    """
    return CustomerServiceTools(
        retriever=FakeRetriever()
    )


def test_search_knowledge_base(
    tools: CustomerServiceTools,
) -> None:
    result = tools.execute(
        tool_name="search_knowledge_base",
        arguments={
            "query": "客服几点上班？",
        },
    )

    assert result.success is True
    assert result.error is None
    assert result.result is not None
    assert result.result["match_count"] == 1

    match = result.result["matches"][0]

    assert match["source"] == "company.txt"
    assert match["score"] == 0.9
    assert "周一到周五" in match["content"]


def test_unknown_tool_is_blocked(
    tools: CustomerServiceTools,
) -> None:
    result = tools.execute(
        tool_name="delete_database",
        arguments={},
    )

    assert result.success is False
    assert result.result is None
    assert result.error is not None
    assert "不允许调用工具" in result.error


def test_invalid_tool_arguments_are_blocked(
    tools: CustomerServiceTools,
) -> None:
    result = tools.execute(
        tool_name="search_knowledge_base",
        arguments={
            "query": "退款政策",
            "top_k": 999,
        },
    )

    assert result.success is False
    assert result.result is None
    assert result.error is not None
    assert "ValidationError" in result.error


def test_create_service_ticket(
    tools: CustomerServiceTools,
) -> None:
    result = tools.execute(
        tool_name="create_service_ticket",
        arguments={
            "request_id": "request_test_001",
            "question": "帮我申请退款",
            "reason": "用户要求执行退款操作",
        },
    )

    assert result.success is True
    assert result.result is not None
    assert result.result["created"] is True
    assert result.result["status"] == "pending"

    assert result.result["ticket_id"].startswith(
        "ticket_"
    )


def test_ticket_creation_is_idempotent(
    tools: CustomerServiceTools,
) -> None:
    arguments = {
        "request_id": "request_same_001",
        "question": "帮我转人工客服",
        "reason": "用户要求人工处理",
    }

    first_result = tools.execute(
        tool_name="create_service_ticket",
        arguments=arguments,
    )

    second_result = tools.execute(
        tool_name="create_service_ticket",
        arguments=arguments,
    )

    assert first_result.success is True
    assert second_result.success is True

    assert first_result.result is not None
    assert second_result.result is not None

    assert first_result.result["created"] is True
    assert second_result.result["created"] is False

    assert (
        first_result.result["ticket_id"]
        == second_result.result["ticket_id"]
    )

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT COUNT(*) AS ticket_count
            FROM service_tickets
            WHERE request_id = ?
            """,
            ("request_same_001",),
        ).fetchone()

        assert row["ticket_count"] == 1

    finally:
        connection.close()


def test_evaluation_record_can_be_saved(
    temporary_database: Path,
) -> None:
    repository = EvaluationRepository()

    record = EvaluationRecord(
        request_id="evaluation_test_001",
        question="客服几点上班？",
        predicted_intent=(
            IntentType.KNOWLEDGE_QUERY
        ),
        intent_confidence=0.95,
        intent_reason="用户正在咨询客服时间",
        retrieved_chunk_ids=[
            "chunk_test_001"
        ],
        retrieval_scores=[0.9],
        tool_names=[
            "search_knowledge_base"
        ],
        answer=(
            "客服工作时间是周一到周五"
            "9:00到18:00。"
        ),
        need_human=False,
        success=True,
        auto_resolved=True,
        total_duration_ms=500.0,
    )

    repository.save(record)

    saved = repository.get(
        "evaluation_test_001"
    )

    assert saved is not None
    assert saved.question == "客服几点上班？"
    assert saved.auto_resolved is True

    assert saved.predicted_intent == (
        IntentType.KNOWLEDGE_QUERY
    )

    assert saved.retrieved_chunk_ids == [
        "chunk_test_001"
    ]


def test_user_feedback_is_preserved(
    temporary_database: Path,
) -> None:
    repository = EvaluationRepository()

    original_record = EvaluationRecord(
        request_id="feedback_test_001",
        question="退款需要什么条件？",
        predicted_intent=(
            IntentType.KNOWLEDGE_QUERY
        ),
        intent_confidence=0.9,
        intent_reason="用户正在咨询退款政策",
        answer="退款需要在7天内申请。",
        need_human=False,
        success=True,
        auto_resolved=True,
        total_duration_ms=600.0,
    )

    repository.save(original_record)

    updated = repository.update_feedback(
        request_id="feedback_test_001",
        feedback=1,
    )

    assert updated is True

    # 模拟同一个request_id重新保存执行记录
    new_record = original_record.model_copy(
        update={
            "answer": "退款申请需要在7天内提交。",
            "user_feedback": None,
        }
    )

    repository.save(new_record)

    saved = repository.get(
        "feedback_test_001"
    )

    assert saved is not None

    # 重新保存不能清除原来的用户评价
    assert saved.user_feedback == 1


def test_evaluation_summary(
    temporary_database: Path,
) -> None:
    repository = EvaluationRepository()

    record = EvaluationRecord(
        request_id="summary_test_001",
        question="客服几点上班？",
        predicted_intent=(
            IntentType.KNOWLEDGE_QUERY
        ),
        intent_confidence=0.95,
        intent_reason="用户正在咨询客服时间",
        answer="客服工作时间是周一到周五。",
        need_human=False,
        success=True,
        auto_resolved=True,
        total_duration_ms=1000.0,
        user_feedback=1,
    )

    repository.save(record)

    summary = repository.get_summary()

    assert summary["total_count"] == 1
    assert summary["success_rate"] == 100.0
    assert summary["auto_resolution_rate"] == 100.0
    assert summary["human_transfer_rate"] == 0.0
    assert summary["helpful_rate"] == 100.0
    assert summary["average_duration_ms"] == 1000.0

    assert summary["intent_distribution"] == {
        "knowledge_query": 1
    }