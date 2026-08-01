from pathlib import Path

import pytest

from app.config import settings
from app.conversation import ConversationRepository


@pytest.fixture
def conversation_repository(tmp_path: Path):
    original_path = settings.database_path
    object.__setattr__(
        settings,
        "database_path",
        tmp_path / "conversation.db",
    )
    try:
        yield ConversationRepository()
    finally:
        object.__setattr__(settings, "database_path", original_path)


def test_conversation_keeps_recent_messages(
    conversation_repository: ConversationRepository,
) -> None:
    conversation_repository.ensure_conversation("conv-1", "customer-1")
    conversation_repository.append_message("conv-1", "user", "XJ-900多少钱？")
    conversation_repository.append_message("conv-1", "assistant", "标准版899元。")

    context = conversation_repository.load_context(
        "conv-1",
        recent_limit=4,
        summary_trigger_tokens=1000,
        max_summary_chars=500,
    )

    assert context.has_history is True
    assert [item.role for item in context.recent_messages] == [
        "user",
        "assistant",
    ]
    assert "XJ-900" in context.as_prompt_data()["recent_messages"][0]["content"]


def test_conversation_is_isolated_by_customer(
    conversation_repository: ConversationRepository,
) -> None:
    conversation_repository.ensure_conversation("conv-private", "customer-1")

    with pytest.raises(PermissionError):
        conversation_repository.ensure_conversation(
            "conv-private",
            "customer-2",
        )


def test_old_messages_are_compacted_into_rolling_summary(
    conversation_repository: ConversationRepository,
) -> None:
    conversation_repository.ensure_conversation("conv-long", "customer-1")
    for index in range(6):
        conversation_repository.append_message(
            "conv-long",
            "user" if index % 2 == 0 else "assistant",
            f"第{index}条较长的会话消息，用于触发摘要。",
        )

    context = conversation_repository.load_context(
        "conv-long",
        recent_limit=2,
        summary_trigger_tokens=1,
        max_summary_chars=500,
    )

    assert "第0条" in context.summary
    assert len(context.recent_messages) == 2
    assert context.recent_messages[0].content.startswith("第4条")
