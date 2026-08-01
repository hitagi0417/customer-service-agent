from dataclasses import dataclass

from app.database import get_connection, initialize_database
from app.schemas import utc_now


def estimate_tokens(text: str) -> int:
    """提供无模型依赖的保守Token估算，供摘要触发使用。"""
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, (len(cleaned) + 1) // 2)


@dataclass(frozen=True)
class ConversationMessage:
    message_id: int
    role: str
    content: str
    token_estimate: int


@dataclass(frozen=True)
class ConversationContext:
    conversation_id: str
    summary: str
    recent_messages: tuple[ConversationMessage, ...]

    @property
    def has_history(self) -> bool:
        return bool(self.summary or self.recent_messages)

    def as_prompt_data(self) -> dict:
        return {
            "summary": self.summary,
            "recent_messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in self.recent_messages
            ],
        }


class ConversationRepository:
    """持久化会话消息，并维护结构化滚动摘要。"""

    def __init__(self) -> None:
        initialize_database()

    def ensure_conversation(
        self,
        conversation_id: str,
        customer_id: str,
    ) -> None:
        now = utc_now().isoformat()
        connection = get_connection()
        try:
            connection.execute(
                """
                INSERT OR IGNORE INTO conversations (
                    conversation_id,
                    customer_id,
                    summary,
                    created_at,
                    updated_at
                ) VALUES (?, ?, '', ?, ?)
                """,
                (conversation_id, customer_id, now, now),
            )
            row = connection.execute(
                """
                SELECT customer_id
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("无法创建会话")
            if row["customer_id"] != customer_id:
                raise PermissionError("会话不属于当前客户")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> int:
        connection = get_connection()
        try:
            cursor = connection.execute(
                """
                INSERT INTO conversation_messages (
                    conversation_id,
                    role,
                    content,
                    token_estimate,
                    summarized,
                    created_at
                ) VALUES (?, ?, ?, ?, 0, ?)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    estimate_tokens(content),
                    utc_now().isoformat(),
                ),
            )
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE conversation_id = ?
                """,
                (utc_now().isoformat(), conversation_id),
            )
            connection.commit()
            return int(cursor.lastrowid)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def load_context(
        self,
        conversation_id: str,
        recent_limit: int,
        summary_trigger_tokens: int,
        max_summary_chars: int,
    ) -> ConversationContext:
        if recent_limit <= 0:
            raise ValueError("recent_limit必须大于0")

        connection = get_connection()
        try:
            conversation = connection.execute(
                """
                SELECT summary
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise ValueError("会话不存在")

            rows = connection.execute(
                """
                SELECT
                    message_id,
                    role,
                    content,
                    token_estimate
                FROM conversation_messages
                WHERE conversation_id = ? AND summarized = 0
                ORDER BY message_id ASC
                """,
                (conversation_id,),
            ).fetchall()

            total_tokens = sum(row["token_estimate"] for row in rows)
            if (
                total_tokens >= summary_trigger_tokens
                and len(rows) > recent_limit
            ):
                compact_rows = rows[:-recent_limit]
                summary = self._merge_summary(
                    existing=conversation["summary"],
                    rows=compact_rows,
                    max_chars=max_summary_chars,
                )
                compact_ids = [row["message_id"] for row in compact_rows]
                placeholders = ",".join("?" for _ in compact_ids)
                connection.execute(
                    """
                    UPDATE conversations
                    SET summary = ?, updated_at = ?
                    WHERE conversation_id = ?
                    """,
                    (
                        summary,
                        utc_now().isoformat(),
                        conversation_id,
                    ),
                )
                connection.execute(
                    f"""
                    UPDATE conversation_messages
                    SET summarized = 1
                    WHERE message_id IN ({placeholders})
                    """,
                    compact_ids,
                )
                connection.commit()
                conversation = {"summary": summary}
                rows = rows[-recent_limit:]

            recent_rows = rows[-recent_limit:]
            return ConversationContext(
                conversation_id=conversation_id,
                summary=conversation["summary"],
                recent_messages=tuple(
                    ConversationMessage(
                        message_id=row["message_id"],
                        role=row["role"],
                        content=row["content"],
                        token_estimate=row["token_estimate"],
                    )
                    for row in recent_rows
                ),
            )
        finally:
            connection.close()

    @staticmethod
    def _merge_summary(existing, rows, max_chars: int) -> str:
        role_names = {
            "user": "用户",
            "assistant": "助手",
            "tool": "工具",
        }
        additions = [
            f"{role_names.get(row['role'], row['role'])}: "
            f"{row['content'].strip()}"
            for row in rows
        ]
        merged = "\n".join(
            part for part in (existing.strip(), *additions) if part
        )
        if len(merged) <= max_chars:
            return merged
        return "[较早内容已压缩]\n" + merged[-max_chars:]
