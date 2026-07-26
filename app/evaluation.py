import json
from typing import Literal

from app.database import (
    get_connection,
    initialize_database,
)
from app.schemas import (
    EvaluationRecord,
    IntentType,
)


class EvaluationRepository:
    """
    评测记录仓库。

    负责：
    1. 保存Agent执行记录；
    2. 查询单条执行记录；
    3. 保存用户反馈；
    4. 统计系统运行指标。
    """

    def __init__(self) -> None:
        # 确保数据库和数据表已经创建
        initialize_database()

    def save(
        self,
        record: EvaluationRecord,
    ) -> None:
        """
        保存一条Agent评测记录。

        如果request_id已经存在，就更新原记录，
        同时保留已经提交的用户反馈。
        """
        connection = get_connection()

        try:
            connection.execute(
                """
                INSERT INTO evaluation_records (
                    request_id,
                    question,
                    predicted_intent,
                    intent_confidence,
                    intent_reason,
                    retrieved_chunk_ids,
                    retrieval_scores,
                    tool_names,
                    answer,
                    need_human,
                    success,
                    auto_resolved,
                    error,
                    total_duration_ms,
                    user_feedback,
                    created_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(request_id) DO UPDATE SET
                    question = excluded.question,
                    predicted_intent = excluded.predicted_intent,
                    intent_confidence = excluded.intent_confidence,
                    intent_reason = excluded.intent_reason,
                    retrieved_chunk_ids = excluded.retrieved_chunk_ids,
                    retrieval_scores = excluded.retrieval_scores,
                    tool_names = excluded.tool_names,
                    answer = excluded.answer,
                    need_human = excluded.need_human,
                    success = excluded.success,
                    auto_resolved = excluded.auto_resolved,
                    error = excluded.error,
                    total_duration_ms = excluded.total_duration_ms,
                    user_feedback = COALESCE(
                        evaluation_records.user_feedback,
                        excluded.user_feedback
                    ),
                    created_at = excluded.created_at
                """,
                (
                    record.request_id,
                    record.question,
                    record.predicted_intent.value,
                    record.intent_confidence,
                    record.intent_reason,

                    json.dumps(
                        record.retrieved_chunk_ids,
                        ensure_ascii=False,
                    ),

                    json.dumps(
                        record.retrieval_scores,
                        ensure_ascii=False,
                    ),

                    json.dumps(
                        record.tool_names,
                        ensure_ascii=False,
                    ),

                    record.answer,
                    int(record.need_human),
                    int(record.success),
                    int(record.auto_resolved),
                    record.error,
                    record.total_duration_ms,
                    record.user_feedback,
                    record.created_at.isoformat(),
                ),
            )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def get(
        self,
        request_id: str,
    ) -> EvaluationRecord | None:
        """
        根据request_id查询一条评测记录。
        """
        connection = get_connection()

        try:
            row = connection.execute(
                """
                SELECT
                    request_id,
                    question,
                    predicted_intent,
                    intent_confidence,
                    intent_reason,
                    retrieved_chunk_ids,
                    retrieval_scores,
                    tool_names,
                    answer,
                    need_human,
                    success,
                    auto_resolved,
                    error,
                    total_duration_ms,
                    user_feedback,
                    created_at
                FROM evaluation_records
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()

            if row is None:
                return None

            return self._row_to_record(row)

        finally:
            connection.close()

    def list_recent(
        self,
        limit: int = 20,
    ) -> list[EvaluationRecord]:
        """
        查询最近的评测记录。
        """
        if limit <= 0 or limit > 100:
            raise ValueError(
                "limit必须在1到100之间"
            )

        connection = get_connection()

        try:
            rows = connection.execute(
                """
                SELECT
                    request_id,
                    question,
                    predicted_intent,
                    intent_confidence,
                    intent_reason,
                    retrieved_chunk_ids,
                    retrieval_scores,
                    tool_names,
                    answer,
                    need_human,
                    success,
                    auto_resolved,
                    error,
                    total_duration_ms,
                    user_feedback,
                    created_at
                FROM evaluation_records
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return [
                self._row_to_record(row)
                for row in rows
            ]

        finally:
            connection.close()

    def update_feedback(
        self,
        request_id: str,
        feedback: Literal[-1, 1],
    ) -> bool:
        """
        保存用户反馈。

        feedback：
            1表示回答有帮助；
            -1表示回答没有帮助。

        返回True表示更新成功，
        返回False表示request_id不存在。
        """
        if feedback not in (-1, 1):
            raise ValueError(
                "feedback只能是-1或1"
            )

        connection = get_connection()

        try:
            cursor = connection.execute(
                """
                UPDATE evaluation_records
                SET user_feedback = ?
                WHERE request_id = ?
                """,
                (
                    feedback,
                    request_id,
                ),
            )

            connection.commit()

            return cursor.rowcount == 1

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def get_summary(self) -> dict:
        """
        统计系统运行指标。

        当前统计：
        - 总请求数；
        - 成功率；
        - 转人工率；
        - 平均响应时间；
        - 用户好评率；
        - 不同意图的数量。
        """
        connection = get_connection()

        try:
            summary_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_count,

                    AVG(success) * 100
                        AS success_rate,

                    AVG(need_human) * 100
                        AS human_transfer_rate,

                    AVG(auto_resolved) * 100
                        AS auto_resolution_rate,

                    AVG(total_duration_ms)
                        AS average_duration_ms,

                    SUM(
                        CASE
                            WHEN user_feedback IS NOT NULL
                            THEN 1
                            ELSE 0
                        END
                    ) AS rated_count,

                    AVG(
                        CASE
                            WHEN user_feedback = 1
                            THEN 1.0
                            WHEN user_feedback = -1
                            THEN 0.0
                            ELSE NULL
                        END
                    ) * 100 AS helpful_rate

                FROM evaluation_records
                """
            ).fetchone()

            intent_rows = connection.execute(
                """
                SELECT
                    predicted_intent,
                    COUNT(*) AS intent_count
                FROM evaluation_records
                GROUP BY predicted_intent
                ORDER BY predicted_intent
                """
            ).fetchall()

            intent_distribution = {
                row["predicted_intent"]: row["intent_count"]
                for row in intent_rows
            }

            return {
                "total_count": (
                    summary_row["total_count"] or 0
                ),

                "success_rate": round(
                    summary_row["success_rate"] or 0.0,
                    2,
                ),

                "human_transfer_rate": round(
                    summary_row["human_transfer_rate"] or 0.0,
                    2,
                ),

                "auto_resolution_rate": round(
                    summary_row["auto_resolution_rate"] or 0.0,
                    2,
                ),

                "average_duration_ms": round(
                    summary_row["average_duration_ms"]
                    or 0.0,
                    2,
                ),

                "rated_count": (
                    summary_row["rated_count"] or 0
                ),

                "helpful_rate": round(
                    summary_row["helpful_rate"] or 0.0,
                    2,
                ),

                "intent_distribution": (
                    intent_distribution
                ),
            }

        finally:
            connection.close()

    def _row_to_record(
        self,
        row,
    ) -> EvaluationRecord:
        """
        把SQLite查询结果转换回EvaluationRecord。
        """
        return EvaluationRecord(
            request_id=row["request_id"],
            question=row["question"],

            predicted_intent=IntentType(
                row["predicted_intent"]
            ),

            intent_confidence=row[
                "intent_confidence"
            ],

            intent_reason=row["intent_reason"],

            retrieved_chunk_ids=json.loads(
                row["retrieved_chunk_ids"]
            ),

            retrieval_scores=json.loads(
                row["retrieval_scores"]
            ),

            tool_names=json.loads(
                row["tool_names"]
            ),

            answer=row["answer"],
            need_human=bool(row["need_human"]),
            success=bool(row["success"]),
            auto_resolved=bool(row["auto_resolved"]),
            error=row["error"],

            total_duration_ms=row[
                "total_duration_ms"
            ],

            user_feedback=row["user_feedback"],
            created_at=row["created_at"],
        )
