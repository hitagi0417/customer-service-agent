import json
from datetime import datetime
from typing import Literal

from sqlalchemy import (
    Integer,
    case,
    cast,
    func,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import (
    insert as postgresql_insert,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.database import (
    database_transaction,
    evaluation_records,
    get_engine,
    initialize_database,
)
from app.schemas import EvaluationRecord, IntentType


class EvaluationRepository:
    """
    Agent执行记录仓储。

    通过SQLAlchemy Core同时支持SQLite和PostgreSQL，
    保存请求链路、召回结果、工具调用和用户反馈。
    """

    def __init__(self) -> None:
        initialize_database()

    def save(
        self,
        record: EvaluationRecord,
    ) -> None:
        """
        幂等保存执行记录。

        request_id重复时更新链路数据，但不会清空用户已经提交的反馈。
        """
        values = {
            "request_id": record.request_id,
            "question": record.question,
            "predicted_intent": (
                record.predicted_intent.value
            ),
            "intent_confidence": (
                record.intent_confidence
            ),
            "intent_reason": record.intent_reason,
            "retrieved_chunk_ids": (
                record.retrieved_chunk_ids
            ),
            "retrieval_scores": record.retrieval_scores,
            "tool_names": record.tool_names,
            "answer": record.answer,
            "need_human": record.need_human,
            "success": record.success,
            "auto_resolved": record.auto_resolved,
            "error": record.error,
            "total_duration_ms": (
                record.total_duration_ms
            ),
            "user_feedback": record.user_feedback,
            "created_at": record.created_at,
        }
        dialect = get_engine().dialect.name

        if dialect == "postgresql":
            statement = postgresql_insert(
                evaluation_records
            ).values(**values)
        else:
            statement = sqlite_insert(
                evaluation_records
            ).values(**values)

        excluded = statement.excluded
        update_values = {
            key: getattr(excluded, key)
            for key in values
            if key not in {
                "request_id",
                "user_feedback",
            }
        }
        update_values["user_feedback"] = func.coalesce(
            evaluation_records.c.user_feedback,
            excluded.user_feedback,
        )
        statement = statement.on_conflict_do_update(
            index_elements=["request_id"],
            set_=update_values,
        )

        with database_transaction() as connection:
            connection.execute(statement)

    def get(
        self,
        request_id: str,
    ) -> EvaluationRecord | None:
        with get_engine().connect() as connection:
            row = (
                connection.execute(
                    select(evaluation_records).where(
                        evaluation_records.c.request_id
                        == request_id
                    )
                )
                .mappings()
                .first()
            )

        if row is None:
            return None

        return self._row_to_record(row)

    def list_recent(
        self,
        limit: int = 20,
    ) -> list[EvaluationRecord]:
        if limit <= 0 or limit > 100:
            raise ValueError("limit必须在1到100之间")

        with get_engine().connect() as connection:
            rows = (
                connection.execute(
                    select(evaluation_records)
                    .order_by(
                        evaluation_records.c.created_at.desc()
                    )
                    .limit(limit)
                )
                .mappings()
                .all()
            )

        return [
            self._row_to_record(row)
            for row in rows
        ]

    def update_feedback(
        self,
        request_id: str,
        feedback: Literal[-1, 1],
    ) -> bool:
        if feedback not in (-1, 1):
            raise ValueError("feedback只能是-1或1")

        with database_transaction() as connection:
            result = connection.execute(
                update(evaluation_records)
                .where(
                    evaluation_records.c.request_id
                    == request_id
                )
                .values(user_feedback=feedback)
            )

        return result.rowcount == 1

    def get_summary(self) -> dict:
        boolean_success = cast(
            evaluation_records.c.success,
            Integer,
        )
        boolean_human = cast(
            evaluation_records.c.need_human,
            Integer,
        )
        boolean_resolved = cast(
            evaluation_records.c.auto_resolved,
            Integer,
        )
        summary_statement = select(
            func.count().label("total_count"),
            (
                func.avg(boolean_success) * 100
            ).label("success_rate"),
            (
                func.avg(boolean_human) * 100
            ).label("human_transfer_rate"),
            (
                func.avg(boolean_resolved) * 100
            ).label("auto_resolution_rate"),
            func.avg(
                evaluation_records.c.total_duration_ms
            ).label("average_duration_ms"),
            func.sum(
                case(
                    (
                        evaluation_records.c.user_feedback
                        .is_not(None),
                        1,
                    ),
                    else_=0,
                )
            ).label("rated_count"),
            (
                func.avg(
                    case(
                        (
                            evaluation_records.c.user_feedback
                            == 1,
                            1.0,
                        ),
                        (
                            evaluation_records.c.user_feedback
                            == -1,
                            0.0,
                        ),
                        else_=None,
                    )
                )
                * 100
            ).label("helpful_rate"),
        )
        intent_statement = (
            select(
                evaluation_records.c.predicted_intent,
                func.count().label("intent_count"),
            )
            .group_by(
                evaluation_records.c.predicted_intent
            )
            .order_by(
                evaluation_records.c.predicted_intent
            )
        )

        with get_engine().connect() as connection:
            summary = (
                connection.execute(summary_statement)
                .mappings()
                .one()
            )
            intents = (
                connection.execute(intent_statement)
                .mappings()
                .all()
            )

        return {
            "total_count": summary["total_count"] or 0,
            "success_rate": round(
                summary["success_rate"] or 0.0,
                2,
            ),
            "human_transfer_rate": round(
                summary["human_transfer_rate"] or 0.0,
                2,
            ),
            "auto_resolution_rate": round(
                summary["auto_resolution_rate"] or 0.0,
                2,
            ),
            "average_duration_ms": round(
                summary["average_duration_ms"] or 0.0,
                2,
            ),
            "rated_count": summary["rated_count"] or 0,
            "helpful_rate": round(
                summary["helpful_rate"] or 0.0,
                2,
            ),
            "intent_distribution": {
                row["predicted_intent"]: (
                    row["intent_count"]
                )
                for row in intents
            },
        }

    def _row_to_record(
        self,
        row,
    ) -> EvaluationRecord:
        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return EvaluationRecord(
            request_id=row["request_id"],
            question=row["question"],
            predicted_intent=IntentType(
                row["predicted_intent"]
            ),
            intent_confidence=row["intent_confidence"],
            intent_reason=row["intent_reason"],
            retrieved_chunk_ids=self._decode_json_list(
                row["retrieved_chunk_ids"]
            ),
            retrieval_scores=self._decode_json_list(
                row["retrieval_scores"]
            ),
            tool_names=self._decode_json_list(
                row["tool_names"]
            ),
            answer=row["answer"],
            need_human=bool(row["need_human"]),
            success=bool(row["success"]),
            auto_resolved=bool(row["auto_resolved"]),
            error=row["error"],
            total_duration_ms=row["total_duration_ms"],
            user_feedback=row["user_feedback"],
            created_at=created_at,
        )

    @staticmethod
    def _decode_json_list(value):
        """兼容早期SQLite表中以TEXT保存的JSON数组。"""
        if isinstance(value, str):
            return json.loads(value)
        return value or []
