import json
from datetime import datetime
from typing import Literal

from sqlalchemy import (
    Integer,
    case,
    cast,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import (
    insert as postgresql_insert,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.database import (
    database_transaction,
    citation_records,
    evaluation_records,
    get_engine,
    initialize_database,
    tool_execution_records,
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
            "tool_calls": [
                tool_call.model_dump(mode="json")
                for tool_call in record.tool_calls
            ],
            "cited_chunk_ids": (
                record.cited_chunk_ids
            ),
            "cited_sources": record.cited_sources,
            "model_calls": (
                record.token_usage.model_calls
            ),
            "usage_available_calls": (
                record.token_usage
                .usage_available_calls
            ),
            "prompt_tokens": (
                record.token_usage.prompt_tokens
            ),
            "completion_tokens": (
                record.token_usage.completion_tokens
            ),
            "total_tokens": (
                record.token_usage.total_tokens
            ),
            "cached_prompt_tokens": (
                record.token_usage
                .cached_prompt_tokens
            ),
            "estimated_cost_usd": (
                record.token_usage
                .estimated_cost_usd
            ),
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
            connection.execute(
                delete(tool_execution_records).where(
                    tool_execution_records.c.request_id
                    == record.request_id
                )
            )
            connection.execute(
                delete(citation_records).where(
                    citation_records.c.request_id
                    == record.request_id
                )
            )

            if record.tool_calls:
                connection.execute(
                    insert(tool_execution_records),
                    [
                        {
                            "tool_call_id": (
                                f"{record.request_id}:tool:{index}"
                            ),
                            "request_id": record.request_id,
                            "call_index": index,
                            "tool_name": tool_call.tool_name,
                            "success": tool_call.success,
                            "duration_ms": (
                                tool_call.duration_ms
                            ),
                            "error": tool_call.error,
                        }
                        for index, tool_call in enumerate(
                            record.tool_calls
                        )
                    ],
                )

            if record.cited_chunk_ids:
                retrieved_ids = set(
                    record.retrieved_chunk_ids
                )
                connection.execute(
                    insert(citation_records),
                    [
                        {
                            "citation_id": (
                                f"{record.request_id}:"
                                f"citation:{index}"
                            ),
                            "request_id": record.request_id,
                            "citation_index": index,
                            "chunk_id": chunk_id,
                            "source": source,
                            "is_retrieved": (
                                chunk_id in retrieved_ids
                            ),
                        }
                        for index, (
                            chunk_id,
                            source,
                        ) in enumerate(
                            zip(
                                record.cited_chunk_ids,
                                record.cited_sources,
                                strict=True,
                            )
                        )
                    ],
                )

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
            func.sum(
                evaluation_records.c.model_calls
            ).label("model_calls"),
            func.sum(
                evaluation_records.c.usage_available_calls
            ).label("usage_available_calls"),
            func.sum(
                evaluation_records.c.prompt_tokens
            ).label("prompt_tokens"),
            func.sum(
                evaluation_records.c.completion_tokens
            ).label("completion_tokens"),
            func.sum(
                evaluation_records.c.total_tokens
            ).label("total_tokens"),
            func.sum(
                evaluation_records.c.cached_prompt_tokens
            ).label("cached_prompt_tokens"),
            func.sum(
                evaluation_records.c.estimated_cost_usd
            ).label("estimated_cost_usd"),
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
        tool_summary_statement = select(
            func.count().label("tool_call_count"),
            func.sum(
                cast(
                    tool_execution_records.c.success,
                    Integer,
                )
            ).label("tool_success_count"),
            func.avg(
                tool_execution_records.c.duration_ms
            ).label("average_tool_duration_ms"),
        )
        tool_distribution_statement = (
            select(
                tool_execution_records.c.tool_name,
                func.count().label("call_count"),
                func.sum(
                    cast(
                        tool_execution_records.c.success,
                        Integer,
                    )
                ).label("success_count"),
            )
            .group_by(
                tool_execution_records.c.tool_name
            )
            .order_by(
                tool_execution_records.c.tool_name
            )
        )
        citation_summary_statement = select(
            func.count().label("citation_count"),
            func.sum(
                cast(
                    citation_records.c.is_retrieved,
                    Integer,
                )
            ).label("valid_citation_count"),
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
            tool_summary = (
                connection.execute(
                    tool_summary_statement
                )
                .mappings()
                .one()
            )
            tool_distribution = (
                connection.execute(
                    tool_distribution_statement
                )
                .mappings()
                .all()
            )
            citation_summary = (
                connection.execute(
                    citation_summary_statement
                )
                .mappings()
                .one()
            )

        tool_call_count = (
            tool_summary["tool_call_count"] or 0
        )
        tool_success_count = (
            tool_summary["tool_success_count"] or 0
        )
        citation_count = (
            citation_summary["citation_count"] or 0
        )
        valid_citation_count = (
            citation_summary["valid_citation_count"]
            or 0
        )
        model_calls = summary["model_calls"] or 0
        usage_available_calls = (
            summary["usage_available_calls"] or 0
        )
        total_count = summary["total_count"] or 0

        return {
            "total_count": total_count,
            "success_rate": round(
                float(
                    summary["success_rate"] or 0.0
                ),
                2,
            ),
            "human_transfer_rate": round(
                float(
                    summary["human_transfer_rate"]
                    or 0.0
                ),
                2,
            ),
            "auto_resolution_rate": round(
                float(
                    summary["auto_resolution_rate"]
                    or 0.0
                ),
                2,
            ),
            "average_duration_ms": round(
                float(
                    summary["average_duration_ms"]
                    or 0.0
                ),
                2,
            ),
            "rated_count": summary["rated_count"] or 0,
            "helpful_rate": round(
                float(
                    summary["helpful_rate"] or 0.0
                ),
                2,
            ),
            "tool_call_count": tool_call_count,
            "tool_success_count": tool_success_count,
            "tool_success_rate": round(
                (
                    tool_success_count
                    / tool_call_count
                    * 100
                )
                if tool_call_count
                else 0.0,
                2,
            ),
            "average_tool_duration_ms": round(
                float(
                    tool_summary[
                        "average_tool_duration_ms"
                    ]
                    or 0.0
                ),
                2,
            ),
            "tool_distribution": {
                row["tool_name"]: {
                    "call_count": row["call_count"],
                    "success_count": (
                        row["success_count"] or 0
                    ),
                    "success_rate": round(
                        (
                            (row["success_count"] or 0)
                            / row["call_count"]
                            * 100
                        ),
                        2,
                    ),
                }
                for row in tool_distribution
            },
            "citation_count": citation_count,
            "valid_citation_count": (
                valid_citation_count
            ),
            "citation_validity_rate": round(
                (
                    valid_citation_count
                    / citation_count
                    * 100
                )
                if citation_count
                else 0.0,
                2,
            ),
            "model_calls": model_calls,
            "usage_available_calls": (
                usage_available_calls
            ),
            "token_usage_coverage_rate": round(
                (
                    usage_available_calls
                    / model_calls
                    * 100
                )
                if model_calls
                else 0.0,
                2,
            ),
            "prompt_tokens": (
                summary["prompt_tokens"] or 0
            ),
            "completion_tokens": (
                summary["completion_tokens"] or 0
            ),
            "total_tokens": (
                summary["total_tokens"] or 0
            ),
            "cached_prompt_tokens": (
                summary["cached_prompt_tokens"] or 0
            ),
            "average_tokens_per_request": round(
                (
                    (summary["total_tokens"] or 0)
                    / total_count
                )
                if total_count
                else 0.0,
                2,
            ),
            "estimated_cost_usd": round(
                summary["estimated_cost_usd"] or 0.0,
                8,
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
            tool_calls=self._decode_json_list(
                row["tool_calls"]
            ),
            cited_chunk_ids=self._decode_json_list(
                row["cited_chunk_ids"]
            ),
            cited_sources=self._decode_json_list(
                row["cited_sources"]
            ),
            token_usage={
                "model_calls": row["model_calls"],
                "usage_available_calls": (
                    row["usage_available_calls"]
                ),
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": (
                    row["completion_tokens"]
                ),
                "total_tokens": row["total_tokens"],
                "cached_prompt_tokens": (
                    row["cached_prompt_tokens"]
                ),
                "estimated_cost_usd": (
                    row["estimated_cost_usd"]
                ),
            },
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
