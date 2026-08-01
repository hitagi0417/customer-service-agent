import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.database import get_connection, initialize_database
from app.retrieval import KnowledgeRetriever
from app.schemas import ToolCallRecord, utc_now


class KnowledgeSearchArguments(BaseModel):
    """search_knowledge_base 工具的参数。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        min_length=1,
        max_length=1000,
        description="需要检索的用户问题",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="最多返回的知识片段数量",
    )


class CreateTicketArguments(BaseModel):
    """create_service_ticket 工具的参数。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    request_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=1000)
    reason: str = Field(min_length=1, max_length=1000)


class OrderArguments(BaseModel):
    """订单类只读工具的公共参数。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    order_id: str = Field(min_length=1, max_length=100)
    customer_id: str = Field(min_length=1, max_length=100)



class CustomerServiceTools:
    """智能客服工具白名单和统一执行入口。"""

    def __init__(self, retriever: KnowledgeRetriever) -> None:
        self.retriever = retriever
        initialize_database()

        self._handlers: dict[
            str,
            Callable[[dict[str, Any]], dict[str, Any]],
        ] = {
            "search_knowledge_base": self._search_knowledge_base,
            "query_order": self._query_order,
            "check_refund_eligibility": (
                self._check_refund_eligibility
            ),
            "create_service_ticket": self._create_service_ticket,
        }

    def available_tool_names(self) -> list[str]:
        """返回 Agent 允许调用的工具名。"""
        return list(self._handlers)

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolCallRecord:
        """校验白名单、执行工具并统一记录结果和耗时。"""
        started_at = time.perf_counter()
        handler = self._handlers.get(tool_name)

        if handler is None:
            duration_ms = (time.perf_counter() - started_at) * 1000
            return ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                error=f"不允许调用工具：{tool_name}",
                duration_ms=round(duration_ms, 2),
            )

        try:
            result = handler(arguments)
            duration_ms = (time.perf_counter() - started_at) * 1000
            return ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                success=True,
                result=result,
                duration_ms=round(duration_ms, 2),
            )
        except Exception as error:
            duration_ms = (time.perf_counter() - started_at) * 1000
            return ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                error=f"{type(error).__name__}: {error}",
                duration_ms=round(duration_ms, 2),
            )

    def _search_knowledge_base(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """执行知识库搜索。"""
        validated = KnowledgeSearchArguments.model_validate(arguments)
        matches = self.retriever.search(
            query=validated.query,
            top_k=validated.top_k,
        )

        return {
            "match_count": len(matches),
            "matches": [
                match.model_dump(mode="json")
                for match in matches
            ],
        }

    def _create_service_ticket(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """创建工单；同一个 request_id 重试时返回原工单。"""
        validated = CreateTicketArguments.model_validate(arguments)
        new_ticket_id = f"ticket_{uuid.uuid4().hex[:12]}"
        created_at = utc_now().isoformat()
        connection = get_connection()

        try:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO service_tickets (
                    ticket_id,
                    request_id,
                    question,
                    reason,
                    status,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    new_ticket_id,
                    validated.request_id,
                    validated.question,
                    validated.reason,
                    "pending",
                    created_at,
                ),
            )
            connection.commit()

            created = cursor.rowcount == 1
            row = connection.execute(
                """
                SELECT
                    ticket_id,
                    request_id,
                    question,
                    reason,
                    status,
                    created_at
                FROM service_tickets
                WHERE request_id = ?
                """,
                (validated.request_id,),
            ).fetchone()

            if row is None:
                raise RuntimeError("创建工单后无法查询到工单")

            return {
                "ticket_id": row["ticket_id"],
                "request_id": row["request_id"],
                "question": row["question"],
                "reason": row["reason"],
                "status": row["status"],
                "created_at": row["created_at"],
                "created": created,
            }
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _query_order(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """按客户权限边界查询订单，不允许跨客户读取。"""
        validated = OrderArguments.model_validate(arguments)
        row = self._get_order(
            order_id=validated.order_id,
            customer_id=validated.customer_id,
        )
        if row is None:
            return {
                "found": False,
                "order_id": validated.order_id,
            }
        return {
            "found": True,
            "order_id": row["order_id"],
            "item_name": row["item_name"],
            "amount": row["amount"],
            "status": row["status"],
            "paid_at": row["paid_at"],
            "shipped_at": row["shipped_at"],
            "delivered_at": row["delivered_at"],
        }

    def _check_refund_eligibility(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """根据订单状态和签收时间判断退款时间条件。"""
        validated = OrderArguments.model_validate(arguments)
        row = self._get_order(
            order_id=validated.order_id,
            customer_id=validated.customer_id,
        )
        if row is None:
            return {
                "found": False,
                "eligible": False,
                "reason": "未找到当前客户的订单",
            }

        if row["status"] in {"cancelled", "refunded"}:
            return {
                "found": True,
                "eligible": False,
                "reason": f"订单状态为{row['status']}，不能重复申请",
            }

        if row["delivered_at"] is None:
            return {
                "found": True,
                "eligible": False,
                "reason": "订单尚未签收，需要按取消订单流程处理",
                "status": row["status"],
            }

        delivered_at = datetime.fromisoformat(row["delivered_at"])
        if delivered_at.tzinfo is None:
            delivered_at = delivered_at.replace(tzinfo=timezone.utc)
        elapsed_days = max(
            0,
            (utc_now() - delivered_at).days,
        )
        refundable_days = int(row["refundable_days"])
        eligible = elapsed_days <= refundable_days
        return {
            "found": True,
            "eligible": eligible,
            "order_id": row["order_id"],
            "status": row["status"],
            "elapsed_days": elapsed_days,
            "refundable_days": refundable_days,
            "reason": (
                "仍在退款申请期限内"
                if eligible
                else "已超过退款申请期限"
            ),
        }

    @staticmethod
    def _get_order(order_id: str, customer_id: str):
        connection = get_connection()
        try:
            return connection.execute(
                """
                SELECT
                    order_id,
                    customer_id,
                    item_name,
                    amount,
                    status,
                    paid_at,
                    shipped_at,
                    delivered_at,
                    refundable_days
                FROM orders
                WHERE order_id = ? AND customer_id = ?
                """,
                (order_id, customer_id),
            ).fetchone()
        finally:
            connection.close()
