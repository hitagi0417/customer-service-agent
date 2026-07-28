import time
import uuid
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval import KnowledgeRetriever
from app.schemas import ToolCallRecord, utc_now
from app.tickets import Ticket, TicketRepository


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

    request_id: str = Field(
        min_length=1,
        max_length=100,
        description="本次请求的唯一编号",
    )
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="用户的原始问题",
    )
    reason: str = Field(
        min_length=1,
        max_length=1000,
        description="创建人工工单的原因",
    )


class CustomerServiceTools:
    """智能客服工具白名单和统一执行入口。"""

    def __init__(self, retriever: KnowledgeRetriever) -> None:
        self.retriever = retriever
        self.ticket_repository = TicketRepository()

        self._handlers: dict[
            str,
            Callable[[dict[str, Any]], dict[str, Any]],
        ] = {
            "search_knowledge_base": self._search_knowledge_base,
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
        saved, created = (
            self.ticket_repository.create_or_get(
                Ticket(
                    ticket_id=new_ticket_id,
                    request_id=validated.request_id,
                    question=validated.question,
                    reason=validated.reason,
                    status="pending",
                    created_at=utc_now(),
                )
            )
        )

        return {
            "ticket_id": saved.ticket_id,
            "request_id": saved.request_id,
            "question": saved.question,
            "reason": saved.reason,
            "status": saved.status,
            "created_at": saved.created_at.isoformat(),
            "created": created,
        }
