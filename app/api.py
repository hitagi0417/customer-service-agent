import asyncio
import logging
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.security import APIKeyHeader

from app.agent import (
    CustomerServiceAgent,
    create_customer_service_agent,
)
from app.config import settings
from app.database import (
    check_database_health,
    dispose_database_engines,
)
from app.schemas import (
    AgentResponse,
    ChatRequest,
    FeedbackRequest,
    FeedbackResponse,
)


logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


@dataclass
class RuntimeMetrics:
    started_at: float = field(
        default_factory=time.time
    )
    requests_total: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    requests_timed_out: int = 0
    total_duration_ms: float = 0.0
    human_transfers: int = 0
    auto_resolved: int = 0
    tool_calls: int = 0
    successful_tool_calls: int = 0
    model_calls: int = 0
    usage_available_calls: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    lock: threading.Lock = field(
        default_factory=threading.Lock
    )

    def record(
        self,
        success: bool,
        timed_out: bool,
        duration_ms: float,
        response: AgentResponse | None = None,
    ) -> None:
        with self.lock:
            self.requests_total += 1
            self.total_duration_ms += duration_ms

            if success:
                self.requests_succeeded += 1
            else:
                self.requests_failed += 1

            if timed_out:
                self.requests_timed_out += 1

            if response is not None:
                if response.need_human:
                    self.human_transfers += 1

                if response.auto_resolved:
                    self.auto_resolved += 1

                self.tool_calls += len(
                    response.tool_calls
                )
                self.successful_tool_calls += sum(
                    1
                    for tool_call in response.tool_calls
                    if tool_call.success
                )
                self.model_calls += (
                    response.token_usage.model_calls
                )
                self.usage_available_calls += (
                    response.token_usage
                    .usage_available_calls
                )
                self.total_tokens += (
                    response.token_usage.total_tokens
                )
                self.estimated_cost_usd += (
                    response.token_usage
                    .estimated_cost_usd
                )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            average_duration = (
                self.total_duration_ms
                / self.requests_total
                if self.requests_total
                else 0.0
            )
            return {
                "uptime_seconds": round(
                    time.time() - self.started_at,
                    2,
                ),
                "requests_total": self.requests_total,
                "requests_succeeded": (
                    self.requests_succeeded
                ),
                "requests_failed": (
                    self.requests_failed
                ),
                "requests_timed_out": (
                    self.requests_timed_out
                ),
                "average_duration_ms": round(
                    average_duration,
                    2,
                ),
                "human_transfer_rate": round(
                    (
                        self.human_transfers
                        / self.requests_succeeded
                        * 100
                    )
                    if self.requests_succeeded
                    else 0.0,
                    2,
                ),
                "auto_resolution_rate": round(
                    (
                        self.auto_resolved
                        / self.requests_succeeded
                        * 100
                    )
                    if self.requests_succeeded
                    else 0.0,
                    2,
                ),
                "tool_call_count": self.tool_calls,
                "tool_success_rate": round(
                    (
                        self.successful_tool_calls
                        / self.tool_calls
                        * 100
                    )
                    if self.tool_calls
                    else 0.0,
                    2,
                ),
                "model_calls": self.model_calls,
                "token_usage_coverage_rate": round(
                    (
                        self.usage_available_calls
                        / self.model_calls
                        * 100
                    )
                    if self.model_calls
                    else 0.0,
                    2,
                ),
                "total_tokens": self.total_tokens,
                "estimated_cost_usd": round(
                    self.estimated_cost_usd,
                    8,
                ),
            }


@dataclass
class ServiceRuntime:
    agent: CustomerServiceAgent
    concurrency_guard: threading.BoundedSemaphore
    metrics: RuntimeMetrics = field(
        default_factory=RuntimeMetrics
    )


def require_api_key(
    api_key: str | None = Depends(
        api_key_header
    ),
) -> None:
    expected = settings.service_api_key

    if expected is None:
        return

    if (
        api_key is None
        or not secrets.compare_digest(
            api_key,
            expected,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或缺少API Key",
        )


def create_app(
    agent_factory: Callable[
        [],
        CustomerServiceAgent,
    ] = create_customer_service_agent,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("正在加载Agent和检索模型")
        agent = await asyncio.to_thread(
            agent_factory
        )
        app.state.runtime = ServiceRuntime(
            agent=agent,
            concurrency_guard=(
                threading.BoundedSemaphore(
                    settings.api_max_concurrency
                )
            ),
        )
        logger.info("API服务准备完成")
        yield
        app.state.runtime = None
        dispose_database_engines()

    application = FastAPI(
        title="智能客服Agent API",
        version="1.0.0",
        docs_url=(
            None
            if settings.app_env == "production"
            else "/docs"
        ),
        redoc_url=None,
        lifespan=lifespan,
    )

    @application.get("/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/health/ready")
    def readiness(
        request: Request,
    ) -> dict[str, str]:
        runtime = getattr(
            request.app.state,
            "runtime",
            None,
        )

        if runtime is None:
            raise HTTPException(
                status_code=503,
                detail="服务尚未准备完成",
            )

        if (
            settings.app_env == "production"
            and not check_database_health()
        ):
            raise HTTPException(
                status_code=503,
                detail="业务数据库不可用",
            )

        if settings.vector_store_backend == "qdrant":
            vector_store = (
                runtime.agent.tools.retriever.vector_store
            )

            if (
                vector_store is None
                or not vector_store.ping()
            ):
                raise HTTPException(
                    status_code=503,
                    detail="向量数据库不可用",
                )

        return {"status": "ready"}

    @application.get(
        "/metrics",
        dependencies=[Depends(require_api_key)],
    )
    def metrics(
        request: Request,
    ) -> dict[str, Any]:
        runtime: ServiceRuntime = (
            request.app.state.runtime
        )
        snapshot = runtime.metrics.snapshot()
        repository = getattr(
            runtime.agent,
            "evaluation_repository",
            None,
        )

        if repository is not None:
            snapshot["evaluation"] = (
                repository.get_summary()
            )

        return snapshot

    @application.post(
        "/api/chat",
        response_model=AgentResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def chat(
        payload: ChatRequest,
        request: Request,
    ) -> AgentResponse:
        runtime: ServiceRuntime = (
            request.app.state.runtime
        )
        started_at = time.perf_counter()
        timed_out = False
        success = False
        agent_response: AgentResponse | None = None

        def execute() -> AgentResponse:
            with runtime.concurrency_guard:
                return runtime.agent.run(
                    question=payload.question,
                    conversation_id=(
                        payload.conversation_id
                    ),
                )

        try:
            agent_response = await asyncio.wait_for(
                asyncio.to_thread(execute),
                timeout=(
                    settings
                    .api_request_timeout_seconds
                ),
            )
            success = True
            return agent_response

        except TimeoutError as error:
            timed_out = True
            raise HTTPException(
                status_code=504,
                detail="Agent处理超时",
            ) from error

        except HTTPException:
            raise

        except Exception as error:
            error_id = uuid.uuid4().hex[:12]
            logger.exception(
                "API请求处理失败，错误编号=%s",
                error_id,
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    "服务内部错误，"
                    f"错误编号：{error_id}"
                ),
            ) from error

        finally:
            duration_ms = (
                time.perf_counter()
                - started_at
            ) * 1000
            runtime.metrics.record(
                success=success,
                timed_out=timed_out,
                duration_ms=duration_ms,
                response=agent_response,
            )

    @application.post(
        "/api/feedback",
        response_model=FeedbackResponse,
        dependencies=[Depends(require_api_key)],
    )
    def save_feedback(
        payload: FeedbackRequest,
        request: Request,
    ) -> FeedbackResponse:
        runtime: ServiceRuntime = (
            request.app.state.runtime
        )
        repository = getattr(
            runtime.agent,
            "evaluation_repository",
            None,
        )

        if repository is None:
            raise HTTPException(
                status_code=503,
                detail="评测存储不可用",
            )

        saved = repository.update_feedback(
            request_id=payload.request_id,
            feedback=payload.feedback,
        )

        if not saved:
            raise HTTPException(
                status_code=404,
                detail="请求记录不存在",
            )

        return FeedbackResponse(
            request_id=payload.request_id,
            feedback=payload.feedback,
            saved=True,
        )

    return application


app = create_app()
