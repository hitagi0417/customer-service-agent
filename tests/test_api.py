from fastapi.testclient import TestClient

from app.api import create_app
from app.config import settings
from app.schemas import (
    AgentResponse,
    IntentType,
    TokenUsage,
    ToolCallRecord,
)


class FakeAgent:
    evaluation_repository = None

    def run(
        self,
        question: str,
        conversation_id: str | None = None,
    ) -> AgentResponse:
        return AgentResponse(
            request_id="request_test",
            answer=f"收到：{question}",
            intent=IntentType.CHITCHAT,
            tool_calls=[
                ToolCallRecord(
                    tool_name="test_tool",
                    success=True,
                    duration_ms=1.0,
                )
            ],
            auto_resolved=True,
            token_usage=TokenUsage(
                model_calls=1,
                usage_available_calls=1,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            total_duration_ms=1.0,
        )


class BrokenAgent:
    def run(
        self,
        question: str,
        conversation_id: str | None = None,
    ) -> AgentResponse:
        raise RuntimeError("不应泄露的内部错误")


class FakeEvaluationRepository:
    def __init__(self) -> None:
        self.feedback: tuple[str, int] | None = None

    def update_feedback(
        self,
        request_id: str,
        feedback: int,
    ) -> bool:
        if request_id == "missing":
            return False

        self.feedback = (
            request_id,
            feedback,
        )
        return True

    def get_summary(self) -> dict:
        return {
            "total_count": 1,
        }


class FeedbackAgent(FakeAgent):
    def __init__(self) -> None:
        self.evaluation_repository = (
            FakeEvaluationRepository()
        )


def set_service_api_key(value: str | None) -> str | None:
    old_value = settings.service_api_key
    object.__setattr__(
        settings,
        "service_api_key",
        value,
    )
    return old_value


def test_health_chat_and_metrics() -> None:
    old_key = set_service_api_key(None)

    try:
        app = create_app(lambda: FakeAgent())

        with TestClient(app) as client:
            assert client.get(
                "/health/live"
            ).json() == {"status": "alive"}
            assert client.get(
                "/health/ready"
            ).json() == {"status": "ready"}

            response = client.post(
                "/api/chat",
                json={"question": "你好"},
            )

            assert response.status_code == 200
            assert response.json()["answer"] == "收到：你好"

            metrics = client.get("/metrics").json()
            assert metrics["requests_total"] == 1
            assert metrics["requests_succeeded"] == 1
            assert metrics["requests_failed"] == 0
            assert metrics["tool_success_rate"] == 100.0
            assert metrics["auto_resolution_rate"] == 100.0
            assert metrics["total_tokens"] == 15
    finally:
        object.__setattr__(
            settings,
            "service_api_key",
            old_key,
        )


def test_api_key_authentication() -> None:
    old_key = set_service_api_key("test-secret")

    try:
        app = create_app(lambda: FakeAgent())

        with TestClient(app) as client:
            assert client.post(
                "/api/chat",
                json={"question": "你好"},
            ).status_code == 401

            response = client.post(
                "/api/chat",
                headers={"X-API-Key": "test-secret"},
                json={"question": "你好"},
            )
            assert response.status_code == 200
    finally:
        object.__setattr__(
            settings,
            "service_api_key",
            old_key,
        )


def test_internal_error_is_not_exposed() -> None:
    old_key = set_service_api_key(None)

    try:
        app = create_app(lambda: BrokenAgent())

        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={"question": "触发异常"},
            )

            assert response.status_code == 500
            assert "错误编号" in response.json()["detail"]
            assert "不应泄露" not in response.text
    finally:
        object.__setattr__(
            settings,
            "service_api_key",
            old_key,
        )


def test_feedback_api() -> None:
    old_key = set_service_api_key(None)
    agent = FeedbackAgent()

    try:
        app = create_app(lambda: agent)

        with TestClient(app) as client:
            response = client.post(
                "/api/feedback",
                json={
                    "request_id": "request_test",
                    "feedback": 1,
                },
            )

            assert response.status_code == 200
            assert response.json()["saved"] is True
            assert (
                agent.evaluation_repository.feedback
                == ("request_test", 1)
            )

            missing = client.post(
                "/api/feedback",
                json={
                    "request_id": "missing",
                    "feedback": -1,
                },
            )
            assert missing.status_code == 404
    finally:
        object.__setattr__(
            settings,
            "service_api_key",
            old_key,
        )
