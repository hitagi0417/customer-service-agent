from fastapi.testclient import TestClient

from app.api import create_app
from app.config import settings
from app.schemas import AgentResponse, IntentType


class FakeAgent:
    def run(
        self,
        question: str,
        conversation_id: str | None = None,
    ) -> AgentResponse:
        return AgentResponse(
            request_id="request_test",
            answer=f"收到：{question}",
            intent=IntentType.CHITCHAT,
            total_duration_ms=1.0,
        )


class BrokenAgent:
    def run(
        self,
        question: str,
        conversation_id: str | None = None,
    ) -> AgentResponse:
        raise RuntimeError("不应泄露的内部错误")


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
