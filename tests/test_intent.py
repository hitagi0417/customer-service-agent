from types import SimpleNamespace

from app.intent import IntentClassifier
from app.schemas import IntentType


class FakeCompletions:
    """
    模拟OpenAI的completions接口。
    """

    def __init__(self, content: str) -> None:
        self.content = content
        self.called = False

    def create(self, **kwargs):
        self.called = True

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.content
                    )
                )
            ]
        )


class FakeClient:
    """
    模拟OpenAI客户端。

    单元测试不应该调用真实模型，
    否则测试会很慢、产生费用并依赖网络。
    """

    def __init__(self, content: str) -> None:
        self.completions = FakeCompletions(
            content
        )

        self.chat = SimpleNamespace(
            completions=self.completions
        )


def test_classify_knowledge_query() -> None:
    client = FakeClient(
        """
        {
          "intent": "knowledge_query",
          "confidence": 0.95,
          "reason": "用户正在咨询退款政策"
        }
        """
    )

    classifier = IntentClassifier(
        client=client
    )

    result = classifier.classify(
        "退款需要什么条件？"
    )

    assert result.intent == (
        IntentType.KNOWLEDGE_QUERY
    )

    assert result.confidence == 0.95
    assert result.fallback_used is False
    assert result.error is None
    assert client.completions.called is True


def test_classify_service_request() -> None:
    client = FakeClient(
        """
        {
          "intent": "service_request",
          "confidence": 0.98,
          "reason": "用户要求执行退款操作"
        }
        """
    )

    classifier = IntentClassifier(
        client=client
    )

    result = classifier.classify(
        "帮我申请退款"
    )

    assert result.intent == (
        IntentType.SERVICE_REQUEST
    )

    assert result.confidence == 0.98
    assert result.fallback_used is False


def test_invalid_model_output_uses_fallback() -> None:
    client = FakeClient(
        "这不是一个合法的JSON对象"
    )

    classifier = IntentClassifier(
        client=client
    )

    result = classifier.classify(
        "退款需要什么条件？"
    )

    assert result.intent == IntentType.UNKNOWN
    assert result.confidence == 0.0
    assert result.fallback_used is True
    assert result.error == "InvalidModelOutput"


def test_invalid_intent_uses_fallback() -> None:
    client = FakeClient(
        """
        {
          "intent": "refund",
          "confidence": 0.9,
          "reason": "错误的意图名称"
        }
        """
    )

    classifier = IntentClassifier(
        client=client
    )

    result = classifier.classify(
        "退款需要什么条件？"
    )

    assert result.intent == IntentType.UNKNOWN
    assert result.fallback_used is True
    assert result.error == "InvalidModelOutput"


def test_empty_question_does_not_call_model() -> None:
    client = FakeClient(
        """
        {
          "intent": "knowledge_query",
          "confidence": 0.9,
          "reason": "不应该使用这个结果"
        }
        """
    )

    classifier = IntentClassifier(
        client=client
    )

    result = classifier.classify("   ")

    assert result.intent == IntentType.UNKNOWN
    assert result.confidence == 0.0
    assert result.fallback_used is False
    assert client.completions.called is False
