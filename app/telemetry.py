from typing import Any

from app.config import settings
from app.schemas import TokenUsage


def _read_value(
    value: Any,
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]

        attribute = getattr(value, name, None)
        if attribute is not None:
            return attribute

    return default


def extract_token_usage(response: Any) -> TokenUsage:
    """
    从OpenAI兼容响应中提取Token数据。

    同时兼容prompt/completion和input/output两套字段名。
    供应商不返回usage时仍记录一次模型调用，但不伪造Token数。
    """
    usage = getattr(response, "usage", None)

    if usage is None:
        return TokenUsage(model_calls=1)

    prompt_tokens = int(
        _read_value(
            usage,
            "prompt_tokens",
            "input_tokens",
            default=0,
        )
        or 0
    )
    completion_tokens = int(
        _read_value(
            usage,
            "completion_tokens",
            "output_tokens",
            default=0,
        )
        or 0
    )
    total_tokens = int(
        _read_value(
            usage,
            "total_tokens",
            default=(
                prompt_tokens
                + completion_tokens
            ),
        )
        or 0
    )
    prompt_details = _read_value(
        usage,
        "prompt_tokens_details",
        "input_tokens_details",
        default=None,
    )
    cached_prompt_tokens = int(
        _read_value(
            prompt_details,
            "cached_tokens",
            default=0,
        )
        or 0
    )
    estimated_cost = (
        prompt_tokens
        * settings.llm_input_cost_per_million_tokens
        / 1_000_000
        + completion_tokens
        * settings.llm_output_cost_per_million_tokens
        / 1_000_000
    )

    return TokenUsage(
        model_calls=1,
        usage_available_calls=1,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        estimated_cost_usd=round(
            estimated_cost,
            8,
        ),
    )


def merge_token_usage(
    *items: TokenUsage,
) -> TokenUsage:
    return TokenUsage(
        model_calls=sum(
            item.model_calls
            for item in items
        ),
        usage_available_calls=sum(
            item.usage_available_calls
            for item in items
        ),
        prompt_tokens=sum(
            item.prompt_tokens
            for item in items
        ),
        completion_tokens=sum(
            item.completion_tokens
            for item in items
        ),
        total_tokens=sum(
            item.total_tokens
            for item in items
        ),
        cached_prompt_tokens=sum(
            item.cached_prompt_tokens
            for item in items
        ),
        estimated_cost_usd=round(
            sum(
                item.estimated_cost_usd
                for item in items
            ),
            8,
        ),
    )


class TokenUsageAccumulator:
    """请求内使用的轻量Token累加器。"""

    def __init__(self) -> None:
        self._usage = TokenUsage()

    def add(
        self,
        usage: TokenUsage,
    ) -> None:
        self._usage = merge_token_usage(
            self._usage,
            usage,
        )

    def snapshot(self) -> TokenUsage:
        return self._usage.model_copy(deep=True)
