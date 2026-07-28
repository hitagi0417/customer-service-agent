from types import SimpleNamespace

from app.config import settings
from app.schemas import TokenUsage
from app.telemetry import (
    extract_token_usage,
    merge_token_usage,
)


def test_extracts_usage_and_estimates_cost() -> None:
    old_input_cost = (
        settings.llm_input_cost_per_million_tokens
    )
    old_output_cost = (
        settings.llm_output_cost_per_million_tokens
    )
    object.__setattr__(
        settings,
        "llm_input_cost_per_million_tokens",
        1.0,
    )
    object.__setattr__(
        settings,
        "llm_output_cost_per_million_tokens",
        2.0,
    )

    try:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=500,
                total_tokens=1500,
                prompt_tokens_details=(
                    SimpleNamespace(
                        cached_tokens=100
                    )
                ),
            )
        )
        usage = extract_token_usage(response)

        assert usage.model_calls == 1
        assert usage.usage_available_calls == 1
        assert usage.prompt_tokens == 1000
        assert usage.completion_tokens == 500
        assert usage.total_tokens == 1500
        assert usage.cached_prompt_tokens == 100
        assert usage.estimated_cost_usd == 0.002
    finally:
        object.__setattr__(
            settings,
            "llm_input_cost_per_million_tokens",
            old_input_cost,
        )
        object.__setattr__(
            settings,
            "llm_output_cost_per_million_tokens",
            old_output_cost,
        )


def test_missing_provider_usage_is_not_fabricated() -> None:
    usage = extract_token_usage(
        SimpleNamespace()
    )

    assert usage.model_calls == 1
    assert usage.usage_available_calls == 0
    assert usage.total_tokens == 0


def test_token_usage_can_be_merged() -> None:
    merged = merge_token_usage(
        TokenUsage(
            model_calls=1,
            usage_available_calls=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            estimated_cost_usd=0.1,
        ),
        TokenUsage(
            model_calls=1,
            usage_available_calls=1,
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
            estimated_cost_usd=0.2,
        ),
    )

    assert merged.model_calls == 2
    assert merged.total_tokens == 45
    assert merged.estimated_cost_usd == 0.3
