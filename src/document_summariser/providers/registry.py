from __future__ import annotations

from collections.abc import Callable

from document_summariser.config import AppConfig, ProviderConfig
from document_summariser.providers.base import (
    AnthropicProvider,
    BaseCloudProvider,
    GeminiProvider,
    MockProvider,
    OpenAICompatibleProvider,
    ProviderAdapter,
    RetryPolicy,
    UnsupportedProvider,
)

ProviderFactory = type[BaseCloudProvider]
RegistryFactory = Callable[[ProviderConfig, float, RetryPolicy], ProviderAdapter]


def _build_mock_provider(provider: ProviderConfig, timeout_seconds: float, retry_policy: RetryPolicy) -> ProviderAdapter:
    return MockProvider(id=provider.id, model=provider.model)


def _build_openai_provider(provider: ProviderConfig, timeout_seconds: float, retry_policy: RetryPolicy) -> ProviderAdapter:
    return OpenAICompatibleProvider(
        id=provider.id,
        model=provider.model,
        config=provider,
        timeout_seconds=timeout_seconds,
        retry_policy=retry_policy,
        max_tokens_parameter="max_completion_tokens",
    )


def _build_deepseek_provider(provider: ProviderConfig, timeout_seconds: float, retry_policy: RetryPolicy) -> ProviderAdapter:
    return OpenAICompatibleProvider(
        id=provider.id,
        model=provider.model,
        config=provider,
        timeout_seconds=timeout_seconds,
        retry_policy=retry_policy,
        max_tokens_parameter="max_tokens",
    )


def _build_cloud_provider(
    provider_class: ProviderFactory,
    provider: ProviderConfig,
    timeout_seconds: float,
    retry_policy: RetryPolicy,
) -> ProviderAdapter:
    return provider_class(
        id=provider.id,
        model=provider.model,
        config=provider,
        timeout_seconds=timeout_seconds,
        retry_policy=retry_policy,
    )


PROVIDER_FACTORIES: dict[str, RegistryFactory] = {
    "mock": _build_mock_provider,
    "anthropic": lambda provider, timeout, retry: _build_cloud_provider(
        AnthropicProvider, provider, timeout, retry
    ),
    "openai": _build_openai_provider,
    "deepseek": _build_deepseek_provider,
    "gemini": lambda provider, timeout, retry: _build_cloud_provider(GeminiProvider, provider, timeout, retry),
    "google_gemini": lambda provider, timeout, retry: _build_cloud_provider(
        GeminiProvider, provider, timeout, retry
    ),
}


def build_provider_registry(config: AppConfig) -> dict[str, ProviderAdapter]:
    registry: dict[str, ProviderAdapter] = {}
    retry_policy = RetryPolicy(
        attempts=int(config.runtime.get("retries", 3)),
        initial_delay_seconds=float(config.runtime.get("retry_initial_delay_seconds", 1)),
    )
    timeout_seconds = float(config.runtime.get("request_timeout_seconds", 120))

    for provider_id, provider in config.providers.items():
        factory = PROVIDER_FACTORIES.get(provider.type)
        if factory is None:
            registry[provider_id] = UnsupportedProvider(
                id=provider_id,
                model=provider.model,
                provider_type=provider.type,
            )
        else:
            registry[provider_id] = factory(provider, timeout_seconds, retry_policy)
    return registry
