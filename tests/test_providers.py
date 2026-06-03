import sys
from types import SimpleNamespace

import pytest

from document_summariser.config import ProviderConfig
from document_summariser.providers.base import AnthropicProvider, OpenAICompatibleProvider, ProviderError, RetryPolicy
from document_summariser.providers.registry import build_provider_registry


def test_cloud_provider_requires_configured_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAICompatibleProvider(
        id="gpt",
        model="gpt-test",
        config=ProviderConfig(
            id="gpt",
            type="openai",
            model="gpt-test",
            api_key_env="OPENAI_API_KEY",
        ),
        timeout_seconds=1,
        retry_policy=RetryPolicy(attempts=1, initial_delay_seconds=0),
    )

    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        provider.generate("Summarise this.")


def test_registry_builds_deepseek_as_openai_compatible_provider():
    config = _provider_registry_config()

    registry = build_provider_registry(config)

    provider = registry["deepseek"]
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.max_tokens_parameter == "max_tokens"


def test_registry_builds_grok_as_openai_compatible_provider():
    config = _provider_registry_config()

    registry = build_provider_registry(config)

    provider = registry["grok"]
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.max_tokens_parameter == "max_tokens"
    assert provider.config.base_url == "https://api.x.ai/v1"


def test_anthropic_provider_passes_thinking_and_effort(monkeypatch):
    captured: dict = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="thinking", thinking=""),
                    SimpleNamespace(type="text", text="final summary"),
                ]
            )

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.messages = FakeMessages()

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=FakeAnthropic),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider(
        id="claude",
        model="claude-opus-4-7",
        config=ProviderConfig(
            id="claude",
            type="anthropic",
            model="claude-opus-4-7",
            api_key_env="ANTHROPIC_API_KEY",
            max_output_tokens=64000,
            extra={
                "thinking": {"type": "adaptive", "display": "omitted"},
                "output_config": {"effort": "xhigh"},
            },
        ),
        timeout_seconds=1,
        retry_policy=RetryPolicy(attempts=1, initial_delay_seconds=0),
    )

    assert provider.generate("Consolidate this.") == "final summary"
    assert captured["model"] == "claude-opus-4-7"
    assert captured["max_tokens"] == 64000
    assert captured["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert captured["output_config"] == {"effort": "xhigh"}


def _provider_registry_config():
    class Config:
        runtime = {"retries": 1, "request_timeout_seconds": 1}
        providers = {
            "deepseek": ProviderConfig(
                id="deepseek",
                type="deepseek",
                model="deepseek-test",
                api_key_env="DEEPSEEK_API_KEY",
                base_url="https://api.deepseek.com",
            ),
            "grok": ProviderConfig(
                id="grok",
                type="grok",
                model="grok-test",
                api_key_env="XAI_API_KEY",
                base_url="https://api.x.ai/v1",
            ),
        }

    return Config()
