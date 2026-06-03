import pytest

from document_summariser.config import ProviderConfig
from document_summariser.providers.base import OpenAICompatibleProvider, ProviderError, RetryPolicy
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
