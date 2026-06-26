import sys
from types import SimpleNamespace

import pytest

from document_summariser.config import ProviderConfig
from document_summariser.errors import ConfigError
from document_summariser.providers.base import (
    AnthropicProvider,
    BaseCloudProvider,
    GeminiProvider,
    OpenAICompatibleProvider,
    ProviderError,
    RetryPolicy,
)
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


def test_retry_skips_non_retryable_errors():
    calls = {"count": 0}

    class FailingProvider(BaseCloudProvider):
        def _generate_once(self, prompt: str) -> str:
            calls["count"] += 1
            raise ProviderError("bad request", retryable=False)

    provider = FailingProvider(
        id="p",
        model="m",
        config=ProviderConfig(id="p", type="openai", model="m"),
        timeout_seconds=1,
        retry_policy=RetryPolicy(attempts=3, initial_delay_seconds=0),
    )

    with pytest.raises(ProviderError, match="bad request"):
        provider.generate("Summarise this.")
    assert calls["count"] == 1


def test_retry_retries_transient_errors():
    calls = {"count": 0}

    class FlakyProvider(BaseCloudProvider):
        def _generate_once(self, prompt: str) -> str:
            calls["count"] += 1
            if calls["count"] < 2:
                raise RuntimeError("transient network error")
            return "ok"

    provider = FlakyProvider(
        id="p",
        model="m",
        config=ProviderConfig(id="p", type="openai", model="m"),
        timeout_seconds=1,
        retry_policy=RetryPolicy(attempts=3, initial_delay_seconds=0),
    )

    assert provider.generate("Summarise this.") == "ok"
    assert calls["count"] == 2


def test_retry_skips_non_retryable_http_status():
    calls = {"count": 0}

    class BadRequestError(Exception):
        status_code = 400

    class FailingProvider(BaseCloudProvider):
        def _generate_once(self, prompt: str) -> str:
            calls["count"] += 1
            raise BadRequestError("invalid request")

    provider = FailingProvider(
        id="p",
        model="m",
        config=ProviderConfig(id="p", type="openai", model="m"),
        timeout_seconds=1,
        retry_policy=RetryPolicy(attempts=3, initial_delay_seconds=0),
    )

    with pytest.raises(ProviderError, match="invalid request"):
        provider.generate("Summarise this.")
    assert calls["count"] == 1


def test_registry_rejects_unknown_provider_type():
    class Config:
        runtime = {"retries": 1, "request_timeout_seconds": 1}
        providers = {
            "watson": ProviderConfig(id="watson", type="watson", model="watson-test"),
        }

    with pytest.raises(ConfigError, match="watson"):
        build_provider_registry(Config())


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
        model="claude-opus-4-8",
        config=ProviderConfig(
            id="claude",
            type="anthropic",
            model="claude-opus-4-8",
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
    assert captured["model"] == "claude-opus-4-8"
    assert captured["max_tokens"] == 64000
    assert captured["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert captured["output_config"] == {"effort": "xhigh"}


def test_gemini_provider_passes_thinking_budget(monkeypatch):
    captured: dict = {}

    class FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="corrected text")

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.models = FakeModels()

    fake_types = SimpleNamespace(
        GenerateContentConfig=FakeGenerateContentConfig,
        ThinkingConfig=FakeThinkingConfig,
    )
    fake_genai = SimpleNamespace(Client=FakeClient, types=fake_types)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = GeminiProvider(
        id="gemini",
        model="gemini-3.1-pro-preview",
        config=ProviderConfig(
            id="gemini",
            type="gemini",
            model="gemini-3.1-pro-preview",
            api_key_env="GEMINI_API_KEY",
            max_output_tokens=16384,
            extra={"thinking_config": {"thinking_budget": 1024}},
        ),
        timeout_seconds=1,
        retry_policy=RetryPolicy(attempts=1, initial_delay_seconds=0),
    )

    assert provider.generate("Correct this.") == "corrected text"
    assert captured["model"] == "gemini-3.1-pro-preview"
    assert captured["client_kwargs"]["http_options"] == {"timeout": 1000}
    assert captured["config_kwargs"]["max_output_tokens"] == 16384
    thinking_config = captured["config_kwargs"]["thinking_config"]
    assert thinking_config.kwargs == {"thinking_budget": 1024}


def test_gemini_provider_rejects_truncated_response(monkeypatch):
    calls = {"count": 0}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            pass

    class FakeModels:
        def generate_content(self, **kwargs):
            calls["count"] += 1
            return SimpleNamespace(
                text="partial output",
                candidates=[SimpleNamespace(finish_reason="FinishReason.MAX_TOKENS")],
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    fake_types = SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig)
    fake_genai = SimpleNamespace(Client=FakeClient, types=fake_types)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = GeminiProvider(
        id="gemini",
        model="gemini-3.1-pro-preview",
        config=ProviderConfig(
            id="gemini",
            type="gemini",
            model="gemini-3.1-pro-preview",
            api_key_env="GEMINI_API_KEY",
            max_output_tokens=16384,
        ),
        timeout_seconds=1,
        retry_policy=RetryPolicy(attempts=3, initial_delay_seconds=0),
    )

    with pytest.raises(ProviderError, match="max_output_tokens"):
        provider.generate("Correct this.")
    assert calls["count"] == 1


def test_gemini_provider_sends_image_attachments(monkeypatch, tmp_path):
    captured: dict = {}
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image-bytes")

    class FakePart:
        @classmethod
        def from_text(cls, text):
            return {"kind": "text", "text": text}

        @classmethod
        def from_bytes(cls, data, mime_type):
            return {"kind": "bytes", "data": data, "mime_type": mime_type}

    class FakeContent:
        def __init__(self, role, parts):
            self.role = role
            self.parts = parts

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="corrected text")

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    fake_types = SimpleNamespace(
        Content=FakeContent,
        GenerateContentConfig=FakeGenerateContentConfig,
        MediaResolution=SimpleNamespace(MEDIA_RESOLUTION_HIGH="MEDIA_RESOLUTION_HIGH"),
        Part=FakePart,
    )
    fake_genai = SimpleNamespace(Client=FakeClient, types=fake_types)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = GeminiProvider(
        id="gemini",
        model="gemini-3.1-pro-preview",
        config=ProviderConfig(
            id="gemini",
            type="gemini",
            model="gemini-3.1-pro-preview",
            api_key_env="GEMINI_API_KEY",
        ),
        timeout_seconds=1,
        retry_policy=RetryPolicy(attempts=1, initial_delay_seconds=0),
    )

    assert provider.generate("Correct this.", attachments=[str(image_path)]) == "corrected text"
    contents = captured["contents"]
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts[0] == {"kind": "text", "text": "Correct this."}
    assert contents[0].parts[1] == {
        "kind": "bytes",
        "data": b"image-bytes",
        "mime_type": "image/png",
    }
    assert captured["config_kwargs"]["media_resolution"] == "MEDIA_RESOLUTION_HIGH"


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
