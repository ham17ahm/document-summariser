import sys
from types import SimpleNamespace

import pytest

from document_summariser.config import ProviderConfig
from document_summariser.errors import ConfigError
from document_summariser.providers.base import (
    AnthropicProvider,
    BaseCloudProvider,
    GenerationResult,
    GeminiProvider,
    OpenAICompatibleProvider,
    ProviderError,
    RetryPolicy,
    XAIProvider,
)
from document_summariser.providers.registry import build_provider_registry
from document_summariser.prompts import PromptRequest


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
        provider.generate(_request("Summarise this."))


def test_retry_skips_non_retryable_errors():
    calls = {"count": 0}

    class FailingProvider(BaseCloudProvider):
        def _generate_once(self, request: PromptRequest) -> GenerationResult:
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
        provider.generate(_request("Summarise this."))
    assert calls["count"] == 1


def test_retry_retries_transient_errors():
    calls = {"count": 0}

    class FlakyProvider(BaseCloudProvider):
        def _generate_once(self, request: PromptRequest) -> GenerationResult:
            calls["count"] += 1
            if calls["count"] < 2:
                raise RuntimeError("transient network error")
            return GenerationResult("ok")

    provider = FlakyProvider(
        id="p",
        model="m",
        config=ProviderConfig(id="p", type="openai", model="m"),
        timeout_seconds=1,
        retry_policy=RetryPolicy(attempts=3, initial_delay_seconds=0),
    )

    assert provider.generate(_request("Summarise this.")).text == "ok"
    assert calls["count"] == 2


def test_retry_skips_non_retryable_http_status():
    calls = {"count": 0}

    class BadRequestError(Exception):
        status_code = 400

    class FailingProvider(BaseCloudProvider):
        def _generate_once(self, request: PromptRequest) -> GenerationResult:
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
        provider.generate(_request("Summarise this."))
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
    assert isinstance(provider, XAIProvider)
    assert provider.max_tokens_parameter == "max_tokens"
    assert provider.config.base_url == "https://api.x.ai/v1"


def test_openai_provider_sends_cache_key_and_records_cache_usage(monkeypatch):
    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))],
                usage=SimpleNamespace(
                    prompt_tokens=2000,
                    completion_tokens=100,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=1500),
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=25),
                ),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAICompatibleProvider(
        id="chatgpt",
        model="gpt-5.2",
        config=ProviderConfig(
            id="chatgpt",
            type="openai",
            model="gpt-5.2",
            api_key_env="OPENAI_API_KEY",
            extra={
                "prompt_cache": {
                    "enabled": True,
                    "retention": "in_memory",
                }
            },
        ),
        timeout_seconds=1,
        retry_policy=RetryPolicy(attempts=1, initial_delay_seconds=0),
    )

    result = provider.generate(_request("Document text"))

    assert result.text == "summary"
    assert captured["messages"][0] == {
        "role": "system",
        "content": "Stable system instructions.",
    }
    assert captured["prompt_cache_key"] == "cache-key"
    assert captured["prompt_cache_retention"] == "in_memory"
    assert result.usage.cached_input_tokens == 1500
    assert result.usage.cache_miss_input_tokens == 500
    assert result.usage.reasoning_tokens == 25


def test_xai_provider_sends_sticky_routing_header():
    provider = XAIProvider(
        id="grok",
        model="grok-4.3",
        config=ProviderConfig(
            id="grok",
            type="grok",
            model="grok-4.3",
            extra={"prompt_cache": {"enabled": True}},
        ),
        timeout_seconds=1,
        retry_policy=RetryPolicy(attempts=1, initial_delay_seconds=0),
        max_tokens_parameter="max_tokens",
    )

    assert provider._cache_request_options(_request("Document text")) == {
        "extra_headers": {
            "x-grok-conv-id": "cache-key",
        }
    }


def test_anthropic_provider_passes_thinking_and_effort(monkeypatch):
    captured: dict = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="thinking", thinking=""),
                    SimpleNamespace(type="text", text="final summary"),
                ],
                usage=SimpleNamespace(
                    input_tokens=200,
                    output_tokens=50,
                    cache_read_input_tokens=1800,
                    cache_creation_input_tokens=0,
                ),
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
                "prompt_cache": {"enabled": True, "ttl": "5m"},
                "thinking": {"type": "adaptive", "display": "omitted"},
                "output_config": {"effort": "xhigh"},
            },
        ),
        timeout_seconds=1,
        retry_policy=RetryPolicy(attempts=1, initial_delay_seconds=0),
    )

    result = provider.generate(_request("Consolidate this."))
    assert result.text == "final summary"
    assert captured["model"] == "claude-opus-4-8"
    assert captured["max_tokens"] == 64000
    assert captured["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert captured["output_config"] == {"effort": "xhigh"}
    assert captured["system"] == [
        {
            "type": "text",
            "text": "Stable system instructions.",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        }
    ]
    assert captured["messages"] == [{"role": "user", "content": "Consolidate this."}]
    assert result.usage.input_tokens == 2000
    assert result.usage.cached_input_tokens == 1800
    assert result.usage.cache_miss_input_tokens == 200


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
            return SimpleNamespace(
                text="corrected text",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=3000,
                    candidates_token_count=500,
                    cached_content_token_count=2000,
                    thoughts_token_count=100,
                ),
            )

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

    result = provider.generate(_request("Correct this."))
    assert result.text == "corrected text"
    assert captured["model"] == "gemini-3.1-pro-preview"
    assert captured["client_kwargs"]["http_options"] == {"timeout": 1000}
    assert captured["config_kwargs"]["max_output_tokens"] == 16384
    thinking_config = captured["config_kwargs"]["thinking_config"]
    assert thinking_config.kwargs == {"thinking_budget": 1024}
    assert captured["config_kwargs"]["system_instruction"] == "Stable system instructions."
    assert result.usage.cached_input_tokens == 2000
    assert result.usage.cache_miss_input_tokens == 1000
    assert result.avg_logprobs is None


def test_gemini_provider_reads_avg_logprobs(monkeypatch):
    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            pass

    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(
                text="corrected text",
                candidates=[
                    SimpleNamespace(finish_reason="FinishReason.STOP", avg_logprobs=-0.15)
                ],
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
        ),
        timeout_seconds=1,
        retry_policy=RetryPolicy(attempts=1, initial_delay_seconds=0),
    )

    result = provider.generate(_request("Correct this."))
    assert result.text == "corrected text"
    assert result.avg_logprobs == -0.15


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
        provider.generate(_request("Correct this."))
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

    assert (
        provider.generate(_request("Correct this."), attachments=[str(image_path)]).text
        == "corrected text"
    )
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


def _request(user: str) -> PromptRequest:
    return PromptRequest(
        system="Stable system instructions.",
        user=user,
        cache_key="cache-key",
    )
