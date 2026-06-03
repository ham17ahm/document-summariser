from __future__ import annotations

from dataclasses import dataclass
import os
from time import sleep
from typing import Protocol

from document_summariser.config import ProviderConfig


class ProviderAdapter(Protocol):
    id: str
    model: str
    supports_attachments: bool

    def generate(self, prompt: str, attachments: list[str] | None = None) -> str:
        ...


@dataclass
class MockProvider:
    id: str
    model: str
    supports_attachments: bool = True

    def generate(self, prompt: str, attachments: list[str] | None = None) -> str:
        excerpt = " ".join(prompt.split())[:600]
        return f"[{self.id} / {self.model}] Mock Urdu summary output.\n\n{excerpt}"


@dataclass
class RetryPolicy:
    attempts: int
    initial_delay_seconds: float


class ProviderError(RuntimeError):
    pass


@dataclass
class BaseCloudProvider:
    id: str
    model: str
    config: ProviderConfig
    timeout_seconds: float
    retry_policy: RetryPolicy
    supports_attachments: bool = False

    def generate(self, prompt: str, attachments: list[str] | None = None) -> str:
        if attachments:
            raise ProviderError(f"{self.id} does not support attachments in this pipeline.")

        last_error: Exception | None = None
        for attempt in range(1, self.retry_policy.attempts + 1):
            try:
                text = self._generate_once(prompt)
                if not text.strip():
                    raise ProviderError(f"{self.id} returned an empty response.")
                return text
            except Exception as exc:  # noqa: BLE001 - provider SDKs expose different exception classes
                last_error = exc
                if attempt >= self.retry_policy.attempts:
                    break
                sleep(self.retry_policy.initial_delay_seconds * (2 ** (attempt - 1)))

        raise ProviderError(f"{self.id} failed after {self.retry_policy.attempts} attempts: {last_error}") from last_error

    def _generate_once(self, prompt: str) -> str:
        raise NotImplementedError

    def _api_key(self) -> str:
        env_name = self.config.api_key_env
        if not env_name:
            raise ProviderError(f"Provider {self.id!r} is missing api_key_env in config.")
        value = os.environ.get(env_name)
        if not value:
            raise ProviderError(f"Missing API key for provider {self.id!r}. Set {env_name}.")
        return value


@dataclass
class AnthropicProvider(BaseCloudProvider):
    def _generate_once(self, prompt: str) -> str:
        try:
            from anthropic import Anthropic
        except ModuleNotFoundError as exc:
            raise ProviderError("Install the 'anthropic' package to use Anthropic providers.") from exc

        client = Anthropic(api_key=self._api_key(), timeout=self.timeout_seconds)
        kwargs = {
            "model": self.model,
            "max_tokens": self.config.max_output_tokens or 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.extra:
            for key in ("thinking", "output_config", "service_tier"):
                if key in self.config.extra:
                    kwargs[key] = self.config.extra[key]

        message = client.messages.create(**kwargs)
        return "\n".join(block.text for block in message.content if getattr(block, "type", None) == "text")


@dataclass
class OpenAICompatibleProvider(BaseCloudProvider):
    max_tokens_parameter: str = "max_completion_tokens"

    def _generate_once(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ProviderError("Install the 'openai' package to use OpenAI-compatible providers.") from exc

        client_kwargs = {"api_key": self._api_key(), "timeout": self.timeout_seconds}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        client = OpenAI(**client_kwargs)

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.config.max_output_tokens is not None:
            kwargs[self.max_tokens_parameter] = self.config.max_output_tokens
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return content or ""


@dataclass
class GeminiProvider(BaseCloudProvider):
    supports_attachments: bool = True

    def generate(self, prompt: str, attachments: list[str] | None = None) -> str:
        return self._generate_with_retry(prompt, attachments)

    def _generate_with_retry(self, prompt: str, attachments: list[str] | None = None) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.retry_policy.attempts + 1):
            try:
                text = self._generate_once(prompt, attachments)
                if not text.strip():
                    raise ProviderError(f"{self.id} returned an empty response.")
                return text
            except Exception as exc:  # noqa: BLE001 - provider SDKs expose different exception classes
                last_error = exc
                if attempt >= self.retry_policy.attempts:
                    break
                sleep(self.retry_policy.initial_delay_seconds * (2 ** (attempt - 1)))

        raise ProviderError(f"{self.id} failed after {self.retry_policy.attempts} attempts: {last_error}") from last_error

    def _generate_once(self, prompt: str, attachments: list[str] | None = None) -> str:
        try:
            from google import genai
            from google.genai import types
        except ModuleNotFoundError as exc:
            raise ProviderError("Install the 'google-genai' package to use Gemini providers.") from exc

        client = genai.Client(api_key=self._api_key())
        config_kwargs = {}
        if self.config.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = self.config.max_output_tokens
        if self.config.temperature is not None:
            config_kwargs["temperature"] = self.config.temperature
        if self.config.extra and "thinking_config" in self.config.extra:
            config_kwargs["thinking_config"] = types.ThinkingConfig(**self.config.extra["thinking_config"])

        contents = prompt
        if attachments:
            parts = [types.Part.from_text(text=prompt)]
            for attachment in attachments:
                path = os.fspath(attachment)
                with open(path, "rb") as handle:
                    data = handle.read()
                parts.append(types.Part.from_bytes(data=data, mime_type=_mime_type_for_path(path)))
            contents = [types.Content(role="user", parts=parts)]

        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
        )
        if getattr(response, "text", None):
            return response.text

        parts: list[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
        text = "\n".join(parts)
        if text.strip():
            return text

        diagnostics: list[str] = []
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            thoughts_tokens = getattr(usage, "thoughts_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)
            diagnostics.append(
                f"usage(prompt={prompt_tokens}, thoughts={thoughts_tokens}, total={total_tokens})"
            )
        finish_reasons = [
            str(getattr(candidate, "finish_reason", "unknown"))
            for candidate in getattr(response, "candidates", []) or []
        ]
        if finish_reasons:
            diagnostics.append(f"finish_reasons={finish_reasons}")
        raise ProviderError(
            f"{self.id} returned no text"
            + (f" ({'; '.join(diagnostics)})" if diagnostics else ".")
        )


@dataclass
class UnsupportedProvider:
    id: str
    model: str
    provider_type: str
    supports_attachments: bool = False

    def generate(self, prompt: str, attachments: list[str] | None = None) -> str:
        raise ProviderError(f"Unsupported provider type {self.provider_type!r} for {self.id!r}.")


def _mime_type_for_path(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(".png"):
        return "image/png"
    if lowered.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lowered.endswith(".pdf"):
        return "application/pdf"
    return "application/octet-stream"
