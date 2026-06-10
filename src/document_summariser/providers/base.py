from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from time import sleep
from typing import Protocol

from document_summariser.config import ProviderConfig
from document_summariser.errors import ProviderError

# Client errors that retrying cannot fix (bad request, auth, not found,
# unprocessable). Rate limits (429) and server errors (5xx) stay retryable.
_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}


def _http_status(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int) and 100 <= value < 600:
        return value
    return None


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ProviderError):
        return exc.retryable
    status = _http_status(exc)
    if status is not None:
        return status not in _NON_RETRYABLE_STATUS_CODES
    return True


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
            raise ProviderError(
                f"{self.id} does not support attachments in this pipeline.",
                details=self._error_details(attachment_count=len(attachments)),
                retryable=False,
            )
        return self._run_with_retry(lambda: self._generate_once(prompt))

    def _run_with_retry(self, call: Callable[[], str], **extra_details: object) -> str:
        last_error: Exception | None = None
        attempts_made = 0
        for attempt in range(1, self.retry_policy.attempts + 1):
            attempts_made = attempt
            try:
                text = call()
                if not text.strip():
                    raise ProviderError(
                        f"{self.id} returned an empty response.",
                        details=self._error_details(attempt=attempt, **extra_details),
                    )
                return text
            except Exception as exc:  # noqa: BLE001 - provider SDKs expose different exception classes
                last_error = exc
                if not _is_retryable(exc) or attempt >= self.retry_policy.attempts:
                    break
                sleep(self.retry_policy.initial_delay_seconds * (2 ** (attempt - 1)))

        if isinstance(last_error, ProviderError) and not last_error.retryable:
            raise last_error
        raise ProviderError(
            f"{self.id} failed after {attempts_made} attempt(s): {last_error}",
            details=self._error_details(attempts=attempts_made, **extra_details),
            cause=last_error,
        ) from last_error

    def _generate_once(self, prompt: str) -> str:
        raise NotImplementedError

    def _api_key(self) -> str:
        env_name = self.config.api_key_env
        if not env_name:
            raise ProviderError(
                f"Provider {self.id!r} is missing api_key_env in config.",
                details=self._error_details(),
                retryable=False,
            )
        value = os.environ.get(env_name)
        if not value:
            raise ProviderError(
                f"Missing API key for provider {self.id!r}. Set {env_name}.",
                details=self._error_details(api_key_env=env_name),
                retryable=False,
            )
        return value

    def _error_details(self, **extra: object) -> dict[str, object]:
        details: dict[str, object] = {
            "provider": self.id,
            "provider_type": self.config.type,
            "model": self.model,
        }
        details.update(extra)
        return details


@dataclass
class AnthropicProvider(BaseCloudProvider):
    def _generate_once(self, prompt: str) -> str:
        try:
            from anthropic import Anthropic
        except ModuleNotFoundError as exc:
            raise ProviderError(
                "Install the 'anthropic' package to use Anthropic providers.",
                details=self._error_details(),
                cause=exc,
                retryable=False,
            ) from exc

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
            raise ProviderError(
                "Install the 'openai' package to use OpenAI-compatible providers.",
                details=self._error_details(),
                cause=exc,
                retryable=False,
            ) from exc

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
        return self._run_with_retry(
            lambda: self._generate_once(prompt, attachments),
            attachment_count=len(attachments or []),
        )

    def _generate_once(self, prompt: str, attachments: list[str] | None = None) -> str:
        try:
            from google import genai
            from google.genai import types
        except ModuleNotFoundError as exc:
            raise ProviderError(
                "Install the 'google-genai' package to use Gemini providers.",
                details=self._error_details(attachment_count=len(attachments or [])),
                cause=exc,
                retryable=False,
            ) from exc

        # google-genai expects the request timeout in milliseconds.
        client = genai.Client(
            api_key=self._api_key(),
            http_options={"timeout": int(self.timeout_seconds * 1000)},
        )
        config_kwargs = {}
        if self.config.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = self.config.max_output_tokens
        if self.config.temperature is not None:
            config_kwargs["temperature"] = self.config.temperature
        if self.config.extra and "thinking_config" in self.config.extra:
            config_kwargs["thinking_config"] = types.ThinkingConfig(**self.config.extra["thinking_config"])

        contents = prompt
        if attachments:
            config_kwargs["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_HIGH
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
        finish_reasons = [
            str(getattr(candidate, "finish_reason", "unknown"))
            for candidate in getattr(response, "candidates", []) or []
        ]
        # A MAX_TOKENS finish means the output was cut off mid-text; a truncated
        # correction would silently poison every downstream summary.
        if any("MAX_TOKENS" in reason for reason in finish_reasons):
            raise ProviderError(
                f"{self.id} hit max_output_tokens and returned a truncated response. "
                "Increase max_output_tokens or lower thinking_config.thinking_budget.",
                details=self._error_details(
                    finish_reasons=finish_reasons,
                    attachment_count=len(attachments or []),
                ),
                retryable=False,
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
        if finish_reasons:
            diagnostics.append(f"finish_reasons={finish_reasons}")
        raise ProviderError(
            f"{self.id} returned no text"
            + (f" ({'; '.join(diagnostics)})" if diagnostics else "."),
            details=self._error_details(attachment_count=len(attachments or [])),
        )


def _mime_type_for_path(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(".png"):
        return "image/png"
    if lowered.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lowered.endswith(".pdf"):
        return "application/pdf"
    return "application/octet-stream"
