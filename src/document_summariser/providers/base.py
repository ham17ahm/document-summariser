from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from time import sleep
from typing import Any, Protocol

from document_summariser.config import ProviderConfig
from document_summariser.errors import ProviderError
from document_summariser.prompts import PromptRequest

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

    def generate(
        self,
        request: PromptRequest,
        attachments: list[str] | None = None,
    ) -> GenerationResult:
        ...


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    cache_miss_input_tokens: int | None = None
    reasoning_tokens: int | None = None

    def to_manifest(self) -> dict[str, int | float]:
        values = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_input_tokens": self.cache_write_input_tokens,
            "cache_miss_input_tokens": self.cache_miss_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }
        result: dict[str, int | float] = {
            key: value for key, value in values.items() if value is not None
        }
        if self.input_tokens and self.cached_input_tokens is not None:
            result["cache_hit_ratio"] = round(self.cached_input_tokens / self.input_tokens, 4)
        return result


@dataclass(frozen=True)
class GenerationResult:
    text: str
    usage: ProviderUsage = ProviderUsage()
    # Length-normalised mean log-probability of the generated tokens. Only
    # populated by providers that expose it (currently Gemini); an uncalibrated
    # confidence proxy, not an accuracy guarantee.
    avg_logprobs: float | None = None


@dataclass
class MockProvider:
    id: str
    model: str
    supports_attachments: bool = True

    def generate(
        self,
        request: PromptRequest,
        attachments: list[str] | None = None,
    ) -> GenerationResult:
        excerpt = " ".join(f"{request.system}{request.user}".split())[:600]
        return GenerationResult(
            f"[{self.id} / {self.model}] Mock Urdu summary output.\n\n{excerpt}"
        )


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

    def generate(
        self,
        request: PromptRequest,
        attachments: list[str] | None = None,
    ) -> GenerationResult:
        if attachments:
            raise ProviderError(
                f"{self.id} does not support attachments in this pipeline.",
                details=self._error_details(attachment_count=len(attachments)),
                retryable=False,
            )
        return self._run_with_retry(lambda: self._generate_once(request))

    def _run_with_retry(
        self,
        call: Callable[[], GenerationResult],
        **extra_details: object,
    ) -> GenerationResult:
        last_error: Exception | None = None
        attempts_made = 0
        for attempt in range(1, self.retry_policy.attempts + 1):
            attempts_made = attempt
            try:
                result = call()
                if not result.text.strip():
                    raise ProviderError(
                        f"{self.id} returned an empty response.",
                        details=self._error_details(attempt=attempt, **extra_details),
                    )
                return result
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

    def _generate_once(self, request: PromptRequest) -> GenerationResult:
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

    def _prompt_cache_config(self) -> dict[str, Any]:
        if not self.config.extra:
            return {}
        value = self.config.extra.get("prompt_cache", {})
        return value if isinstance(value, dict) else {}


@dataclass
class AnthropicProvider(BaseCloudProvider):
    def _generate_once(self, request: PromptRequest) -> GenerationResult:
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
        system_block: dict[str, object] = {
            "type": "text",
            "text": request.system,
        }
        cache_config = self._prompt_cache_config()
        if cache_config.get("enabled", False):
            cache_control = {"type": "ephemeral"}
            ttl = cache_config.get("ttl")
            if ttl:
                cache_control["ttl"] = str(ttl)
            system_block["cache_control"] = cache_control

        kwargs = {
            "model": self.model,
            "max_tokens": self.config.max_output_tokens or 4096,
            "system": [system_block],
            "messages": [{"role": "user", "content": request.user}],
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.extra:
            for key in ("thinking", "output_config", "service_tier"):
                if key in self.config.extra:
                    kwargs[key] = self.config.extra[key]

        message = client.messages.create(**kwargs)
        text = "\n".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        )
        usage = getattr(message, "usage", None)
        cache_read = _optional_int_attr(usage, "cache_read_input_tokens")
        cache_write = _optional_int_attr(usage, "cache_creation_input_tokens")
        uncached_input = _optional_int_attr(usage, "input_tokens")
        total_input = _sum_optional(uncached_input, cache_read, cache_write)
        return GenerationResult(
            text=text,
            usage=ProviderUsage(
                input_tokens=total_input,
                output_tokens=_optional_int_attr(usage, "output_tokens"),
                cached_input_tokens=cache_read,
                cache_write_input_tokens=cache_write,
                cache_miss_input_tokens=(
                    (uncached_input or 0) + (cache_write or 0)
                    if uncached_input is not None or cache_write is not None
                    else None
                ),
            ),
        )


@dataclass
class OpenAICompatibleProvider(BaseCloudProvider):
    max_tokens_parameter: str = "max_completion_tokens"

    def _generate_once(self, request: PromptRequest) -> GenerationResult:
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
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        }
        if self.config.max_output_tokens is not None:
            kwargs[self.max_tokens_parameter] = self.config.max_output_tokens
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        kwargs.update(self._cache_request_options(request))

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        cached_tokens = _optional_int_attr(prompt_details, "cached_tokens")
        if cached_tokens is None:
            cached_tokens = _optional_int_attr(usage, "prompt_cache_hit_tokens")
        input_tokens = _optional_int_attr(usage, "prompt_tokens")
        cache_miss_tokens = _optional_int_attr(usage, "prompt_cache_miss_tokens")
        if cache_miss_tokens is None and input_tokens is not None and cached_tokens is not None:
            cache_miss_tokens = input_tokens - cached_tokens
        return GenerationResult(
            text=content or "",
            usage=ProviderUsage(
                input_tokens=input_tokens,
                output_tokens=_optional_int_attr(usage, "completion_tokens"),
                cached_input_tokens=cached_tokens,
                cache_miss_input_tokens=cache_miss_tokens,
                reasoning_tokens=_optional_int_attr(completion_details, "reasoning_tokens"),
            ),
        )

    def _cache_request_options(self, request: PromptRequest) -> dict[str, object]:
        cache_config = self._prompt_cache_config()
        if not cache_config.get("enabled", False):
            return {}
        options: dict[str, object] = {
            "prompt_cache_key": request.cache_key,
        }
        retention = cache_config.get("retention")
        if retention:
            options["prompt_cache_retention"] = str(retention)
        return options


@dataclass
class XAIProvider(OpenAICompatibleProvider):
    def _cache_request_options(self, request: PromptRequest) -> dict[str, object]:
        cache_config = self._prompt_cache_config()
        if not cache_config.get("enabled", False):
            return {}
        return {
            "extra_headers": {
                "x-grok-conv-id": request.cache_key,
            }
        }


@dataclass
class GeminiProvider(BaseCloudProvider):
    supports_attachments: bool = True

    def generate(
        self,
        request: PromptRequest,
        attachments: list[str] | None = None,
    ) -> GenerationResult:
        return self._run_with_retry(
            lambda: self._generate_once(request, attachments),
            attachment_count=len(attachments or []),
        )

    def _generate_once(
        self,
        request: PromptRequest,
        attachments: list[str] | None = None,
    ) -> GenerationResult:
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
        config_kwargs = {"system_instruction": request.system}
        if self.config.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = self.config.max_output_tokens
        if self.config.temperature is not None:
            config_kwargs["temperature"] = self.config.temperature
        if self.config.extra and "thinking_config" in self.config.extra:
            config_kwargs["thinking_config"] = types.ThinkingConfig(**self.config.extra["thinking_config"])

        contents = request.user
        if attachments:
            config_kwargs["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_HIGH
            parts = [types.Part.from_text(text=request.user)]
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
        parts: list[str] = []
        response_text = getattr(response, "text", None)
        if response_text:
            text = response_text
        else:
            for candidate in getattr(response, "candidates", []) or []:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", []) or []:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        parts.append(part_text)
            text = "\n".join(parts)
        if text.strip():
            usage = getattr(response, "usage_metadata", None)
            input_tokens = _optional_int_attr(usage, "prompt_token_count")
            cached_tokens = _optional_int_attr(usage, "cached_content_token_count")
            return GenerationResult(
                text=text,
                avg_logprobs=_first_candidate_avg_logprobs(response),
                usage=ProviderUsage(
                    input_tokens=input_tokens,
                    output_tokens=_optional_int_attr(usage, "candidates_token_count"),
                    cached_input_tokens=cached_tokens,
                    cache_miss_input_tokens=(
                        input_tokens - cached_tokens
                        if input_tokens is not None and cached_tokens is not None
                        else None
                    ),
                    reasoning_tokens=_optional_int_attr(usage, "thoughts_token_count"),
                ),
            )

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


def _optional_int_attr(value: object, name: str) -> int | None:
    candidate = getattr(value, name, None)
    return candidate if isinstance(candidate, int) else None


def _first_candidate_avg_logprobs(response: object) -> float | None:
    # avg_logprobs is absent on some model/thinking configurations, so read it
    # defensively and treat anything non-numeric as unavailable.
    for candidate in getattr(response, "candidates", []) or []:
        value = getattr(candidate, "avg_logprobs", None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None
    return None


def _sum_optional(*values: int | None) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None
