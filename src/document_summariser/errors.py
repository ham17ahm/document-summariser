from __future__ import annotations

from dataclasses import dataclass, field
import re
import traceback
from typing import Any, TextIO


_SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|token|secret|password|authorization|credential)",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization|credential)\b\s*[:=]\s*[^\s,;]+"
)
_OPENAI_STYLE_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_GOOGLE_STYLE_KEY_PATTERN = re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b")


@dataclass
class DocumentSummariserError(Exception):
    message: str
    category: str = "Runtime error"
    details: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 1
    cause: BaseException | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": type(self).__name__,
            "category": self.category,
            "message": redact(self.message),
        }
        if self.details:
            payload["details"] = redact(self.details)
        if self.cause is not None:
            payload["cause"] = exception_summary(self.cause)
        return payload


class ConfigError(DocumentSummariserError, ValueError):
    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message=message,
            category="Configuration error",
            details=details or {},
            cause=cause,
        )


class InputFileError(DocumentSummariserError, ValueError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            category="Input file error",
            details=details or {},
        )


class OcrError(DocumentSummariserError):
    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message=message,
            category="OCR error",
            details=details or {},
            cause=cause,
        )


class ProviderError(DocumentSummariserError):
    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(
            message=message,
            category="Provider API error",
            details=details or {},
            cause=cause,
        )
        # Retrying cannot fix config/auth/validation failures; the retry loop
        # in BaseCloudProvider checks this flag to fail fast on those.
        self.retryable = retryable


class PipelineStageError(DocumentSummariserError):
    def __init__(
        self,
        stage: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        merged_details = {"stage": stage, **(details or {})}
        super().__init__(
            message=message,
            category="Pipeline stage error",
            details=merged_details,
            cause=cause,
        )


def exception_summary(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, DocumentSummariserError):
        return exc.to_dict()
    return {
        "type": type(exc).__name__,
        "message": redact(str(exc) or type(exc).__name__),
    }


def render_cli_error(exc: BaseException, *, debug: bool = False) -> str:
    error = exc if isinstance(exc, DocumentSummariserError) else _wrap_unhandled(exc)
    lines = [
        f"Error: {error.category}",
        redact(error.message),
    ]

    if error.details:
        lines.append("")
        lines.append("Details:")
        for key, value in redact(error.details).items():
            lines.append(f"  - {_label(key)}: {_format_value(value)}")

    cause = error.cause
    if cause is not None:
        lines.append("")
        lines.append("Cause:")
        cause_summary = exception_summary(cause)
        lines.append(f"  - Type: {cause_summary['type']}")
        if cause_summary.get("message"):
            lines.append(f"  - Message: {cause_summary['message']}")

    if debug:
        lines.append("")
        lines.append("Traceback:")
        lines.extend(traceback.format_exception(type(exc), exc, exc.__traceback__))

    return "\n".join(line.rstrip("\n") for line in lines)


def print_cli_error(exc: BaseException, stream: TextIO, *, debug: bool = False) -> int:
    error = exc if isinstance(exc, DocumentSummariserError) else _wrap_unhandled(exc)
    print(render_cli_error(error, debug=debug), file=stream)
    return error.exit_code


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            redacted[key_text] = (
                "[redacted]"
                if _SENSITIVE_KEY_PATTERN.search(key_text) and not key_text.lower().endswith("_env")
                else redact(item)
            )
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _wrap_unhandled(exc: BaseException) -> DocumentSummariserError:
    return DocumentSummariserError(
        message=str(exc) or type(exc).__name__,
        category="Unhandled error",
        details={"type": type(exc).__name__},
        cause=exc,
    )


def _redact_text(text: str) -> str:
    redacted = _SECRET_ASSIGNMENT_PATTERN.sub(lambda match: f"{match.group(1)}=[redacted]", text)
    redacted = _OPENAI_STYLE_KEY_PATTERN.sub("[redacted]", redacted)
    redacted = _GOOGLE_STYLE_KEY_PATTERN.sub("[redacted]", redacted)
    return redacted


def _label(key: str) -> str:
    return key.replace("_", " ").title()


def _format_value(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{_label(str(key))}={_format_value(item)}" for key, item in value.items())
    if isinstance(value, list):
        return ", ".join(_format_value(item) for item in value)
    return str(value)
