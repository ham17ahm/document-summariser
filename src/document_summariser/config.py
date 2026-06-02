from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised when dependencies are not installed
    yaml = None


@dataclass(frozen=True)
class ProviderConfig:
    id: str
    type: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class AppConfig:
    source_path: Path
    raw: dict[str, Any]
    summary_language: str
    ocr: dict[str, Any]
    summarisers: list[str]
    correction_provider: str
    consolidator: str
    min_summaries: int
    providers: dict[str, ProviderConfig]
    prompts: dict[str, Path]
    output: dict[str, Any]
    runtime: dict[str, Any]


def load_config(path: str | Path) -> AppConfig:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as handle:
        if yaml is not None:
            raw = yaml.safe_load(handle) or {}
        else:
            raw = _simple_yaml_load(handle.read())

    providers = {
        provider_id: _load_provider_config(provider_id, settings)
        for provider_id, settings in raw.get("providers", {}).items()
    }

    pipeline = raw.get("pipeline", {})
    prompts = {
        name: _resolve_path(source_path.parent, prompt_path)
        for name, prompt_path in raw.get("prompts", {}).items()
    }

    config = AppConfig(
        source_path=source_path,
        raw=raw,
        summary_language=str(raw.get("language", {}).get("summary_language", "ur")),
        ocr=raw.get("ocr", {}),
        summarisers=list(pipeline.get("summarisers", [])),
        correction_provider=str(pipeline.get("correction_provider", pipeline.get("consolidator", ""))),
        consolidator=str(pipeline.get("consolidator", "")),
        min_summaries=int(pipeline.get("min_summaries", 1)),
        providers=providers,
        prompts=prompts,
        output=raw.get("output", {}),
        runtime=raw.get("runtime", {}),
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    missing_summarisers = [item for item in config.summarisers if item not in config.providers]
    if missing_summarisers:
        raise ValueError(f"Missing provider config for summarisers: {', '.join(missing_summarisers)}")
    if config.correction_provider not in config.providers:
        raise ValueError(f"Missing provider config for correction_provider: {config.correction_provider}")
    if config.consolidator not in config.providers:
        raise ValueError(f"Missing provider config for consolidator: {config.consolidator}")
    if config.min_summaries < 1:
        raise ValueError("pipeline.min_summaries must be at least 1")
    if config.min_summaries > len(config.summarisers):
        raise ValueError("pipeline.min_summaries cannot exceed configured summarisers")
    for name, prompt_path in config.prompts.items():
        if not prompt_path.exists():
            raise ValueError(f"Prompt file for {name!r} does not exist: {prompt_path}")
    for required_prompt in ("correction", "summarise", "consolidate"):
        if required_prompt not in config.prompts:
            raise ValueError(f"Missing prompt config for {required_prompt!r}")
    if int(config.runtime.get("concurrency", 1)) < 1:
        raise ValueError("runtime.concurrency must be at least 1")
    if int(config.runtime.get("retries", 1)) < 1:
        raise ValueError("runtime.retries must be at least 1")
    if float(config.runtime.get("request_timeout_seconds", 1)) <= 0:
        raise ValueError("runtime.request_timeout_seconds must be greater than 0")


def _load_provider_config(provider_id: str, settings: dict[str, Any]) -> ProviderConfig:
    known_keys = {"type", "model", "api_key_env", "base_url", "max_output_tokens", "temperature"}
    extra = {key: value for key, value in settings.items() if key not in known_keys}
    return ProviderConfig(
        id=provider_id,
        type=str(settings["type"]),
        model=str(settings["model"]),
        api_key_env=_optional_str(settings.get("api_key_env")),
        base_url=_optional_str(settings.get("base_url")),
        max_output_tokens=_optional_int(settings.get("max_output_tokens")),
        temperature=_optional_float(settings.get("temperature")),
        extra=extra,
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base.parent / path).resolve()


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by the default config.

    This keeps the CLI smoke-testable before project dependencies are installed.
    Install PyYAML for full YAML support.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value.strip() == "":
            nested: dict[str, Any] = {}
            current[key] = nested
            stack.append((indent, nested))
        else:
            current[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].strip()
        if not items:
            return []
        return [_parse_scalar(item.strip()) for item in items.split(",")]
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if value.isdigit():
        return int(value)
    return value
