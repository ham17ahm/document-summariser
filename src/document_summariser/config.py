from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from document_summariser.errors import ConfigError


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
    prompt_sets_dir: Path
    selected_prompt_set: str | None
    output: dict[str, Any]
    runtime: dict[str, Any]
    # Short CLI names for provider IDs, e.g. {"argon": "gemini_argon_medium"}.
    model_aliases: dict[str, str] = field(default_factory=dict)


def default_config_path() -> Path:
    return Path(str(resources.files("document_summariser").joinpath("defaults/config.yaml")))


def preferred_config_path() -> Path:
    local_master = Path("config/master_config.yaml").resolve()
    if local_master.exists():
        return local_master
    return default_config_path()


def load_config(path: str | Path | None = None) -> AppConfig:
    source_path = Path(path).expanduser().resolve() if path is not None else default_config_path()
    try:
        with source_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise ConfigError(
            "Could not read config file.",
            details={"config_file": str(source_path)},
            cause=exc,
        ) from exc
    except Exception as exc:
        raise ConfigError(
            "Could not parse config file.",
            details={"config_file": str(source_path)},
            cause=exc,
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            "Config file must contain a YAML mapping at the top level.",
            details={"config_file": str(source_path)},
        )

    providers: dict[str, ProviderConfig] = {}
    try:
        provider_settings = raw.get("providers", {})
        if not isinstance(provider_settings, dict):
            raise ConfigError("Config key 'providers' must be a mapping.")
        vendors = raw.get("vendors") or {}
        if not isinstance(vendors, dict):
            raise ConfigError(
                "Config key 'vendors' must be a mapping.",
                details={"config_file": str(source_path)},
            )
        for provider_id, settings in provider_settings.items():
            if not isinstance(settings, dict):
                raise ConfigError(
                    f"Provider config for {provider_id!r} must be a mapping.",
                    details={"config_file": str(source_path), "provider": provider_id},
                )
            settings = _apply_vendor_defaults(provider_id, settings, vendors, source_path)
            providers[provider_id] = _load_provider_config(provider_id, settings)
    except ConfigError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(
            "Could not load provider config.",
            details={"config_file": str(source_path)},
            cause=exc,
        ) from exc

    pipeline = raw.get("pipeline", {})
    prompts = {
        name: _resolve_path(source_path.parent, prompt_path)
        for name, prompt_path in raw.get("prompts", {}).items()
    }
    prompt_sets = raw.get("prompt_sets", {})
    if prompt_sets is None:
        prompt_sets = {}
    if not isinstance(prompt_sets, dict):
        raise ConfigError(
            "Config key 'prompt_sets' must be a mapping.",
            details={"config_file": str(source_path)},
        )
    output = raw.get("output", {})
    model_aliases = raw.get("model_aliases", {})
    if not isinstance(model_aliases, dict):
        raise ConfigError(
            "Config key 'model_aliases' must be a mapping of short name to provider name.",
            details={"config_file": str(source_path)},
        )

    try:
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
            prompt_sets_dir=_resolve_path(source_path.parent, str(prompt_sets.get("directory", "prompts/sets"))),
            selected_prompt_set=None,
            output=output,
            runtime=raw.get("runtime", {}),
            model_aliases=dict(model_aliases),
        )
        validate_config(config)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Config contains a value with an invalid type.",
            details={"config_file": str(source_path)},
            cause=exc,
        ) from exc
    return config


def validate_config(config: AppConfig) -> None:
    missing_summarisers = [item for item in config.summarisers if item not in config.providers]
    if missing_summarisers:
        raise ConfigError(
            f"Missing provider config for summarisers: {', '.join(missing_summarisers)}",
            details={"config_file": str(config.source_path), "summarisers": missing_summarisers},
        )
    if config.correction_provider not in config.providers:
        raise ConfigError(
            f"Missing provider config for correction_provider: {config.correction_provider}",
            details={"config_file": str(config.source_path), "correction_provider": config.correction_provider},
        )
    if config.consolidator not in config.providers:
        raise ConfigError(
            f"Missing provider config for consolidator: {config.consolidator}",
            details={"config_file": str(config.source_path), "consolidator": config.consolidator},
        )
    for alias, target in config.model_aliases.items():
        if any(
            not isinstance(value, str) or not value.strip() or value != value.strip()
            for value in (alias, target)
        ):
            raise ConfigError(
                "model_aliases names and targets must be nonempty strings without surrounding whitespace.",
                details={"config_file": str(config.source_path)},
            )
        if alias in config.providers and alias != target:
            raise ConfigError(
                f"Model alias {alias!r} conflicts with an existing provider ID.",
                details={"config_file": str(config.source_path), "alias": alias, "target": target},
            )
    broken_aliases = {
        alias: target for alias, target in config.model_aliases.items() if target not in config.providers
    }
    if broken_aliases:
        raise ConfigError(
            "model_aliases point to providers that do not exist: "
            + ", ".join(f"{alias} -> {target}" for alias, target in broken_aliases.items()),
            details={"config_file": str(config.source_path), "model_aliases": broken_aliases},
        )
    if config.min_summaries < 1:
        raise ConfigError(
            "pipeline.min_summaries must be at least 1",
            details={"config_file": str(config.source_path), "min_summaries": config.min_summaries},
        )
    if config.min_summaries > len(config.summarisers):
        raise ConfigError(
            "pipeline.min_summaries cannot exceed configured summarisers",
            details={
                "config_file": str(config.source_path),
                "min_summaries": config.min_summaries,
                "summariser_count": len(config.summarisers),
            },
        )
    for name, prompt_path in config.prompts.items():
        if not prompt_path.exists():
            raise ConfigError(
                f"Prompt file for {name!r} does not exist: {prompt_path}",
                details={"config_file": str(config.source_path), "prompt": name, "prompt_path": str(prompt_path)},
            )
    for required_prompt in ("correction", "summarise", "consolidate"):
        if required_prompt not in config.prompts:
            raise ConfigError(
                f"Missing prompt config for {required_prompt!r}",
                details={"config_file": str(config.source_path), "prompt": required_prompt},
            )
    if int(config.runtime.get("concurrency", 1)) < 1:
        raise ConfigError(
            "runtime.concurrency must be at least 1",
            details={"config_file": str(config.source_path), "concurrency": config.runtime.get("concurrency")},
        )
    if int(config.runtime.get("retries", 1)) < 1:
        raise ConfigError(
            "runtime.retries must be at least 1",
            details={"config_file": str(config.source_path), "retries": config.runtime.get("retries")},
        )
    if float(config.runtime.get("request_timeout_seconds", 1)) <= 0:
        raise ConfigError(
            "runtime.request_timeout_seconds must be greater than 0",
            details={
                "config_file": str(config.source_path),
                "request_timeout_seconds": config.runtime.get("request_timeout_seconds"),
            },
        )


def apply_prompt_set(config: AppConfig, prompt_set: str | None) -> AppConfig:
    if prompt_set is None:
        return config

    prompt_set = prompt_set.strip()
    if not prompt_set:
        raise ConfigError(
            "Prompt set name cannot be empty.",
            details={"config_file": str(config.source_path)},
        )

    prompt_set_path = Path(prompt_set)
    if prompt_set_path.is_absolute() or prompt_set_path.name != prompt_set:
        raise ConfigError(
            "Prompt set name must be a single folder name, not a path.",
            details={"config_file": str(config.source_path), "prompt_set": prompt_set},
        )

    root = config.prompt_sets_dir.resolve()
    selected_dir = (root / prompt_set).resolve()
    if selected_dir.parent != root:
        raise ConfigError(
            "Prompt set must be inside the configured prompt_sets directory.",
            details={"config_file": str(config.source_path), "prompt_set": prompt_set},
        )

    required = {
        "summarise": selected_dir / "summarise.prompt.txt",
        "consolidate": selected_dir / "consolidate.prompt.txt",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise ConfigError(
            f"Prompt set {prompt_set!r} is missing required prompt file(s): {', '.join(missing)}",
            details={
                "config_file": str(config.source_path),
                "prompt_set": prompt_set,
                "prompt_set_dir": str(selected_dir),
                "missing": missing,
            },
        )

    prompts = dict(config.prompts)
    prompts.update(required)
    updated = replace(config, prompts=prompts, selected_prompt_set=prompt_set)
    validate_config(updated)
    return updated


def resolve_model_name(config: AppConfig, name: str) -> str:
    """Turn a CLI model name into a provider ID.

    Accepts a short name from ``model_aliases`` or a provider ID from
    ``providers``. Aliases cannot redirect an existing provider ID.
    """
    name = name.strip()
    if not name:
        raise ConfigError(
            "Model name cannot be empty.",
            details={"config_file": str(config.source_path)},
        )
    if name in config.model_aliases:
        return config.model_aliases[name]
    if name in config.providers:
        return name
    aliases = sorted(config.model_aliases)
    raise ConfigError(
        f"Unknown model {name!r}. "
        f"Short names: {', '.join(aliases) or 'none defined'}. "
        f"Provider names: {', '.join(sorted(config.providers))}.",
        details={
            "config_file": str(config.source_path),
            "model": name,
            "model_aliases": aliases,
            "providers": sorted(config.providers),
        },
    )


def apply_consolidator(config: AppConfig, model: str | None) -> AppConfig:
    """Override ``pipeline.consolidator`` with a CLI-selected model, if given."""
    if model is None:
        return config
    updated = replace(config, consolidator=resolve_model_name(config, model))
    validate_config(updated)
    return updated


def _apply_vendor_defaults(
    provider_id: str,
    settings: dict[str, Any],
    vendors: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    """Fill in shared vendor settings for a provider entry that names a ``vendor``.

    Values written on the provider entry win over the vendor's. Entries without
    a ``vendor`` key are returned unchanged, so fully spelled-out entries keep working.
    """
    if "vendor" not in settings:
        return settings
    vendor_name = settings["vendor"]
    vendor_settings = vendors.get(vendor_name)
    if not isinstance(vendor_settings, dict):
        raise ConfigError(
            f"Provider {provider_id!r} uses unknown vendor {vendor_name!r}. "
            f"Vendors defined in the config: {', '.join(sorted(map(str, vendors))) or 'none'}.",
            details={
                "config_file": str(source_path),
                "provider": provider_id,
                "vendor": vendor_name,
                "available_vendors": sorted(map(str, vendors)),
            },
        )
    merged = {**vendor_settings, **settings}
    del merged["vendor"]
    return merged


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
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = (
        base / path,
        base.parent / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return (base / path).resolve()
