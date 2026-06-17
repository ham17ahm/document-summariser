from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from document_summariser.artifacts import ArtifactStore
from document_summariser.config import AppConfig, apply_prompt_set, load_config
from document_summariser.errors import ConfigError
from document_summariser.ocr import build_ocr_adapter
from document_summariser.providers.registry import build_provider_registry
from document_summariser.stages import Pipeline
from document_summariser.stages.context import RunContext


@dataclass(frozen=True)
class SummaryRunResult:
    input_pdf: Path
    config: AppConfig
    artifacts: ArtifactStore
    context: RunContext
    output_text_path: Path


def run_document_summary(
    pdf_path: str | Path,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    prompt_set: str | None = None,
) -> SummaryRunResult:
    config = apply_prompt_set(load_config(config_path), prompt_set)
    _preflight_check(config)
    input_pdf = Path(pdf_path).expanduser().resolve()
    artifacts = ArtifactStore.create(_resolve_output_dir(config, output_dir), input_pdf)

    context = RunContext(
        input_pdf=input_pdf,
        config=config,
        artifacts=artifacts,
        ocr=build_ocr_adapter(config),
        providers=build_provider_registry(config),
        manifest=_build_initial_manifest(input_pdf, config),
    )
    Pipeline().run(context)

    return SummaryRunResult(
        input_pdf=input_pdf,
        config=config,
        artifacts=artifacts,
        context=context,
        output_text_path=artifacts.root / "05_output.txt",
    )


def _preflight_check(config: AppConfig) -> None:
    """Fail before OCR starts if any pipeline provider cannot possibly succeed.

    Without this, a missing consolidator key is only discovered after OCR,
    correction, and all summaries have already run and been paid for.
    """
    pipeline_provider_ids = dict.fromkeys(
        [config.correction_provider, *config.summarisers, config.consolidator]
    )
    missing_keys: dict[str, str] = {}
    for provider_id in pipeline_provider_ids:
        provider = config.providers[provider_id]
        if provider.type == "mock":
            continue
        if not provider.api_key_env:
            raise ConfigError(
                f"Provider {provider_id!r} is missing api_key_env in config.",
                details={"config_file": str(config.source_path), "provider": provider_id},
            )
        if not os.environ.get(provider.api_key_env):
            missing_keys[provider_id] = provider.api_key_env
    if missing_keys:
        raise ConfigError(
            "Missing API keys for providers: "
            + ", ".join(f"{provider_id} (set {env})" for provider_id, env in missing_keys.items()),
            details={"config_file": str(config.source_path), "required_env_vars": missing_keys},
        )


def _resolve_output_dir(config: AppConfig, output_dir: str | Path | None) -> str | Path:
    if output_dir is not None:
        return output_dir
    return os.environ.get("DOCUMENT_SUMMARISER_OUTPUT_DIR") or config.output.get("destination", "./runs/")


def _build_initial_manifest(input_pdf: Path, config: AppConfig) -> dict[str, object]:
    return {
        "input_file": str(input_pdf),
        "config_file": str(config.source_path),
        "prompt_set": config.selected_prompt_set,
        "ocr_provider": config.ocr.get("provider"),
        "providers": {
            provider_id: {"type": provider.type, "model": provider.model}
            for provider_id, provider in config.providers.items()
        },
    }
