from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

from document_summariser.errors import PipelineStageError, exception_summary
from document_summariser.prompts import load_prompt
from document_summariser.providers.base import ProviderAdapter
from document_summariser.stages.context import RunContext


class Pipeline:
    def __init__(self) -> None:
        pass

    def run(self, context: RunContext) -> RunContext:
        self._stage("ocr", context, self._ocr)
        self._stage("correction", context, self._correct)
        self._stage("summarisation", context, self._summarise)
        self._stage("consolidation", context, self._consolidate)
        self._stage("write_text", context, self._write_text_output)
        context.artifacts.write_json("manifest.json", context.manifest)
        return context

    def _stage(self, name: str, context: RunContext, func) -> None:
        start = perf_counter()
        context.manifest.setdefault("stages", {})[name] = {"status": "running"}
        try:
            func(context)
            context.manifest["stages"][name] = {
                "status": "succeeded",
                "duration_seconds": round(perf_counter() - start, 3),
            }
        except Exception as exc:
            manifest_path = context.artifacts.root / "manifest.json"
            context.manifest["stages"][name] = {
                "status": "failed",
                "duration_seconds": round(perf_counter() - start, 3),
                "error": exception_summary(exc),
            }
            context.artifacts.write_json("manifest.json", context.manifest)
            if isinstance(exc, PipelineStageError):
                exc.details.setdefault("artifacts", str(context.artifacts.root))
                exc.details.setdefault("manifest", str(manifest_path))
                raise
            raise PipelineStageError(
                name,
                f"Pipeline stage {name!r} failed.",
                details={
                    "artifacts": str(context.artifacts.root),
                    "manifest": str(manifest_path),
                },
                cause=exc,
            ) from exc

    def _ocr(self, context: RunContext) -> None:
        result = context.ocr.extract(context.input_pdf, context.artifacts)
        context.ocr_text = result.text
        context.ocr_page_images = [
            str(page.image_path)
            for page in result.pages
            if page.image_path
        ]
        if not context.ocr_text.strip():
            raise RuntimeError("OCR completed but produced no text.")
        context.artifacts.write_json("01_ocr.json", result.to_json())

    def _correct(self, context: RunContext) -> None:
        prompt = load_prompt(context.config.prompts["correction"])
        provider = context.providers[context.config.correction_provider]
        attachments = self._attachments_for_provider(provider, context.ocr_page_images)
        if context.ocr_page_images:
            context.manifest["correction_attachments"] = {
                "available": len(context.ocr_page_images),
                "sent": len(attachments or []),
                "provider_supports_attachments": bool(getattr(provider, "supports_attachments", False)),
            }
        corrected = provider.generate(
            prompt.render(ocr_text=context.ocr_text),
            attachments=attachments,
        )
        context.corrected_text = corrected
        context.artifacts.write_text("02_corrected.txt", corrected)
        # Correction should roughly preserve length; a much shorter result
        # usually means the provider truncated its output.
        ocr_characters = len(context.ocr_text.strip())
        corrected_characters = len(corrected.strip())
        if ocr_characters and corrected_characters < 0.5 * ocr_characters:
            context.manifest["correction_warning"] = {
                "reason": (
                    "Corrected text is much shorter than the OCR text; "
                    "the correction output may be truncated."
                ),
                "ocr_characters": ocr_characters,
                "corrected_characters": corrected_characters,
            }
        context.manifest.setdefault("prompts", {})["correction"] = {
            "path": str(prompt.path),
            "sha256": prompt.sha256,
        }

    def _attachments_for_provider(self, provider: ProviderAdapter, page_images: list[str]) -> list[str] | None:
        if not page_images:
            return None

        supports_attachments = bool(getattr(provider, "supports_attachments", False))
        if not supports_attachments:
            return None
        return page_images

    def _summarise(self, context: RunContext) -> None:
        prompt = load_prompt(context.config.prompts["summarise"])
        successes: dict[str, str] = {}
        failures: dict[str, dict[str, object]] = {}
        max_workers = min(
            int(context.config.runtime.get("concurrency", 4)),
            max(len(context.config.summarisers), 1),
        )
        rendered = prompt.render(
            document=context.corrected_text,
            summary_language=context.config.summary_language,
        )

        def call(provider_id: str) -> tuple[str, str]:
            provider = context.providers[provider_id]
            return provider_id, provider.generate(rendered)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(call, provider_id): provider_id for provider_id in context.config.summarisers}
            for future in as_completed(futures):
                provider_id = futures[future]
                try:
                    _, summary = future.result()
                    successes[provider_id] = summary
                    context.artifacts.write_text(f"03_summaries/{provider_id}.txt", summary)
                except Exception as exc:
                    provider = context.providers[provider_id]
                    provider_error = exception_summary(exc)
                    provider_error.setdefault(
                        "details",
                        {
                            "provider": provider_id,
                            "model": getattr(provider, "model", "unknown"),
                        },
                    )
                    failures[provider_id] = provider_error

        context.manifest.setdefault("prompts", {})["summarise"] = {
            "path": str(prompt.path),
            "sha256": prompt.sha256,
        }
        context.manifest["summarisation_providers"] = {
            "succeeded": sorted(successes),
            "failed": failures,
        }

        if len(successes) < context.config.min_summaries:
            raise PipelineStageError(
                "summarisation",
                f"Only {len(successes)} summaries succeeded; min_summaries is {context.config.min_summaries}.",
                details={
                    "succeeded": sorted(successes),
                    "failed": sorted(failures),
                    "min_summaries": context.config.min_summaries,
                },
            )

        context.summaries = {
            provider_id: successes[provider_id]
            for provider_id in context.config.summarisers
            if provider_id in successes
        }

    def _consolidate(self, context: RunContext) -> None:
        prompt = load_prompt(context.config.prompts["consolidate"])
        ordered_summaries = [
            context.summaries[provider_id]
            for provider_id in context.config.summarisers
            if provider_id in context.summaries
        ]
        labelled = "\n\n".join(
            f"## {provider_id}\n{context.summaries[provider_id]}"
            for provider_id in context.config.summarisers
            if provider_id in context.summaries
        )
        summary_variables = {
            f"summary{index}": ordered_summaries[index - 1] if index <= len(ordered_summaries) else ""
            for index in range(1, 5)
        }
        provider = context.providers[context.config.consolidator]
        consolidated = provider.generate(
            prompt.render(
                summaries=labelled,
                summary_language=context.config.summary_language,
                **summary_variables,
            )
        )
        context.consolidated_summary = consolidated
        context.artifacts.write_text("04_consolidated.txt", consolidated)
        context.manifest.setdefault("prompts", {})["consolidate"] = {
            "path": str(prompt.path),
            "sha256": prompt.sha256,
        }

    def _write_text_output(self, context: RunContext) -> None:
        output_path = context.artifacts.write_text("05_output.txt", context.consolidated_summary)
        context.manifest["output_text"] = str(output_path)
