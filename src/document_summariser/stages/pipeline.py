from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

from document_summariser.prompts import load_prompt
from document_summariser.rendering import DocxRenderer
from document_summariser.stages.context import RunContext


class Pipeline:
    def __init__(self, renderer: DocxRenderer | None = None) -> None:
        self.renderer = renderer or DocxRenderer()

    def run(self, context: RunContext) -> RunContext:
        self._stage("ocr", context, self._ocr)
        self._stage("correction", context, self._correct)
        self._stage("summarisation", context, self._summarise)
        self._stage("consolidation", context, self._consolidate)
        self._stage("render", context, self._render_docx)
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
            context.manifest["stages"][name] = {
                "status": "failed",
                "duration_seconds": round(perf_counter() - start, 3),
                "error": str(exc),
            }
            context.artifacts.write_json("manifest.json", context.manifest)
            raise

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
        corrected = provider.generate(
            prompt.render(ocr_text=context.ocr_text),
            attachments=context.ocr_page_images or None,
        )
        context.corrected_text = corrected
        context.artifacts.write_text("02_corrected.txt", corrected)
        context.manifest.setdefault("prompts", {})["correction"] = {
            "path": str(prompt.path),
            "sha256": prompt.sha256,
        }

    def _summarise(self, context: RunContext) -> None:
        prompt = load_prompt(context.config.prompts["summarise"])
        successes: dict[str, str] = {}
        failures: dict[str, str] = {}
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
                    failures[provider_id] = str(exc)

        if len(successes) < context.config.min_summaries:
            raise RuntimeError(
                f"Only {len(successes)} summaries succeeded; min_summaries is {context.config.min_summaries}."
            )

        context.summaries = {
            provider_id: successes[provider_id]
            for provider_id in context.config.summarisers
            if provider_id in successes
        }
        context.manifest.setdefault("prompts", {})["summarise"] = {
            "path": str(prompt.path),
            "sha256": prompt.sha256,
        }
        context.manifest["summarisation_providers"] = {
            "succeeded": sorted(successes),
            "failed": failures,
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

    def _render_docx(self, context: RunContext) -> None:
        output_path = context.artifacts.root / "05_output.docx"
        self.renderer.render(context.consolidated_summary, output_path, context.config)
        context.manifest["output_docx"] = str(output_path)
