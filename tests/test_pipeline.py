from pathlib import Path
from time import sleep

from document_summariser.artifacts import ArtifactStore
from document_summariser.config import load_config
from document_summariser.ocr import OcrPage, OcrResult
from document_summariser.ocr import build_ocr_adapter
from document_summariser.prompts import PromptRequest
from document_summariser.providers.base import GenerationResult, ProviderUsage
from document_summariser.providers.registry import build_provider_registry
from document_summariser.stages.context import RunContext
from document_summariser.stages.pipeline import Pipeline


def test_pipeline_writes_expected_artifacts(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = load_config(_write_mock_config(tmp_path))
    artifacts = ArtifactStore(tmp_path / "run")

    context = RunContext(
        input_pdf=pdf,
        config=config,
        artifacts=artifacts,
        ocr=build_ocr_adapter(config),
        providers=build_provider_registry(config),
        manifest={"input_file": str(pdf), "config_file": str(config.source_path)},
    )

    Pipeline().run(context)

    assert (artifacts.root / "01_ocr.json").exists()
    assert (artifacts.root / "02_corrected.txt").exists()
    assert (artifacts.root / "03_summaries" / "claude.txt").exists()
    assert (artifacts.root / "04_consolidated.txt").exists()
    assert (artifacts.root / "05_output.txt").exists()
    assert (artifacts.root / "manifest.json").exists()
    assert (artifacts.root / "05_output.txt").read_text(encoding="utf-8").strip()


def test_pipeline_consolidates_summaries_in_configured_order(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = load_config(_write_order_config(tmp_path))
    artifacts = ArtifactStore(tmp_path / "run")
    consolidator = RecordingProvider("consolidator", "test-model")

    context = RunContext(
        input_pdf=pdf,
        config=config,
        artifacts=artifacts,
        ocr=build_ocr_adapter(config),
        providers={
            "corrector": StaticProvider("corrector", "test-model", "corrected text"),
            "slow": DelayedProvider("slow", "test-model", "slow summary", delay_seconds=0.05),
            "fast": StaticProvider("fast", "test-model", "fast summary"),
            "consolidator": consolidator,
        },
        manifest={"input_file": str(pdf), "config_file": str(config.source_path)},
    )

    Pipeline().run(context)

    assert list(context.summaries) == ["slow", "fast"]
    assert consolidator.last_prompt.index("<summary1>\nslow summary") < consolidator.last_prompt.index(
        "<summary2>\nfast summary"
    )


def test_pipeline_passes_ocr_page_images_to_correction_provider(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = load_config(_write_order_config(tmp_path))
    artifacts = ArtifactStore(tmp_path / "run")
    image_path = artifacts.write_bytes("page_images/page_0001.png", b"png")
    corrector = RecordingProvider("corrector", "test-model", supports_attachments=True)

    context = RunContext(
        input_pdf=pdf,
        config=config,
        artifacts=artifacts,
        ocr=StaticOcrAdapter(str(image_path)),
        providers={
            "corrector": corrector,
            "slow": StaticProvider("slow", "test-model", "slow summary"),
            "fast": StaticProvider("fast", "test-model", "fast summary"),
            "consolidator": StaticProvider("consolidator", "test-model", "final summary"),
        },
        manifest={"input_file": str(pdf), "config_file": str(config.source_path)},
    )

    Pipeline().run(context)

    assert corrector.last_attachments == [str(image_path)]


def test_pipeline_skips_ocr_page_images_for_text_only_correction_provider(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = load_config(_write_order_config(tmp_path))
    artifacts = ArtifactStore(tmp_path / "run")
    image_path = artifacts.write_bytes("page_images/page_0001.png", b"png")
    corrector = RecordingProvider("corrector", "test-model", supports_attachments=False)

    context = RunContext(
        input_pdf=pdf,
        config=config,
        artifacts=artifacts,
        ocr=StaticOcrAdapter(str(image_path)),
        providers={
            "corrector": corrector,
            "slow": StaticProvider("slow", "test-model", "slow summary"),
            "fast": StaticProvider("fast", "test-model", "fast summary"),
            "consolidator": StaticProvider("consolidator", "test-model", "final summary"),
        },
        manifest={"input_file": str(pdf), "config_file": str(config.source_path)},
    )

    Pipeline().run(context)

    assert corrector.last_attachments is None
    assert context.manifest["correction_attachments"] == {
        "available": 1,
        "sent": 0,
        "provider_supports_attachments": False,
    }


def test_pipeline_warns_when_correction_is_suspiciously_short(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = load_config(_write_order_config(tmp_path))
    artifacts = ArtifactStore(tmp_path / "run")
    long_ocr_text = "This is a long page of OCR text. " * 50

    context = RunContext(
        input_pdf=pdf,
        config=config,
        artifacts=artifacts,
        ocr=StaticOcrAdapter(None, text=long_ocr_text),
        providers={
            "corrector": StaticProvider("corrector", "test-model", "short"),
            "slow": StaticProvider("slow", "test-model", "slow summary"),
            "fast": StaticProvider("fast", "test-model", "fast summary"),
            "consolidator": StaticProvider("consolidator", "test-model", "final summary"),
        },
        manifest={"input_file": str(pdf), "config_file": str(config.source_path)},
    )

    Pipeline().run(context)

    warning = context.manifest["correction_warning"]
    assert warning["corrected_characters"] == len("short")
    assert warning["ocr_characters"] == len(long_ocr_text.strip())


def test_pipeline_records_cache_usage_without_prompt_content(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = load_config(_write_order_config(tmp_path))
    artifacts = ArtifactStore(tmp_path / "run")

    context = RunContext(
        input_pdf=pdf,
        config=config,
        artifacts=artifacts,
        ocr=StaticOcrAdapter(None),
        providers={
            "corrector": StaticProvider(
                "corrector",
                "test-model",
                "corrected text",
                usage=ProviderUsage(input_tokens=100, cached_input_tokens=80),
            ),
            "slow": StaticProvider("slow", "test-model", "slow summary"),
            "fast": StaticProvider("fast", "test-model", "fast summary"),
            "consolidator": StaticProvider("consolidator", "test-model", "final summary"),
        },
        manifest={"input_file": str(pdf), "config_file": str(config.source_path)},
    )

    Pipeline().run(context)

    usage = context.manifest["provider_usage"]["correction"]["corrector"]
    assert usage == {
        "input_tokens": 100,
        "cached_input_tokens": 80,
        "cache_hit_ratio": 0.8,
    }
    assert "corrected text" not in str(context.manifest["provider_usage"])


def test_pipeline_records_ocr_confidence_in_manifest(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = load_config(_write_order_config(tmp_path))
    artifacts = ArtifactStore(tmp_path / "run")
    pages = [
        OcrPage(page_number=1, text="page one", confidence=0.95, low_confidence=False, image_path=None),
        OcrPage(page_number=2, text="page two", confidence=0.6, low_confidence=True, image_path=None),
    ]

    context = RunContext(
        input_pdf=pdf,
        config=config,
        artifacts=artifacts,
        ocr=StaticOcrAdapter(None, pages=pages),
        providers={
            "corrector": StaticProvider("corrector", "test-model", "corrected text"),
            "slow": StaticProvider("slow", "test-model", "slow summary"),
            "fast": StaticProvider("fast", "test-model", "fast summary"),
            "consolidator": StaticProvider("consolidator", "test-model", "final summary"),
        },
        manifest={"input_file": str(pdf), "config_file": str(config.source_path)},
    )

    Pipeline().run(context)

    assert context.manifest["quality"]["ocr"] == {
        "pages": 2,
        "average_confidence": 0.775,
        "low_confidence_pages": [2],
        "low_confidence_threshold": 0.8,
    }
    # The default StaticProvider reports no avg_logprobs; that must degrade to None.
    assert context.manifest["quality"]["correction"]["avg_logprobs"] is None
    assert context.manifest["quality"]["correction"]["avg_token_probability"] is None


def test_pipeline_records_correction_confidence_in_manifest(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = load_config(_write_order_config(tmp_path))
    artifacts = ArtifactStore(tmp_path / "run")

    context = RunContext(
        input_pdf=pdf,
        config=config,
        artifacts=artifacts,
        ocr=StaticOcrAdapter(None, text="one two three four"),
        providers={
            "corrector": StaticProvider(
                "corrector",
                "test-model",
                "one two three five",
                avg_logprobs=-0.15,
            ),
            "slow": StaticProvider("slow", "test-model", "slow summary"),
            "fast": StaticProvider("fast", "test-model", "fast summary"),
            "consolidator": StaticProvider("consolidator", "test-model", "final summary"),
        },
        manifest={"input_file": str(pdf), "config_file": str(config.source_path)},
    )

    Pipeline().run(context)

    correction = context.manifest["quality"]["correction"]
    assert correction["avg_logprobs"] == -0.15
    assert correction["avg_token_probability"] == 0.8607
    # Three of four words survive the correction: ratio 2*3/(4+4) = 0.75.
    assert correction["similarity_to_ocr"] == 0.75
    assert correction["change_ratio"] == 0.25


class StaticProvider:
    def __init__(
        self,
        provider_id: str,
        model: str,
        response: str,
        supports_attachments: bool = False,
        usage: ProviderUsage = ProviderUsage(),
        avg_logprobs: float | None = None,
    ) -> None:
        self.id = provider_id
        self.model = model
        self.response = response
        self.supports_attachments = supports_attachments
        self.usage = usage
        self.avg_logprobs = avg_logprobs

    def generate(
        self,
        request: PromptRequest,
        attachments: list[str] | None = None,
    ) -> GenerationResult:
        return GenerationResult(self.response, self.usage, avg_logprobs=self.avg_logprobs)


class DelayedProvider(StaticProvider):
    def __init__(self, provider_id: str, model: str, response: str, delay_seconds: float) -> None:
        super().__init__(provider_id, model, response)
        self.delay_seconds = delay_seconds

    def generate(
        self,
        request: PromptRequest,
        attachments: list[str] | None = None,
    ) -> GenerationResult:
        sleep(self.delay_seconds)
        return super().generate(request, attachments)


class RecordingProvider(StaticProvider):
    def __init__(self, provider_id: str, model: str, supports_attachments: bool = False) -> None:
        super().__init__(provider_id, model, "final summary", supports_attachments=supports_attachments)
        self.last_prompt = ""
        self.last_attachments = None

    def generate(
        self,
        request: PromptRequest,
        attachments: list[str] | None = None,
    ) -> GenerationResult:
        self.last_prompt = f"{request.system}{request.user}"
        self.last_attachments = attachments
        return super().generate(request, attachments)


class StaticOcrAdapter:
    def __init__(
        self,
        image_path: str | None,
        text: str = "OCR text",
        pages: list[OcrPage] | None = None,
    ) -> None:
        self.image_path = image_path
        self.text = text
        self.pages = pages

    def extract(self, pdf_path: Path, artifacts: ArtifactStore) -> OcrResult:
        pages = self.pages or [
            OcrPage(
                page_number=1,
                text=self.text,
                confidence=None,
                low_confidence=False,
                image_path=self.image_path,
            )
        ]
        return OcrResult(provider="static", pages=pages)


def _write_mock_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    correction = Path("prompts/correction.prompt.txt").resolve()
    summarise = Path("prompts/summarise.prompt.txt").resolve()
    consolidate = Path("prompts/consolidate.prompt.txt").resolve()
    config_path.write_text(
        f"""
language:
  summary_language: ur

ocr:
  provider: mock

pipeline:
  correction_provider: gemini
  summarisers: [claude, gpt, gemini, deepseek]
  consolidator: claude
  min_summaries: 2

providers:
  claude:
    type: mock
    model: claude-test
  gpt:
    type: mock
    model: gpt-test
  gemini:
    type: mock
    model: gemini-test
  deepseek:
    type: mock
    model: deepseek-test

prompts:
  correction: {correction}
  summarise: {summarise}
  consolidate: {consolidate}

output:
  destination: {tmp_path / "runs"}

runtime:
  concurrency: 2
  retries: 1
""",
        encoding="utf-8",
    )
    return config_path


def _write_order_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "order-config.yaml"
    correction = Path("prompts/correction.prompt.txt").resolve()
    summarise = Path("prompts/summarise.prompt.txt").resolve()
    consolidate = Path("prompts/consolidate.prompt.txt").resolve()
    config_path.write_text(
        f"""
language:
  summary_language: ur

ocr:
  provider: mock

pipeline:
  correction_provider: corrector
  summarisers: [slow, fast]
  consolidator: consolidator
  min_summaries: 2

providers:
  corrector:
    type: mock
    model: corrector-test
  slow:
    type: mock
    model: slow-test
  fast:
    type: mock
    model: fast-test
  consolidator:
    type: mock
    model: consolidator-test

prompts:
  correction: {correction}
  summarise: {summarise}
  consolidate: {consolidate}

runtime:
  concurrency: 2
  retries: 1
""",
        encoding="utf-8",
    )
    return config_path
