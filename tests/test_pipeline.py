from pathlib import Path
from time import sleep
from zipfile import ZipFile

from document_summariser.artifacts import ArtifactStore
from document_summariser.config import load_config
from document_summariser.ocr import build_ocr_adapter
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
    assert (artifacts.root / "05_output.docx").exists()
    assert (artifacts.root / "manifest.json").exists()

    with ZipFile(artifacts.root / "05_output.docx") as docx:
        document_xml = docx.read("word/document.xml").decode("utf-8")
    assert "w:bidi" in document_xml


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
    assert consolidator.last_prompt.index("## slow") < consolidator.last_prompt.index("## fast")


class StaticProvider:
    def __init__(self, provider_id: str, model: str, response: str) -> None:
        self.id = provider_id
        self.model = model
        self.response = response

    def generate(self, prompt: str, attachments: list[str] | None = None) -> str:
        return self.response


class DelayedProvider(StaticProvider):
    def __init__(self, provider_id: str, model: str, response: str, delay_seconds: float) -> None:
        super().__init__(provider_id, model, response)
        self.delay_seconds = delay_seconds

    def generate(self, prompt: str, attachments: list[str] | None = None) -> str:
        sleep(self.delay_seconds)
        return super().generate(prompt, attachments)


class RecordingProvider(StaticProvider):
    def __init__(self, provider_id: str, model: str) -> None:
        super().__init__(provider_id, model, "final summary")
        self.last_prompt = ""

    def generate(self, prompt: str, attachments: list[str] | None = None) -> str:
        self.last_prompt = prompt
        return super().generate(prompt, attachments)


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
