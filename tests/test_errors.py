from pathlib import Path

import pytest

from document_summariser import cli
from document_summariser.artifacts import ArtifactStore
from document_summariser.config import load_config
from document_summariser.errors import PipelineStageError, ProviderError
from document_summariser.ocr import OcrPage, OcrResult
from document_summariser.stages.context import RunContext
from document_summariser.stages.pipeline import Pipeline


def test_cli_renders_user_facing_error_for_missing_pdf(tmp_path, capsys):
    config = _write_mock_config(tmp_path)
    missing_pdf = tmp_path / "missing.pdf"

    exit_code = cli.main(
        [
            str(missing_pdf),
            "--config",
            str(config),
            "--out",
            str(tmp_path / "runs"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "Error: Pipeline stage error" in captured.err
    assert "Stage: ocr" in captured.err
    assert "Manifest:" in captured.err
    assert "Input file does not exist" in captured.err


def test_pipeline_records_redacted_summariser_failures(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = load_config(_write_mock_config(tmp_path, summarisers="[good, bad]", min_summaries=2))
    artifacts = ArtifactStore(tmp_path / "run")

    context = RunContext(
        input_pdf=pdf,
        config=config,
        artifacts=artifacts,
        ocr=StaticOcrAdapter(),
        providers={
            "corrector": StaticProvider("corrector", "test-model", "corrected text"),
            "good": StaticProvider("good", "test-model", "good summary"),
            "bad": FailingProvider("bad", "test-model"),
            "consolidator": StaticProvider("consolidator", "test-model", "final summary"),
        },
        manifest={"input_file": str(pdf), "config_file": str(config.source_path)},
    )

    with pytest.raises(PipelineStageError, match="min_summaries"):
        Pipeline().run(context)

    failures = context.manifest["summarisation_providers"]["failed"]
    assert failures["bad"]["type"] == "ProviderError"
    assert failures["bad"]["details"]["provider"] == "bad"
    assert "[redacted]" in failures["bad"]["message"]
    assert "sk-test-secret" not in failures["bad"]["message"]
    assert (artifacts.root / "manifest.json").exists()


class StaticProvider:
    def __init__(self, provider_id: str, model: str, response: str) -> None:
        self.id = provider_id
        self.model = model
        self.supports_attachments = False
        self.response = response

    def generate(self, prompt: str, attachments: list[str] | None = None) -> str:
        return self.response


class FailingProvider(StaticProvider):
    def __init__(self, provider_id: str, model: str) -> None:
        super().__init__(provider_id, model, "")

    def generate(self, prompt: str, attachments: list[str] | None = None) -> str:
        raise ProviderError(
            "API request failed: api_key=sk-test-secret-123456789012",
            details={"provider": self.id, "model": self.model},
        )


class StaticOcrAdapter:
    def extract(self, pdf_path: Path, artifacts: ArtifactStore) -> OcrResult:
        return OcrResult(
            provider="static",
            pages=[
                OcrPage(
                    page_number=1,
                    text="OCR text",
                    confidence=None,
                    low_confidence=False,
                    image_path=None,
                )
            ],
        )


def _write_mock_config(
    tmp_path: Path,
    *,
    summarisers: str = "[good]",
    min_summaries: int = 1,
) -> Path:
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
  correction_provider: corrector
  summarisers: {summarisers}
  consolidator: consolidator
  min_summaries: {min_summaries}

providers:
  corrector:
    type: mock
    model: corrector-test
  good:
    type: mock
    model: good-test
  bad:
    type: mock
    model: bad-test
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
