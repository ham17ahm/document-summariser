from pathlib import Path
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
