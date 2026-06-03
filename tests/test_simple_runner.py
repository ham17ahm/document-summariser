from pathlib import Path

from summarise_pdf import main


def test_simple_runner_writes_text_next_to_pdf(tmp_path, monkeypatch):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = _write_mock_config(tmp_path)
    monkeypatch.setenv("DOCUMENT_SUMMARISER_OUTPUT_DIR", str(tmp_path / "runs"))

    exit_code = main([str(pdf), "--config", str(config)])

    assert exit_code == 0
    assert (tmp_path / "sample.txt").exists()


def test_simple_runner_uses_configured_final_text_directory(tmp_path, monkeypatch):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    final_dir = tmp_path / "final"
    config = _write_mock_config(tmp_path, final_text_directory=final_dir)
    monkeypatch.setenv("DOCUMENT_SUMMARISER_OUTPUT_DIR", str(tmp_path / "runs"))

    exit_code = main([str(pdf), "--config", str(config)])

    assert exit_code == 0
    assert (final_dir / "sample.txt").exists()
    assert not (tmp_path / "sample.txt").exists()


def _write_mock_config(tmp_path: Path, final_text_directory: Path | None = None) -> Path:
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
{f"  final_text_directory: {final_text_directory}" if final_text_directory else ""}

runtime:
  concurrency: 2
  retries: 1
""",
        encoding="utf-8",
    )
    return config_path
