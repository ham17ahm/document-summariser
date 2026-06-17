from pathlib import Path

import pytest

from document_summariser.application import run_document_summary
from document_summariser.errors import ConfigError


def test_run_aborts_before_artifacts_when_api_key_missing(tmp_path, monkeypatch):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = _write_config(tmp_path, provider_block="    api_key_env: OPENAI_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    runs_dir = tmp_path / "runs"

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        run_document_summary(pdf, config_path=config, output_dir=runs_dir)

    assert not runs_dir.exists()


def test_run_aborts_when_pipeline_provider_lacks_api_key_env(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = _write_config(tmp_path, provider_block="")
    runs_dir = tmp_path / "runs"

    with pytest.raises(ConfigError, match="api_key_env"):
        run_document_summary(pdf, config_path=config, output_dir=runs_dir)

    assert not runs_dir.exists()


def test_run_uses_selected_prompt_set(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = _write_config(tmp_path, provider_block="", provider_type="mock")
    prompt_set_dir = tmp_path / "prompt-sets" / "pr1"
    prompt_set_dir.mkdir(parents=True)
    (prompt_set_dir / "summarise.prompt.txt").write_text(
        "CUSTOM SUMMARY PROMPT {{document}} {{summary_language}}",
        encoding="utf-8",
    )
    (prompt_set_dir / "consolidate.prompt.txt").write_text(
        "CUSTOM CONSOLIDATE PROMPT {{summary1}}",
        encoding="utf-8",
    )

    result = run_document_summary(pdf, config_path=config, output_dir=tmp_path / "runs", prompt_set="pr1")
    output = result.output_text_path.read_text(encoding="utf-8")

    assert result.config.selected_prompt_set == "pr1"
    assert result.context.manifest["prompt_set"] == "pr1"
    assert "CUSTOM CONSOLIDATE PROMPT" in output
    assert "CUSTOM SUMMARY PROMPT" in output


def test_run_aborts_before_artifacts_when_prompt_set_missing(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = _write_config(tmp_path, provider_block="", provider_type="mock")
    runs_dir = tmp_path / "runs"

    with pytest.raises(ConfigError, match="missing"):
        run_document_summary(pdf, config_path=config, output_dir=runs_dir, prompt_set="missing")

    assert not runs_dir.exists()


def _write_config(tmp_path: Path, provider_block: str, provider_type: str = "openai") -> Path:
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
  correction_provider: gpt
  summarisers: [gpt]
  consolidator: gpt
  min_summaries: 1

providers:
  gpt:
    type: {provider_type}
    model: gpt-test
{provider_block}

prompts:
  correction: {correction}
  summarise: {summarise}
  consolidate: {consolidate}

prompt_sets:
  directory: {tmp_path / "prompt-sets"}

runtime:
  concurrency: 1
  retries: 1
""",
        encoding="utf-8",
    )
    return config_path
