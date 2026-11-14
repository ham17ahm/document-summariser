import json
from pathlib import Path

import pytest
import yaml

from document_summariser.application import run_document_summary
from document_summariser.cli import main
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


def test_run_uses_selected_consolidator(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = _write_config(tmp_path, provider_block="", provider_type="mock")

    default = run_document_summary(pdf, config_path=config, output_dir=tmp_path / "runs")
    selected = run_document_summary(
        pdf, config_path=config, output_dir=tmp_path / "runs", consolidator="alt"
    )

    assert default.config.consolidator == "gpt"
    assert default.context.manifest["consolidator"] == "gpt"
    assert selected.config.consolidator == "gpt_alt"
    assert selected.context.manifest["consolidator"] == "gpt_alt"
    for result, provider in [(default, "gpt"), (selected, "gpt_alt")]:
        assert result.output_text_path.read_text(encoding="utf-8").startswith(f"[{provider} / ")
        manifest = json.loads((result.artifacts.root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["consolidator"] == provider
    for relative_path in ["01_ocr.json", "02_corrected.txt", "03_summaries/gpt.txt"]:
        assert (default.artifacts.root / relative_path).read_bytes() == (
            selected.artifacts.root / relative_path
        ).read_bytes()
    assert selected.config.prompts == default.config.prompts


@pytest.mark.parametrize(
    ("model", "message"),
    [("missing", "Unknown model 'missing'"), ("", "Model name cannot be empty"),
     ("   ", "Model name cannot be empty")],
)
def test_invalid_model_fails_before_ocr_or_artifacts(tmp_path, monkeypatch, model, message):
    config = _write_config(tmp_path, provider_block="", provider_type="mock")

    def unexpected_ocr(*args):
        pytest.fail("OCR must not be built for an invalid model")

    monkeypatch.setattr("document_summariser.application.build_ocr_adapter", unexpected_ocr)
    runs_dir = tmp_path / "runs"
    with pytest.raises(ConfigError, match=message):
        run_document_summary(tmp_path / "absent.pdf", config, runs_dir, consolidator=model)
    assert not runs_dir.exists()


def test_missing_alias_target_fails_before_ocr_or_artifacts(tmp_path, monkeypatch):
    config = _write_config(tmp_path, provider_block="", provider_type="mock")
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["model_aliases"] = {"alt": "missing"}
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    def unexpected_ocr(*args):
        pytest.fail("OCR must not be built for invalid aliases")

    monkeypatch.setattr("document_summariser.application.build_ocr_adapter", unexpected_ocr)
    runs_dir = tmp_path / "runs"
    with pytest.raises(ConfigError, match="alt -> missing"):
        run_document_summary(tmp_path / "absent.pdf", config, runs_dir)
    assert not runs_dir.exists()


@pytest.mark.parametrize(("pdf_count", "model"), [(1, None), (1, "gpt_alt"), (2, "alt")])
def test_cli_runs_mock_pipeline_with_prompt_set_and_model(tmp_path, monkeypatch, pdf_count, model):
    config = _write_config(tmp_path, provider_block="", provider_type="mock")
    prompt_set = tmp_path / "prompt-sets" / "custom"
    prompt_set.mkdir(parents=True)
    (prompt_set / "summarise.prompt.txt").write_text("CUSTOM SUMMARY {{document}}", encoding="utf-8")
    (prompt_set / "consolidate.prompt.txt").write_text("CUSTOM FINAL {{summary1}}", encoding="utf-8")
    pdfs = [tmp_path / f"sample{index}.pdf" for index in range(pdf_count)]
    for pdf in pdfs:
        pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    monkeypatch.setattr("document_summariser.cli.load_local_env", lambda: None)
    runs_dir = tmp_path / "runs"
    args = [str(pdf) for pdf in pdfs] + [
        "--config", str(config), "--out", str(runs_dir), "-p", "custom", "--no-publish-final",
    ]
    if model is not None:
        args.extend(["-m", model])

    assert main(args) == 0

    manifests = list(runs_dir.glob("*/manifest.json"))
    assert len(manifests) == pdf_count
    expected_provider = "gpt" if model is None else "gpt_alt"
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["consolidator"] == expected_provider
        assert manifest["prompt_set"] == "custom"
        output = (path.parent / "05_output.txt").read_text(encoding="utf-8")
        assert output.startswith(f"[{expected_provider} / ")
        assert "CUSTOM FINAL" in output
        assert "CUSTOM SUMMARY" in output


def test_run_preflight_checks_selected_consolidator_key(tmp_path, monkeypatch):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n% placeholder\n")
    config = _write_config(tmp_path, provider_block="", provider_type="mock")
    monkeypatch.delenv("ALT_KEY_FOR_TEST", raising=False)
    runs_dir = tmp_path / "runs"

    with pytest.raises(ConfigError, match="ALT_KEY_FOR_TEST"):
        run_document_summary(pdf, config_path=config, output_dir=runs_dir, consolidator="keyed")

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

model_aliases:
  alt: gpt_alt

providers:
  gpt:
    type: {provider_type}
    model: gpt-test
{provider_block}
  gpt_alt:
    type: mock
    model: gpt-alt-test
  keyed:
    type: openai
    model: keyed-test
    api_key_env: ALT_KEY_FOR_TEST

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
