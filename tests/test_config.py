from pathlib import Path

import pytest

from document_summariser.config import load_config


def test_default_config_loads():
    config = load_config("config/config.yaml")

    assert config.summary_language == "ur"
    assert config.min_summaries == 2
    assert config.correction_provider == "gemini"
    assert config.summarisers == ["claude", "gpt", "gemini", "deepseek"]
    assert config.consolidator == "claude"
    assert config.ocr["provider"] == "google_cloud_vision"
    assert config.providers["deepseek"].type == "deepseek"


def test_config_requires_correction_provider(tmp_path):
    config_path = tmp_path / "config.yaml"
    correction = Path("prompts/correction.prompt.txt").resolve()
    summarise = Path("prompts/summarise.prompt.txt").resolve()
    consolidate = Path("prompts/consolidate.prompt.txt").resolve()
    config_path.write_text(
        f"""
pipeline:
  correction_provider: missing
  summarisers: [claude]
  consolidator: claude
providers:
  claude:
    type: mock
    model: claude-test
prompts:
  correction: {correction}
  summarise: {summarise}
  consolidate: {consolidate}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="correction_provider"):
        load_config(config_path)
