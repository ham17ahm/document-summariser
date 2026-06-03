import pytest
from pathlib import Path

from document_summariser.config import default_config_path, load_config, preferred_config_path


def test_master_config_loads():
    config = load_config("config/master_config.yaml")

    assert config.summary_language == "ur"
    assert config.min_summaries == 4
    assert config.correction_provider == "gemini"
    assert config.summarisers == ["chatgpt", "gemini", "grok", "deepseek"]
    assert config.consolidator == "claude"
    assert config.ocr["provider"] == "google_cloud_vision"
    assert config.providers["claude"].model == "claude-opus-4-7"
    assert config.providers["claude"].max_output_tokens == 64000
    assert config.providers["claude"].extra == {
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": "xhigh"},
    }
    assert config.providers["chatgpt"].model == "gpt-5.2"
    assert config.providers["gemini"].model == "gemini-2.5-pro"
    assert config.providers["gemini"].max_output_tokens == 16384
    assert config.providers["gemini"].extra == {"thinking_config": {"thinking_budget": 1024}}
    assert config.providers["grok"].type == "grok"
    assert config.providers["grok"].base_url == "https://api.x.ai/v1"
    assert config.providers["deepseek"].model == "deepseek-v4-pro"
    assert config.providers["deepseek"].type == "deepseek"
    assert config.output["format"] == "txt"


def test_packaged_default_config_loads():
    config = load_config()

    assert default_config_path().exists()
    assert config.source_path == default_config_path()
    assert config.summarisers == ["chatgpt", "gemini", "grok", "deepseek"]
    assert config.prompts["correction"].exists()
    assert config.output["format"] == "txt"


def test_preferred_config_path_uses_repository_master_config():
    assert preferred_config_path() == Path("config/master_config.yaml").resolve()


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
