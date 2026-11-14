import pytest
from pathlib import Path

import yaml

from document_summariser.config import (
    apply_consolidator,
    apply_prompt_set,
    default_config_path,
    load_config,
    preferred_config_path,
)
from document_summariser.errors import ConfigError


def test_master_config_loads():
    config = load_config("config/master_config.yaml")

    assert config.summary_language == "ur"
    assert config.min_summaries == 4
    assert config.correction_provider == "gemini"
    assert config.summarisers == ["chatgpt_gpt6", "claude", "grok", "deepseek"]
    assert config.consolidator == "gemini_argon_medium"
    assert config.ocr["provider"] == "google_cloud_vision"
    assert config.providers["claude"].type == "anthropic"
    assert config.providers["claude"].api_key_env == "ANTHROPIC_API_KEY"
    assert config.providers["gemini"].type == "gemini"
    assert config.providers["gemini"].api_key_env == "GEMINI_API_KEY"
    assert config.providers["chatgpt"].type == "openai"
    assert config.providers["chatgpt"].api_key_env == "OPENAI_API_KEY"
    assert config.providers["grok"].api_key_env == "XAI_API_KEY"
    assert config.providers["deepseek"].api_key_env == "DEEPSEEK_API_KEY"
    assert config.providers["deepseek"].base_url == "https://api.deepseek.com"
    assert config.providers["claude"].model == "claude-opus-5-5"
    assert config.providers["claude"].max_output_tokens == 64000
    assert config.providers["claude"].extra == {
        "prompt_cache": {"enabled": True, "ttl": "5m"},
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": "medium"},
    }
    assert config.providers["chatgpt"].model == "gpt-5.2"
    assert config.providers["chatgpt"].extra == {
        "prompt_cache": {"enabled": True}
    }
    assert config.providers["gemini"].model == "gemini-2.5-flash"
    assert config.providers["gemini"].max_output_tokens == 16384
    assert config.providers["gemini"].extra == {"thinking_config": {"thinking_budget": 1024}}
    assert config.providers["gemini_consolidate"].model == "gemini-3.1-pro-preview"
    assert config.providers["gemini_consolidate"].max_output_tokens == 16384
    assert config.providers["gemini_consolidate"].extra == {"thinking_config": {"thinking_budget": 1024}}
    assert config.providers["grok"].type == "grok"
    assert config.providers["grok"].base_url == "https://api.x.ai/v1"
    assert config.providers["grok"].extra == {"prompt_cache": {"enabled": True}}
    assert config.providers["deepseek"].model == "deepseek-v4-pro"
    assert config.providers["deepseek"].type == "deepseek"
    assert config.providers["deepseek"].max_output_tokens == 32768
    assert config.prompt_sets_dir == Path("prompts/sets").resolve()
    assert config.selected_prompt_set is None
    assert config.output["format"] == "txt"


def test_packaged_default_config_loads():
    config = load_config()

    assert default_config_path().exists()
    assert config.source_path == default_config_path()
    assert config.summarisers == ["chatgpt", "claude", "grok", "deepseek"]
    assert config.providers["deepseek"].max_output_tokens == 32768
    assert config.prompts["correction"].exists()
    assert config.prompt_sets_dir.exists()
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


def _write_vendor_config(tmp_path: Path, providers_yaml: str) -> Path:
    config_path = tmp_path / "config.yaml"
    correction = Path("prompts/correction.prompt.txt").resolve()
    summarise = Path("prompts/summarise.prompt.txt").resolve()
    consolidate = Path("prompts/consolidate.prompt.txt").resolve()
    config_path.write_text(
        f"""
pipeline:
  correction_provider: fast
  summarisers: [fast]
  consolidator: fast
vendors:
  acme:
    type: mock
    api_key_env: ACME_API_KEY
    base_url: https://api.acme.test
providers:
{providers_yaml}
prompts:
  correction: {correction}
  summarise: {summarise}
  consolidate: {consolidate}
""",
        encoding="utf-8",
    )
    return config_path


def test_provider_inherits_vendor_settings(tmp_path):
    config = load_config(
        _write_vendor_config(
            tmp_path,
            """
  fast:
    vendor: acme
    model: acme-fast
    max_output_tokens: 100
    reasoning_effort: low
""",
        )
    )

    provider = config.providers["fast"]
    assert provider.type == "mock"
    assert provider.model == "acme-fast"
    assert provider.api_key_env == "ACME_API_KEY"
    assert provider.base_url == "https://api.acme.test"
    assert provider.max_output_tokens == 100
    # "vendor" is a config shortcut, not a setting passed on to the provider.
    assert provider.extra == {"reasoning_effort": "low"}


def test_provider_settings_override_vendor_settings(tmp_path):
    config = load_config(
        _write_vendor_config(
            tmp_path,
            """
  fast:
    vendor: acme
    model: acme-fast
    api_key_env: OTHER_KEY
""",
        )
    )

    assert config.providers["fast"].api_key_env == "OTHER_KEY"
    assert config.providers["fast"].base_url == "https://api.acme.test"


def test_provider_with_unknown_vendor_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown vendor 'nope'"):
        load_config(
            _write_vendor_config(
                tmp_path,
                """
  fast:
    vendor: nope
    model: acme-fast
""",
            )
        )


def test_apply_prompt_set_overrides_summarise_and_consolidate_only(tmp_path):
    config = load_config(_write_prompt_set_config(tmp_path))
    set_dir = tmp_path / "prompt-sets" / "pr1"
    set_dir.mkdir(parents=True)
    custom_summarise = set_dir / "summarise.prompt.txt"
    custom_consolidate = set_dir / "consolidate.prompt.txt"
    custom_summarise.write_text("custom summarise {{document}} {{summary_language}}", encoding="utf-8")
    custom_consolidate.write_text("custom consolidate {{summary1}}", encoding="utf-8")

    updated = apply_prompt_set(config, "pr1")

    assert updated.selected_prompt_set == "pr1"
    assert updated.prompts["correction"] == config.prompts["correction"]
    assert updated.prompts["summarise"] == custom_summarise.resolve()
    assert updated.prompts["consolidate"] == custom_consolidate.resolve()


def test_apply_prompt_set_rejects_missing_required_files(tmp_path):
    config = load_config(_write_prompt_set_config(tmp_path))
    set_dir = tmp_path / "prompt-sets" / "pr1"
    set_dir.mkdir(parents=True)
    (set_dir / "summarise.prompt.txt").write_text("custom {{document}}", encoding="utf-8")

    with pytest.raises(ValueError, match="consolidate"):
        apply_prompt_set(config, "pr1")


def test_apply_prompt_set_rejects_path_names(tmp_path):
    config = load_config(_write_prompt_set_config(tmp_path))

    with pytest.raises(ValueError, match="single folder name"):
        apply_prompt_set(config, "../pr1")


def test_master_config_model_aliases_resolve():
    config = load_config("config/master_config.yaml")

    assert apply_consolidator(config, None) is config
    assert apply_consolidator(config, "argon").consolidator == "gemini_argon_medium"
    assert apply_consolidator(config, "carbon").consolidator == "gemini_carbon_medium"
    # A full provider name works without an alias.
    assert apply_consolidator(config, "gemini_argon_high").consolidator == "gemini_argon_high"


def test_apply_consolidator_changes_only_the_consolidator(tmp_path):
    config = load_config(_write_alias_config(tmp_path, "quick: other"))

    updated = apply_consolidator(config, "quick")

    assert updated.consolidator == "other"
    assert updated.correction_provider == config.correction_provider
    assert updated.summarisers == config.summarisers
    assert updated.prompts == config.prompts


def test_config_without_model_aliases_still_loads(tmp_path):
    config = load_config(_write_prompt_set_config(tmp_path))

    assert config.model_aliases == {}
    assert apply_consolidator(config, "claude").consolidator == "claude"


@pytest.mark.parametrize("aliases", [[], "other", None])
def test_model_aliases_require_a_mapping(tmp_path, aliases):
    path = _write_alias_config(tmp_path, "quick: other")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["model_aliases"] = aliases
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(path)


@pytest.mark.parametrize(
    "alias_line",
    [
        '"": other', '" quick ": other', "123: other",
        'quick: "   "', 'quick: " other "', "quick: 123", "quick: [other]",
    ],
)
def test_model_alias_names_and_targets_require_nonempty_strings(tmp_path, alias_line):
    with pytest.raises(ConfigError, match="must be nonempty strings"):
        load_config(_write_alias_config(tmp_path, alias_line))


def test_model_alias_cannot_redirect_a_provider_id(tmp_path):
    with pytest.raises(ConfigError, match="conflicts with an existing provider ID"):
        load_config(_write_alias_config(tmp_path, "claude: other"))


def test_model_alias_can_repeat_its_own_provider_id(tmp_path):
    config = load_config(_write_alias_config(tmp_path, "other: other"))

    assert apply_consolidator(config, "other").consolidator == "other"


def _write_alias_config(tmp_path: Path, alias_line: str) -> Path:
    config_path = tmp_path / "config.yaml"
    correction = Path("prompts/correction.prompt.txt").resolve()
    summarise = Path("prompts/summarise.prompt.txt").resolve()
    consolidate = Path("prompts/consolidate.prompt.txt").resolve()
    config_path.write_text(
        f"""
pipeline:
  correction_provider: claude
  summarisers: [claude]
  consolidator: claude
model_aliases:
  {alias_line}
providers:
  claude:
    type: mock
    model: claude-test
  other:
    type: mock
    model: other-test
prompts:
  correction: {correction}
  summarise: {summarise}
  consolidate: {consolidate}
""",
        encoding="utf-8",
    )
    return config_path


def _write_prompt_set_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    correction = Path("prompts/correction.prompt.txt").resolve()
    summarise = Path("prompts/summarise.prompt.txt").resolve()
    consolidate = Path("prompts/consolidate.prompt.txt").resolve()
    config_path.write_text(
        f"""
pipeline:
  correction_provider: claude
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
prompt_sets:
  directory: {tmp_path / "prompt-sets"}
runtime:
  concurrency: 1
  retries: 1
""",
        encoding="utf-8",
    )
    return config_path
