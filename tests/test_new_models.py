import json
import sys
from types import SimpleNamespace

import pytest
import yaml

from document_summariser.application import run_document_summary
from document_summariser.config import load_config
from document_summariser.errors import ProviderError
from document_summariser.ocr import OcrPage, OcrResult
from document_summariser.prompts import PromptRequest
from document_summariser.providers.registry import build_provider_registry


def test_optional_models_match_packaged_defaults():
    master = load_config("config/master_config.yaml")
    packaged = load_config()
    for name in ("gemini_argon_medium", "gemini_argon_high", "chatgpt_gpt6"):
        assert master.providers[name] == packaged.providers[name]
        assert name not in packaged.summarisers
        assert name not in (packaged.correction_provider, packaged.consolidator)


@pytest.fixture
def fake_apis(monkeypatch):
    calls = {"gemini": [], "openai": [], "finish_reason": "stop"}

    def gemini_generate(**kwargs):
        calls["gemini"].append(kwargs)
        return SimpleNamespace(text="یہ ایک آزمائشی خلاصہ ہے۔")

    def openai_generate(**kwargs):
        calls["openai"].append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="آزمائشی خلاصہ"),
                finish_reason=calls["finish_reason"],
            )],
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=50,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=20),
            ),
        )

    # Keep the real Gemini SDK config and image types to validate request schemas.
    from google import genai
    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(
        models=SimpleNamespace(generate_content=gemini_generate),
    ))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(
        OpenAI=lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=openai_generate)),
        ),
    ))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return calls


@pytest.mark.parametrize("level", ["medium", "high"])
def test_new_models_through_full_pipeline(tmp_path, monkeypatch, fake_apis, level):
    master = load_config("config/master_config.yaml")
    raw = master.raw
    argon = f"gemini_argon_{level}"
    raw["pipeline"].update(
        correction_provider=argon,
        summarisers=["chatgpt_gpt6", argon, "grok", "deepseek"],
        consolidator=argon,
    )
    for name in ("grok", "deepseek"):
        raw["providers"][name]["type"] = "mock"
    raw["prompts"] = {name: str(path) for name, path in master.prompts.items()}
    raw["prompt_sets"]["directory"] = str(master.prompt_sets_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    class FakeOcr:
        def extract(self, pdf_path, artifacts):
            image = artifacts.write_bytes("page_images/page_0001.png", b"test-image")
            return OcrResult(provider="mock", pages=[OcrPage(
                page_number=1, text="یہ ایک آزمائشی متن ہے۔", image_path=str(image),
                confidence=0.99, low_confidence=False,
            )])

    monkeypatch.setattr("document_summariser.application.build_ocr_adapter", lambda config: FakeOcr())
    result = run_document_summary(
        pdf, config_path=config_path, output_dir=tmp_path / "runs", prompt_set="default",
    )
    assert result.output_text_path.read_text(encoding="utf-8") == "یہ ایک آزمائشی خلاصہ ہے۔"
    manifest = json.loads((result.artifacts.root / "manifest.json").read_text(encoding="utf-8"))
    assert all(stage["status"] == "succeeded" for stage in manifest["stages"].values())
    assert manifest["prompt_set"] == "default"
    assert manifest["correction_attachments"]["sent"] == 1
    assert manifest["summarisation_providers"]["failed"] == {}
    assert len(result.context.summaries) == 4
    assert len(fake_apis["gemini"]) == 3
    for call in fake_apis["gemini"]:
        assert call["model"] == "gemini-4-argon"
        assert call["config"].thinking_config.thinking_level.value.lower() == level
        assert call["config"].thinking_config.thinking_budget is None
        assert call["config"].max_output_tokens == (32768 if level == "medium" else 65536)
    image_part = fake_apis["gemini"][0]["contents"][0].parts[1]
    assert image_part.inline_data.data == b"test-image"
    assert image_part.inline_data.mime_type == "image/png"
    assert len(fake_apis["openai"]) == 1
    request = fake_apis["openai"][0]
    assert request["model"] == "gpt-6-astra"
    assert request["reasoning_effort"] == "medium"
    assert request["max_completion_tokens"] == 32768
    assert request["prompt_cache_key"]
    assert "temperature" not in request
    assert "prompt_cache_retention" not in request


def test_gpt6_rejects_truncation_without_retry(fake_apis):
    fake_apis["finish_reason"] = "length"
    provider = build_provider_registry(load_config("config/master_config.yaml"))["chatgpt_gpt6"]
    with pytest.raises(ProviderError, match="truncated"):
        provider.generate(PromptRequest(system="Summarise", user="Test", cache_key="test"))
    assert len(fake_apis["openai"]) == 1
