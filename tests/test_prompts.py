import pytest

from document_summariser.prompts import PromptRenderError, PromptTemplate


def test_prompt_render_rejects_missing_variables(tmp_path):
    prompt = PromptTemplate(path=tmp_path / "prompt.txt", text="Hello {{name}} from {{place}}.", sha256="sha")

    with pytest.raises(PromptRenderError, match="place"):
        prompt.render(name="Aisha")


def test_prompt_render_replaces_whitespace_padded_variables(tmp_path):
    prompt = PromptTemplate(path=tmp_path / "prompt.txt", text="Hello {{ name }}.", sha256="sha")

    assert prompt.render(name="Aisha") == "Hello Aisha."
