import pytest

from document_summariser.prompts import PromptRenderError, PromptTemplate


def test_prompt_render_rejects_missing_variables(tmp_path):
    prompt = PromptTemplate(path=tmp_path / "prompt.txt", text="Hello {{name}} from {{place}}.", sha256="sha")

    with pytest.raises(PromptRenderError, match="place"):
        prompt.render(name="Aisha")


def test_prompt_render_replaces_whitespace_padded_variables(tmp_path):
    prompt = PromptTemplate(path=tmp_path / "prompt.txt", text="Hello {{ name }}.", sha256="sha")

    assert prompt.render(name="Aisha") == "Hello Aisha."


def test_prompt_render_request_splits_stable_prefix_from_dynamic_input(tmp_path):
    prompt = PromptTemplate(
        path=tmp_path / "prompt.txt",
        text="Summarise in {{language}}.\n\nDocument:\n{{document}}",
        sha256="sha",
    )

    request = prompt.render_request(
        dynamic_variables={"document"},
        language="Urdu",
        document="Variable text",
    )

    assert request.system == "Summarise in Urdu.\n\nDocument:\n"
    assert request.user == "Variable text"
    assert len(request.cache_key) == 64


def test_prompt_render_request_requires_a_dynamic_placeholder(tmp_path):
    prompt = PromptTemplate(
        path=tmp_path / "prompt.txt",
        text="Hello {{name}}.",
        sha256="sha",
    )

    with pytest.raises(PromptRenderError, match="no dynamic variables"):
        prompt.render_request(dynamic_variables={"document"}, name="Aisha")
