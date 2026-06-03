import os

from document_summariser.env import load_local_env


def test_load_local_env_accepts_export_and_inline_comments(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        """
export FIRST=value
SECOND="two words" # comment
THIRD='keeps # hash'
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("FIRST", raising=False)
    monkeypatch.delenv("SECOND", raising=False)
    monkeypatch.delenv("THIRD", raising=False)

    load_local_env(env_path)

    assert os.environ["FIRST"] == "value"
    assert os.environ["SECOND"] == "two words"
    assert os.environ["THIRD"] == "keeps # hash"
