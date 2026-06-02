import pytest

from document_summariser.artifacts import ArtifactStore
from document_summariser.ocr import MockOcrAdapter


def test_ocr_rejects_non_pdf(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("not a pdf", encoding="utf-8")
    artifacts = ArtifactStore(tmp_path / "run")

    with pytest.raises(ValueError, match="PDF"):
        MockOcrAdapter().extract(text_file, artifacts)
