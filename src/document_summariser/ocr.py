from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from document_summariser.artifacts import ArtifactStore
from document_summariser.config import AppConfig
from document_summariser.errors import InputFileError, OcrError


class OcrAdapter(Protocol):
    def extract(self, pdf_path: Path, artifacts: ArtifactStore) -> "OcrResult":
        ...


@dataclass(frozen=True)
class OcrPage:
    page_number: int
    text: str
    confidence: float | None
    low_confidence: bool
    image_path: str | None


@dataclass(frozen=True)
class OcrResult:
    provider: str
    pages: list[OcrPage]

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    def to_json(self) -> dict[str, Any]:
        return {"provider": self.provider, "pages": [asdict(page) for page in self.pages]}


@dataclass
class MockOcrAdapter:
    provider_name: str = "mock"

    def extract(self, pdf_path: Path, artifacts: ArtifactStore) -> OcrResult:
        validate_pdf(pdf_path)
        page = OcrPage(
            page_number=1,
            text="Mock OCR text for tests.",
            confidence=None,
            low_confidence=False,
            image_path=None,
        )
        return OcrResult(provider=self.provider_name, pages=[page])


@dataclass
class GoogleCloudVisionOcrAdapter:
    language_hints: list[str]
    page_image_dpi: int
    low_confidence_threshold: float

    def extract(self, pdf_path: Path, artifacts: ArtifactStore) -> OcrResult:
        validate_pdf(pdf_path)
        try:
            from google.cloud import vision
        except ModuleNotFoundError as exc:
            raise OcrError(
                "Install the 'google-cloud-vision' package to use Google Cloud Vision OCR.",
                details={"provider": "google_cloud_vision"},
                cause=exc,
            ) from exc

        try:
            client = vision.ImageAnnotatorClient()
        except Exception as exc:  # noqa: BLE001 - Google auth/client errors vary by environment
            raise OcrError(
                "Could not initialise Google Cloud Vision OCR client.",
                details={"provider": "google_cloud_vision"},
                cause=exc,
            ) from exc
        pages: list[OcrPage] = []
        for page_number, image_bytes in enumerate(_render_pdf_pages(pdf_path, self.page_image_dpi), start=1):
            image_path = artifacts.write_bytes(f"page_images/page_{page_number:04d}.png", image_bytes)
            image = vision.Image(content=image_bytes)
            image_context = {"language_hints": self.language_hints} if self.language_hints else None
            try:
                response = client.document_text_detection(image=image, image_context=image_context)
            except Exception as exc:  # noqa: BLE001 - Google API exceptions are SDK-specific
                raise OcrError(
                    f"Google Cloud Vision OCR request failed on page {page_number}.",
                    details={
                        "provider": "google_cloud_vision",
                        "page": page_number,
                        "language_hints": self.language_hints,
                    },
                    cause=exc,
                ) from exc
            if response.error.message:
                raise OcrError(
                    f"Google Cloud Vision failed on page {page_number}: {response.error.message}",
                    details={"provider": "google_cloud_vision", "page": page_number},
                )

            text = response.full_text_annotation.text or ""
            confidence = _average_word_confidence(response.full_text_annotation)
            low_confidence = confidence is not None and confidence < self.low_confidence_threshold
            pages.append(
                OcrPage(
                    page_number=page_number,
                    text=text,
                    confidence=confidence,
                    low_confidence=low_confidence,
                    image_path=str(image_path),
                )
            )

        if not pages:
            raise OcrError("PDF did not contain any pages.", details={"input_file": str(pdf_path)})
        return OcrResult(provider="google_cloud_vision", pages=pages)


def build_ocr_adapter(config: AppConfig) -> OcrAdapter:
    provider = str(config.ocr.get("provider", "google_cloud_vision"))
    if provider == "mock":
        return MockOcrAdapter(provider_name=provider)
    if provider != "google_cloud_vision":
        raise OcrError(f"Unsupported OCR provider {provider!r}.", details={"provider": provider})

    return GoogleCloudVisionOcrAdapter(
        language_hints=[str(item) for item in config.ocr.get("language_hints", [])],
        page_image_dpi=int(config.ocr.get("page_image_dpi", 300)),
        low_confidence_threshold=float(config.ocr.get("low_confidence_threshold", 0.8)),
    )


def validate_pdf(path: Path) -> None:
    if not path.exists():
        raise InputFileError(f"Input file does not exist: {path}", details={"input_file": str(path)})
    if not path.is_file():
        raise InputFileError(f"Input path is not a file: {path}", details={"input_file": str(path)})
    if path.suffix.lower() != ".pdf":
        raise InputFileError("Input must be a PDF file.", details={"input_file": str(path)})
    try:
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise InputFileError(
                    "Input does not appear to be a readable PDF.",
                    details={"input_file": str(path)},
                )
    except InputFileError:
        raise
    except OSError as exc:
        raise InputFileError(
            "Could not read input PDF.",
            details={"input_file": str(path)},
        ) from exc


def _render_pdf_pages(pdf_path: Path, dpi: int) -> list[bytes]:
    try:
        import fitz
    except ModuleNotFoundError as exc:
        raise OcrError(
            "Install the 'PyMuPDF' package to render PDF pages for OCR.",
            details={"input_file": str(pdf_path), "dpi": dpi},
            cause=exc,
        ) from exc

    images: list[bytes] = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    try:
        with fitz.open(pdf_path) as document:
            for page in document:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                images.append(pixmap.tobytes("png"))
    except Exception as exc:  # noqa: BLE001 - PyMuPDF can raise several document/render exceptions
        raise OcrError(
            "Could not render PDF pages for OCR.",
            details={"input_file": str(pdf_path), "dpi": dpi},
            cause=exc,
        ) from exc
    return images


def _average_word_confidence(annotation: Any) -> float | None:
    confidences: list[float] = []
    for page in getattr(annotation, "pages", []) or []:
        for block in getattr(page, "blocks", []) or []:
            for paragraph in getattr(block, "paragraphs", []) or []:
                for word in getattr(paragraph, "words", []) or []:
                    confidence = getattr(word, "confidence", None)
                    if confidence is not None:
                        confidences.append(float(confidence))
    if not confidences:
        return None
    return round(sum(confidences) / len(confidences), 4)
