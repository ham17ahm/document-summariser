from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from document_summariser.artifacts import ArtifactStore
from document_summariser.config import AppConfig
from document_summariser.ocr import OcrAdapter
from document_summariser.providers.base import ProviderAdapter


@dataclass
class RunContext:
    input_pdf: Path
    config: AppConfig
    artifacts: ArtifactStore
    ocr: OcrAdapter
    providers: dict[str, ProviderAdapter]
    manifest: dict[str, Any] = field(default_factory=dict)
    ocr_text: str = ""
    ocr_page_images: list[str] = field(default_factory=list)
    corrected_text: str = ""
    summaries: dict[str, str] = field(default_factory=dict)
    consolidated_summary: str = ""
