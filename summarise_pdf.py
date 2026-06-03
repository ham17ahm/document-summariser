from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from document_summariser.artifacts import ArtifactStore
from document_summariser.cli import copy_final_text
from document_summariser.config import load_config, preferred_config_path
from document_summariser.env import load_local_env
from document_summariser.ocr import build_ocr_adapter
from document_summariser.providers.registry import build_provider_registry
from document_summariser.stages import Pipeline
from document_summariser.stages.context import RunContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarise a PDF and write the final TXT beside the input file.",
    )
    parser.add_argument("pdf", help="Path to the PDF file.")
    parser.add_argument(
        "--config",
        default=os.environ.get("DOCUMENT_SUMMARISER_CONFIG") or str(preferred_config_path()),
        help="Path to config YAML.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    args = build_parser().parse_args(argv)
    input_pdf = Path(args.pdf).resolve()
    config = load_config(args.config)
    artifacts = ArtifactStore.create(
        os.environ.get("DOCUMENT_SUMMARISER_OUTPUT_DIR") or config.output.get("destination", "./runs/"),
        input_pdf,
    )

    context = RunContext(
        input_pdf=input_pdf,
        config=config,
        artifacts=artifacts,
        ocr=build_ocr_adapter(config),
        providers=build_provider_registry(config),
        manifest={
            "input_file": str(input_pdf),
            "config_file": str(config.source_path),
            "ocr_provider": config.ocr.get("provider"),
            "providers": {
                provider_id: {"type": provider.type, "model": provider.model}
                for provider_id, provider in config.providers.items()
            },
        },
    )

    Pipeline().run(context)
    final_path = copy_final_text(artifacts.root / "05_output.txt", input_pdf.with_suffix(".txt"))
    print(final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
