from __future__ import annotations

import argparse
import shutil
import os
from pathlib import Path

from document_summariser.artifacts import ArtifactStore
from document_summariser.config import load_config, preferred_config_path
from document_summariser.env import load_local_env
from document_summariser.ocr import build_ocr_adapter
from document_summariser.providers.registry import build_provider_registry
from document_summariser.stages import Pipeline
from document_summariser.stages.context import RunContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="summarise")
    parser.add_argument("pdf", help="Path to a single PDF to summarise.")
    parser.add_argument(
        "--config",
        default=os.environ.get("DOCUMENT_SUMMARISER_CONFIG") or str(preferred_config_path()),
        help="Path to config YAML.",
    )
    parser.add_argument("--out", default=None, help="Output runs directory.")
    parser.add_argument(
        "--final-text",
        default=None,
        help="Optional path for a copy of the final TXT output. Defaults to run artifacts only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    input_pdf = Path(args.pdf).resolve()
    destination = args.out or os.environ.get("DOCUMENT_SUMMARISER_OUTPUT_DIR") or config.output.get("destination", "./runs/")
    artifacts = ArtifactStore.create(destination, input_pdf)
    ocr = build_ocr_adapter(config)
    providers = build_provider_registry(config)

    context = RunContext(
        input_pdf=input_pdf,
        config=config,
        artifacts=artifacts,
        ocr=ocr,
        providers=providers,
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
    if args.final_text:
        final_text = copy_final_text(artifacts.root / "05_output.txt", Path(args.final_text))
        print(f"Wrote final TXT to {final_text}")
    print(f"Wrote run artifacts to {artifacts.root}")
    return 0


def copy_final_text(source: Path, destination: Path) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


if __name__ == "__main__":
    raise SystemExit(main())
