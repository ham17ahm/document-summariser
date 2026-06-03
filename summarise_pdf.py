from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from document_summariser.application import run_document_summary
from document_summariser.cli import copy_final_text
from document_summariser.config import preferred_config_path
from document_summariser.env import load_local_env
from document_summariser.errors import print_cli_error


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
    parser.add_argument(
        "--debug-errors",
        action="store_true",
        help="Print a traceback after the user-facing error summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        load_local_env()
        args = build_parser().parse_args(argv)
        result = run_document_summary(args.pdf, config_path=args.config)
        final_output_dir = Path(result.config.output.get("final_text_directory", result.input_pdf.parent)).expanduser()
        final_path = copy_final_text(
            result.output_text_path,
            final_output_dir / result.input_pdf.with_suffix(".txt").name,
        )
        print(final_path)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary must render all runtime failures
        debug = "--debug-errors" in (argv or sys.argv[1:])
        return print_cli_error(exc, sys.stderr, debug=debug)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
