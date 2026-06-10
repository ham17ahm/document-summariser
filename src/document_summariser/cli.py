from __future__ import annotations

import argparse
import shutil
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from document_summariser.application import run_document_summary
from document_summariser.config import preferred_config_path
from document_summariser.env import load_local_env
from document_summariser.errors import print_cli_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="summarise")
    parser.add_argument("pdf", nargs="+", help="Path to one or more PDFs to summarise.")
    parser.add_argument(
        "--config",
        default=os.environ.get("DOCUMENT_SUMMARISER_CONFIG") or str(preferred_config_path()),
        help="Path to config YAML.",
    )
    parser.add_argument("--out", default=None, help="Output runs directory.")
    parser.add_argument(
        "--final-text",
        default=None,
        help="Optional path for a copy of the final TXT output. Single PDF only.",
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
    except Exception as exc:  # noqa: BLE001
        debug = "--debug-errors" in (argv or sys.argv[1:])
        return print_cli_error(exc, sys.stderr, debug=debug)

    debug: bool = args.debug_errors

    if len(args.pdf) == 1:
        try:
            result = run_document_summary(args.pdf[0], config_path=args.config, output_dir=args.out)
            if args.final_text:
                final_text = copy_final_text(result.output_text_path, Path(args.final_text))
                print(f"Wrote final TXT to {final_text}")
            print(f"Wrote run artifacts to {result.artifacts.root}")
            return 0
        except Exception as exc:  # noqa: BLE001 - CLI boundary must render all runtime failures
            return print_cli_error(exc, sys.stderr, debug=debug)

    if args.final_text:
        print("Error: --final-text cannot be used with multiple input PDFs.", file=sys.stderr)
        return 1

    return _run_concurrent(args.pdf, args.config, args.out, debug)


def _run_concurrent(pdf_paths: list[str], config: str, out: str | None, debug: bool) -> int:
    any_failed = False
    with ThreadPoolExecutor(max_workers=len(pdf_paths)) as executor:
        futures = {
            executor.submit(run_document_summary, path, config, out): path
            for path in pdf_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            label = Path(path).name
            try:
                result = future.result()
                print(f"[{label}] Wrote run artifacts to {result.artifacts.root}")
            except Exception as exc:  # noqa: BLE001 - report per-PDF failures without stopping others
                any_failed = True
                print(f"[{label}] Failed:", file=sys.stderr)
                print_cli_error(exc, sys.stderr, debug=debug)
    return 1 if any_failed else 0


def copy_final_text(source: Path, destination: Path) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


if __name__ == "__main__":
    raise SystemExit(main())
