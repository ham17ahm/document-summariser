from __future__ import annotations

import argparse
import shutil
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from textwrap import fill

from document_summariser.application import SummaryRunResult, run_document_summary
from document_summariser.config import AppConfig, available_prompt_sets, load_config, preferred_config_path
from document_summariser.env import load_local_env
from document_summariser.errors import print_cli_error, redact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="summarise", add_help=False)
    parser.add_argument("-h", "--help", action="store_true", help="Show help and available options.")
    parser.add_argument("pdf", nargs="*", help="Path to one or more PDFs to summarise.")
    parser.add_argument(
        "--config",
        default=os.environ.get("DOCUMENT_SUMMARISER_CONFIG") or str(preferred_config_path()),
        help="Path to config YAML.",
    )
    parser.add_argument("--out", default=None, help="Output runs directory.")
    parser.add_argument(
        "-p",
        "--prompt-set",
        default=None,
        help="Prompt set folder under prompt_sets.directory to use for summarise and consolidate prompts.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help=(
            "Model for the consolidation step: a short name from model_aliases in the "
            "config or a provider ID. Defaults to pipeline.consolidator."
        ),
    )
    parser.add_argument(
        "--final-text",
        default=None,
        help="Optional path for a copy of the final TXT output. Single PDF only.",
    )
    parser.add_argument(
        "--publish-final",
        dest="publish_final",
        action="store_true",
        default=True,
        help=(
            "Copy each final TXT to output.final_text_directory from the config "
            "(defaults to beside the input PDF) and print its path. This is the default."
        ),
    )
    parser.add_argument(
        "--no-publish-final",
        dest="publish_final",
        action="store_false",
        help="Only write run artifacts; do not copy the final TXT to the final output folder.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="Maximum number of PDFs to process concurrently when multiple PDFs are given.",
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
        parser = build_parser()
        args = parser.parse_args(argv)
    except Exception as exc:  # noqa: BLE001
        debug = "--debug-errors" in (argv or sys.argv[1:])
        return print_cli_error(exc, sys.stderr, debug=debug)

    debug: bool = args.debug_errors
    show_help = args.help or not args.pdf
    if show_help:
        parser.print_help()
    elif len(args.pdf) > 1 and args.final_text:
        print("Error: --final-text cannot be used with multiple input PDFs.", file=sys.stderr)
        return 1

    try:
        options = _format_available_options(load_config(args.config))
    except Exception as exc:  # noqa: BLE001 - use the CLI error renderer for config failures
        exit_code = print_cli_error(exc, sys.stderr, debug=debug)
        return 0 if show_help else exit_code

    if show_help:
        print(f"\n{options}")
        return 0

    print(options, file=sys.stderr)

    if len(args.pdf) == 1:
        try:
            result = run_document_summary(
                args.pdf[0],
                config_path=args.config,
                output_dir=args.out,
                prompt_set=args.prompt_set,
                consolidator=args.model,
            )
            if args.final_text:
                final_text = copy_final_text(result.output_text_path, Path(args.final_text))
                print(f"Wrote final TXT to {final_text}")
            if args.publish_final:
                print(f"Wrote final TXT to {publish_final_text(result)}")
            _print_quality_summary(result)
            print(f"Wrote run artifacts to {result.artifacts.root}")
            return 0
        except Exception as exc:  # noqa: BLE001 - CLI boundary must render all runtime failures
            return print_cli_error(exc, sys.stderr, debug=debug)

    return _run_concurrent(
        args.pdf,
        args.config,
        args.out,
        debug,
        args.parallel,
        args.publish_final,
        args.prompt_set,
        args.model,
    )


def _format_available_options(config: AppConfig) -> str:
    try:
        prompt_sets = ", ".join(available_prompt_sets(config)) or "none"
    except OSError:
        prompt_sets = "unavailable (could not read the prompt sets directory)"

    rows = {
        "Prompt sets (-p)": prompt_sets,
        "Model aliases (-m)": ", ".join(sorted(config.model_aliases)) or "none",
        "Provider IDs (-m)": ", ".join(sorted(key for key in config.providers if isinstance(key, str))),
        "Default model": config.consolidator,
    }
    lines = ["Available options"]
    for label, value in rows.items():
        indent = f"  {label:<21}"
        lines.append(fill(
            redact(value),
            width=80,
            initial_indent=indent,
            subsequent_indent=" " * len(indent),
            break_long_words=False,
            break_on_hyphens=False,
        ))
    return "\n".join(lines)


def _run_concurrent(
    pdf_paths: list[str],
    config: str,
    out: str | None,
    debug: bool,
    parallel: int,
    publish_final: bool = False,
    prompt_set: str | None = None,
    consolidator: str | None = None,
) -> int:
    any_failed = False
    # Each pipeline already fans out its own summariser threads; running every
    # PDF at once multiplies that into a rate-limit storm against shared keys.
    max_workers = max(1, min(parallel, len(pdf_paths)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_document_summary, path, config, out, prompt_set, consolidator): path
            for path in pdf_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            label = Path(path).name
            try:
                result = future.result()
                if publish_final:
                    print(f"[{label}] Wrote final TXT to {publish_final_text(result)}")
                _print_quality_summary(result, label=label)
                print(f"[{label}] Wrote run artifacts to {result.artifacts.root}")
            except Exception as exc:  # noqa: BLE001 - report per-PDF failures without stopping others
                any_failed = True
                print(f"[{label}] Failed:", file=sys.stderr)
                print_cli_error(exc, sys.stderr, debug=debug)
    return 1 if any_failed else 0


def _print_quality_summary(result: SummaryRunResult, label: str | None = None) -> None:
    prefix = f"[{label}] " if label else ""
    quality = result.context.manifest.get("quality") or {}

    ocr = quality.get("ocr") or {}
    ocr_parts: list[str] = []
    average = ocr.get("average_confidence")
    if average is not None:
        ocr_parts.append(f"{average * 100:.1f}% average over {ocr.get('pages', '?')} pages")
        flagged = ocr.get("low_confidence_pages") or []
        threshold = ocr.get("low_confidence_threshold")
        if flagged:
            pages = ", ".join(str(page) for page in flagged)
            ocr_parts.append(f"low-confidence pages: {pages}")
        elif threshold is not None:
            ocr_parts.append(f"all pages above {threshold * 100:.0f}% threshold")
    if ocr_parts:
        print(f"{prefix}OCR confidence: {'; '.join(ocr_parts)}")

    correction = quality.get("correction") or {}
    correction_parts: list[str] = []
    token_probability = correction.get("avg_token_probability")
    if token_probability is not None:
        correction_parts.append(
            f"model token confidence ~{token_probability * 100:.0f}% "
            f"(avg_logprobs {correction.get('avg_logprobs'):.2f})"
        )
    change_ratio = correction.get("change_ratio")
    if change_ratio is not None:
        correction_parts.append(f"{change_ratio * 100:.1f}% of text changed vs OCR")
    if correction_parts:
        print(f"{prefix}Correction: {'; '.join(correction_parts)}")

    warning = result.context.manifest.get("correction_warning")
    if warning:
        print(f"{prefix}Warning: {warning.get('reason', 'correction quality warning')}")


def copy_final_text(source: Path, destination: Path) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def publish_final_text(result: SummaryRunResult) -> Path:
    final_dir = Path(
        result.config.output.get("final_text_directory", result.input_pdf.parent)
    ).expanduser()
    return copy_final_text(
        result.output_text_path,
        final_dir / result.input_pdf.with_suffix(".txt").name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
