"""Simple runner: summarise one or more PDFs and publish the final TXT(s).

Thin wrapper over the main CLI. Each final TXT goes to
output.final_text_directory from the config, defaulting to the directory of
the input PDF.
"""

from __future__ import annotations

import sys

from document_summariser.cli import main as cli_main


def main(argv: list[str] | None = None) -> int:
    forwarded = list(argv) if argv is not None else sys.argv[1:]
    return cli_main([*forwarded, "--publish-final"])


if __name__ == "__main__":
    raise SystemExit(main())
