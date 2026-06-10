"""Simple runner: summarise a PDF and publish the final TXT.

Thin wrapper over the main CLI; equivalent to `summarise <pdf> --publish-final`.
The final TXT goes to output.final_text_directory from the config, defaulting
to the directory of the input PDF.
"""

from __future__ import annotations

import sys

from document_summariser.cli import main as cli_main


def main(argv: list[str] | None = None) -> int:
    forwarded = list(argv) if argv is not None else sys.argv[1:]
    return cli_main([*forwarded, "--publish-final"])


if __name__ == "__main__":
    raise SystemExit(main())
