# Project Memory: Document Summariser

This repo is a Python CLI app for PDF OCR, Urdu summarisation, and right-to-left DOCX output.

## Repository

- Local path: `/Users/hamadahmadmobeen/programming/document-summariser`
- GitHub remote: `https://github.com/ham17ahm/document-summariser.git`
- Branch: `main`
- Initial pushed commit: `2de0a14 Initial production-ready document summariser`

## Current App Flow

1. Validate input PDF.
2. Render PDF pages to PNG with PyMuPDF.
3. Run Google Cloud Vision document OCR.
4. Correct OCR text using the configured correction provider.
5. Summarise corrected text in parallel with Claude, OpenAI, Gemini, and DeepSeek.
6. Require at least `pipeline.min_summaries` successful summaries.
7. Consolidate provider summaries.
8. Render final Urdu summary as a right-to-left DOCX.
9. Write intermediate artifacts and `manifest.json`.

## Key Modules

- `src/document_summariser/cli.py`: CLI entry point and dependency wiring.
- `src/document_summariser/config.py`: YAML loading and validation.
- `src/document_summariser/ocr.py`: OCR protocol, Google Cloud Vision OCR, mock OCR.
- `src/document_summariser/providers/base.py`: provider protocol, retry behavior, cloud provider adapters.
- `src/document_summariser/providers/registry.py`: provider factory map.
- `src/document_summariser/stages/pipeline.py`: pipeline orchestration.
- `src/document_summariser/rendering.py`: right-to-left DOCX rendering.
- `src/document_summariser/prompts.py`: prompt loading and strict placeholder rendering.

## Credentials Still Needed

The user will provide credentials later. Do not commit real keys or credential JSON files.

Expected `.env` values:

```bash
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/google-service-account.json
GOOGLE_CLOUD_PROJECT=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
```

## Verification Commands

Use the existing virtual environment when present:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests
.venv/bin/summarise --help
```

Expected current test result:

```text
8 passed
```

Live OCR/model calls have not been run yet because credentials are intentionally absent.

## Design Notes

- The code was refactored to better follow `Agents.md`.
- `Pipeline` is orchestration-focused.
- DOCX rendering is extracted into `DocxRenderer`.
- Providers are created through a factory map, not a long conditional chain.
- Prompt rendering fails loudly via `PromptRenderError` when variables are missing.
- Tests use mocks and do not require cloud credentials.
