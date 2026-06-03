# Project Memory: Document Summariser

This repo is a Python CLI app for PDF OCR, Urdu summarisation, multi-provider consolidation, and plain UTF-8 text output.

## Repository

- Local path: `C:\Users\Hamad.Mobeen\Downloads\programming\document-summariser`
- GitHub remote: `https://github.com/ham17ahm/document-summariser.git`
- Branch: `main`
- Canonical editable config: `config/master_config.yaml`

## Current App Flow

1. Validate input PDF.
2. Render PDF pages to PNG with PyMuPDF and save them under `page_images/`.
3. Run Google Cloud Vision document OCR.
4. Correct OCR text using the configured correction provider. The correction stage passes the OCR text plus rendered page images to Gemini so it can compare OCR against the handwritten source.
5. Summarise corrected text in parallel with ChatGPT, Gemini, Grok, and DeepSeek.
6. Require at least `pipeline.min_summaries` successful summaries. The current master config requires all four.
7. Consolidate provider summaries with Claude.
8. Write the final Urdu summary as plain UTF-8 text:
   - run artifact: `05_output.txt`
   - simple runner copy: `<input-pdf-stem>.txt` in `output.final_text_directory`
9. Write intermediate artifacts and `manifest.json`.

## Current Default Providers

- OCR: Google Cloud Vision
- Correction: Gemini `gemini-2.5-pro`
  - `max_output_tokens: 16384`
  - `thinking_config.thinking_budget: 1024`
- Summarisers:
  - ChatGPT: `gpt-5.2`
  - Gemini: `gemini-2.5-pro`
  - Grok: `grok-4.3`
  - DeepSeek: `deepseek-v4-pro`
- Consolidator: Claude `claude-opus-4-7`
  - adaptive thinking
  - `output_config.effort: xhigh`

## Key Modules

- `summarise_pdf.py`: simple runner that writes final `.txt` beside the input PDF.
- `src/document_summariser/cli.py`: advanced CLI entry point and dependency wiring.
- `src/document_summariser/config.py`: YAML loading, validation, and default/master config resolution.
- `src/document_summariser/ocr.py`: OCR protocol, Google Cloud Vision OCR, mock OCR, PDF page rendering.
- `src/document_summariser/providers/base.py`: provider protocol, retry behavior, cloud provider adapters, Gemini image attachments.
- `src/document_summariser/providers/registry.py`: provider factory map, including OpenAI-compatible Grok and DeepSeek.
- `src/document_summariser/stages/pipeline.py`: pipeline orchestration and artifact writing.
- `src/document_summariser/prompts.py`: prompt loading and strict placeholder rendering.

## Prompt Files

- `prompts/correction.prompt.txt`: OCR correction against attached handwritten page images.
- `prompts/summarise.prompt.txt`: first-person, conversational/respectful Urdu correspondence summary style.
- `prompts/consolidate.prompt.txt`: four-summary consolidation using `summary1` through `summary4`.
- Packaged default copies live under `src/document_summariser/defaults/prompts/`.

## Credentials And Secrets

Real secrets are present locally but must not be committed.

Expected `.env` values:

```bash
GOOGLE_APPLICATION_CREDENTIALS=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
XAI_API_KEY=...
DEEPSEEK_API_KEY=...
DOCUMENT_SUMMARISER_CONFIG=config/master_config.yaml
DOCUMENT_SUMMARISER_OUTPUT_DIR=./runs/
```

`GOOGLE_CLOUD_PROJECT` is optional when the service-account JSON already contains `project_id`.

`.gitignore` protects `.env`, `.env.*`, and local Google credential JSON files such as `vision-ocr-*.json`.

## Verification Commands

Use the active Python environment or virtual environment:

```bash
python -m compileall -q src tests summarise_pdf.py
python -m pytest -q --basetemp=pytest-tmp -p no:cacheprovider
summarise --help
```

Expected current test result:

```text
18 passed
```

On this Windows/sandbox setup, pytest may need to run outside the sandbox because pytest-created temp directories can receive ACLs that the sandbox cannot scan. Clean `pytest-tmp/` after test runs if it remains.

## Operational Notes

- Live OCR with Google Vision has succeeded on `X:\PS Office\Scan-Hamad\20260603133539.pdf`.
- A previous Gemini correction failure was caused by Gemini spending the full output budget on thinking tokens. The fix was to raise Gemini `max_output_tokens` and set `thinking_config.thinking_budget`.
- Final output is now text only. DOCX rendering and the `python-docx` dependency were removed.
- The default final text directory is `C:\Users\Hamad.Mobeen\Downloads`.
- Tests use mock OCR and mock model providers, so they do not require API keys.
