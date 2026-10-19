# Project Memory: Document Summariser

This repo is a Python CLI app for PDF OCR, Urdu summarisation, multi-provider consolidation, and plain UTF-8 text output.

## Repository

- Local path: `C:\Users\Hamad.Mobeen\Downloads\programming\document-summariser`
- GitHub remote: `https://github.com/ham17ahm/document-summariser.git`
- Branch: `main`
- Canonical editable config: `config/master_config.yaml`

## Current App Flow

1. Load config and optionally apply a CLI-selected prompt set with `-p/--prompt-set`.
2. Validate input PDF.
3. Render PDF pages to PNG with PyMuPDF and save them under `page_images/`.
4. Run Google Cloud Vision document OCR.
5. Correct OCR text using the configured correction provider. The correction stage passes the OCR text plus rendered page images to Gemini so it can compare OCR against the handwritten source.
6. Summarise corrected text in parallel with ChatGPT, Claude, Grok, and DeepSeek.
7. Require at least `pipeline.min_summaries` successful summaries. The current master config requires all four.
8. Consolidate provider summaries with the configured Gemini consolidator.
9. Write the final Urdu summary as plain UTF-8 text:
   - run artifact: `05_output.txt`
   - simple runner copy: `<input-pdf-stem>.txt` in `output.final_text_directory`
10. Write intermediate artifacts and `manifest.json`, including `prompt_set`.

## Current Master Configuration (verified 2026-10-19)

Read `config/master_config.yaml` as the source of truth; packaged defaults differ.

- OCR: Google Cloud Vision.
- Correction: Gemini `gemini-2.5-flash`, `max_output_tokens: 16384`, `thinking_config.thinking_budget: 1024`.
- Summarisers: `chatgpt_gpt6` (`gpt-6-astra`), Claude (`claude-opus-4-7`), Grok (`grok-4.3`), DeepSeek (`deepseek-v4-pro`).
- DeepSeek: `max_output_tokens: 32768` in both master and packaged configs; the adapter sends this as `max_tokens`.
- Consolidator: `gemini_argon_high` (`gemini-4-argon`), `max_output_tokens: 65536`, `thinking_config.thinking_level: high`.
- `pipeline.min_summaries: 4` remains unchanged.

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
- Selectable prompt sets live under `prompts/sets/<name>/` and packaged copies under `src/document_summariser/defaults/prompts/sets/<name>/`.
- A prompt set must contain `summarise.prompt.txt` and `consolidate.prompt.txt`; the OCR correction prompt remains fixed by config.
- Use `summarise input.pdf -p <name>` or `summarise input.pdf --prompt-set <name>` to apply a prompt set to one PDF or the whole batch.
- `prompts/sets/default/` is the baseline selectable set copied from the current summarise/consolidate prompts.

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
56 passed (verified 2026-10-19)
```

On this Windows/sandbox setup, pytest may need to run outside the sandbox because pytest-created temp directories can receive ACLs that the sandbox cannot scan. Clean `pytest-tmp/` after test runs if it remains.

## Operational Notes

- Live OCR with Google Vision has succeeded on `X:\PS Office\Scan-Hamad\20260603133539.pdf`.
- A previous Gemini correction failure was caused by Gemini spending the full output budget on thinking tokens. The fix was to raise Gemini `max_output_tokens` and set `thinking_config.thinking_budget`.
- Final output is now text only. DOCX rendering and the `python-docx` dependency were removed.
- The default final text directory is `C:\Users\Hamad.Mobeen\Downloads`.
- Tests use mock OCR and mock model providers, so they do not require API keys.

## DeepSeek Truncation Fix (2026-10-19)

- A summarisation failure with three successes was caused by DeepSeek returning `finish_reason: length` at the former 4096-token output limit. This indicates per-request truncation, not depleted API credits or account quota.
- Raised DeepSeek's output budget to 32768 in master and packaged configs. The budget covers reasoning and the final answer; truncated responses remain rejected without retry. All four summaries are still required.
- Replaced deprecated `fitz` imports and references with `pymupdf` in `ocr.py`. The import warning was separate from the DeepSeek failure.
- Verification: 56 tests passed; compile check, CLI help, and diff checks passed. Mocked requests confirmed both configs send `max_tokens=32768` and reject truncation without retry. A synthetic two-page PDF rendered correctly at the requested DPI without the fitz warning.
- Live paid API pipeline was not rerun; success on the original document with the increased budget remains unverified.
- Removed generated pytest and bytecode cache artifacts after checks.
