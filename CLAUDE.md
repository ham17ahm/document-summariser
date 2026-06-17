# AGENTS.md

## Purpose

This repository is a Python CLI app for PDF OCR, OCR correction, Urdu summarisation, multi-provider consolidation, and plain UTF-8 text output.

AI agents working in this repo should make focused, safe, well-tested changes while preserving the existing pipeline unless explicitly asked to change it.

Priorities:

1. Preserve existing behaviour.
2. Make the smallest safe change.
3. Keep code clear, testable, and maintainable.
4. Keep provider-specific logic isolated.
5. Verify changes before reporting completion.
6. Never expose or commit secrets.

---

## Codebase Overview

The app flow is:

1. Load config and optionally apply a CLI-selected prompt set with `-p/--prompt-set`.
2. Validate input PDF.
3. Render PDF pages to PNG using PyMuPDF.
4. Save page images under `page_images/`.
5. Run Google Cloud Vision OCR.
6. Correct OCR text using Gemini with rendered page images attached.
7. Summarise corrected text in parallel with ChatGPT, Gemini, Grok, and DeepSeek.
8. Require at least `pipeline.min_summaries` successful summaries.
9. Consolidate summaries with Claude.
10. Write final Urdu output as plain UTF-8 text.
11. Write intermediate artifacts and `manifest.json`, including `prompt_set`.

Before step 1, `application.py` runs a preflight check: every provider used by the pipeline must have its `api_key_env` configured and present in the environment, and the provider registry rejects unknown provider types at startup. Failures raise `ConfigError` before any OCR cost is incurred.

Canonical editable config:

```text
config/master_config.yaml
```

Final output is text only. Do not reintroduce DOCX output or `python-docx` unless explicitly requested.

---

## Key Files

```text
summarise_pdf.py
```

Simple runner that writes the final `.txt` beside the input PDF (or `output.final_text_directory`). Thin shim that forwards to `cli.main` with `--publish-final`.

```text
src/document_summariser/cli.py
```

CLI entry point and dependency wiring.

```text
src/document_summariser/config.py
```

YAML loading, validation, and config resolution.

```text
src/document_summariser/ocr.py
```

OCR protocol, Google Cloud Vision OCR, mock OCR, and PDF page rendering.

```text
src/document_summariser/providers/base.py
```

Provider protocol, retry behaviour, cloud adapters, and Gemini image attachments.

```text
src/document_summariser/providers/registry.py
```

Provider factory map, including Grok and DeepSeek.

```text
src/document_summariser/stages/pipeline.py
```

Pipeline orchestration and artifact writing.

```text
src/document_summariser/prompts.py
```

Prompt loading and strict placeholder rendering.

Prompt files live in:

```text
prompts/
prompts/sets/<name>/
src/document_summariser/defaults/prompts/
src/document_summariser/defaults/prompts/sets/<name>/
```

---

## Design Principles

Apply these principles without over-engineering.

### Keep responsibilities focused

Separate OCR, correction, summarisation, consolidation, config, provider wiring, prompt rendering, CLI handling, and artifact writing.

Do not mix provider API details into pipeline orchestration.

### Prefer extension points

Add new providers through adapters and the provider registry.

Avoid hard-coding provider-specific logic into unrelated modules.

### Keep contracts narrow

Provider implementations should present consistent behaviour to the pipeline even when vendor APIs differ.

Do not expose broad interfaces or surprising return shapes.

### Inject dependencies

Prefer explicit dependency wiring through the CLI or composition layer.

Avoid hidden globals, scattered environment reads, and hard-coded cloud clients inside business logic.

### Reduce coupling

Pipeline code should orchestrate stages.

Provider adapters should hide vendor-specific request and response details.

### Keep one source of truth

Use:

```text
config/master_config.yaml
```

for runtime configuration.

Use prompt files for prompt text.

Avoid duplicating provider settings, prompt text, business rules, or output behaviour across modules.

---

## Implementation Rules

Before editing:

1. Inspect the relevant files.
2. Follow existing patterns.
3. Make the smallest safe change.
4. Preserve public behaviour unless the task requires changing it.
5. Add or update tests when behaviour changes.
6. Run relevant verification.
7. Report what changed and what was verified.

Prefer simple, readable Python.

Avoid:

- broad rewrites
- unrelated cleanup
- premature abstractions
- silent exception swallowing
- mutable global state
- manually editing generated artifacts
- reformatting unrelated files

Add comments only for non-obvious design decisions, provider quirks, or operational constraints.

---

## Provider and Prompt Rules

Current default providers are configured in `config/master_config.yaml`.

Defaults:

```text
OCR:          Google Cloud Vision
Correction:   Gemini gemini-2.5-pro
Summarisers:  ChatGPT gpt-5.2
              Gemini gemini-2.5-pro
              Grok grok-4.3
              DeepSeek deepseek-v4-pro
Consolidator: Claude claude-opus-4-7
```

Important Gemini correction settings:

```yaml
max_output_tokens: 16384
thinking_config.thinking_budget: 1024
```

These settings fixed a previous Gemini issue where the model spent the output budget on thinking tokens. Do not reduce or remove them without a specific reason.

Provider retry and timeout behaviour:

- `ProviderError` carries a `retryable` flag. Auth/config/missing-SDK errors and HTTP 400/401/403/404/422 fail fast; 429/5xx/timeouts/empty responses retry with exponential backoff via the shared `BaseCloudProvider._run_with_retry`.
- `runtime.request_timeout_seconds` applies to all providers, including Gemini (passed as `http_options={"timeout": <milliseconds>}`).
- Gemini raises a non-retryable error when a response finishes with `MAX_TOKENS` (truncation guard), and the pipeline records a `correction_warning` in the manifest when corrected text is under half the OCR text length.

Prompt rules:

- Keep prompt text in prompt files.
- Preserve strict placeholder rendering.
- Keep editable prompts and packaged default prompts in sync when needed.
- Selectable prompt sets live under `prompts/sets/<name>/` and must contain `summarise.prompt.txt` and `consolidate.prompt.txt`.
- Users select a prompt set with `summarise input.pdf -p <name>` or `summarise input.pdf --prompt-set <name>`; the same set applies to the whole batch.
- Prompt sets override only summarise and consolidate prompts. Keep the OCR correction prompt fixed by config unless explicitly asked to change that contract.
- Reject prompt-set names that are paths or leave the configured `prompt_sets.directory`.
- If changing the number of summariser providers, check `consolidate.prompt.txt` and `pipeline.min_summaries`.

---

## Security Rules

Never commit or expose:

```text
.env
.env.*
API keys
service-account JSON files
Google credential files
private document contents
generated outputs containing sensitive text
```

Expected local environment variables may include:

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

Tests must not require real API keys.

Use mocks for OCR and model providers.

---

## Verification Commands

Use the active Python environment or virtual environment.

Compile check:

```bash
python -m compileall -q src tests summarise_pdf.py
```

Run tests:

```bash
python -m pytest -q --basetemp=pytest-tmp -p no:cacheprovider
```

Check CLI:

```bash
summarise --help
```

Expected current test result:

```text
40 passed
```

On Windows/sandbox setups, pytest may need to run outside the sandbox because temp directories can receive ACLs the sandbox cannot scan.

Do not claim tests passed unless they were actually run.

---

## Definition of Done

A task is complete when:

- the requested change is implemented
- behaviour is preserved unless intentionally changed
- the diff is focused
- relevant tests or checks were run where practical
- config and prompt changes are kept in sync
- no secrets or sensitive generated outputs are exposed
- the final response states what changed, what was verified, and any remaining risks
