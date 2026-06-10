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

1. Validate input PDF.
2. Render PDF pages to PNG using PyMuPDF.
3. Save page images under `page_images/`.
4. Run Google Cloud Vision OCR.
5. Correct OCR text using Gemini with rendered page images attached.
6. Summarise corrected text in parallel with ChatGPT, Gemini, Grok, and DeepSeek.
7. Require at least `pipeline.min_summaries` successful summaries.
8. Consolidate summaries with Claude.
9. Write final Urdu output as plain UTF-8 text.
10. Write intermediate artifacts and `manifest.json`.

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

Simple runner that writes the final `.txt` beside the input PDF.

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
src/document_summariser/defaults/prompts/
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

Prompt rules:

- Keep prompt text in prompt files.
- Preserve strict placeholder rendering.
- Keep editable prompts and packaged default prompts in sync when needed.
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
18 passed
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
