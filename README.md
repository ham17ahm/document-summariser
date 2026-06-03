# Document Summariser

CLI pipeline for OCR, correction, multi-provider summarisation, consolidation, and plain Urdu text output.

## What The Pipeline Does

1. Validates the input PDF.
2. Renders each PDF page to a PNG artifact.
3. Sends each page image to Google Cloud Vision document OCR.
4. Corrects OCR text with the configured correction provider.
5. Sends the corrected text to ChatGPT, Gemini, Grok, and DeepSeek in parallel.
6. Consolidates successful summaries with Claude into one final Urdu summary.
7. Writes a plain UTF-8 text file plus intermediate artifacts and `manifest.json`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
cp .env.example .env
```

Fill in `.env` later with:

- `GOOGLE_APPLICATION_CREDENTIALS`
- `GOOGLE_CLOUD_PROJECT`
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `XAI_API_KEY`
- `DEEPSEEK_API_KEY`

Do not commit `.env` or credential JSON files.

## Simple Run

To create the final text file in the configured final output folder:

```bash
.venv/bin/python summarise_pdf.py /absolute/path/to/input.pdf
```

For example, this command:

```bash
.venv/bin/python summarise_pdf.py ~/Documents/report.pdf
```

creates:

```text
C:\Users\Hamad.Mobeen\Downloads\report.txt
```

The script prints the final TXT path when it finishes. The default final output folder is configured by `output.final_text_directory` in `config/master_config.yaml`.

## Advanced Run

```bash
.venv/bin/summarise path/to/input.pdf
```

Optional output directory:

```bash
.venv/bin/summarise path/to/input.pdf --out ./runs
```

Each run creates a timestamped directory containing:

- `01_ocr.json`
- `page_images/`
- `02_corrected.txt`
- `03_summaries/`
- `04_consolidated.txt`
- `05_output.txt`
- `manifest.json`

## Configuration

The canonical editable configuration lives in `config/master_config.yaml`.

- OCR provider: Google Cloud Vision
- Correction provider: Gemini
- Summarisers: ChatGPT, Gemini, Grok, DeepSeek
- Consolidator: Claude Opus 4.7 with adaptive thinking and `xhigh` effort
- Output: plain UTF-8 Urdu text

Provider credentials are read from environment variables named by each provider's `api_key_env`.
The installed CLI also ships with the same default config, prompts, and output template, so `summarise` can run from outside the repository when no `--config` value is provided. When run from this repository, the CLI prefers `config/master_config.yaml`.

Common changes should be made in:

- `pipeline.summarisers` to choose which services produce independent summaries.
- `pipeline.consolidator` to choose the final consolidation service.
- `providers.<name>.model` to change a provider model.
- `runtime.concurrency`, `runtime.retries`, and `runtime.request_timeout_seconds` for operational tuning.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Tests use mock OCR and mock model providers, so they do not require API keys.
