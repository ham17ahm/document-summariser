# Document Summariser

CLI pipeline for OCR, correction, multi-provider summarisation, consolidation, and right-to-left Urdu DOCX output.

## What The Pipeline Does

1. Validates the input PDF.
2. Renders each PDF page to a PNG artifact.
3. Sends each page image to Google Cloud Vision document OCR.
4. Corrects OCR text with the configured correction provider.
5. Sends the corrected text to Claude, OpenAI, Gemini, and DeepSeek in parallel.
6. Consolidates successful summaries into one final Urdu summary.
7. Writes a right-to-left DOCX plus intermediate artifacts and `manifest.json`.

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
- `DEEPSEEK_API_KEY`

Do not commit `.env` or credential JSON files.

## Run

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
- `05_output.docx`
- `manifest.json`

## Configuration

Production defaults live in `config/config.yaml`.

- OCR provider: Google Cloud Vision
- Correction provider: Gemini
- Summarisers: Claude, OpenAI, Gemini, DeepSeek
- Consolidator: Claude
- Output: right-to-left DOCX using `Noto Nastaliq Urdu`

Provider credentials are read from environment variables named by each provider's `api_key_env`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Tests use mock OCR and mock model providers, so they do not require API keys.
