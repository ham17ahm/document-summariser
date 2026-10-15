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

The installed `summarise` CLI also copies the final TXT to the configured final
output folder by default. Use `--no-publish-final` only when you want run
artifacts without that extra final TXT copy.

Optional output directory:

```bash
.venv/bin/summarise path/to/input.pdf --out ./runs
```

Select a prompt set for one PDF or a batch:

```bash
.venv/bin/summarise path/to/input.pdf --prompt-set default
.venv/bin/summarise a.pdf b.pdf -p default
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
- Summarisers: ChatGPT, Claude Opus 4.8, Grok, DeepSeek
- Consolidator: Gemini 3.1 Pro Preview
- Output: plain UTF-8 Urdu text

Provider credentials are read from environment variables named by each provider's `api_key_env`.
The installed CLI also ships with the same default config, prompts, and output template, so `summarise` can run from outside the repository when no `--config` value is provided. When run from this repository, the CLI prefers `config/master_config.yaml`.

Common changes should be made in:

- `pipeline.summarisers` to choose which services produce independent summaries.
- `pipeline.consolidator` to choose the final consolidation service.
- `providers.<name>.model` to change a provider model.
- `runtime.concurrency`, `runtime.retries`, and `runtime.request_timeout_seconds` for operational tuning.

### Optional Gemini 4 Argon and GPT-6 providers

The repository config currently selects GPT-6 for summarisation and Gemini 4 Argon
High for consolidation. The previous providers remain available, and packaged
defaults are unchanged. Three additional provider IDs are available:

| Provider ID | Model | Thinking / reasoning |
| --- | --- | --- |
| `gemini_argon_medium` | `gemini-4-argon` | medium |
| `gemini_argon_high` | `gemini-4-argon` | high |
| `chatgpt_gpt6` | `gpt-6-astra` | medium |

To opt in, edit the existing `pipeline` block in `config/master_config.yaml`, for example:

```yaml
pipeline:
  correction_provider: gemini_argon_medium
  summarisers: [chatgpt_gpt6, claude, grok, deepseek]
  consolidator: gemini_argon_high
  min_summaries: 4
```

Then run `summarise input.pdf` or `python summarise_pdf.py input.pdf` as usual.
Selections also apply to the whole batch and work with `--prompt-set`.
Each stage can be selected independently. To use Argon for an independent
summary, replace one entry in `summarisers` with either Argon ID. Keep four
entries with the supplied consolidation prompts, which have four summary slots.
Restore `gemini`, `[chatgpt, claude, grok, deepseek]`, and `gemini_consolidate`
to return to the repository's original pipeline. Unselected entries make no API calls.

The new entries reuse `GEMINI_API_KEY` and `OPENAI_API_KEY`. GPT-6 uses the text-only
OpenAI adapter; use Gemini for correction with page images. GPT-6 reasoning can
be changed with `providers.chatgpt_gpt6.reasoning_effort` (for example, `high`);
do not add `temperature`. Argon uses `thinking_config.thinking_level`, without
`thinking_budget`. Existing Gemini budget settings are preserved.

Update dependencies in your active environment with `python -m pip install -e ".[dev]"`.
Argon's high preset needs `google-genai>=2.30.1` for automatic continuation above
the non-streaming token cap. The presets allow 32,768 tokens for medium and
65,536 for high, including thinking tokens. High thinking can take longer and
cost more; increase `runtime.request_timeout_seconds` if requests time out.
Actual model access depends on your provider account.

API references: [Gemini 4 Argon](https://ai.google.dev/gemini-api/docs/generate-content/latest-model)
and [GPT-6](https://developers.openai.com/api/docs/guides/latest-model).

Model requests separate stable instructions from variable document content so provider-side
prompt caches can reuse the common prefix. OpenAI and xAI routing hints and Claude's
five-minute cache are enabled through each provider's `prompt_cache` config. Gemini and
DeepSeek use their automatic caches. Cache reads, writes, and hit ratios are recorded under
`provider_usage` in `manifest.json` when the provider returns usage metadata.

Prompt sets live under `prompts/sets/<name>/`. Each set must contain:

- `summarise.prompt.txt`
- `consolidate.prompt.txt`

The correction prompt remains fixed by config. To add a new selectable prompt style, copy `prompts/sets/default/` to a new folder such as `prompts/sets/pr1/`, edit the two prompt files, then run with `--prompt-set pr1`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Tests use mock OCR and mock model providers, so they do not require API keys.
