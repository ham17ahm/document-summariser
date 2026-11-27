from pathlib import Path
from types import SimpleNamespace

import yaml

from document_summariser.cli import _print_quality_summary, build_parser, main
from document_summariser.config import default_config_path


def _write_discovery_config(tmp_path):
    template = default_config_path()
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["prompts"] = {name: str(template.parent / path) for name, path in config["prompts"].items()}
    config["prompt_sets"] = {"directory": str(tmp_path / "sets")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, config


def test_parser_caps_multi_pdf_parallelism_by_default():
    args = build_parser().parse_args(["a.pdf", "b.pdf"])

    assert args.parallel == 2
    assert args.publish_final is True
    assert args.prompt_set is None


def test_parser_can_disable_default_publish():
    args = build_parser().parse_args(["a.pdf", "--no-publish-final"])

    assert args.publish_final is False


def test_parser_accepts_prompt_set_aliases():
    long_args = build_parser().parse_args(["a.pdf", "--prompt-set", "pr1"])
    short_args = build_parser().parse_args(["a.pdf", "-p", "pr2"])

    assert long_args.prompt_set == "pr1"
    assert short_args.prompt_set == "pr2"


def test_parser_accepts_model_aliases():
    default_args = build_parser().parse_args(["a.pdf"])
    long_args = build_parser().parse_args(["a.pdf", "--model", "argon"])
    short_args = build_parser().parse_args(["a.pdf", "-m", "carbon"])

    assert default_args.model is None
    assert long_args.model == "argon"
    assert short_args.model == "carbon"


def test_help_uses_current_prompt_folders_and_model_config(tmp_path, monkeypatch, capsys):
    config_path, config = _write_discovery_config(tmp_path)
    monkeypatch.setenv("DOCUMENT_SUMMARISER_CONFIG", str(tmp_path / "unused.yaml"))

    complete = tmp_path / "sets" / "complete-set"
    complete.mkdir(parents=True)
    (complete / "summarise.prompt.txt").touch()
    (complete / "consolidate.prompt.txt").touch()
    incomplete = tmp_path / "sets" / "incomplete-set"
    incomplete.mkdir()
    (incomplete / "summarise.prompt.txt").touch()
    (incomplete / "consolidate.prompt.txt").mkdir()

    assert main(["--help", "--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "complete-set" in out
    assert "incomplete-set" not in out
    assert "argonhigh" in out
    assert "gemini_argon_high" in out

    complete.rename(complete.with_name("new-set"))
    config["model_aliases"] = {"new-alias": "new-provider"}
    config["providers"]["new-provider"] = {"type": "mock", "model": "test-model"}
    config["pipeline"]["consolidator"] = "new-provider"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("DOCUMENT_SUMMARISER_CONFIG", str(config_path))

    assert main([]) == 0
    captured = capsys.readouterr()
    assert "Prompt sets (-p)" in captured.out and "new-set" in captured.out
    assert "Model aliases (-m)" in captured.out and "new-alias" in captured.out
    assert "Provider IDs (-m)" in captured.out and "new-provider" in captured.out
    assert "Default model" in captured.out
    assert "complete-set" not in captured.out
    assert "argonhigh" not in captured.out
    assert captured.err == ""


def test_help_still_shows_usage_with_unreadable_config(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "missing.yaml"), "-h"]) == 0
    captured = capsys.readouterr()
    assert "usage: summarise" in captured.out
    assert "Could not read config file" in captured.err


def test_final_text_rejected_for_multiple_pdfs(capsys):
    exit_code = main(["a.pdf", "b.pdf", "--final-text", "out.txt"])

    assert exit_code == 1
    assert "--final-text" in capsys.readouterr().err


def test_publish_final_supported_for_multiple_pdfs(tmp_path, monkeypatch, capsys):
    seen_prompt_sets = []
    seen_models = []
    config_path, _ = _write_discovery_config(tmp_path)

    def fake_run(pdf_path, config=None, output_dir=None, prompt_set=None, consolidator=None):
        seen_prompt_sets.append(prompt_set)
        seen_models.append(consolidator)
        pdf = tmp_path / Path(pdf_path).name
        text_path = tmp_path / "runs" / f"{pdf.stem}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(f"summary of {pdf.stem}", encoding="utf-8")
        return SimpleNamespace(
            input_pdf=pdf,
            config=SimpleNamespace(output={}),
            artifacts=SimpleNamespace(root=tmp_path / "runs" / pdf.stem),
            context=SimpleNamespace(manifest={}),
            output_text_path=text_path,
        )

    monkeypatch.setattr("document_summariser.cli.run_document_summary", fake_run)

    exit_code = main(["a.pdf", "b.pdf", "--config", str(config_path), "--prompt-set", "pr1", "-m", "carbon"])

    assert exit_code == 0
    assert seen_prompt_sets == ["pr1", "pr1"]
    assert seen_models == ["carbon", "carbon"]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "summary of a"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "summary of b"
    captured = capsys.readouterr()
    assert "[a.pdf]" in captured.out and "[b.pdf]" in captured.out
    assert captured.err.count("Available options") == 1


def test_single_pdf_forwards_prompt_set(tmp_path, monkeypatch, capsys):
    seen = {}
    config_path, _ = _write_discovery_config(tmp_path)

    def fake_run(pdf_path, config_path=None, output_dir=None, prompt_set=None, consolidator=None):
        text_path = tmp_path / "runs" / "summary.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("summary", encoding="utf-8")
        seen.update(
            {
                "pdf_path": pdf_path,
                "config_path": config_path,
                "output_dir": output_dir,
                "prompt_set": prompt_set,
                "consolidator": consolidator,
            }
        )
        return SimpleNamespace(
            input_pdf=tmp_path / Path(pdf_path).name,
            config=SimpleNamespace(output={}),
            artifacts=SimpleNamespace(root=tmp_path / "runs"),
            context=SimpleNamespace(manifest={}),
            output_text_path=text_path,
        )

    monkeypatch.setattr("document_summariser.cli.run_document_summary", fake_run)

    exit_code = main(["a.pdf", "--config", str(config_path), "--out", "runs", "-p", "pr2", "--model", "argon"])

    assert exit_code == 0
    assert seen == {
        "pdf_path": "a.pdf",
        "config_path": str(config_path),
        "output_dir": "runs",
        "prompt_set": "pr2",
        "consolidator": "argon",
    }
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "summary"
    out = capsys.readouterr().out
    assert "Wrote final TXT" in out
    assert "Wrote run artifacts" in out


def _quality_result(manifest: dict) -> SimpleNamespace:
    return SimpleNamespace(context=SimpleNamespace(manifest=manifest))


def test_print_quality_summary_reports_both_stages(capsys):
    manifest = {
        "quality": {
            "ocr": {
                "pages": 12,
                "average_confidence": 0.912,
                "low_confidence_pages": [3, 7],
                "low_confidence_threshold": 0.8,
            },
            "correction": {
                "avg_logprobs": -0.14,
                "avg_token_probability": 0.8694,
                "similarity_to_ocr": 0.959,
                "change_ratio": 0.041,
            },
        },
        "correction_warning": {"reason": "Corrected text is much shorter than the OCR text."},
    }

    _print_quality_summary(_quality_result(manifest), label="a.pdf")

    out = capsys.readouterr().out
    assert "[a.pdf] OCR confidence: 91.2% average over 12 pages; low-confidence pages: 3, 7" in out
    assert "[a.pdf] Correction: model token confidence ~87% (avg_logprobs -0.14); 4.1% of text changed vs OCR" in out
    assert "[a.pdf] Warning: Corrected text is much shorter than the OCR text." in out


def test_print_quality_summary_handles_missing_confidence(capsys):
    manifest = {
        "quality": {
            "ocr": {
                "pages": 2,
                "average_confidence": None,
                "low_confidence_pages": [],
                "low_confidence_threshold": 0.8,
            },
            "correction": {
                "avg_logprobs": None,
                "avg_token_probability": None,
                "similarity_to_ocr": 0.98,
                "change_ratio": 0.02,
            },
        }
    }

    _print_quality_summary(_quality_result(manifest))

    out = capsys.readouterr().out
    assert "OCR confidence" not in out
    assert "avg_logprobs" not in out
    assert "Correction: 2.0% of text changed vs OCR" in out


def test_print_quality_summary_is_silent_without_quality_data(capsys):
    _print_quality_summary(_quality_result({}))

    assert capsys.readouterr().out == ""
