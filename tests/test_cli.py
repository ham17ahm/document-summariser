from pathlib import Path
from types import SimpleNamespace

from document_summariser.cli import build_parser, main


def test_parser_caps_multi_pdf_parallelism_by_default():
    args = build_parser().parse_args(["a.pdf", "b.pdf"])

    assert args.parallel == 2
    assert args.publish_final is False
    assert args.prompt_set is None


def test_parser_accepts_prompt_set_aliases():
    long_args = build_parser().parse_args(["a.pdf", "--prompt-set", "pr1"])
    short_args = build_parser().parse_args(["a.pdf", "-p", "pr2"])

    assert long_args.prompt_set == "pr1"
    assert short_args.prompt_set == "pr2"


def test_final_text_rejected_for_multiple_pdfs(capsys):
    exit_code = main(["a.pdf", "b.pdf", "--final-text", "out.txt"])

    assert exit_code == 1
    assert "--final-text" in capsys.readouterr().err


def test_publish_final_supported_for_multiple_pdfs(tmp_path, monkeypatch, capsys):
    seen_prompt_sets = []

    def fake_run(pdf_path, config=None, output_dir=None, prompt_set=None):
        seen_prompt_sets.append(prompt_set)
        pdf = tmp_path / Path(pdf_path).name
        text_path = tmp_path / "runs" / f"{pdf.stem}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(f"summary of {pdf.stem}", encoding="utf-8")
        return SimpleNamespace(
            input_pdf=pdf,
            config=SimpleNamespace(output={}),
            artifacts=SimpleNamespace(root=tmp_path / "runs" / pdf.stem),
            output_text_path=text_path,
        )

    monkeypatch.setattr("document_summariser.cli.run_document_summary", fake_run)

    exit_code = main(["a.pdf", "b.pdf", "--publish-final", "--prompt-set", "pr1"])

    assert exit_code == 0
    assert seen_prompt_sets == ["pr1", "pr1"]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "summary of a"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "summary of b"
    out = capsys.readouterr().out
    assert "[a.pdf]" in out and "[b.pdf]" in out


def test_single_pdf_forwards_prompt_set(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run(pdf_path, config_path=None, output_dir=None, prompt_set=None):
        text_path = tmp_path / "runs" / "summary.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("summary", encoding="utf-8")
        seen.update(
            {
                "pdf_path": pdf_path,
                "config_path": config_path,
                "output_dir": output_dir,
                "prompt_set": prompt_set,
            }
        )
        return SimpleNamespace(
            input_pdf=tmp_path / Path(pdf_path).name,
            config=SimpleNamespace(output={}),
            artifacts=SimpleNamespace(root=tmp_path / "runs"),
            output_text_path=text_path,
        )

    monkeypatch.setattr("document_summariser.cli.run_document_summary", fake_run)

    exit_code = main(["a.pdf", "--config", "config.yaml", "--out", "runs", "-p", "pr2"])

    assert exit_code == 0
    assert seen == {
        "pdf_path": "a.pdf",
        "config_path": "config.yaml",
        "output_dir": "runs",
        "prompt_set": "pr2",
    }
    assert "Wrote run artifacts" in capsys.readouterr().out
