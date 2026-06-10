from document_summariser.cli import build_parser, main


def test_parser_caps_multi_pdf_parallelism_by_default():
    args = build_parser().parse_args(["a.pdf", "b.pdf"])

    assert args.parallel == 2
    assert args.publish_final is False


def test_publish_final_rejected_for_multiple_pdfs(capsys):
    exit_code = main(["a.pdf", "b.pdf", "--publish-final"])

    assert exit_code == 1
    assert "--publish-final" in capsys.readouterr().err
