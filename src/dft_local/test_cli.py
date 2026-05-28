from __future__ import annotations

import dft_local.__main__ as cli


def test_cli_test_command_returns_success() -> None:
    assert cli.main(["test", "--timeout", "120"]) == 0


def test_cli_parser_help_is_available(capsys) -> None:
    try:
        cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    out = capsys.readouterr().out

    assert "serve" in out
    assert "test" in out
