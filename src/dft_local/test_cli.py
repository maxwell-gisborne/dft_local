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



def test_cli_help_lists_export_typst(capsys) -> None:
    try:
        cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    out = capsys.readouterr().out

    assert "export-typst" in out
    assert "bundle" in out


def test_cli_export_typst_command_writes_bundle(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    code = cli.main([
        "export-typst",
        "transport.boltzmann.calculation.overview",
        "--no-context",
    ])

    assert code == 0
    out = capsys.readouterr().out
    assert "diagnostic_bundles/transport_boltzmann_calculation_overview" in out
    assert (tmp_path / "diagnostic_bundles" / "transport_boltzmann_calculation_overview" / "diagnostics.typ").exists()
    assert (tmp_path / "diagnostic_bundles" / "transport_boltzmann_calculation_overview" / "generated" / "components.typ").exists()


def test_cli_export_typst_rejects_malformed_input(capsys) -> None:
    code = cli.main([
        "export-typst",
        "transport.boltzmann.calculation.overview",
        "--input",
        "not-an-assignment",
        "--no-context",
    ])

    assert code == 2
    err = capsys.readouterr().err
    assert "expected name=value" in err



def test_cli_export_typst_command_writes_custom_bundle(tmp_path, capsys) -> None:
    out_dir = tmp_path / "custom_bundle"

    code = cli.main([
        "export-typst",
        "transport.boltzmann.calculation.overview",
        "--no-context",
        "--out",
        str(out_dir),
    ])

    assert code == 0
    out = capsys.readouterr().out
    assert str(out_dir) in out
    assert (out_dir / "diagnostics.typ").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "generated" / "components.typ").exists()
    assert not (out_dir / "data").exists()


def test_cli_bundle_new_and_refresh_preserves_root_document(tmp_path, capsys) -> None:
    out_dir = tmp_path / "editable_bundle"

    code = cli.main([
        "bundle",
        "new",
        "transport.boltzmann.calculation.overview",
        str(out_dir),
        "--no-context",
        "--lib-mode",
        "none",
    ])

    assert code == 0
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "diagnostics.typ").exists()
    assert (out_dir / "generated" / "components.typ").exists()
    assert (out_dir / "generated" / "diagnostics.json").exists()

    original = (out_dir / "diagnostics.typ").read_text()
    (out_dir / "diagnostics.typ").write_text(original + "\n// manual edit\n")

    code = cli.main(["bundle", str(out_dir)])

    assert code == 0
    assert "// manual edit" in (out_dir / "diagnostics.typ").read_text()
    assert "Refreshed Typst diagnostic bundle" in capsys.readouterr().out
