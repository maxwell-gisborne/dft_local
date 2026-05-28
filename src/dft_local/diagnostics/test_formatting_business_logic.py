# Copied from old diagnostics panel tests during package migration.
# These tests should target dft_local only.

from dft_local.diagnostics.formatting import fmt, safe_json_for_script


def test_fmt_formats_none_and_numbers():
    assert fmt(None) == "—"
    assert fmt(0) == "0"
    assert "e" in fmt(1e-9)


def test_safe_json_for_script_escapes_html_sensitive_chars():
    out = str(safe_json_for_script({"x": "<tag>&"}))
    assert "<tag>" not in out
    assert "\\u003c" in out
    assert "\\u0026" in out
