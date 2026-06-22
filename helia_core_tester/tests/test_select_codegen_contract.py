from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_select_v2_template_uses_bool_condition() -> None:
    template = (_repo_root() / "assets" / "templates" / "SelectFunctions" / "select_v2" / "select_v2.h.j2").read_text()
    source = (_repo_root() / "helia_core_tester" / "generation" / "ops" / "SelectFunctions" / "select_v2.py").read_text()

    assert "static const {{ condition_c_type }} {{ name }}_condition[]" in template
    assert '"condition_c_type": "bool"' in source
    assert "condition_int8" not in source


def test_where_templates_use_int64_coordinates() -> None:
    header = (_repo_root() / "assets" / "templates" / "SelectFunctions" / "where" / "where.h.j2").read_text()
    source = (_repo_root() / "assets" / "templates" / "SelectFunctions" / "where" / "where.c.j2").read_text()
    op_source = (_repo_root() / "helia_core_tester" / "generation" / "ops" / "SelectFunctions" / "where.py").read_text()

    assert "static const int64_t {{ name }}_expected_output[]" in header
    assert "static {{ output_c_type }} {{ name }}_output" in source
    assert '"output_c_type": "int64_t"' in op_source
    assert "dtype=np.int64" in op_source
