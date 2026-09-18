"""Tests for the nightly workflow's board-matrix script.

The script is a standalone reader of `assets/hardware_boards.yaml` so the
workflow's hosted plan job can resolve the matrix without installing the
project's dependency set. That independence is the thing worth testing: the
reader must agree with `helia_core_tester.hardware.boards.load_board_table()`,
must not need the package to be importable, and must turn an unusable
selection into a non-zero exit rather than an empty matrix.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from helia_core_tester.hardware.boards import load_board_table
from helia_core_tester.scripts.board_matrix import (
    BoardSelectionError,
    board_ids,
    main,
    repo_root,
    select_boards,
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "board_matrix.py"


def _table(tmp_path: Path, *ids: str) -> Path:
    rows = "\n".join(f"  - id: {board}\n    cpu: cortex-m55" for board in ids)
    path = tmp_path / "hardware_boards.yaml"
    path.write_text(
        "schema: hct.hardware_boards\nschema_version: 1\nboards:\n" + rows + "\n",
        encoding="utf-8",
    )
    return path


def test_reader_agrees_with_the_board_table_loader() -> None:
    """The one duplication the standalone reader introduces, pinned."""
    assert board_ids() == [board.id for board in load_board_table()]


def test_repo_root_points_at_the_checkout() -> None:
    assert (repo_root() / "assets" / "hardware_boards.yaml").is_file()


def test_no_selection_means_every_board(tmp_path: Path) -> None:
    table = _table(tmp_path, "apollo510_evb", "apollo330mP_evb")
    for requested in (None, "", "   "):
        assert select_boards(requested, table) == ["apollo510_evb", "apollo330mP_evb"]


def test_selection_keeps_the_requested_order_and_drops_repeats(tmp_path: Path) -> None:
    table = _table(tmp_path, "apollo510_evb", "apollo330mP_evb")
    # Two jobs for one board would race for its probe.
    assert select_boards("apollo330mP_evb, apollo510_evb ,apollo330mP_evb", table) == [
        "apollo330mP_evb",
        "apollo510_evb",
    ]


def test_an_unknown_board_is_an_error_not_a_queued_job(tmp_path: Path) -> None:
    """A board id no runner advertises as a label would queue forever."""
    table = _table(tmp_path, "apollo510_evb")
    with pytest.raises(BoardSelectionError, match="unknown board 'apollo3p_evb'"):
        select_boards("apollo3p_evb", table)


def test_a_selection_of_only_separators_is_an_error(tmp_path: Path) -> None:
    table = _table(tmp_path, "apollo510_evb")
    with pytest.raises(BoardSelectionError, match="selected no board"):
        select_boards(",,", table)


@pytest.mark.parametrize(
    "document, match",
    [
        ("schema: hct.hardware_boards\nschema_version: 2\nboards: []\n", "expected"),
        ("schema: something.else\nschema_version: 1\nboards: []\n", "expected"),
        ("schema: hct.hardware_boards\nschema_version: 1\nboards: []\n", "lists no boards"),
        ("- not-a-mapping\n", "not a mapping"),
    ],
)
def test_an_unusable_table_is_refused(tmp_path: Path, document: str, match: str) -> None:
    path = tmp_path / "table.yaml"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(BoardSelectionError, match=match):
        board_ids(path)


def test_a_missing_table_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(BoardSelectionError, match="board table not found"):
        board_ids(tmp_path / "absent.yaml")


def test_main_prints_a_json_array_the_matrix_can_consume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    table = _table(tmp_path, "apollo510_evb", "apollo330mP_evb")
    assert main(["--boards", "apollo510_evb", "--table", str(table)]) == 0
    assert json.loads(capsys.readouterr().out) == ["apollo510_evb"]


def test_main_reports_a_bad_selection_as_a_workflow_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    table = _table(tmp_path, "apollo510_evb")
    assert main(["--boards", "nope", "--table", str(table)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("::error::")


def test_the_script_imports_nothing_but_the_standard_library_and_pyyaml() -> None:
    """The plan job runs it as `uv run --no-project --with pyyaml`, where the
    project and everything it depends on is absent. An import of
    `helia_core_tester...` added here would only fail on a hosted runner."""
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"yaml"}
    assert imported <= allowed, f"unexpected imports: {sorted(imported - allowed)}"


def test_the_script_runs_as_a_subprocess_from_any_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--boards", ""],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [board.id for board in load_board_table()]
