"""Resolve the nightly workflow's board matrix from ``assets/hardware_boards.yaml``.

The ``plan`` job of ``.github/workflows/hardware-nightly.yml`` runs this and
feeds the JSON array it prints to ``strategy.matrix.board``, so that adding a
row to the board table is all it takes for the nightly to cover a new board --
the workflow names no board of its own.

It is deliberately a standalone reader rather than a caller of
``helia_core_tester.hardware.boards``: the plan job runs on ``ubuntu-latest``
with ``uv run --no-project --with pyyaml`` and must not pay for the project's
full dependency set (TensorFlow among them) to turn a comma-separated string
into a JSON list. ``test_board_matrix.py`` asserts this reader and
``load_board_table()`` agree on the ids, so the two cannot drift apart.

Usage::

    python helia_core_tester/scripts/board_matrix.py            # every board
    python helia_core_tester/scripts/board_matrix.py --boards apollo510_evb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import yaml

TABLE_RELATIVE_PATH = Path("assets/hardware_boards.yaml")
EXPECTED_SCHEMA = "hct.hardware_boards"
EXPECTED_SCHEMA_VERSION = 1


class BoardSelectionError(ValueError):
    """A board table that cannot be read, or a selection it does not contain."""


def repo_root() -> Path:
    """This file is ``<repo>/helia_core_tester/scripts/board_matrix.py``."""
    return Path(__file__).resolve().parents[2]


def board_ids(table_path: Path | None = None) -> list[str]:
    """Every board id in the table, in table order."""
    path = table_path or (repo_root() / TABLE_RELATIVE_PATH)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BoardSelectionError(f"board table not found: {path}") from exc
    if not isinstance(document, dict):
        raise BoardSelectionError(f"board table {path} is not a mapping")
    schema = document.get("schema")
    version = document.get("schema_version")
    if schema != EXPECTED_SCHEMA or version != EXPECTED_SCHEMA_VERSION:
        raise BoardSelectionError(
            f"board table {path} declares schema {schema!r} v{version!r}; "
            f"expected {EXPECTED_SCHEMA!r} v{EXPECTED_SCHEMA_VERSION}"
        )
    rows = document.get("boards") or []
    ids = [str(row["id"]) for row in rows if isinstance(row, dict) and row.get("id")]
    if not ids:
        raise BoardSelectionError(f"board table {path} lists no boards")
    return ids


def select_boards(requested: str | None, table_path: Path | None = None) -> list[str]:
    """The boards the matrix should carry.

    An empty (or whitespace-only) request means every board in the table: the
    scheduled nightly passes no inputs at all, and a workflow-level default
    list would silently keep a newly added board out of it. Duplicates in the
    request collapse -- two jobs for one board would race for its probe -- and
    an id the table does not carry is an error rather than a job that queues
    forever against a label no runner advertises.
    """
    known = board_ids(table_path)
    if requested is None or not requested.strip():
        return known
    selected: list[str] = []
    for chunk in requested.split(","):
        board = chunk.strip()
        if not board:
            continue
        if board not in known:
            raise BoardSelectionError(
                f"unknown board {board!r}; the board table carries {', '.join(known)}"
            )
        if board not in selected:
            selected.append(board)
    if not selected:
        raise BoardSelectionError(
            f"the boards input {requested!r} selected no board; "
            f"the board table carries {', '.join(known)}"
        )
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boards",
        default="",
        help="Comma-separated board ids; empty selects every board in the table.",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=None,
        help=f"Board table to read (default: <repo>/{TABLE_RELATIVE_PATH}).",
    )
    args = parser.parse_args(argv)
    try:
        selected = select_boards(args.boards, args.table)
    except BoardSelectionError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2
    print(json.dumps(selected))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
