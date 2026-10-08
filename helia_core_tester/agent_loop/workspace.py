"""Workspace layout and saved campaign state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Campaign, from_json

STATE_FILE = "campaign.json"
LIB_REL = Path("_nsx/nsx_cmsis_nn/nsx/libnsx_cmsis_nn.a")


@dataclass(frozen=True)
class Workspace:
    """Every path a campaign uses."""

    root: Path

    @property
    def state(self) -> Path:
        return self.root / STATE_FILE

    @property
    def tester(self) -> Path:
        return self.root / "tester"

    @property
    def base(self) -> Path:
        return self.root / "base"

    @property
    def agent(self) -> Path:
        return self.root / "agent"

    @property
    def results(self) -> Path:
        return self.root / "agent-results"

    @property
    def ledger(self) -> Path:
        return self.root / "ledger"

    @property
    def bin(self) -> Path:
        return self.root / "bin"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def check_dir(self) -> Path:
        return self.root / "check"

    @property
    def check_tree(self) -> Path:
        return self.check_dir / "tree"

    @property
    def check_build(self) -> Path:
        return self.check_dir / "build"

    @property
    def size_ref(self) -> Path:
        return self.root / "size-ref.json"

    @property
    def size_build(self) -> Path:
        return self.root / "size-ref-build"

    @property
    def prompt(self) -> Path:
        return self.root / "prompt.md"

    @property
    def settings(self) -> Path:
        return self.root / "agent-settings.json"

    @property
    def run_jsonl(self) -> Path:
        return self.logs / "agent-run.jsonl"

    @property
    def run_meta(self) -> Path:
        return self.root / "agent-run.json"

    def baseline(self, leg: str) -> Path:
        return self.root / "baselines" / leg

    def hidden_dir(self, campaign: Campaign) -> Path:
        return campaign.secrets_dir / "hidden"

    def seed_file(self, campaign: Campaign) -> Path:
        return campaign.secrets_dir / "seed"

    def tester_cmd(self) -> list[str]:
        """The pinned tester's CLI."""
        return ["uv", "--directory", str(self.tester), "run", "-q", "helia_core_tester"]

    def load(self) -> tuple[Campaign, dict[str, Any]]:
        """Saved campaign and init facts."""
        try:
            data = json.loads(self.state.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FileNotFoundError(f"{self.root}: no campaign; run agent-loop init") from exc
        return from_json(data["campaign"]), data

    def save(self, campaign: Campaign, facts: dict[str, Any]) -> None:
        data = {"schema": "hct.agent_loop", "schema_version": 1, "campaign": campaign.to_json(), **facts}
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.state)


def kernel_lib(build_dir: Path) -> Path:
    """The built kernel library."""
    path = build_dir / LIB_REL
    if path.is_file():
        return path
    found = sorted(build_dir.glob("**/libnsx_cmsis_nn.a"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no kernel library under {build_dir}")
    return found[-1]
