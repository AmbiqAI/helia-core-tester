"""Flash the firmware through the J-Link recipe NSX generates for the app.

`nsx_finalize_app()` writes a ready-made commander script per target at
`<build>/jlink/<target>/flash_cmds.jlink`:

    ExitOnError 1
    Reset
    LoadFile "<build>/hct_benchmark_server.bin", 0x00410000
    Reset
    Go
    Exit

That recipe is the proven one -- NSX baked the load address out of the board's
own linker configuration -- so it is run verbatim through `JLinkExe` rather than
re-derived here. heliaPROFILER found that hand-rolling `loadfile` on the
extension-less ELF instead *silently programmed nothing* on Apollo510 (the
measured behaviour of the board never changed), which is the worst failure this
tool has: every number afterwards describes firmware that was never flashed.

Running someone else's script verbatim means vetting it first, the way NSX's own
`validate_flash_recipe` does and hpx's `target/probe/flash.py` does after it:

* `ExitOnError 1` must be present *and armed before the first* `LoadFile`.
  Without it JLinkExe can fail a command and still exit zero, so a failed flash
  would look like a success; arming it afterwards protects nothing, because the
  commander runs a script top to bottom.
* Some `LoadFile` must name *this build's* `.bin` and carry an explicit address.
  Recipes bake absolute paths, so a recipe left behind by another build can
  flash a stale image while the tester reports the new build's id. An
  addressless `LoadFile` programs flash at a destination taken from the image
  format, which is a destination this module cannot check -- refused, because
  checking it is the entire point.

Unlike hpx there is no hand-rolled fallback for a missing recipe: the recipe is
emitted by the same `nsx build` that produces the `.bin`, so a build dir without
one is an incomplete build, and `hardware build` is the fix.

Afterwards the flash is verified twice: JLinkExe's exit status (trustworthy
because of `ExitOnError 1`), and the flash *bank* J-Link names in its own
output, which must be the bank the recipe's address lives in.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from . import jlink_cli
from .boards import BoardSpec

#: Every refusal below has to say the board was left alone: "refused" and
#: "failed halfway through programming" call for opposite next steps.
NOTHING_PROGRAMMED = "Nothing was programmed -- the recipe was refused before JLinkExe ran."

# Ported from hpx's `_ADDRESSED_LOAD_FILE_RE` (itself ported from NSX's
# `validate_flash_recipe`) rather than imported: `neuralspotx.operations._hardware`
# is private, and the recipe this module runs is emitted by NSX's own
# `flash_cmds.jlink.in`, so the two grammars must agree by inspection. Accepts the
# quoted form NSX generates and the unquoted form a hand-edited recipe may use,
# plus the optional `, reset|noreset` tail and `//` comment the commander accepts.
_ADDRESSED_LOAD_FILE_RE = re.compile(
    r'^\s*LoadFile\s+(?:"(?P<quoted>[^"]+)"|(?P<plain>.+?))\s*,\s*'
    r"(?P<address>0x[0-9a-fA-F]+|[0-9]+)"
    r"(?:\s*,\s*(?:no)?reset)?\s*(?://.*)?$",
    re.IGNORECASE | re.MULTILINE,
)
_FAIL_FAST_RE = re.compile(r"^\s*ExitOnError\s+1\s*(?://.*)?$", re.IGNORECASE | re.MULTILINE)
# Deliberately looser: this answers "has anything been programmed yet?" for the
# ordering check, so it must also see a `LoadFile` the addressed regex rejects --
# such a line still programs flash, and fail-fast must be armed before it runs.
_ANY_LOAD_FILE_RE = re.compile(r"^\s*LoadFile\b", re.IGNORECASE | re.MULTILINE)
# A flash-BANK identity check, not a destination check: J-Link's format string is
# `Bank %d @ 0x%.8X: %d range%s affected` -- one address for N ranges -- so it
# names the base of the bank it programmed, never where the image landed inside
# it. What it does catch is an address in a bank J-Link never touched, i.e. a
# build dir configured for another part. Anchored on `Flash download:` so only a
# programming confirmation counts: three other J-Link strings carry the same
# `Bank %d @` shape and are not confirmations.
_BANK_ADDR_RE = re.compile(r"flash\s+download:\s*bank\s+\d+\s*@\s*(0x[0-9a-fA-F]+)", re.IGNORECASE)
#: Exactly two markers confirm a flash. A bare connection "O.K." is printed
#: before any programming and must not count.
_CONFIRMATIONS = ("flash download: total", "skipped. contents already match")


class FlashRecipeError(RuntimeError):
    """The NSX flash recipe is missing, unreadable, or not one we will run."""


def recipe_path(build_dir: Path, board: BoardSpec) -> Path:
    """`<build>/jlink/<target>/flash_cmds.jlink` for this board's firmware."""
    from .firmware_build import output_dir
    from .nsx_app import SERVER_TARGET

    return output_dir(build_dir, board) / "jlink" / SERVER_TARGET / "flash_cmds.jlink"


def _parse_addr(text: str) -> int:
    return int(text, 16) if text.lower().startswith("0x") else int(text, 10)


def _same_file(candidate: Path, expected: Path) -> bool:
    """Do the two paths name the same file on disk?

    Path equality is the fast path and settles every recipe NSX generates;
    `samefile` settles the rest on stat identity, which is case-correct on a
    case-insensitive volume and equally correct on a case-sensitive one. It
    stats both paths and raises if either is unreadable -- which here means the
    recipe names something that is not there, i.e. a stale recipe.
    """
    if candidate == expected:
        return True
    try:
        return candidate.samefile(expected)
    except (OSError, ValueError):
        return False


def recipe_load_address(script: str, *, script_path: Path, bin_path: Path) -> int:
    """Validate a recipe and return the address its `LoadFile` programs at.

    On this path the *recipe* is the authority on where the image lands: it is
    run verbatim and NSX baked the address from the board's linker
    configuration, so the expected address is read back out of it.
    """
    fail_fast = _FAIL_FAST_RE.search(script)
    if fail_fast is None:
        raise FlashRecipeError(
            f"The NSX flash recipe ({script_path}) is missing `ExitOnError 1`; without it "
            "JLinkExe can fail a command and still exit successfully, so a failed flash "
            f"would look like a success. {NOTHING_PROGRAMMED} Re-run `hardware build` to "
            "regenerate the recipe from NSX."
        )
    first_load = _ANY_LOAD_FILE_RE.search(script)
    if first_load is not None and fail_fast.start() > first_load.start():
        raise FlashRecipeError(
            f"The NSX flash recipe ({script_path}) puts `ExitOnError 1` after its first "
            "`LoadFile`, so the flash itself runs with fail-fast off -- JLinkExe executes "
            "the script in order, so enabling it afterwards protects nothing. "
            f"{NOTHING_PROGRAMMED} Move it above the first `LoadFile` (NSX's own recipes "
            "open with it) or re-run `hardware build`."
        )

    expected_bin = bin_path.resolve()
    loaded: list[Path] = []
    for match in _ADDRESSED_LOAD_FILE_RE.finditer(script):
        quoted = match.group("quoted")
        candidate = Path(quoted if quoted is not None else match.group("plain").strip())
        if not candidate.is_absolute():
            candidate = script_path.parent / candidate
        try:
            resolved = candidate.resolve()
        except (OSError, ValueError) as exc:
            raise FlashRecipeError(
                f"The NSX flash recipe ({script_path}) names a `LoadFile` path this host "
                f"cannot resolve ({candidate!r}): {exc}. {NOTHING_PROGRAMMED}"
            ) from exc
        if _same_file(resolved, expected_bin):
            return _parse_addr(match.group("address"))
        loaded.append(candidate)

    if not loaded:
        if first_load is None:
            raise FlashRecipeError(
                f"The NSX flash recipe ({script_path}) has no `LoadFile` command at all, so "
                f"there is no address to verify a flash against. {NOTHING_PROGRAMMED} Re-run "
                "`hardware build` so NSX regenerates it."
            )
        raise FlashRecipeError(
            f"The NSX flash recipe ({script_path}) has a `LoadFile` command, but none in the "
            "`LoadFile <image>, <address>` form this tool can read a destination out of. "
            "JLinkExe accepts an addressless `LoadFile` -- it takes the destination from the "
            "image format -- so this recipe would program flash somewhere the flash cannot be "
            f"checked, which is the one thing this gate exists to refuse. {NOTHING_PROGRAMMED}"
        )
    listed = ", ".join(str(path) for path in loaded)
    raise FlashRecipeError(
        f"The NSX flash recipe ({script_path}) loads {listed}, not this build's image "
        f"({bin_path}). Recipes bake absolute paths, so a stale recipe flashes an older image "
        f"while the run is attributed to the current build id. {NOTHING_PROGRAMMED} Re-run "
        "`hardware build` (or delete the build dir) so NSX regenerates the recipe."
    )


def read_recipe(script_path: Path, bin_path: Path) -> str:
    """The recipe's text, with every pre-flight refusal that does not need parsing."""
    if not script_path.is_file():
        raise FlashRecipeError(
            f"No NSX flash recipe at {script_path}. The recipe is generated by the same "
            f"`nsx build` that links the firmware, so this build dir is incomplete. "
            f"{NOTHING_PROGRAMMED} Run `hardware build` first."
        )
    if not bin_path.is_file():
        raise FlashRecipeError(
            f"The NSX flash recipe ({script_path}) exists but this build's image ({bin_path}) "
            f"does not, so the recipe could only flash something other than what was built. "
            f"{NOTHING_PROGRAMMED} Run `hardware build` first."
        )
    try:
        # Explicit utf-8, never the locale codec: this decides whether the flash
        # runs at all, so a mis-decode would refuse a correct flash.
        return script_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        fault = "is not valid UTF-8" if isinstance(exc, UnicodeDecodeError) else "cannot be read"
        raise FlashRecipeError(
            f"The NSX flash recipe ({script_path}) {fault} ({exc}), so the recipe that would "
            f"have run cannot be read. {NOTHING_PROGRAMMED} Re-run `hardware build`."
        ) from exc


def verify_flash_bank(output: str, *, expected_addr: int, echo: Callable[[str], None]) -> list[int]:
    """Require J-Link to name the flash bank the recipe's address lives in.

    A flash to a wrong-but-writable address prints the same `Total:` summary as a
    correct one, so the summary alone proves only that *something* was
    programmed, never where. This narrows that to the bank, which is all J-Link
    reports. When J-Link names no bank at all this warns instead of raising: the
    bank line corroborates the exit-status gate, and turning a J-Link rewording
    into a hard stop would block correct flashes with no evidence of a wrong one.
    """
    observed = [_parse_addr(addr) for addr in _BANK_ADDR_RE.findall(output)]
    if not observed:
        echo(
            "[hardware] WARNING: UNVERIFIED FLASH DESTINATION: JLinkExe confirmed a flash but "
            f"named no bank address, so where the image landed could not be checked against the "
            f"recipe's 0x{expected_addr:08X}. A wrong-address flash would look exactly like this."
        )
        return observed
    if expected_addr in observed:
        return observed
    seen = ", ".join(f"0x{addr:08X}" for addr in observed)
    raise FlashRecipeError(
        f"JLinkExe programmed the flash bank(s) based at {seen}, but the recipe requested "
        f"0x{expected_addr:08X}, which is in none of them. J-Link names the bank it programmed, "
        "so the image landed somewhere other than the requested address: the board boots stale "
        "firmware from its real entry point while this run would be attributed to the new build. "
        "A build dir carried over from another board is the usual cause."
    )


def flash_image(
    *,
    script_path: Path,
    bin_path: Path,
    device: str,
    serial_no: int,
    speed_khz: int = 4000,
    timeout_s: float = jlink_cli.FLASH_TIMEOUT_S,
    echo: Callable[[str], None] = print,
    runner=None,
) -> int:
    """Run the validated recipe through JLinkExe and return the address it loaded at."""
    script = read_recipe(script_path, bin_path)
    expected_addr = recipe_load_address(script, script_path=script_path, bin_path=bin_path)
    echo(
        f"[hardware] Flashing via the NSX recipe {script_path} at 0x{expected_addr:08X} "
        f"(J-Link serial {serial_no}, device {device})."
    )
    kwargs = {"runner": runner} if runner is not None else {}
    proc = jlink_cli.run_script(
        script,
        device=device,
        serial_no=serial_no,
        speed_khz=speed_khz,
        timeout_s=timeout_s,
        op_label="JLinkExe flash",
        **kwargs,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if not any(marker in combined.lower() for marker in _CONFIRMATIONS):
        raise FlashRecipeError(
            "JLinkExe printed no recognized flash confirmation -- either J-Link reworded its "
            "summary, or nothing was programmed and the session that follows would measure "
            f"stale firmware. JLinkExe output tail: {combined.strip()[-800:]}"
        )
    banks = verify_flash_bank(combined, expected_addr=expected_addr, echo=echo)
    where = ", ".join(f"0x{addr:08X}" for addr in banks) if banks else "unverified"
    echo(f"[hardware] Flash complete at 0x{expected_addr:08X} (J-Link programmed bank(s) {where}).")
    return expected_addr


def describe_recipe(build_dir: Path, board: BoardSpec) -> str:
    """One line for `doctor`: whether this board's build dir carries a flash recipe."""
    path = recipe_path(build_dir, board)
    if not path.is_file():
        return f"{path}: missing (run `hardware build --board {board.id}`)"
    try:
        script = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        return f"{path}: unreadable ({exc})"
    match = _ADDRESSED_LOAD_FILE_RE.search(script)
    if match is None:
        return f"{path}: present, but no addressed `LoadFile` line"
    return f"{path}: loads at {match.group('address')}"
