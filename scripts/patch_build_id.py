#!/usr/bin/env python3
"""Stamp a linked benchmark-server firmware image with a content-derived build id.

Run by CMake as a POST_BUILD step of `hct_benchmark_server`, after the link and
after objcopy has produced the `.bin` (see CMakeLists.txt). It

1. assembles the flash image from the ELF's PT_LOAD segments (LMA order, zero
   gap fill -- the same bytes `objcopy -O binary` emits, which is checked
   against the `.bin` when one is given),
2. finds the build-id slot (cmake/hardware/hct_build_id.c: the marker
   `HCT-BUILD-ID:` followed by a zeroed id area) exactly once in that image,
3. hashes the *whole* image with the id area zeroed -> `hct-<sha256[:48]>`,
4. writes the id into the slot of the ELF and the `.bin` in place (fixed-size
   slot, so nothing in the image moves), and
5. writes `hct_build_id.txt` next to the build for the host.

Because the hash spans every loadable byte, it covers every linked input --
the server objects, cmsis-nn, the NSX board/core/perf/startup libraries and
the linker script's layout -- by construction. Identical images get identical
ids whichever build dir produced them; any differing byte gives a different
id. Re-running on an already-patched image yields the same id (the id area is
zeroed before hashing).

Usage:
    patch_build_id.py --elf <elf> [--bin <bin>] --output-txt <path>
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path
from typing import List, Optional, Tuple

BUILD_ID_PREFIX = "hct-"
BUILD_ID_HEX_CHARS = 48
# Must match cmake/hardware/hct_build_id.c.
MARKER = b"HCT-BUILD-ID:"
SLOT_BYTES = 80
ID_AREA = SLOT_BYTES - len(MARKER)

_ELF_MAGIC = b"\x7fELF"
_ELFCLASS32 = 1
_ELFDATA2LSB = 1
_PT_LOAD = 1


class PatchError(RuntimeError):
    pass


class Segment:
    """One PT_LOAD segment's file bytes: where they sit in the ELF and in the image."""

    __slots__ = ("lma", "file_offset", "size")

    def __init__(self, lma: int, file_offset: int, size: int) -> None:
        self.lma, self.file_offset, self.size = lma, file_offset, size


def load_segments(elf: bytes) -> List[Segment]:
    """PT_LOAD segments with file-backed bytes, in LMA order (ELF32 little-endian only)."""
    if elf[:4] != _ELF_MAGIC:
        raise PatchError("not an ELF file")
    if elf[4] != _ELFCLASS32 or elf[5] != _ELFDATA2LSB:
        raise PatchError("only ELF32 little-endian images are supported")
    (e_phoff, e_phentsize, e_phnum) = struct.unpack_from("<I", elf, 28) + struct.unpack_from("<HH", elf, 42)
    segments: List[Segment] = []
    for index in range(e_phnum):
        p_type, p_offset, _p_vaddr, p_paddr, p_filesz = struct.unpack_from("<IIIII", elf, e_phoff + index * e_phentsize)
        if p_type == _PT_LOAD and p_filesz > 0:
            if p_offset + p_filesz > len(elf):
                raise PatchError(f"PT_LOAD segment {index} extends past the end of the file")
            segments.append(Segment(p_paddr, p_offset, p_filesz))
    if not segments:
        raise PatchError("no PT_LOAD segments with file contents")
    segments.sort(key=lambda s: s.lma)
    return segments


def assemble_image(elf: bytes, segments: List[Segment]) -> Tuple[bytearray, int]:
    """The flat flash image (`objcopy -O binary` layout) and its base address."""
    base = segments[0].lma
    end = max(s.lma + s.size for s in segments)
    image = bytearray(end - base)
    for segment in segments:
        start = segment.lma - base
        image[start:start + segment.size] = elf[segment.file_offset:segment.file_offset + segment.size]
    return image, base


def find_slot(image: bytes, what: str) -> int:
    """Offset of the build-id slot in `image`; exactly one marker must be present."""
    first = image.find(MARKER)
    if first < 0:
        raise PatchError(f"build-id marker {MARKER!r} not found in {what} (is cmake/hardware/hct_build_id.c linked in?)")
    if image.find(MARKER, first + 1) >= 0:
        raise PatchError(f"build-id marker {MARKER!r} occurs more than once in {what}")
    if first + SLOT_BYTES > len(image):
        raise PatchError(f"build-id slot runs past the end of {what}")
    area = bytes(image[first + len(MARKER):first + SLOT_BYTES])
    if area.strip(b"\0") and not area.startswith(BUILD_ID_PREFIX.encode()):
        raise PatchError(f"build-id slot in {what} holds unexpected bytes {area[:16]!r}; slot layout mismatch")
    return first


def compute_build_id(image: bytes, slot: int) -> str:
    """`hct-<sha256 of the image with the id area zeroed>`, truncated to 52 characters."""
    hashed = bytearray(image)
    hashed[slot + len(MARKER):slot + SLOT_BYTES] = bytes(ID_AREA)
    return BUILD_ID_PREFIX + hashlib.sha256(hashed).hexdigest()[:BUILD_ID_HEX_CHARS]


def patch_slot(data: bytearray, slot: int, build_id: str) -> None:
    encoded = build_id.encode("ascii")
    if len(encoded) >= ID_AREA:
        raise PatchError(f"build id {build_id!r} does not fit the {ID_AREA}-byte id area")
    data[slot + len(MARKER):slot + SLOT_BYTES] = encoded + bytes(ID_AREA - len(encoded))


def image_file_offset(segments: List[Segment], base: int, image_offset: int) -> int:
    """Map an image offset back to the ELF file offset of the segment holding the whole slot."""
    lma = base + image_offset
    for segment in segments:
        if segment.lma <= lma and lma + SLOT_BYTES <= segment.lma + segment.size:
            return segment.file_offset + (lma - segment.lma)
    raise PatchError("build-id slot is not contained in a single PT_LOAD segment")


def stamp(elf_path: Path, bin_path: Optional[Path], output_txt: Path) -> str:
    elf = bytearray(elf_path.read_bytes())
    segments = load_segments(bytes(elf))
    image, base = assemble_image(bytes(elf), segments)
    slot = find_slot(image, str(elf_path))
    build_id = compute_build_id(image, slot)
    # Patch the ELF in memory now; both files are only written once the .bin
    # (when given) has been checked against the image assembled from the ELF.
    patch_slot(elf, image_file_offset(segments, base, slot), build_id)

    if bin_path is not None:
        binary = bytearray(bin_path.read_bytes())
        bin_slot = find_slot(binary, str(bin_path))
        # The flashed bytes must be the image we hashed: same length, same content
        # outside the id area, slot at the same place.
        patched_image = bytearray(image)
        patch_slot(patched_image, slot, build_id)
        patch_slot(binary, bin_slot, build_id)
        if bin_slot != slot or binary != patched_image:
            raise PatchError(
                f"{bin_path} does not match the image assembled from {elf_path}'s PT_LOAD segments "
                f"({len(binary)} vs {len(image)} bytes, slot at {bin_slot:#x} vs {slot:#x})"
            )
    elf_path.write_bytes(elf)
    if bin_path is not None:
        bin_path.write_bytes(binary)

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(build_id + "\n", encoding="utf-8")
    return build_id


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--elf", required=True, type=Path, help="linked ELF; patched in place")
    parser.add_argument("--bin", type=Path, help="objcopy -O binary output of --elf; patched in place when given")
    parser.add_argument("--output-txt", required=True, type=Path, help="where to write the id for the host")
    args = parser.parse_args(argv)

    for path in (args.elf, args.bin):
        if path is not None and not path.is_file():
            print(f"patch_build_id.py: missing input {path}", file=sys.stderr)
            return 1
    try:
        build_id = stamp(args.elf, args.bin, args.output_txt)
    except PatchError as exc:
        print(f"patch_build_id.py: {exc}", file=sys.stderr)
        return 1
    print(f"[hct] firmware build id {build_id} ({args.elf.stat().st_size} byte ELF, image hashed post-link)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
