"""Render the hardware firmware as an NSX app.

Renderer behind `hardware build`: writes ``nsx.yml``, a placeholder
``cmake/nsx/modules.cmake`` and ``CMakeLists.txt`` into an app directory,
plus ``modules/nsx-cmsis-nn`` for a local kernel checkout, vendored the way
helia-profiler does it. ``nsx lock``/``nsx sync`` own ``cmake/nsx/`` and the
other ``modules/`` from there; the firmware sources stay in this checkout.
NSX copies the rest of ``cmake/nsx/`` (bootstrap, helpers, toolchain flags)
out of its own wheel on every lock and sync, so the app never ships them.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import jinja2
import yaml

from ..core.cpu_targets import get_cpu_profile
from ..core.discovery import find_tester_templates_dir
from . import nsx_cli
from .boards import BoardSpec
from .boards import repo_root as tester_repo_root
from .firmware_build import BUILD_ID_TXT, IMAGE_SUBDIR, SERVER_TARGET
from .pathutil import is_relative_to
from .toolchain import DEFAULT_TOOLCHAIN, TOOLCHAINS, toolchain_spec

APP_NAME = "hct_benchmark_server"
SIZE_PROBE_TARGET = "hct_universal_size_probe"

# Kernel identity per neuralspotx registry.lock.yaml.
CMSIS_NN_MODULE = "nsx-cmsis-nn"
CMSIS_NN_PROJECT = "ns-cmsis-nn"
CMSIS_NN_METADATA = "modules/ns-cmsis-nn/nsx/nsx-module.yaml"
CMSIS_NN_REF = "v7.43.0"
# The pin while records lacked the flag.
_PRE_FLAG_PIN = "v7.35.1"

# Not in the registry yet; declared inline.
SEGGER_RTT_MODULE = "nsx-segger-rtt"
SEGGER_RTT_URL = "https://github.com/AmbiqAI/nsx-segger-rtt.git"
SEGGER_RTT_METADATA = "nsx-module.yaml"
SEGGER_RTT_REF = "v0.1.2"

PMU_MODULE = "nsx-pmu-armv8m"

# NSX linker scripts without heap bounds.
HEAPLESS_SOCS = ("apollo2", "apollo3", "apollo3p")

# Copied from a local checkout, like hpx.
KERNEL_TREES = ("Include", "Source", "cmake")
# What makes a dir a checkout.
CHECKOUT_DIRS = ("Include", "Source")
CHECKOUT_FILES = ("nsx/CMakeLists.txt", "nsx/nsx-module.yaml")
KERNEL_SHIM = "# Shim: delegates to the native ns-cmsis-nn NSX build.\nadd_subdirectory(nsx)\n"

# Kernel entry alignment; 64 is the max.
KERNEL_ALIGN_BYTES = 64

# tcm: all operands in DTCM. mram: weights, bias in MRAM.
PLACEMENTS = ("tcm", "mram")

# Options the last successful build used.
OPTIONS_FILE = ".hct-options.json"

# Holds one flush burst; rarely blocks.
RTT_BUFFER_SIZE_UP = 8192
# Holds one BLOB_CHUNK frame.
RTT_BUFFER_SIZE_DOWN = 512

_SERVER_SOURCES = (
    "benchmark_server_main.c",
    "hctp_protocol.c",
    "benchmark_server_catalog.c",
    "benchmark_server_messages.c",
    "benchmark_server_adapter.c",
    "benchmark_server_session.c",
    "benchmark_server_adapters.gen.c",
    "benchmark_server_transport_rtt.c",
    "hct_build_id.c",
)


class AppRenderError(RuntimeError):
    """The app could not be rendered."""


@dataclass(frozen=True)
class AppOptions:
    """Kernel source, kernel switches, target choice."""

    cmsis_nn_ref: str = CMSIS_NN_REF
    # False: the ref follows a pin bump.
    cmsis_nn_ref_explicit: bool = False
    cmsis_nn_root: Optional[Path] = None
    # ON matches hpx and shipping builds.
    requantize_inline_asm: bool = True
    enable_f32: bool = True
    enable_f16: bool = True
    build_size_probe: bool = False
    placement: str = "tcm"
    toolchain: str = DEFAULT_TOOLCHAIN

    def __post_init__(self) -> None:
        if self.placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of: {', '.join(PLACEMENTS)}")
        toolchain_spec(self.toolchain)
        # One spelling per checkout.
        if self.cmsis_nn_root is not None:
            object.__setattr__(self, "cmsis_nn_root", Path(self.cmsis_nn_root).expanduser().resolve())

    def kernel_source(self) -> str:
        """Kernel source, as printed."""
        return str(self.cmsis_nn_root or f"ns-cmsis-nn {self.cmsis_nn_ref}")

    def kernel_id(self) -> str:
        """Short hash of the kernel source."""
        return hashlib.sha256(self.kernel_source().encode("utf-8")).hexdigest()[:12]

    def cache_vars(self) -> dict[str, str]:
        """Switches forced before the NSX bootstrap."""
        switches = {
            "NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM": self.requantize_inline_asm,
            "ARM_NN_ENABLE_F32": self.enable_f32,
            "ARM_NN_ENABLE_F16": self.enable_f16,
        }
        return {name: "ON" if on else "OFF" for name, on in switches.items()}

    def summary(self) -> str:
        """Kernel source and inline asm, as printed."""
        return (
            f"{self.kernel_source()}, inline asm {_on_off(self.requantize_inline_asm)}, "
            f"placement {self.placement}, toolchain {self.toolchain}"
        )

    def changes_from(self, old: "AppOptions") -> list[str]:
        """What differs from old, as printed."""
        changes = []
        if old.kernel_source() != self.kernel_source():
            changes.append(f"kernels {old.kernel_source()} -> {self.kernel_source()}")
        for field in dataclasses.fields(self):
            before, after = getattr(old, field.name), getattr(self, field.name)
            if field.name.startswith("cmsis_nn_") or before == after:
                continue
            label = field.name.replace("_", " ")
            if isinstance(after, bool):
                before, after = _on_off(before), _on_off(after)
            changes.append(f"{label} {before} -> {after}")
        return changes

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, default=str) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "AppOptions":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise TypeError("options record is not an object")
        kept = {}
        for field in dataclasses.fields(cls):
            if field.name not in data:
                continue
            value = data[field.name]
            # Reject wrong types as corrupt.
            if not _field_type_ok(field.name, value):
                raise TypeError(f"bad type for {field.name}")
            kept[field.name] = value
        # Older records: only the old pin defaulted.
        ref = kept.get("cmsis_nn_ref")
        kept.setdefault("cmsis_nn_ref_explicit", ref is not None and ref != _PRE_FLAG_PIN)
        return cls(**kept)


def _field_type_ok(name: str, value: Any) -> bool:
    """Match the JSON type to the field."""
    if name == "cmsis_nn_ref":
        return isinstance(value, str) and bool(value)
    if name == "cmsis_nn_root":
        return value is None or (isinstance(value, str) and bool(value))
    if name == "placement":
        return value in PLACEMENTS
    if name == "toolchain":
        return value in TOOLCHAINS
    return isinstance(value, bool)


def _on_off(value: bool) -> str:
    return "on" if value else "off"


def saved_options(app_dir: Path) -> Optional[AppOptions]:
    """Options the last build used, if readable."""
    try:
        return AppOptions.from_json((app_dir / OPTIONS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def save_options(app_dir: Path, options: AppOptions) -> None:
    """Replace the record atomically."""
    tmp = app_dir / f"{OPTIONS_FILE}.tmp"
    tmp.write_text(options.to_json(), encoding="utf-8")
    os.replace(tmp, app_dir / OPTIONS_FILE)


def resolve_options(
    app_dir: Path,
    repo_root: Path,
    *,
    cmsis_nn_ref: Optional[str] = None,
    cmsis_nn_root: Optional[Path] = None,
    inline_asm: Optional[bool] = None,
    placement: Optional[str] = None,
    toolchain: Optional[str] = None,
    follow_pin: bool = True,
) -> AppOptions:
    """Flags win; kernel source then saved, switches then defaults.

    follow_pin=False resolves the flashed build: saved ref and switches.
    """
    saved = saved_options(app_dir)
    if saved and follow_pin:
        # Unpassed switches reset each build.
        saved = AppOptions(**{
            f.name: getattr(saved, f.name) for f in dataclasses.fields(saved) if f.name.startswith("cmsis_nn_")
        })
    base = saved or AppOptions(cmsis_nn_root=nested_kernel_root(repo_root))
    if cmsis_nn_ref or cmsis_nn_root:
        base = dataclasses.replace(
            base, cmsis_nn_ref=cmsis_nn_ref or CMSIS_NN_REF, cmsis_nn_ref_explicit=bool(cmsis_nn_ref),
            cmsis_nn_root=cmsis_nn_root,
        )
    elif saved and saved.cmsis_nn_root and not saved.cmsis_nn_root.is_dir():
        raise AppRenderError(f"Last build's kernel root is gone: {saved.cmsis_nn_root}")
    # Unpassed refs follow a pin bump.
    elif follow_pin and not base.cmsis_nn_ref_explicit:
        base = dataclasses.replace(base, cmsis_nn_ref=CMSIS_NN_REF)
    if inline_asm is not None:
        base = dataclasses.replace(base, requantize_inline_asm=inline_asm)
    if placement is not None:
        base = dataclasses.replace(base, placement=placement)
    if toolchain is not None:
        base = dataclasses.replace(base, toolchain=toolchain)
    return base


@dataclass(frozen=True)
class AppRender:
    """The rendered texts and where they went."""

    app_dir: Path
    modules: tuple[str, ...]
    nsx_yml: str
    modules_cmake: str
    cmakelists: str


def module_names(board: BoardSpec, profile: dict[str, Any]) -> list[str]:
    """Profile modules, then kernels, RTT, PMU."""
    names = [str(name) for name in profile.get("modules") or []]
    extras = [CMSIS_NN_MODULE, SEGGER_RTT_MODULE]
    if board.pmu_tier == "armv8m":
        extras.append(PMU_MODULE)
    names.extend(name for name in extras if name not in names)
    return names


def module_registry(options: AppOptions) -> dict[str, Any]:
    """Overrides for the modules the profile lacks."""
    projects: dict[str, Any] = {SEGGER_RTT_MODULE: {"url": SEGGER_RTT_URL, "revision": SEGGER_RTT_REF}}
    modules: dict[str, Any] = {
        SEGGER_RTT_MODULE: {"project": SEGGER_RTT_MODULE, "revision": SEGGER_RTT_REF, "metadata": SEGGER_RTT_METADATA},
    }
    # A local checkout is vendored instead.
    if options.cmsis_nn_root is None:
        ref = options.cmsis_nn_ref
        projects[CMSIS_NN_PROJECT] = {"revision": ref}
        modules[CMSIS_NN_MODULE] = {"project": CMSIS_NN_PROJECT, "revision": ref, "metadata": CMSIS_NN_METADATA}
    return {"projects": projects, "modules": modules}


def _checkout_missing(root: Path) -> list[str]:
    """Checkout files absent under root."""
    dirs = [name for name in CHECKOUT_DIRS if not (root / name).is_dir()]
    return dirs + [name for name in CHECKOUT_FILES if not (root / name).is_file()]


def nested_kernel_root(repo_root: Path) -> Optional[Path]:
    """The enclosing ns-cmsis-nn checkout, if any."""
    # Layout: ns-cmsis-nn/Tests/helia-core-tester.
    root = repo_root.resolve().parent.parent
    return None if _checkout_missing(root) else root


def kernel_dir(app_dir: Path, options: AppOptions) -> Path:
    """Where the build reads kernels."""
    # NSX vendors by module, clones by project.
    name = CMSIS_NN_MODULE if options.cmsis_nn_root else CMSIS_NN_PROJECT
    return app_dir / "modules" / name


def _write_if_absent(path: Path, text: str) -> None:
    """NSX rewrites this file on sync; seed it once."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_if_changed(path: Path, text: str) -> None:
    """Skip unchanged files: keeps mtimes."""
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except (OSError, UnicodeDecodeError):
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _check_no_overlap(root: Path, module_dir: Path) -> None:
    """Refuse copies that would delete sources."""
    # Lexical and resolved, both directions.
    sources = {Path(os.path.abspath(root)), root.resolve()}
    targets = {Path(os.path.abspath(module_dir)), module_dir.resolve(), module_dir.parent.resolve() / module_dir.name}
    copied = (*KERNEL_TREES, "nsx")
    for src in sources:
        for dst in targets:
            inside_tree = any(is_relative_to(dst, src / name) for name in copied)
            if dst == src or is_relative_to(src, dst) or inside_tree:
                raise AppRenderError(f"Kernel root overlaps the app: {root}")


def _remove(path: Path) -> None:
    """Delete a path; never follow links."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif os.path.lexists(path):
        path.unlink()


class SwapError(AppRenderError):
    """A swap failed and left the old module aside."""


def _swap_in(fresh: Path, module_dir: Path) -> None:
    """Replace module_dir with fresh in one step."""
    old = None
    # Move links and dirs aside; never follow.
    if os.path.lexists(module_dir):
        old = fresh.with_name(fresh.name + ".old")
        os.rename(module_dir, old)
    try:
        os.replace(fresh, module_dir)
    except BaseException as exc:
        if old is not None:
            try:
                os.rename(old, module_dir)
            except OSError:
                error = SwapError(f"Old module left at {old}; {module_dir} taken by another writer")
                error.stranded = old
                raise error from exc
        raise
    if old is not None:
        _remove(old)


@contextlib.contextmanager
def _module_lock(module_dir: Path) -> Iterator[None]:
    """Serialize writers of one module dir."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - no flock on Windows
        yield
        return
    lock = module_dir.with_name(module_dir.name + ".lock")
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def write_kernels(root: Path, module_dir: Path) -> None:
    """Vendor a local checkout, as hpx does.

    Under a per-module lock, builds a fresh sibling dir, then swaps it in.
    """
    missing = _checkout_missing(root)
    if missing:
        raise AppRenderError(f"Not an ns-cmsis-nn checkout: {root} lacks {missing[0]}")
    _check_no_overlap(root, module_dir)
    module_dir.parent.mkdir(parents=True, exist_ok=True)
    with _module_lock(module_dir):
        _check_no_overlap(root, module_dir)
        _vendor(root, module_dir)


def _stamp_path(module_dir: Path) -> Path:
    """Marks a module vendored with fresh mtimes."""
    return module_dir.with_name(module_dir.name + ".fresh")


def _file_hashes(module_dir: Path) -> dict[str, str]:
    """sha256 per regular file."""
    return {
        str(path.relative_to(module_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in module_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def _vendored_hashes(module_dir: Path) -> Optional[dict[str, str]]:
    """Hashes this code vendored, if current."""
    try:
        stamp = json.loads(_stamp_path(module_dir).read_text(encoding="utf-8"))
        if stamp.get("ino") != module_dir.stat().st_ino or not isinstance(stamp.get("files"), dict):
            return None
        return stamp["files"]
    except (OSError, ValueError, AttributeError):
        return None


def _keep_mtimes(fresh: Path, old: Path) -> None:
    """Same bytes keep the old mtime.

    Old mtimes are trusted only from a fresh vendor,
    and only for files still holding vendored bytes.
    Added paths keep none: headers may shadow.
    """
    if old.is_symlink() or not old.is_dir():
        return
    vendored = _vendored_hashes(old)
    if vendored is None:
        return
    paths = sorted(path.relative_to(fresh) for path in fresh.rglob("*"))
    if not set(paths) <= {path.relative_to(old) for path in old.rglob("*")}:
        return
    for rel in paths:
        path, prior = fresh / rel, old / rel
        if path.is_symlink() or prior.is_symlink() or not (path.is_file() and prior.is_file()):
            continue
        # In-place edits break the match.
        digest = hashlib.sha256(prior.read_bytes()).hexdigest()
        if digest == vendored.get(str(rel)) == hashlib.sha256(path.read_bytes()).hexdigest():
            info = prior.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))


def _vendor(root: Path, module_dir: Path) -> None:
    """Build fresh, then swap in."""
    import tempfile

    fresh = Path(tempfile.mkdtemp(prefix=f".{module_dir.name}.", dir=module_dir.parent))
    try:
        (fresh / "nsx").mkdir()
        # Native manifest at the module root.
        shutil.copy(root / "nsx" / "nsx-module.yaml", fresh / "nsx-module.yaml")
        shutil.copy(root / "nsx" / "CMakeLists.txt", fresh / "nsx" / "CMakeLists.txt")
        # Keep the shim's mtime: CMake reruns otherwise.
        shim = module_dir / "CMakeLists.txt"
        if not module_dir.is_symlink() and shim.is_file() and not shim.is_symlink() and shim.read_text(
            encoding="utf-8", errors="replace",
        ) == KERNEL_SHIM:
            shutil.copy2(shim, fresh / "CMakeLists.txt")
        else:
            (fresh / "CMakeLists.txt").write_text(KERNEL_SHIM, encoding="utf-8")
        # Fresh mtimes; same bytes get old ones.
        for name in KERNEL_TREES:
            if (root / name).is_dir():
                shutil.copytree(root / name, fresh / name, copy_function=shutil.copy)
        _keep_mtimes(fresh, module_dir)
        _stamp_path(module_dir).unlink(missing_ok=True)
        _swap_in(fresh, module_dir)
        stamp = {"ino": module_dir.stat().st_ino, "files": _file_hashes(module_dir)}
        _stamp_path(module_dir).write_text(json.dumps(stamp) + "\n", encoding="utf-8")
    except BaseException:
        _remove(fresh)
        raise


def checkout_hash(root: Path) -> Optional[str]:
    """Tree hash a vendored build records."""
    import tempfile

    from .nsx_cli import tree_hash

    if _checkout_missing(root):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        module = Path(tmp) / CMSIS_NN_MODULE
        write_kernels(root, module)
        return tree_hash(module)


def kernels_match(root: Path, module_dir: Path) -> bool:
    """The checkout still equals the vendored copy."""
    from .nsx_cli import tree_hash

    for name in KERNEL_TREES:
        src, dst = root / name, module_dir / name
        if src.is_dir() != dst.is_dir():
            return False
        if src.is_dir() and tree_hash(src) != tree_hash(dst):
            return False
    pairs = (
        (root / "nsx" / "nsx-module.yaml", module_dir / "nsx-module.yaml"),
        (root / "nsx" / "CMakeLists.txt", module_dir / "nsx" / "CMakeLists.txt"),
    )
    return all(src.is_file() and dst.is_file() and src.read_bytes() == dst.read_bytes() for src, dst in pairs)


def render_app(
    board: BoardSpec,
    options: AppOptions,
    app_dir: Path,
    *,
    repo_root: Optional[Path] = None,
) -> AppRender:
    """Write nsx.yml, modules.cmake, CMakeLists.txt, local kernels."""
    if options.placement == "mram" and not board.has_mram:
        raise AppRenderError(f"{board.id} has no cached MRAM; use tcm")
    if options.cmsis_nn_root is not None:
        # App files must not land in root.
        _check_no_overlap(options.cmsis_nn_root, app_dir)
    repo_root = (repo_root or tester_repo_root()).resolve()
    profile = nsx_cli.starter_profile(board.nsx_board)
    if profile is None:
        raise AppRenderError(f"No NSX starter profile for {board.nsx_board}")
    modules = module_names(board, profile)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(find_tester_templates_dir(repo_root) / "hardware" / "nsx")),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )

    registry_yaml = yaml.safe_dump({"module_registry": module_registry(options)}, sort_keys=False)
    nsx_yml = env.get_template("nsx.yml.j2").render(
        app_name=APP_NAME,
        board=board.nsx_board,
        toolchain=toolchain_spec(options.toolchain).name,
        channel=profile.get("channel"),
        modules=modules,
        vendored=[CMSIS_NN_MODULE] if options.cmsis_nn_root else [],
        module_registry_yaml=registry_yaml,
    )
    modules_cmake = env.get_template("modules.cmake.j2").render(modules=modules)

    probe = options.build_size_probe
    cmakelists = env.get_template("CMakeLists.txt.j2").render(
        app_name=APP_NAME,
        board=board,
        target=SIZE_PROBE_TARGET if probe else SERVER_TARGET,
        build_size_probe=probe,
        sources=("universal_size_probe.c",) if probe else _SERVER_SOURCES,
        cache_vars=options.cache_vars(),
        enable_f32=options.enable_f32,
        enable_f16=options.enable_f16,
        hardware_dir=repo_root / "cmake" / "hardware",
        scripts_dir=repo_root / "scripts",
        kernel_dir=kernel_dir(app_dir, options).name,
        kernel_id=options.kernel_id(),
        kernel_align=KERNEL_ALIGN_BYTES,
        image_dir="probe" if probe else IMAGE_SUBDIR,
        build_id_txt=BUILD_ID_TXT,
        link_pmu=PMU_MODULE in modules,
        empty_heap=board.soc in HEAPLESS_SOCS,
        fp16_storage=not get_cpu_profile(board.cpu).supports_execution_dtype("FP16"),
        rtt_buffer_size_up=RTT_BUFFER_SIZE_UP,
        rtt_buffer_size_down=RTT_BUFFER_SIZE_DOWN,
        placement=options.placement,
    )

    if options.cmsis_nn_root is not None:
        write_kernels(options.cmsis_nn_root, kernel_dir(app_dir, options))
    _write_if_changed(app_dir / "nsx.yml", nsx_yml)
    _write_if_changed(app_dir / "CMakeLists.txt", cmakelists)
    _write_if_absent(app_dir / "cmake" / "nsx" / "modules.cmake", modules_cmake)
    return AppRender(app_dir, tuple(modules), nsx_yml, modules_cmake, cmakelists)
