#!/usr/bin/env python3
"""
setup_dependencies.py — Download and setup build dependencies.

This script downloads and sets up the dependencies needed for building and running
CMSIS-NN unit tests, similar to the Setup_Environment() function in build_and_run_tests.sh.

Dependencies downloaded:
- Corstone300 FVP (Fixed Virtual Platform)
- ARM GCC toolchain
- CMSIS-5 library
- Ethos-U core platform
- Python virtual environment with requirements

Usage:
    python3 scripts/setup_dependencies.py [--downloads-dir DOWNLOADS_DIR] [--force]
        [--gcc-version 13.2.rel1] [--gcc-sha256 <hex>]
"""

import argparse
import contextlib
import hashlib
import os
import platform
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

import certifi

try:
    import fcntl
except ImportError:
    fcntl = None
try:
    import msvcrt
except ImportError:
    msvcrt = None

from helia_core_tester.core.discovery import find_repo_root


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file's SHA-256 does not match the pinned value."""


GCC_VERSION_ENV = "HELIA_GCC_VERSION"
GCC_SHA256_ENV = "HELIA_GCC_SHA256"
GCC_VERSION_RE = re.compile(r"^\d+\.\d+\.rel\d+$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Releases up to 15.2.rel1 are served from developer.arm.com; from 15.3.rel1 Arm
# publishes to its GitLab package registry instead and the old path 404s.
_GCC_GITLAB_FIRST_RELEASE = (15, 3)
_GCC_DEVELOPER_URL = "https://developer.arm.com/-/media/Files/downloads/gnu/{version}/binrel/{archive}"
_GCC_GITLAB_URL = (
    "https://gitlab.arm.com/api/v4/projects/tooling%2Fgnu-toolchains-for-arm/packages/generic/"
    "gnu-toolchain/{version}/{archive}"
)

# Pinned SHA-256 digests for all downloaded dependency archives. Values were
# obtained from Arm's official release artifacts:
#   - ARM GCC toolchain, keyed (dependency, version, architecture): published by
#     Arm alongside each release as <archive>.sha256asc next to the archive (see
#     arm_gcc_url() for the host). A release not listed here can still
#     be selected with --gcc-version; its digest is then taken from
#     --gcc-sha256 / HELIA_GCC_SHA256 or, failing that, fetched from that same
#     sidecar at install time (see resolve_gcc_sha256()).
#   - Corstone-300 FVP 11.24_13, keyed (dependency, architecture): Arm does not
#     publish a SHA-256 sidecar for this archive; the digest below was computed
#     directly from a fresh download of the official Arm URL referenced in
#     setup_corstone300() and should be re-verified/updated whenever the pinned
#     Corstone300 version changes.
PINNED_SHA256 = {
    ("arm_gcc", "13.2.rel1", "x86_64"): "6cd1bbc1d9ae57312bcd169ae283153a9572bd6a8e4eeae2fedfbc33b115fdbb",
    ("arm_gcc", "13.2.rel1", "aarch64"): "8fd8b4a0a8d44ab2e195ccfbeef42223dfb3ede29d80f14dcf2183c34b8d199a",
    ("arm_gcc", "13.3.rel1", "x86_64"): "95c011cee430e64dd6087c75c800f04b9c49832cc1000127a92a97f9c8d83af4",
    ("arm_gcc", "13.3.rel1", "aarch64"): "c8824bffd057afce2259f7618254e840715f33523a3d4e4294f471208f976764",
    ("arm_gcc", "14.2.rel1", "x86_64"): "62a63b981fe391a9cbad7ef51b17e49aeaa3e7b0d029b36ca1e9c3b2a9b78823",
    ("arm_gcc", "14.2.rel1", "aarch64"): "87330bab085dd8749d4ed0ad633674b9dc48b237b61069e3b481abd364d0a684",
    ("arm_gcc", "14.3.rel1", "x86_64"): "8f6903f8ceb084d9227b9ef991490413014d991874a1e34074443c2a72b14dbd",
    ("arm_gcc", "14.3.rel1", "aarch64"): "2d465847eb1d05f876270494f51034de9ace9abe87a4222d079f3360240184d3",
    ("arm_gcc", "15.2.rel1", "x86_64"): "597893282ac8c6ab1a4073977f2362990184599643b4c5ee34870a8215783a16",
    ("arm_gcc", "15.2.rel1", "aarch64"): "d061559d814b205ed30c5b7c577c03317ec447ca51cd5a159d26b12a5bbeb20c",
    ("arm_gcc", "15.3.rel1", "x86_64"): "563bebb2b97d53382b956d6ee1fe61e2cae26699901417234a37df505ef9b5fa",
    ("arm_gcc", "15.3.rel1", "aarch64"): "06979e0c8171de58e5dc2a2b2019330a290f30930f27728af98a83e1a7369b3a",
    ("corstone300", "x86_64"): "6ea4096ecf8a8c06d6e76e21cae494f0c7139374cb33f6bc3964d189b84539a9",
    ("corstone300", "aarch64"): "9b43da6a688220c707cd1801baf9cf4f5fb37d6dc77587b9071347411a64fd56",
}


# The default Arm GNU Toolchain release, matching heliaAOT's tests/e2e/tools/deps.py.
# Bumping it needs pinned arm_gcc digests for it above.
ARM_GCC_VERSION = "14.3.rel1"
# Written into the install once it is complete. An install without it, or with another
# version, is stale: a restored CI cache or a reused volume must not keep an old compiler.
ARM_GCC_VERSION_MARKER = ".hct-arm-gcc-version"
ARM_GCC_STAGING_PREFIX = ".arm_gcc_staging-"


def normalize_gcc_version(version: str) -> str:
    """Lowercase (Arm's URL path uses `rel1`, not `Rel1`) and validate the format."""
    normalized = str(version).strip().lower()
    if not GCC_VERSION_RE.match(normalized):
        raise ValueError(
            f"Invalid ARM GCC version {version!r}: expected <major>.<minor>.rel<N> "
            f"(e.g. {ARM_GCC_VERSION})"
        )
    return normalized


def resolve_gcc_version(cli_value: Optional[str] = None) -> str:
    """Selected GCC release: --gcc-version > HELIA_GCC_VERSION > ARM_GCC_VERSION."""
    raw = cli_value or os.environ.get(GCC_VERSION_ENV) or ARM_GCC_VERSION
    return normalize_gcc_version(raw)


def arm_gcc_url(arch: str, version: str = ARM_GCC_VERSION) -> str:
    if arch not in ("x86_64", "aarch64"):
        raise RuntimeError(f"Unsupported architecture for ARM GCC: {arch}")
    version = normalize_gcc_version(version)
    major, minor = (int(part) for part in version.split(".")[:2])
    template = _GCC_GITLAB_URL if (major, minor) >= _GCC_GITLAB_FIRST_RELEASE else _GCC_DEVELOPER_URL
    return template.format(version=version, archive=f"arm-gnu-toolchain-{version}-{arch}-arm-none-eabi.tar.xz")


def parse_sha256_sidecar(text: str) -> str:
    """Extract the digest from Arm's `<hex>  <filename>` .sha256asc content."""
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if not _SHA256_HEX_RE.match(tokens[0]):
            raise ValueError(f"Malformed SHA-256 sidecar line: {line!r}")
        return tokens[0].lower()
    raise ValueError("Empty SHA-256 sidecar")


def fetch_sha256_sidecar(archive_url: str) -> str:
    with urllib.request.urlopen(archive_url + ".sha256asc", context=tls_context()) as response:
        return parse_sha256_sidecar(response.read().decode("utf-8", errors="replace"))


def resolve_gcc_sha256(version: str, arch: str, override: Optional[str] = None) -> str:
    """Expected digest: explicit override > pinned table > Arm's .sha256asc sidecar."""
    if override:
        digest = override.strip().lower()
        if not _SHA256_HEX_RE.match(digest):
            raise ValueError(f"Invalid --gcc-sha256 value {override!r}: expected 64 hex characters")
        return digest

    pinned = PINNED_SHA256.get(("arm_gcc", version, arch))
    if pinned:
        return pinned

    url = arm_gcc_url(arch, version)
    print(
        f"NOTE: no pinned SHA-256 for ARM GCC {version} ({arch}); fetching {url}.sha256asc. "
        "This digest comes from the same origin as the archive, so it guards against a "
        "corrupted download but is not an independent pin. Pass --gcc-sha256 (or set "
        f"{GCC_SHA256_ENV}) to pin it explicitly."
    )
    try:
        return fetch_sha256_sidecar(url)
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch the SHA-256 sidecar for ARM GCC {version} ({arch}): {e}. "
            f"Pass --gcc-sha256 <hex> (or set {GCC_SHA256_ENV}) with the expected digest."
        ) from e


def installed_arm_gcc_version(gcc_dir: Path) -> Optional[str]:
    """The version an install records, or None when it records none."""
    try:
        return (gcc_dir / ARM_GCC_VERSION_MARKER).read_text().strip() or None
    except OSError:
        return None


def get_architecture() -> str:
    """Get the system architecture (x86_64 or aarch64)."""
    machine = platform.machine().lower()
    if machine in ['x86_64', 'amd64']:
        return 'x86_64'
    elif machine in ['aarch64', 'arm64']:
        return 'aarch64'
    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")


def get_os() -> str:
    """Get the operating system."""
    system = platform.system().lower()
    if system == 'linux':
        return 'linux'
    else:
        raise RuntimeError(f"Unsupported operating system: {system}")




def tls_context() -> ssl.SSLContext:
    """Return a verifying TLS context for downloads."""
    ctx = ssl.create_default_context()
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return ctx
    # uv's CPython on NixOS lacks CAs.
    ctx.load_verify_locations(cafile=certifi.where())
    return ctx


def download_file(url: str, dest_path: Path, description: str, expected_sha256: str) -> None:
    """Download a file from URL to destination path, verifying its SHA-256 digest.

    Streams the hash incrementally while writing to disk, compares it against
    `expected_sha256` once the download completes, and deletes the file and
    raises ChecksumMismatchError on any mismatch. Never returns successfully
    for a file whose digest does not match.
    """
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")

    hasher = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, context=tls_context()) as response:
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    f.write(chunk)
        print("Downloaded successfully")
    except Exception as e:
        print(f"Download failed: {e}")
        dest_path.unlink(missing_ok=True)
        raise

    actual_sha256 = hasher.hexdigest()
    if actual_sha256.lower() != expected_sha256.lower():
        dest_path.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"SHA-256 mismatch for {description} ({url}): "
            f"expected {expected_sha256}, got {actual_sha256}. "
            "The downloaded file has been deleted and will not be extracted or executed."
        )
    print(f"Checksum verified (sha256={actual_sha256})")


def extract_tar_gz(archive_path: Path, extract_to: Path, strip_components: int = 0) -> None:
    """Extract a .tar.gz or .tar.xz file."""
    print(f"Extracting {archive_path.name} to {extract_to}")
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(archive_path, 'r:*') as tar:
        # Get all members
        members = tar.getmembers()
        
        if strip_components > 0:
            # Remove the specified number of path components
            for member in members:
                path_parts = member.name.split('/')
                if len(path_parts) > strip_components:
                    member.name = '/'.join(path_parts[strip_components:])
                else:
                    member.name = '.'
        
        # Extract all members at once (more robust for complex archives)
        try:
            tar.extractall(extract_to, members=members)
        except (OSError, IOError) as e:
            print(f"Warning: Some files could not be extracted: {e}")
            # Try extracting members one by one for better error handling
            for member in members:
                try:
                    tar.extract(member, extract_to)
                except (OSError, IOError) as member_error:
                    if member.issym() or member.islnk():
                        print(f"Warning: Skipping symlink {member.name}: {member_error}")
                    else:
                        print(f"Warning: Skipping {member.name}: {member_error}")
                    continue
    
    print("Extracted successfully")


def run_command(
    cmd: list[str],
    cwd: Optional[Path] = None,
    description: str = "",
    *,
    env: Optional[dict] = None,
    stdin: Optional[int] = None,
    quiet: bool = False,
) -> str:
    """Run a command and return its stdout; quiet leaves output and failure details to the caller."""
    if description:
        print(f"Running: {description}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env=env,
            stdin=stdin,
        )
        if result.stdout and not quiet:
            print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        if not quiet:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Return code: {e.returncode}")
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
        raise


def setup_corstone300(downloads_dir: Path, force: bool = False) -> None:
    """Download and setup Corstone300 FVP (mirrors the bash flow)."""
    corstone_dir = downloads_dir / "corstone300_download"

    # Existing install?
    if corstone_dir.exists() and not force:
        print("Corstone300 already installed. If you wish to install a new version, please delete the old folder.")
        return

    # Force re-install
    if force and corstone_dir.exists():
        print("Removing existing Corstone300 installation...")
        shutil.rmtree(corstone_dir)

    arch = get_architecture()
    if arch == 'x86_64':
        corstone_url = "https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.24_13_Linux64.tgz"
    elif arch == 'aarch64':
        corstone_url = "https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.24_13_Linux64_armv8l.tgz"
    else:
        raise RuntimeError(f"Unsupported architecture for Corstone300: {arch}")

    expected_sha256 = PINNED_SHA256[("corstone300", arch)]

    # Work in temp dirs like the bash script
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        archive_file = temp_path / "corstone300.tgz"

        # Download (equivalent to: wget -q "${CORSTONE_URL}" -O "${TEMPFILE}")
        try:
            download_file(corstone_url, archive_file, "Corstone300", expected_sha256)
        except ChecksumMismatchError:
            # Fail closed: do not extract or execute an archive that failed
            # integrity verification.
            raise
        except Exception:
            # Match the bash error message
            raise RuntimeError("Download Corstone300 failed!")

        # Extract (equivalent to: tar -C ${TEMPDIR} -xzf ${TEMPFILE})
        # Use system 'tar' to mirror bash behavior closely.
        run_command(
            ["tar", "-C", str(temp_path), "-xzf", str(archive_file)],
            description=f"Extracting {archive_file.name}"
        )

        # Find the installer script (bash expects ${TEMPDIR}/FVP_Corstone_SSE-300.sh)
        installer_script = None
        for item in temp_path.iterdir():
            if item.name.startswith("FVP_Corstone_SSE-300") and item.suffix == ".sh":
                installer_script = item
                break
        if not installer_script:
            raise RuntimeError("Could not find Corstone300 installer script after extraction")

        # Ensure destination exists (equivalent to: mkdir ${WORKING_DIR}/corstone300_download)
        corstone_dir.mkdir(parents=True, exist_ok=True)

        # Make sure script is executable (some environments lose +x)
        try:
            current_mode = installer_script.stat().st_mode
            installer_script.chmod(current_mode | 0o111)
        except Exception:
            pass  # best-effort

        # Run installer (equiv: ${TEMPDIR}/FVP_Corstone_SSE-300.sh --i-agree... -d <dir>)
        print("Installing Corstone300...")
        run_command(
            [
                str(installer_script),
                "--i-agree-to-the-contained-eula",
                "--no-interactive",
                "-q",
                "-d", str(corstone_dir),
            ],
            description="Installing Corstone300 FVP"
        )

    print("Corstone300 setup complete")



def setup_arm_gcc(
    downloads_dir: Path,
    force: bool = False,
    version: Optional[str] = None,
    sha256: Optional[str] = None,
) -> None:
    """Download and set up the selected ARM GCC release, replacing a stale install.

    `version` / `sha256` fall back to HELIA_GCC_VERSION / HELIA_GCC_SHA256 and then to
    ARM_GCC_VERSION / the pinned table (see resolve_gcc_sha256()).
    """
    gcc_dir = downloads_dir / "arm_gcc_download"
    version = resolve_gcc_version(version)

    if gcc_dir.exists() and not force:
        installed = installed_arm_gcc_version(gcc_dir)
        if installed == version:
            print(f"Arm GCC {version} already installed.")
            return
        print(f"Arm GCC at {gcc_dir} is {installed or 'an unrecorded version'}; installing {version}.")

    arch = get_architecture()
    gcc_url = arm_gcc_url(arch, version)
    expected_sha256 = resolve_gcc_sha256(version, arch, override=sha256 or os.environ.get(GCC_SHA256_ENV))

    downloads_dir.mkdir(parents=True, exist_ok=True)
    for leftover in downloads_dir.glob(f"{ARM_GCC_STAGING_PREFIX}*"):
        shutil.rmtree(leftover, ignore_errors=True)
    # Staged next to the install so the swap below is a rename, not a cross-device copy.
    with tempfile.TemporaryDirectory(prefix=ARM_GCC_STAGING_PREFIX, dir=downloads_dir) as temp_dir:
        temp_path = Path(temp_dir)
        archive_file = temp_path / "arm_gcc.tar.xz"

        download_file(gcc_url, archive_file, f"ARM GCC toolchain {version}", expected_sha256)

        temp_extract = temp_path / "extracted"
        extract_tar_gz(archive_file, temp_extract, strip_components=0)

        toolchain_dirs = [d for d in temp_extract.iterdir() if d.is_dir()]
        if len(toolchain_dirs) != 1:
            raise RuntimeError(f"Expected one toolchain directory in the archive, found {len(toolchain_dirs)}")
        toolchain_dir = toolchain_dirs[0]
        if not (toolchain_dir / "bin").is_dir():
            raise RuntimeError(f"Toolchain archive has no bin/ under {toolchain_dir.name}")
        (toolchain_dir / ARM_GCC_VERSION_MARKER).write_text(version + "\n")

        # The old install goes only once the new one is verified, extracted and marked.
        if gcc_dir.exists():
            print(f"Replacing existing ARM GCC installation at {gcc_dir}...")
            os.replace(gcc_dir, temp_path / "previous")
        os.replace(toolchain_dir, gcc_dir)

    print(f"ARM GCC {version} setup complete")


class SourcePin(NamedTuple):
    label: str
    url: str
    commit: str
    # Top-level directory every install of the repo has, git checkout or not.
    marker: str


# Reviewed source revisions, fetched by commit and verified after checkout.
# Bumping one is a deliberate change to this table; CI's download cache is keyed
# on this file's hash, so it refetches.
SOURCE_PINS = {
    # ARM-software/CMSIS_5 develop, 2024-09-03.
    "CMSIS_5": SourcePin(
        "CMSIS-5",
        "https://github.com/ARM-software/CMSIS_5.git",
        "55b19837f5703e418ca37894d5745b1dc05e4c91",
        "CMSIS",
    ),
    # ethos-u-core-platform main, 2026-10-06.
    "ethos-u-core-platform": SourcePin(
        "Ethos-U core platform",
        "https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform.git",
        "8405e3fc7e6fc684917a6b0cdfc906ce8a94f204",
        "core_platform",
    ),
}

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")

# enforce (default, and what every build/run uses): put every checkout at its pin or fail.
# warn (opt-in, --source-pins warn): fetch missing sources at the pin, but only
# report an existing checkout that is off its pin.
PIN_MODES = ("enforce", "warn")


class SourcePinError(RuntimeError):
    """Raised when a source dependency is not, or cannot be put, at its pinned commit."""


def _git(args: list[str], cwd: Path) -> str:
    """git without prompts, trusting cwd even when another user owns it.

    downloads_dir is ours, but in a bind-mounted container it is owned by the
    host user and git would otherwise refuse it as dubious ownership.
    """
    cmd = ["git", "-c", f"safe.directory={Path(cwd).resolve()}", *args]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        return run_command(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, quiet=True).strip()
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise SourcePinError(f"git {' '.join(args)} failed in {cwd}: {detail}") from exc


def _checkout_head(path: Path) -> Optional[str]:
    """HEAD of the checkout at path, or None when it is not a usable git checkout."""
    if not (path / ".git").exists():
        return None
    try:
        return _git(["rev-parse", "--verify", "HEAD^{commit}"], path)
    except SourcePinError:
        return None


def _has_local_changes(path: Path) -> bool:
    """Tracked edits, untracked files, or index entries git status skips.

    An assume-unchanged or skip-worktree entry hides edits from git status and
    keeps its bytes across a checkout, so it is treated as a change (the same
    rule as hardware/harness_lock.py).
    """
    if _git(["status", "--porcelain"], path):
        return True
    # ls-files -v tags: S = skip-worktree, lowercase = assume-unchanged.
    entries = _git(["ls-files", "-v", "-z"], path).split("\0")
    return any(entry[:1] == "S" or entry[:1].islower() for entry in entries if entry)


def _has_commit(path: Path, commit: str) -> bool:
    try:
        _git(["cat-file", "-e", f"{commit}^{{commit}}"], path)
    except SourcePinError:
        return False
    return True


def _fetch_pinned(path: Path, pin: SourcePin) -> None:
    if not _has_commit(path, pin.commit):
        # Keep full clones full; only shallow (or new) repos get a depth-1 fetch.
        shallow = _git(["rev-parse", "--is-shallow-repository"], path) == "true"
        depth = ["--depth=1"] if shallow or not _git(["rev-list", "-n1", "--all"], path) else []
        _git(["fetch", "--quiet", *depth, pin.url, pin.commit], path)
    _git(["checkout", "--quiet", "--detach", pin.commit], path)
    head = _checkout_head(path)
    if head != pin.commit:
        raise SourcePinError(f"{pin.label} checkout is at {head}, expected pinned {pin.commit}")


@contextlib.contextmanager
def _setup_lock(downloads_dir: Path, dirname: str) -> Iterator[bool]:
    """Serialize setup of one source across processes (parallel runs share downloads_dir).

    Yields whether a cross-process lock is held: flock on POSIX, msvcrt on
    Windows, none elsewhere.
    """
    with open(downloads_dir / f".{dirname}.lock", "a") as handle:
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield True
        elif msvcrt is not None:
            handle.seek(0)
            while True:
                try:
                    # LK_LOCK retries for about 10 s before raising; keep waiting.
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
            try:
                yield True
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            yield False


def _remove_stale_staging(downloads_dir: Path, dirname: str) -> None:
    """Drop staging dirs a killed setup left behind; safe only under _setup_lock."""
    for leftover in downloads_dir.glob(f".{dirname}-*"):
        if leftover.is_dir():
            print(f"Removing stale staging directory {leftover}")
            shutil.rmtree(leftover, ignore_errors=True)


def _swap_in(stage: Path, dest: Path, scratch: Path) -> None:
    """Move stage to dest; an existing dest is set aside first and restored if the move fails."""
    backup = None
    if dest.exists() or dest.is_symlink():
        backup = scratch / f"{dest.name}.old"
        os.replace(dest, backup)
    try:
        os.replace(stage, dest)
    except OSError:
        if backup is not None:
            os.replace(backup, dest)
        raise
    if backup is not None:
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            print(f"WARNING: could not remove the old checkout at {backup} ({exc}); the next setup removes it")


def _existing_checkout_problem(dest: Path, pin: SourcePin) -> tuple[Optional[str], Optional[str], bool]:
    """(head, problem, usable) for an existing dest; problem is None when it can be moved to the pin."""
    head = _checkout_head(dest)
    if head is None:
        problem = (
            f"{dest} exists but is not a git checkout git can read (a non-git directory or an "
            f"empty repo), so its revision cannot be verified against {pin.commit}"
        )
        return None, problem, (dest / pin.marker).is_dir()
    if _has_local_changes(dest):
        return head, (
            f"{dest} has local changes (edits, untracked files, or assume-unchanged/skip-worktree "
            f"entries) at {head}"
        ), True
    return head, None, True


def setup_pinned_source(
    downloads_dir: Path, dirname: str, force: bool = False, pin_mode: str = "enforce"
) -> None:
    """Put downloads_dir/dirname at its SOURCE_PINS commit.

    A fresh install (or a --force reinstall) is fetched by commit into a staging
    directory and swapped in only once verified; the old checkout is set aside
    and restored if the swap fails, so a failed setup leaves it untouched. In
    enforce mode an existing clean checkout at another commit is moved to the
    pin, and one with local changes, or a directory whose revision cannot be
    read, is rejected unless force replaces it. In warn mode an existing
    checkout off its pin is reported and used as is (a non-git install only if
    it has the repo's marker directory).
    """
    if pin_mode not in PIN_MODES:
        raise ValueError(f"pin_mode must be one of {PIN_MODES}, got {pin_mode!r}")
    pin = SOURCE_PINS[dirname]
    if not _COMMIT_RE.fullmatch(pin.commit):
        raise SourcePinError(f"{pin.label} pin {pin.commit!r} is not a full 40-hex commit")
    dest = downloads_dir / dirname
    downloads_dir.mkdir(parents=True, exist_ok=True)

    with _setup_lock(downloads_dir, dirname) as locked:
        # Without a lock a concurrent setup's live staging dir would look stale.
        if locked:
            _remove_stale_staging(downloads_dir, dirname)

        if dest.exists() and not force:
            head, problem, usable = _existing_checkout_problem(dest, pin)
            if problem is None and head == pin.commit:
                print(f"{pin.label} already at pinned {pin.commit[:12]}")
                return
            if pin_mode == "warn" and usable:
                reason = problem or f"{dest} is at {head[:12]}, not pinned {pin.commit[:12]}"
                print(
                    f"WARNING: {reason}; using it as is. setup_dependencies.py moves a clean "
                    "checkout to the pin; --force replaces any other."
                )
                return
            if problem is not None:
                raise SourcePinError(
                    f"{problem}; refusing to move it to {pin.commit}. Delete it or rerun with --force."
                )
            print(f"Moving {pin.label} from {head[:12]} to pinned {pin.commit[:12]}...")
            _fetch_pinned(dest, pin)
            print(f"{pin.label} setup complete")
            return

        print(f"Fetching {pin.label} at {pin.commit[:12]}...")
        with tempfile.TemporaryDirectory(
            dir=downloads_dir, prefix=f".{dirname}-", ignore_cleanup_errors=True
        ) as tmp:
            stage = Path(tmp) / dirname
            stage.mkdir()
            _git(["init", "--quiet"], stage)
            _fetch_pinned(stage, pin)
            if dest.exists() or dest.is_symlink():
                print(f"Replacing existing {pin.label} at {dest}...")
            _swap_in(stage, dest, Path(tmp))
    print(f"{pin.label} setup complete")


def setup_cmsis5(downloads_dir: Path, force: bool = False, pin_mode: str = "enforce") -> None:
    """Fetch CMSIS-5 at its pinned commit."""
    setup_pinned_source(downloads_dir, "CMSIS_5", force, pin_mode)


def setup_ethos_u_platform(downloads_dir: Path, force: bool = False, pin_mode: str = "enforce") -> None:
    """Fetch the Ethos-U core platform at its pinned commit."""
    setup_pinned_source(downloads_dir, "ethos-u-core-platform", force, pin_mode)


DEFAULT_DOWNLOADS_DIR = Path("artifacts/downloads")


def setup_python_venv(downloads_dir: Path, force: bool = False) -> None:
    """Setup Python virtual environment using uv (required)."""
    venv_dir = downloads_dir / "cmsis_nn_venv"
    
    if venv_dir.exists() and not force:
        print("Python venv already installed. If you wish to install a new version, please delete the old folder.")
        return
    
    if force and venv_dir.exists():
        print("Removing existing Python venv...")
        shutil.rmtree(venv_dir)
    
    print("Setting up Python virtual environment with uv...")
    
    # Check if uv is available (required)
    uv_available = shutil.which("uv") is not None
    if not uv_available:
        raise RuntimeError(
            "uv is required but not found. Please install uv:\n"
            "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
    
    # Create virtual environment with uv
    run_command(
        ["uv", "venv", str(venv_dir)],
        description="Creating Python virtual environment with uv"
    )
    
    # Determine Python executable in venv
    if sys.platform.startswith('win'):
        python_cmd = str(venv_dir / "Scripts" / "python.exe")
    else:
        python_cmd = str(venv_dir / "bin" / "python")
    
    # Install dependencies using uv pip
    repo_root = find_repo_root()
    requirements_file = repo_root / "requirements.txt"
    pyproject_file = repo_root / "pyproject.toml"
    
    if pyproject_file.exists():
        print("Installing Python requirements from pyproject.toml with uv...")
        run_command(
            ["uv", "pip", "install", "--python", python_cmd, "-e", str(repo_root)],
            description="Installing Python requirements with uv"
        )
    elif requirements_file.exists():
        print("Installing Python requirements from requirements.txt with uv...")
        run_command(
            ["uv", "pip", "install", "--python", python_cmd, "-r", str(requirements_file)],
            description="Installing Python requirements with uv"
        )
    else:
        raise RuntimeError("Neither pyproject.toml nor requirements.txt found")
    
    print("✓ Python virtual environment setup complete")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and setup build dependencies for CMSIS-NN unit tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 scripts/setup_dependencies.py
    python3 scripts/setup_dependencies.py --downloads-dir ./my_downloads
    python3 scripts/setup_dependencies.py --force
    python3 scripts/setup_dependencies.py --gcc-version 13.2.rel1
    HELIA_GCC_VERSION=15.2.rel1 python3 scripts/setup_dependencies.py
    python3 scripts/setup_dependencies.py --gcc-version 16.1.rel1 --gcc-sha256 <hex>
        """
    )

    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=DEFAULT_DOWNLOADS_DIR,
        help="Directory to store downloaded dependencies (default: artifacts/downloads)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and reinstall all dependencies"
    )
    parser.add_argument(
        "--gcc-version",
        default=None,
        help=(
            "ARM GCC release to install, e.g. 13.2.rel1 (default: $%s or %s). "
            "An install of another release is replaced."
            % (GCC_VERSION_ENV, ARM_GCC_VERSION)
        ),
    )
    parser.add_argument(
        "--gcc-sha256",
        default=None,
        help=(
            "Expected SHA-256 of the ARM GCC archive (default: $%s, then the pinned table, "
            "then Arm's published .sha256asc sidecar)" % GCC_SHA256_ENV
        ),
    )
    parser.add_argument(
        "--skip-corstone",
        action="store_true",
        help="Skip Corstone300 FVP download"
    )
    parser.add_argument(
        "--skip-gcc",
        action="store_true",
        help="Skip ARM GCC toolchain download"
    )
    parser.add_argument(
        "--skip-cmsis5",
        action="store_true",
        help="Skip CMSIS-5 download"
    )
    parser.add_argument(
        "--skip-ethos",
        action="store_true",
        help="Skip Ethos-U core platform download"
    )
    parser.add_argument(
        "--source-pins",
        choices=PIN_MODES,
        default="enforce",
        help=(
            "enforce (default): move CMSIS-5 / Ethos-U checkouts to their pinned commits or fail; "
            "warn: fetch missing ones at the pin, report but keep existing ones that are off it"
        ),
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Skip Python virtual environment setup"
    )
    # Retired flags; accepted so old callers work.
    for retired in ("--skip-nsx-sdk", "--with-hardware"):
        parser.add_argument(retired, action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()
    
    # Validate system. Only the two prebuilt-binary downloads (Corstone-300 FVP
    # and the ARM GCC tarball) are Linux-only; the git dependencies
    # (CMSIS-5, Ethos-U core platform, fetched at pinned commits) install anywhere. Treat an
    # unsupported OS as "skip those two", not as a hard failure.
    try:
        get_architecture()
        gcc_version = resolve_gcc_version(args.gcc_version)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    host_os_supported = True
    try:
        get_os()
    except RuntimeError as e:
        host_os_supported = False
        print(f"NOTE: {e}.")
        print("      Skipping the Linux-only prebuilt downloads (Corstone-300 FVP, ARM GCC);")
        print("      supply those yourself. All other dependencies still install.")
        print()
    
    # Create downloads directory
    args.downloads_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Setting up CMSIS-NN build dependencies")
    print(f"Downloads directory: {args.downloads_dir}")
    print(f"Force mode: {args.force}")
    print(f"ARM GCC version: {gcc_version}")
    print("=" * 80)

    try:
        if host_os_supported and not args.skip_corstone:
            setup_corstone300(args.downloads_dir, args.force)
            print()

        if host_os_supported and not args.skip_gcc:
            setup_arm_gcc(args.downloads_dir, args.force, version=gcc_version, sha256=args.gcc_sha256)
            print()
        
        if not args.skip_cmsis5:
            setup_cmsis5(args.downloads_dir, args.force, args.source_pins)
            print()
        
        if not args.skip_ethos:
            setup_ethos_u_platform(args.downloads_dir, args.force, args.source_pins)
            print()
        
        if not args.skip_python:
            setup_python_venv(args.downloads_dir, args.force)
            print()

        print("=" * 80)
        print("All dependencies setup complete!")
        print("=" * 80)
        
        # Print summary
        print("\nInstalled dependencies:")
        for item in args.downloads_dir.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
