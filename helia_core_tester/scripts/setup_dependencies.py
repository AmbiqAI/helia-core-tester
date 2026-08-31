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
"""

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from helia_core_tester.core.discovery import find_repo_root


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file's SHA-256 does not match the pinned value."""


# Pinned SHA-256 digests for all downloaded dependency archives, keyed by
# (dependency, architecture). Values were obtained from Arm's official release
# artifacts:
#   - ARM GCC toolchain 14.2.rel1: published by Arm alongside the release at
#     https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/<file>.sha256asc
#   - Corstone-300 FVP 11.24_13: Arm does not publish a SHA-256 sidecar for this
#     archive; the digest below was computed directly from a fresh download of
#     the official Arm URL referenced in setup_corstone300() and should be
#     re-verified/updated whenever the pinned Corstone300 version changes.
PINNED_SHA256 = {
    ("arm_gcc", "x86_64"): "62a63b981fe391a9cbad7ef51b17e49aeaa3e7b0d029b36ca1e9c3b2a9b78823",
    ("arm_gcc", "aarch64"): "87330bab085dd8749d4ed0ad633674b9dc48b237b61069e3b481abd364d0a684",
    ("corstone300", "x86_64"): "6ea4096ecf8a8c06d6e76e21cae494f0c7139374cb33f6bc3964d189b84539a9",
    ("corstone300", "aarch64"): "9b43da6a688220c707cd1801baf9cf4f5fb37d6dc77587b9071347411a64fd56",
}


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
        with urllib.request.urlopen(url) as response:
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


def run_command(cmd: list[str], cwd: Optional[Path] = None, description: str = "") -> None:
    """Run a command and handle errors."""
    if description:
        print(f"Running: {description}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
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



def setup_arm_gcc(downloads_dir: Path, force: bool = False) -> None:
    """Download and setup ARM GCC toolchain."""
    gcc_dir = downloads_dir / "arm_gcc_download"
    
    if gcc_dir.exists() and not force:
        print("Arm GCC already installed. If you wish to install a new version, please delete the old folder.")
        return
    
    if force and gcc_dir.exists():
        print("Removing existing ARM GCC installation...")
        shutil.rmtree(gcc_dir)
    
    arch = get_architecture()
    if arch == 'x86_64':
        gcc_url = "https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi.tar.xz"
    elif arch == 'aarch64':
        gcc_url = "https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-aarch64-arm-none-eabi.tar.xz"
    else:
        raise RuntimeError(f"Unsupported architecture for ARM GCC: {arch}")

    expected_sha256 = PINNED_SHA256[("arm_gcc", arch)]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        archive_file = temp_path / "arm_gcc.tar.xz"
        
        download_file(gcc_url, archive_file, "ARM GCC toolchain", expected_sha256)
        
        # Extract to temporary directory first
        temp_extract = temp_path / "extracted"
        extract_tar_gz(archive_file, temp_extract, strip_components=0)
        
        # Find the toolchain directory (should be the only subdirectory)
        toolchain_dirs = [d for d in temp_extract.iterdir() if d.is_dir()]
        if not toolchain_dirs:
            raise RuntimeError("Could not find toolchain directory in archive")
        
        toolchain_dir = toolchain_dirs[0]
        
        # Move contents to final destination
        print(f"Moving toolchain from {toolchain_dir.name} to {gcc_dir}")
        shutil.move(str(toolchain_dir), str(gcc_dir))
    
    print("ARM GCC setup complete")


def setup_cmsis5(downloads_dir: Path, force: bool = False) -> None:
    """Clone CMSIS-5 repository."""
    cmsis5_dir = downloads_dir / "CMSIS_5"
    
    # Check if it's a valid installation (has .git or CMSIS subdirectory)
    is_valid_install = cmsis5_dir.exists() and (
        (cmsis5_dir / ".git").exists() or 
        (cmsis5_dir / "CMSIS").exists()
    )
    
    if is_valid_install and not force:
        print("CMSIS-5 already installed. If you wish to install a new version, please delete the old folder.")
        return
    
    if force and cmsis5_dir.exists():
        print("Removing existing CMSIS-5 installation...")
        shutil.rmtree(cmsis5_dir)
    elif cmsis5_dir.exists() and not is_valid_install:
        # Directory exists but is empty/invalid, remove it
        print("Removing invalid/empty CMSIS-5 directory...")
        shutil.rmtree(cmsis5_dir)
    
    print("Cloning CMSIS-5...")
    run_command(
        ["git", "clone", "--quiet", "--depth=1", "https://github.com/ARM-software/CMSIS_5.git"],
        cwd=downloads_dir,
        description="Cloning CMSIS-5"
    )
    
    print("CMSIS-5 setup complete")


def setup_ethos_u_platform(downloads_dir: Path, force: bool = False) -> None:
    """Clone Ethos-U core platform repository."""
    ethos_dir = downloads_dir / "ethos-u-core-platform"
    
    # Check if it's a valid installation (has .git or core_platform subdirectory)
    is_valid_install = ethos_dir.exists() and (
        (ethos_dir / ".git").exists() or 
        (ethos_dir / "core_platform").exists()
    )
    
    if is_valid_install and not force:
        print("Ethos-U core platform already installed. If you wish to install a new version, please delete the old folder.")
        return
    
    if force and ethos_dir.exists():
        print("Removing existing Ethos-U core platform installation...")
        shutil.rmtree(ethos_dir)
    elif ethos_dir.exists() and not is_valid_install:
        # Directory exists but is empty/invalid, remove it
        print("Removing invalid/empty Ethos-U core platform directory...")
        shutil.rmtree(ethos_dir)
    
    print("Cloning Ethos-U core platform...")
    run_command(
        ["git", "clone", "--quiet", "--depth=1", "https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform.git"],
        cwd=downloads_dir,
        description="Cloning Ethos-U core platform"
    )
    
    print("Ethos-U core platform setup complete")


def _ensure_nsx_sdk_symlinks(project_root: Path, sdk_dir: Path) -> None:
    """(Re-)create the local symlinks the top-level CMakeLists.txt/cmake/nsx glue
    expects at fixed, repo-relative paths, pointing into the vendored SDK checkout:

      boards/apollo510_evb   -> modules/nsx-ambiq-sdk/boards/apollo510_evb
      cmake/socs             -> modules/nsx-ambiq-sdk/cmake/socs
      cmake/nsx_soc_facts.cmake -> modules/nsx-ambiq-sdk/cmake/nsx_soc_facts.cmake

    Idempotent and always re-run (not just on a fresh clone) so an existing
    nsx-ambiq-sdk checkout that predates these symlinks still gets them.
    """
    links = {
        project_root / "boards" / "apollo510_evb": sdk_dir / "boards" / "apollo510_evb",
        project_root / "cmake" / "socs": sdk_dir / "cmake" / "socs",
        project_root / "cmake" / "nsx_soc_facts.cmake": sdk_dir / "cmake" / "nsx_soc_facts.cmake",
    }
    for link_path, target_path in links.items():
        if link_path.is_symlink() or link_path.exists():
            continue
        link_path.parent.mkdir(parents=True, exist_ok=True)
        relative_target = os.path.relpath(target_path, start=link_path.parent)
        link_path.symlink_to(relative_target, target_is_directory=target_path.is_dir())
        print(f"Linked {link_path.relative_to(project_root)} -> {relative_target}")


def setup_nsx_ambiq_sdk(project_root: Path, force: bool = False) -> None:
    """Clone the vendor NSX Ambiq SDK (board bring-up, HAL/BSP, CMSIS-Core, SoC
    descriptors) needed for HELIA_HARDWARE_BUILD=ON. Not a fixed-checksum tarball
    like the other deps here -- it's a large, actively-developed monorepo, so this
    pins to a shallow clone of its default branch rather than a specific commit.
    Lives under modules/ (see .gitignore's `modules/*` rule), not artifacts/downloads/,
    to match CMakeLists.txt's NSX_AMBIQ_SDK_DIR="modules/nsx-ambiq-sdk" expectation.
    Also (re-)creates the local symlinks (boards/apollo510_evb, cmake/socs,
    cmake/nsx_soc_facts.cmake) that redirect into it -- see
    _ensure_nsx_sdk_symlinks().
    """
    sdk_dir = project_root / "modules" / "nsx-ambiq-sdk"

    is_valid_install = sdk_dir.exists() and (
        (sdk_dir / ".git").exists() or
        (sdk_dir / "modules").exists()
    )

    if is_valid_install and not force:
        print("nsx-ambiq-sdk already installed. If you wish to install a new version, please delete the old folder.")
        _ensure_nsx_sdk_symlinks(project_root, sdk_dir)
        return

    if force and sdk_dir.exists():
        print("Removing existing nsx-ambiq-sdk installation...")
        shutil.rmtree(sdk_dir)
    elif sdk_dir.exists() and not is_valid_install:
        print("Removing invalid/empty nsx-ambiq-sdk directory...")
        shutil.rmtree(sdk_dir)

    sdk_dir.parent.mkdir(parents=True, exist_ok=True)
    print("Cloning nsx-ambiq-sdk (this is a large monorepo; may take a while)...")
    run_command(
        ["git", "clone", "--quiet", "--depth=1", "https://github.com/AmbiqAI/nsx-ambiq-sdk.git"],
        cwd=sdk_dir.parent,
        description="Cloning nsx-ambiq-sdk"
    )
    _ensure_nsx_sdk_symlinks(project_root, sdk_dir)

    print("nsx-ambiq-sdk setup complete")


def setup_neuralspotx(downloads_dir: Path, force: bool = False) -> None:
    """Clone the neuralspotx (NSX workspace tooling) repo for its RTT sources
    (examples/coremark/src/rtt/SEGGER_RTT.*), consumed by the perf-stream hardware
    firmware build. NOT AmbiqAI/neuralSPOT (capital SPOT, the separate C SDK/example
    repo) -- that repo has a different layout and lacks this path entirely.
    """
    neuralspotx_dir = downloads_dir / "neuralspotx"

    is_valid_install = neuralspotx_dir.exists() and (
        (neuralspotx_dir / ".git").exists() or
        (neuralspotx_dir / "examples").exists()
    )

    if is_valid_install and not force:
        print("neuralspotx already installed. If you wish to install a new version, please delete the old folder.")
        return

    if force and neuralspotx_dir.exists():
        print("Removing existing neuralspotx installation...")
        shutil.rmtree(neuralspotx_dir)
    elif neuralspotx_dir.exists() and not is_valid_install:
        print("Removing invalid/empty neuralspotx directory...")
        shutil.rmtree(neuralspotx_dir)

    print("Cloning neuralspotx...")
    run_command(
        ["git", "clone", "--quiet", "--depth=1", "https://github.com/AmbiqAI/neuralspotx.git"],
        cwd=downloads_dir,
        description="Cloning neuralspotx"
    )

    print("neuralspotx setup complete")


def setup_nsx_toolchain(project_root: Path, downloads_dir: Path, force: bool = False) -> None:
    """Generate cmake/nsx/toolchains/arm-none-eabi-gcc.cmake from the already-downloaded
    ARM GCC toolchain, via nsx-ambiq-sdk's own generator script. Requires
    setup_arm_gcc() and setup_nsx_ambiq_sdk() to have already run. This file is
    machine-specific (bakes in an absolute --gcc-root path) so it stays gitignored
    (*.cmake) and is always regenerated here rather than tracked.
    """
    toolchain_file = project_root / "cmake" / "nsx" / "toolchains" / "arm-none-eabi-gcc.cmake"
    gcc_dir = downloads_dir / "arm_gcc_download"
    generator = project_root / "modules" / "nsx-ambiq-sdk" / "tools" / "nsx_toolchain_file.py"

    if toolchain_file.exists() and not force:
        print("NSX arm-none-eabi-gcc toolchain file already generated. Pass --force to regenerate.")
        return
    if not gcc_dir.exists():
        raise RuntimeError(f"setup_nsx_toolchain: {gcc_dir} does not exist -- run setup_arm_gcc() first.")
    if not generator.exists():
        raise RuntimeError(f"setup_nsx_toolchain: {generator} does not exist -- run setup_nsx_ambiq_sdk() first.")

    toolchain_file.parent.mkdir(parents=True, exist_ok=True)
    print("Generating NSX arm-none-eabi-gcc toolchain file...")
    run_command(
        [
            sys.executable, str(generator),
            "--toolchain-family", "gcc",
            "--gcc-root", str(gcc_dir.resolve()),
            "--output", str(toolchain_file),
        ],
        cwd=project_root,
        description="Generating NSX toolchain file"
    )
    # Not emitted by nsx_toolchain_file.py itself: nsx-ambiq-sdk's own
    # cmake/nsx_toolchain_flags.cmake requires this to resolve prebuilt AmbiqSuite
    # artifact/library paths and toolchain-specific compile/link flags.
    with toolchain_file.open("a", encoding="utf-8") as f:
        f.write(
            "\n"
            "# Appended (not generated by nsx_toolchain_file.py): required by\n"
            "# nsx-ambiq-sdk's cmake/nsx_toolchain_flags.cmake.\n"
            'set(NSX_TOOLCHAIN_FAMILY "gcc")\n'
        )

    print("NSX toolchain file generated")


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
        """
    )
    
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path("artifacts/downloads"),
        help="Directory to store downloaded dependencies (default: artifacts/downloads)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and reinstall all dependencies"
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
        "--skip-python",
        action="store_true",
        help="Skip Python virtual environment setup"
    )
    parser.add_argument(
        "--with-hardware",
        action="store_true",
        help="Also fetch the real Apollo510/Cortex-M55 hardware-build dependencies "
        "(nsx-ambiq-sdk, neuralspotx, generated NSX toolchain file). Opt-in: nsx-ambiq-sdk "
        "is a large monorepo most FVP-only development doesn't need. `helia_core_tester "
        "perf-stream flash/build-firmware/run-generated` also fetch these lazily on first "
        "use if missing, so this flag is only needed to pre-fetch them ahead of time."
    )

    args = parser.parse_args()
    
    # Validate system
    try:
        get_os()
        get_architecture()
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    
    # Create downloads directory
    args.downloads_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Setting up CMSIS-NN build dependencies")
    print(f"Downloads directory: {args.downloads_dir}")
    print(f"Force mode: {args.force}")
    print("=" * 80)
    
    try:
        if not args.skip_corstone:
            setup_corstone300(args.downloads_dir, args.force)
            print()
        
        if not args.skip_gcc:
            setup_arm_gcc(args.downloads_dir, args.force)
            print()
        
        if not args.skip_cmsis5:
            setup_cmsis5(args.downloads_dir, args.force)
            print()
        
        if not args.skip_ethos:
            setup_ethos_u_platform(args.downloads_dir, args.force)
            print()
        
        if not args.skip_python:
            setup_python_venv(args.downloads_dir, args.force)
            print()

        if args.with_hardware:
            project_root = find_repo_root()
            setup_nsx_ambiq_sdk(project_root, args.force)
            print()
            setup_neuralspotx(args.downloads_dir, args.force)
            print()
            setup_nsx_toolchain(project_root, args.downloads_dir, args.force)
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
