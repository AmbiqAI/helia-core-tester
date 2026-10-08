"""
Pytest configuration and fixtures for Helia-Core Tester.
"""

import pytest
import os
import shutil
import sys
from pathlib import Path

from helia_core_tester.core.cpu_targets import normalize_cpu
from helia_core_tester.core.discovery import find_generated_tests_dir, find_repo_root
from helia_core_tester.core.path_layout import generated_tests_dir


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--op", action="store", default=None,
                    help="Filter by operators, comma-separated (e.g., FullyConnected)")
    parser.addoption("--dtype", action="store", default=None,
                    help="Filter by case dtype: activations, or S4 weights")
    parser.addoption("--wtype", action="store", default=None,
                    help="Filter by weight dtype (S8, S4)")
    parser.addoption("--name", action="store", default=None,
                    help="Filter by exact test names, comma-separated")
    parser.addoption("--limit", action="store", type=int, default=None,
                    help="Limit number of tests to run")
    parser.addoption("--seed", action="store", type=int, default=None,
                    help="Run seed every case's draw derives from (default: HCT_SEED, else a fresh draw)")
    parser.addoption("--fresh-seed", action="store_true", default=False,
                    help="The --seed was drawn by the pipeline, not chosen: cases are not reused")
    parser.addoption("--cpu", action="store", default="cortex-m55",
                    help="Target CPU for code generation")
    parser.addoption("--generated-tests-dir", action="store", default=None,
                    help="Override generated tests output directory")
    parser.addoption("--suite", action="store", default="int",
                    help="Suite selection: int or float")
    parser.addoption("--float-precision", action="store", default="both",
                    help="Float precision for float suite: f16, f32, or both")
    parser.addoption("--force-generate", action="store_true", default=False,
                    help="Regenerate every case, ignoring reuse stamps")
    parser.addoption("--keep-unselected", action="store_true", default=False,
                    help="Keep cases outside the filter instead of pruning")
    parser.addoption("--random-shapes", action="store", type=int, default=None,
                    help="Draw N random shapes per op instead")
    parser.addoption("--shape-seed", action="store", type=int, default=None,
                    help="Seed for --random-shapes")
    parser.addoption("--hidden-dir", action="store", default=None,
                    help="Write secret-seeded shapes here instead")


def _keeps_unselected(config) -> bool:
    """Public random shapes join the fixed tree."""
    if config.getoption("--hidden-dir"):
        # Hidden trees hold one draw.
        return False
    return bool(config.getoption("--keep-unselected") or config.getoption("--random-shapes"))


def _generated_override(config):
    """Explicit output dir, else the hidden tree."""
    hidden = config.getoption("--hidden-dir")
    if not hidden:
        return config.getoption("--generated-tests-dir")
    cpu = normalize_cpu(config.getoption("--cpu") or "cortex-m55")
    return str(generated_tests_dir(Path(hidden), cpu, suite=config.getoption("--suite") or "int"))


def _guard_hidden(config) -> None:
    """Keep hidden output and secrets contained."""
    hidden = config.getoption("--hidden-dir")
    if not hidden:
        return
    # Verbose tracebacks print the seed.
    config.option.tbstyle = "native"
    config.option.showlocals = False
    config.option.fulltrace = False
    from helia_core_tester.generation.random_shapes import check_hidden_paths

    # Else public cases replace the draw.
    if (config.getoption("--random-shapes") or 0) < 1:
        raise pytest.UsageError("--hidden-dir needs --random-shapes N >= 1")
    if config.getoption("--shape-seed") is not None:
        raise pytest.UsageError("--hidden-dir takes a secret, not --shape-seed")

    # Hidden outputs all derive from DIR.
    if config.getoption("--generated-tests-dir"):
        raise pytest.UsageError("--hidden-dir takes no --generated-tests-dir")
    try:
        check_hidden_paths(
            Path(hidden), find_repo_root(), config.getoption("--cpu") or "cortex-m55",
            config.getoption("--suite") or "int",
        )
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc


def pytest_configure(config):
    """Configure pytest with custom options."""
    _guard_hidden(config)
    generated_override = _generated_override(config)
    target_cpu = config.getoption("--cpu") or "cortex-m55"
    target_suite = config.getoption("--suite") or "int"
    generated_tests_dir = (
        Path(generated_override).resolve()
        if generated_override
        else find_generated_tests_dir(cpu=target_cpu, suite=target_suite, create=False)
    )

    # Without --force-generate the tree is the reuse cache: cases still matching
    # their stamp are kept and the run prunes whatever falls outside the active
    # filter (see generation/reuse.py). Only a forced run starts from empty,
    # unless --keep-unselected keeps other cases.
    keep = _keeps_unselected(config)
    if not config.getoption("--force-generate") or keep:
        generated_tests_dir.mkdir(parents=True, exist_ok=True)
        print("Reusing generated tests directory (stamp-checked per case)")
        return

    # Clean generated tests directory before running
    if generated_tests_dir.exists():
        print(f"\nCleaning existing generated tests directory...")
        try:
            # Count existing files before deletion
            existing_count = sum(1 for _ in generated_tests_dir.rglob("*.tflite"))
            if existing_count > 0:
                print(f"   Removing {existing_count} existing TFLite model(s)")
            
            shutil.rmtree(generated_tests_dir)
            print(f"Directory cleaned")
        except OSError as e:
            print(f"Warning: Could not remove entire directory, trying individual files...")
            # If rmtree fails, try to remove individual files
            for item in generated_tests_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            print(f"   Individual files removed")
    
    # Create fresh directory
    generated_tests_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created generated tests directory\n")


@pytest.fixture
def test_filters(request):
    """Provide test filters from command line options."""
    return {
        'op': request.config.getoption("--op"),
        'dtype': request.config.getoption("--dtype"),
        'wtype': request.config.getoption("--wtype"),
        'name': request.config.getoption("--name"),
        'limit': request.config.getoption("--limit"),
        'seed': request.config.getoption("--seed"),
        'fresh_seed': request.config.getoption("--fresh-seed"),
        'cpu': request.config.getoption("--cpu"),
        'suite': request.config.getoption("--suite"),
        'float_precision': request.config.getoption("--float-precision"),
        'generated_tests_dir': _generated_override(request.config),
        'force_generate': request.config.getoption("--force-generate"),
        'keep_unselected': _keeps_unselected(request.config),
        'random_shapes': request.config.getoption("--random-shapes"),
        'shape_seed': request.config.getoption("--shape-seed"),
        'hidden_dir': request.config.getoption("--hidden-dir"),
    }
