"""
Pytest configuration and fixtures for Helia-Core Tester.
"""

import pytest
import os
import shutil
import sys
from pathlib import Path

from helia_core_tester.core.discovery import find_generated_tests_dir


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--op", action="store", default=None,
                    help="Filter by operator (e.g., FullyConnected)")
    parser.addoption("--dtype", action="store", default=None,
                    help="Filter by activation dtype (S8, S16)")
    parser.addoption("--wtype", action="store", default=None,
                    help="Filter by weight dtype (S8, S4)")
    parser.addoption("--name", action="store", default=None,
                    help="Filter by exact test name")
    parser.addoption("--limit", action="store", type=int, default=None,
                    help="Limit number of tests to run")
    parser.addoption("--seed", action="store", type=int, default=None,
                    help="Random seed for test generation (default: hash of test name)")
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


def pytest_configure(config):
    """Configure pytest with custom options."""
    generated_override = config.getoption("--generated-tests-dir")
    target_cpu = config.getoption("--cpu") or "cortex-m55"
    target_suite = config.getoption("--suite") or "int"
    generated_tests_dir = (
        Path(generated_override).resolve()
        if generated_override
        else find_generated_tests_dir(cpu=target_cpu, suite=target_suite, create=False)
    )

    # Without --force-generate the tree is the reuse cache: cases still matching
    # their stamp are kept and the run prunes whatever falls outside the active
    # filter (see generation/reuse.py). Only a forced run starts from empty.
    if not config.getoption("--force-generate"):
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
        'cpu': request.config.getoption("--cpu"),
        'suite': request.config.getoption("--suite"),
        'float_precision': request.config.getoption("--float-precision"),
        'generated_tests_dir': request.config.getoption("--generated-tests-dir"),
        'force_generate': request.config.getoption("--force-generate"),
    }
