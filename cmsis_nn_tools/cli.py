"""
Command-line interface for CMSIS-NN Tools.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Allow running this file directly without packaging by setting up sys.path
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Prefer absolute imports; fall back to local when run as a script
try:
    from cmsis_nn_tools.core.config import Config
    from cmsis_nn_tools.core.logger import setup_logger
    from cmsis_nn_tools.core.pipeline import FullTestPipeline
except Exception:  # noqa: BLE001 - broad to allow script-mode fallback
    from core.config import Config
    from core.logger import setup_logger
    from core.pipeline import FullTestPipeline


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="CMSIS-NN Tools - Complete testing and simulation toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  cmsis-nn-tools

  # Run with specific filters
  cmsis-nn-tools --op Conv2D --dtype S8

  # Skip generation, only build and run
  cmsis-nn-tools --skip-generation

  # Dry run to see what would be done
  cmsis-nn-tools --dry-run

  # Verbose output with custom CPU
  cmsis-nn-tools --cpu cortex-m3 --verbose
        """
    )
    
    # Pipeline control
    parser.add_argument("--skip-generation", action="store_true",
                       help="Skip TFLite model generation (assume models exist)")
    parser.add_argument("--skip-conversion", action="store_true",
                       help="Skip TFLite to C conversion (assume C modules exist)")
    parser.add_argument("--skip-runners", action="store_true",
                       help="Skip test runner generation (assume runners exist)")
    parser.add_argument("--skip-build", action="store_true",
                       help="Skip FVP build (assume binaries exist)")
    parser.add_argument("--skip-run", action="store_true",
                       help="Skip FVP test execution")
    
    # Generation filters
    parser.add_argument("--op", type=str, default=None,
                       help="Generate only specific operator (e.g., 'Conv2D')")
    parser.add_argument("--dtype", type=str, default=None,
                       help="Generate only specific dtype (e.g., 'S8', 'S16')")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of models to generate")
    
    # Build options
    parser.add_argument("--cpu", type=str, default="cortex-m55",
                       help="Target CPU (default: cortex-m55)")
    parser.add_argument("--opt", type=str, default="-Ofast",
                       help="Optimization level (default: -Ofast)")
    parser.add_argument("--jobs", type=int, default=None,
                       help="Parallel build jobs (default: auto)")
    parser.add_argument("--quiet-build", action="store_true",
                       help="Reduce FVP build verbosity (default: full verbosity)")
    
    # Run options
    parser.add_argument("--timeout", type=float, default=0.0,
                       help="Per-test timeout in seconds (0 = none)")
    parser.add_argument("--no-fail-fast", action="store_true",
                       help="Don't stop on first test failure")
    
    # Reporting options
    parser.add_argument("--no-report", action="store_true",
                       help="Disable comprehensive test reporting (enabled by default)")
    parser.add_argument("--report-formats", nargs="+", 
                       choices=["json", "html", "md"], default=["json"],
                       help="Report formats to generate (default: json)")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"),
                       help="Directory to save reports (default: reports)")
    parser.add_argument("--show-latest-report", action="store_true",
                       help="Show the latest test report summary")
    
    # General options
    parser.add_argument("--setup", action="store_true",
                       help="Initialize git submodules and install Python dependencies")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without actually doing it")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Reduce output verbosity")
    parser.add_argument("--log-file", type=Path,
                       help="Log file path")
    
    # Path options
    parser.add_argument("--project-root", type=Path,
                       help="Project root directory")
    parser.add_argument("--downloads-dir", type=Path,
                       help="Downloads directory")
    parser.add_argument("--generated-tests-dir", type=Path,
                       help="Generated tests directory")
    parser.add_argument("--tflite-generator-dir", type=Path,
                       help="TFLite generator directory")
    
    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup: Initialize submodules and install dependencies (only if --setup is passed)
    if args.setup:
        repo_root = Path(__file__).resolve().parent.parent
        
        print("Initializing git submodules...")
        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=repo_root,
                check=True
            )
            print("Git submodules initialized successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to initialize git submodules: {e}", file=sys.stderr)
            # Continue anyway - submodules might already be initialized
        except FileNotFoundError:
            print("Warning: git not found, skipping submodule initialization", file=sys.stderr)
        
        print("Installing Python dependencies...")
        requirements_file = repo_root / "requirements.txt"
        if requirements_file.exists():
            try:
                subprocess.run(
                    ["python3", "-m", "pip", "install", "-r", str(requirements_file)],
                    cwd=repo_root,
                    check=True
                )
                print("Python dependencies installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error: Failed to install dependencies: {e}", file=sys.stderr)
                return 1
        else:
            print(f"Warning: requirements.txt not found at {requirements_file}", file=sys.stderr)
    
    # Create configuration
    config = Config()
    
    # Override with command-line arguments
    if args.project_root:
        config.project_root = args.project_root
    if args.downloads_dir:
        config.downloads_dir = args.downloads_dir
    if args.generated_tests_dir:
        config.generated_tests_dir = args.generated_tests_dir
    if args.tflite_generator_dir:
        config.tflite_generator_dir = args.tflite_generator_dir
    
    # Set other options
    config.cpu = args.cpu
    config.optimization = args.opt
    config.jobs = args.jobs
    config.timeout = args.timeout
    config.fail_fast = not args.no_fail_fast
    config.verbose = args.verbose and not args.quiet
    config.build_verbose = not args.quiet_build  # Default: True (full verbosity)
    config.dry_run = args.dry_run
    config.op_filter = args.op
    config.dtype_filter = args.dtype
    config.limit = args.limit
    config.skip_generation = args.skip_generation
    config.skip_conversion = args.skip_conversion
    config.skip_runners = args.skip_runners
    config.skip_build = args.skip_build
    config.skip_run = args.skip_run
    
    # Reporting configuration
    config.enable_reporting = not args.no_report
    config.report_formats = args.report_formats
    config.report_dir = args.report_dir
    
    # Handle show-latest-report
    if args.show_latest_report:
        try:
            from .reporting.storage import ReportStorage
        except ImportError:
            # Fallback for when running as standalone script
            import sys
            sys.path.append(str(Path(__file__).parent))
            from reporting.storage import ReportStorage
        storage = ReportStorage(config.report_dir)
        latest_report = storage.get_latest_report(config.cpu)
        
        if latest_report:
            print(f"\nLatest Test Report for {config.cpu}")
            print("=" * 50)
            print(f"Run ID: {latest_report.run_id}")
            print(f"Start Time: {latest_report.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Duration: {latest_report.duration:.2f} seconds")
            print(f"Summary: {latest_report.summary}")
            print()
            
            # Show failed tests if any
            failed_tests = latest_report.get_failed_tests()
            if failed_tests:
                print("Failed Tests:")
                for test in failed_tests:
                    print(f"  - {test.test_name}: {test.failure_reason}")
                print()
            
            return 0
        else:
            print(f"No reports found for CPU: {config.cpu}")
            return 1
    
    # Setup logging
    logger = setup_logger(
        level=20 if args.quiet else 10,  # WARNING if quiet, INFO otherwise
        log_file=args.log_file,
        verbose=config.verbose
    )
    
    # Create and run pipeline
    pipeline = FullTestPipeline(config)
    success = pipeline.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
