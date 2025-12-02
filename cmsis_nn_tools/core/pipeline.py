"""
Main pipeline orchestration for CMSIS-NN Tools.
"""

import subprocess
import time

from .config import Config
from .logger import get_logger
from ..utils.command_runner import run_command


class FullTestPipeline:
    """Main pipeline class for CMSIS-NN testing workflow."""
    
    def __init__(self, config: Config):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def _get_platform_name(self, cpu: str) -> str:
        """
        Map CPU type to helia-aot platform name.
        
        Args:
            cpu: CPU type (e.g., 'cortex-m4', 'cortex-m55')
            
        Returns:
            Platform name for helia-aot
        """
        platform_mapping = {
            'cortex-m4': 'apollo4l_evb',
            'cortex-m55': 'apollo510_evb',
        }
        return platform_mapping.get(cpu, 'apollo510_evb')  # Default to apollo510_evb
    
    def run(self) -> bool:
        """
        Run the complete test pipeline.
        
        Returns:
            True if all steps succeeded, False otherwise
        """
        self.logger.info("Starting CMSIS-NN Full Test Pipeline")
        
        if self.config.dry_run:
            self.logger.warning("DRY RUN MODE - No actual changes will be made")
        
        start_time = time.time()
        overall_success = True
        
        try:
            # Step 1: Generate TFLite models
            if not self.config.skip_generation:
                success = self._step1_generate_tflite_models()
                if not success:
                    overall_success = False
                    if self.config.fail_fast:
                        self.logger.error("Stopping due to failure in TFLite generation")
                        return False
            else:
                self.logger.info("Skipping TFLite model generation")
            
            # Step 2: Convert TFLite models
            if not self.config.skip_conversion and overall_success:
                success = self._step2_convert_tflite_models()
                if not success:
                    overall_success = False
                    if self.config.fail_fast:
                        self.logger.error("Stopping due to failure in TFLite conversion")
                        return False
            else:
                self.logger.info("Skipping TFLite to C conversion")
            
            # Step 3: Generate test runners (first attempt)
            if not self.config.skip_runners and overall_success:
                success = self._step3_generate_test_runners()
                if not success:
                    overall_success = False
                    if self.config.fail_fast:
                        self.logger.error("Stopping due to failure in test runner generation")
                        return False
            else:
                self.logger.info("Skipping test runner generation")
            
            # Step 3.5: Generate test runners after conversion (if needed)
            if not self.config.skip_runners and overall_success and not self.config.skip_conversion:
                success = self._step3_5_generate_test_runners_after_conversion()
                if not success:
                    overall_success = False
                    if self.config.fail_fast:
                        self.logger.error("Stopping due to failure in test runner generation")
                        return False
            
            # Step 4: Build for FVP
            if not self.config.skip_build and overall_success:
                success = self._step4_build_fvp()
                if not success:
                    overall_success = False
                    if self.config.fail_fast:
                        self.logger.error("Stopping due to failure in FVP build")
                        return False
            else:
                self.logger.info("Skipping FVP build")
            
            # Step 5: Run tests
            if not self.config.skip_run and overall_success:
                success = self._step5_run_tests()
                if not success:
                    overall_success = False
            else:
                self.logger.info("Skipping FVP test execution")
            
            # Print summary
            end_time = time.time()
            duration = end_time - start_time
            
            if overall_success:
                self.logger.info(f"Pipeline completed successfully in {duration:.1f} seconds")
                return True
            else:
                self.logger.error(f"Pipeline failed after {duration:.1f} seconds")
                return False
                
        except Exception as e:
            self.logger.error(f"Pipeline failed with exception: {e}")
            return False
    
    def _step1_generate_tflite_models(self) -> bool:
        """Step 1: Generate TFLite models."""
        self.logger.info("Step 1/5: Generate TFLite Models")
        self.logger.info("Generating TensorFlow Lite models using pytest in tflite generator")
        
        if not self.config.tflite_generator_dir.exists():
            self.logger.error(f"tflite generator directory not found: {self.config.tflite_generator_dir.absolute()}")
            return False
        
        # Build pytest command
        cmd = ["pytest", "test_ops.py::test_generation", "-v"]
        
        # Add filters
        if self.config.op_filter:
            cmd.extend(["--op", self.config.op_filter])
        if self.config.dtype_filter:
            cmd.extend(["--dtype", self.config.dtype_filter])
        if self.config.limit:
            cmd.extend(["--limit", str(self.config.limit)])
        
        try:
            run_command(cmd, cwd=self.config.tflite_generator_dir, verbose=self.config.verbose)
            self.logger.info("TFLite models generated successfully")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error(f"Failed to generate TFLite models: {e}")
            return False
    
    def _step2_convert_tflite_models(self) -> bool:
        """Step 2: Convert TFLite models to C modules."""
        self.logger.info("Step 2/5: Convert TFLite to C Modules")
        self.logger.info("Converting TensorFlow Lite models to standalone C inference modules")
        
        if not self.config.generated_tests_dir.exists():
            self.logger.error(f"Generated tests directory not found: {self.config.generated_tests_dir}")
            return False
        
        # Find all TFLite files
        tflite_files = list(self.config.generated_tests_dir.rglob("*.tflite"))
        if not tflite_files:
            self.logger.error(f"No TFLite files found in {self.config.generated_tests_dir}")
            return False
        
        self.logger.info(f"Found {len(tflite_files)} TFLite files to convert")
        
        if self.config.dry_run:
            self.logger.info("Dry run mode - would convert the following files:")
            for tflite_file in tflite_files:
                self.logger.info(f"  - {tflite_file}")
            return True

        platform_name = self._get_platform_name(self.config.cpu)
        self.logger.info(f"Using platform: {platform_name} for CPU: {self.config.cpu}")
    
        # Convert each TFLite file
        success_count = 0
        for tflite_file in tflite_files:
            # Determine output directory (same as TFLite file directory)
            output_dir = tflite_file.parent
            
            # Extract module name from filename
            module_name = tflite_file.stem
            
            # Build helia-aot command (use Python module from modules directory)
            helia_aot_path = self.config.project_root / "modules" / "helia-aot"
            cmd = [
                "python3", "-m", "helia_aot.cli.main",
                "convert",
                "--model.path", str(tflite_file),
                "--module.path", str(output_dir),
                "--module.name", module_name,
                "--module.prefix", module_name,
                "--platform.name",  platform_name,
                "--test.enabled",    
            ]
            
            try:
                if self.config.verbose:
                    run_command(cmd, verbose=True, cwd=helia_aot_path)
                else:
                    run_command(cmd, capture_output=True, cwd=helia_aot_path)
                self.logger.info(f"Converted: {module_name}")
                success_count += 1
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                self.logger.error(f"Failed to convert {module_name}: {e}")
                continue
        
        if success_count == 0:
            self.logger.error("No TFLite files were successfully converted")
            return False
        
        self.logger.info(f"Successfully converted {success_count}/{len(tflite_files)} TFLite files")
        return True
    
    def _step3_generate_test_runners(self) -> bool:
        """Step 3: Generate Unity test runners."""
        self.logger.info("Step 3/5: Generate Test Runners")
        self.logger.info("Generating Unity test runners for all test directories")
        
        script_path = self.config.project_root / "cmsis_nn_tools" / "scripts" / "generate_test_runners.py"
        if not script_path.exists():
            self.logger.error(f"Test runner script not found: {script_path}")
            return False
        
        # Check if there are any model headers to generate runners for
        model_headers = list(self.config.generated_tests_dir.rglob("includes-api/*_model.h"))
        if not model_headers:
            self.logger.warning("No model headers found - test runners will be generated after TFLite conversion")
            self.logger.info("This step will be skipped for now and run after conversion")
            return True
        
        # Build command
        cmd = ["python3", str(script_path), "--root", str(self.config.generated_tests_dir)]
        if self.config.verbose:
            cmd.append("--verbose")
        if self.config.dry_run:
            cmd.append("--dry-run")
        
        try:
            run_command(cmd, verbose=self.config.verbose)
            self.logger.info("Test runners generated successfully")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error(f"Failed to generate test runners: {e}")
            return False
    
    def _step3_5_generate_test_runners_after_conversion(self) -> bool:
        """Step 3.5: Generate test runners after conversion if needed."""
        # Check if we need to generate runners after conversion
        model_headers = list(self.config.generated_tests_dir.rglob("includes-api/*_model.h"))
        if not model_headers:
            self.logger.info("Generating test runners after TFLite conversion...")
            return self._step3_generate_test_runners()
        return True
    
    def _step4_build_fvp(self) -> bool:
        """Step 4: Build for FVP."""
        self.logger.info("Step 4/5: Build for FVP")
        
        if self.config.dry_run:
            self.logger.info("Dry run mode - would build for FVP")
            return True
        
        # Use the existing build_and_run_fvp.py script but only build
        script_path = self.config.project_root / "cmsis_nn_tools" / "build_and_run_fvp.py"
        if not script_path.exists():
            self.logger.error(f"Build script not found: {script_path}")
            return False
        
        # Build command (only build, don't run)
        # Use CMake definition instead of --opt to avoid argument parsing issues
        cmd = [
            "python3", str(script_path),
            "--cpu", self.config.cpu,
            "--cmake-def", f"CMSIS_OPTIMIZATION_LEVEL={self.config.optimization}",
            "--no-run",  # Only build, don't run
        ]
        
        if self.config.jobs:
            cmd.extend(["--jobs", str(self.config.jobs)])
        if not self.config.build_verbose:
            cmd.append("--quiet")
        
        try:
            run_command(cmd, verbose=self.config.verbose)
            self.logger.info(f"Successfully built for {self.config.cpu}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to build for FVP (exit code {e.returncode})")
            # Re-run with output capture to get error details
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.stdout:
                    self.logger.error(f"stdout: {result.stdout}")
                if result.stderr:
                    self.logger.error(f"stderr: {result.stderr}")
            except Exception:
                pass  # Ignore errors in error reporting
            return False
        except FileNotFoundError as e:
            self.logger.error(f"Failed to build for FVP: {e}")
            return False
    
    def _step5_run_tests(self) -> bool:
        """Step 5: Run tests on FVP."""
        self.logger.info("Step 5/5: Run Tests on FVP")
        
        if self.config.dry_run:
            self.logger.info("Dry run mode - would run tests on FVP")
            return True
        
        # Use the existing build_and_run_fvp.py script but only run
        script_path = self.config.project_root / "cmsis_nn_tools" / "build_and_run_fvp.py"
        if not script_path.exists():
            self.logger.error(f"Build script not found: {script_path}")
            return False
        
        # Build command (only run, don't build)
        cmd = [
            "python3", str(script_path),
            "--cpu", self.config.cpu,
            "--no-build",  # Only run, don't build
        ]
        
        if self.config.timeout > 0:
            cmd.extend(["--timeout-run", str(self.config.timeout)])
        if not self.config.fail_fast:
            cmd.append("--no-fail-fast")
        
        # Reporting configuration (enabled by default, use --no-report to disable)
        if hasattr(self.config, 'enable_reporting'):
            if not self.config.enable_reporting:
                cmd.append("--no-report")
            if hasattr(self.config, 'report_formats') and self.config.report_formats:
                cmd.extend(["--report-formats"] + self.config.report_formats)
            if hasattr(self.config, 'report_dir') and self.config.report_dir:
                cmd.extend(["--report-dir", str(self.config.report_dir)])
        
        # Always show real-time output for test runs (don't use --quiet)
        
        try:
            # Run with real-time output streaming
            self.logger.info("Running tests on FVP (real-time output):")
            self.logger.info("=" * 60)
            
            subprocess.run(
                cmd,
                check=True,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            self.logger.info("=" * 60)
            self.logger.info("All tests completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error("=" * 60)
            self.logger.error(f"Some tests failed (exit code: {e.returncode})")
            return False
        except FileNotFoundError as e:
            self.logger.error(f"Failed to run tests: {e}")
            return False
