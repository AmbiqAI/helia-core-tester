"""
FVP test execution step.
"""

import subprocess
import sys
from typing import Optional

from helia_core_tester.core.runtime_env import RuntimeEnvContext, bootstrap_runtime_env, build_locked_fvp_flags
from helia_core_tester.core.steps.base import StepBase, StepPlan, StepResult, StepStatus
from helia_core_tester.core.errors import FVPRunError
from helia_core_tester.core.logging import get_logger
from helia_core_tester.core.discovery import find_fvp_script_path


class RunStep(StepBase):
    """Step for running tests on FVP."""
    
    def __init__(self, config, runtime_env: Optional[RuntimeEnvContext] = None):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.runtime_env = runtime_env

    def _ensure_runtime_env(self) -> RuntimeEnvContext:
        if self.runtime_env is None:
            self.runtime_env = bootstrap_runtime_env(
                downloads_dir=self.config.downloads_dir,
                ensure_setup=True,
                toolchain=self.config.toolchain,
            )
        return self.runtime_env

    def _locked_fvp_flags(self) -> list[str]:
        return build_locked_fvp_flags(self.runtime_env, self.config.downloads_dir, self.config.toolchain)
    
    @property
    def name(self) -> str:
        return "run"
    
    def should_skip(self) -> bool:
        """Check if test execution should be skipped."""
        return self.config.skip_run
    
    def validate(self) -> str | None:
        """Validate prerequisites for test execution."""
        script_path = find_fvp_script_path(self.config.project_root)
        if not script_path.exists():
            return f"FVP script not found: {script_path}"
        missing_builds = []
        for suite in self.config.suites:
            for group in self.config.cpu_groups_for_suite(suite):
                for cpu in group["cpus"]:
                    build_dir = self.config.build_dir_for(cpu, suite=suite)
                    if not build_dir.exists():
                        missing_builds.append(str(build_dir))
        if missing_builds:
            return f"Build directory not found: {', '.join(missing_builds)}. Run 'build' step first."
        return None

    def plan_validate(self) -> str | None:
        """Lenient validation for plan mode."""
        script_path = find_fvp_script_path(self.config.project_root)
        if not script_path.exists():
            return f"FVP script not found: {script_path}"
        return None

    def _run_commands(self) -> list[list[str]]:
        commands: list[list[str]] = []
        for suite in self.config.suites:
            for group in self.config.cpu_groups_for_suite(suite):
                cpu_csv = ",".join(group["cpus"])
                cmd = [
                    sys.executable,
                    "-m",
                    "helia_core_tester.fvp.build_and_run_fvp",
                    *self._locked_fvp_flags(),
                    "--cpu",
                    cpu_csv,
                    "--suite",
                    suite,
                    "--no-build",
                ]
                if getattr(self.config, "coverage", False):
                    cmd.append("--coverage")
                    if suite == "float" and getattr(self.config, "coverage_mve_float", False):
                        cmd.extend(["--coverage-report-suite", "float-mve"])

                # Always forwarded, 0 included: the child's own default is
                # non-zero, so withholding the flag would turn the explicit
                # opt-out back into the default.
                cmd.extend(["--timeout-run", str(self.config.timeout)])
                if self.config.fail_fast:
                    cmd.append("--fail-fast")
                else:
                    cmd.append("--no-fail-fast")
                cmd.extend(["--run-jobs", str(self.config.run_jobs)])

                # Pass verbosity
                cmd.extend(["--verbosity", str(self.config.verbosity)])

                # Reporting options
                if not self.config.enable_reporting:
                    cmd.append("--no-report")
                if self.config.report_formats:
                    cmd.extend(["--report-formats"] + self.config.report_formats)
                commands.append(cmd)
        return commands

    def _do_execute(self) -> StepResult:
        """Execute FVP test execution."""
        if self.config.verbosity >= 1:
            self.logger.info(f"Running tests on FVP suites={','.join(self.config.suites)} cpus={','.join(self.config.cpus)}")
        commands = self._run_commands()
        
        try:
            runtime_env = self._ensure_runtime_env()
            commands = self._run_commands()
            for cmd in commands:
                if self.config.verbosity >= 2:
                    self.logger.info(f"Running command: {' '.join(cmd)}")
                    self.logger.info("=" * 60)

                # Use subprocess.run directly for better output handling
                subprocess.run(
                    cmd,
                    cwd=self.config.project_root,
                    check=True,
                    text=True,
                    bufsize=1,
                    env=runtime_env.child_env,
                )
            
            if self.config.verbosity >= 2:
                self.logger.info("=" * 60)
            
            if self.config.verbosity >= 1:
                self.logger.info("All tests completed successfully")
            
            return StepResult(
                name=self.name,
                status=StepStatus.SUCCESS,
                message="All tests completed successfully",
                outputs={
                    "build_dir": str(self.config.project_root / "artifacts"),
                    "reports_root": str(self.config.reports_root),
                },
                details={"commands": commands},
            )
        except subprocess.CalledProcessError as e:
            if self.config.verbosity >= 2:
                self.logger.error("=" * 60)
            
            error_msg = f"Some tests failed (exit code: {e.returncode})"
            self.logger.error(error_msg)
            
            fvp_error = FVPRunError(error_msg)
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                message=error_msg,
                error=fvp_error,
                outputs={
                    "build_dir": str(self.config.project_root / "artifacts"),
                    "reports_root": str(self.config.reports_root),
                },
                details={"commands": commands},
            )
        except FileNotFoundError as e:
            error_msg = f"Failed to run tests: {e}"
            self.logger.error(error_msg)
            fvp_error = FVPRunError(error_msg)
            fvp_error.__cause__ = e
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                message=error_msg,
                error=fvp_error,
                outputs={
                    "build_dir": str(self.config.project_root / "artifacts"),
                    "reports_root": str(self.config.reports_root),
                },
                details={"commands": commands},
            )

    def dry_run(self) -> StepResult:
        """Dry run of run step."""
        cmd_preview = self._run_commands()
        
        return StepResult(
            name=self.name,
            status=StepStatus.SKIPPED,
            message=f"DRY RUN: Would run {len(cmd_preview)} FVP command(s)",
            outputs={
                "build_dir": str(self.config.project_root / "artifacts"),
                "reports_root": str(self.config.reports_root),
            },
            details={"commands": cmd_preview},
        )

    def _plan_details(self) -> StepPlan:
        commands = self._run_commands()
        return StepPlan(
            name=self.name,
            will_run=True,
            reason="ready",
            commands=commands,
            outputs={
                "build_dir": str(self.config.project_root / "artifacts"),
                "reports_root": str(self.config.reports_root),
            }
        )
