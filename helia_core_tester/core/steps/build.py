"""
CMake build step for FVP.
"""

import subprocess
import sys
from typing import Optional

from helia_core_tester.core.runtime_env import RuntimeEnvContext, bootstrap_runtime_env, build_locked_fvp_flags
from helia_core_tester.core.steps.base import StepBase, StepPlan, StepResult, StepStatus
from helia_core_tester.core.errors import BuildError
from helia_core_tester.core.logging import get_logger
from helia_core_tester.core.discovery import find_fvp_script_path
from helia_core_tester.utils.command_runner import run_command


class BuildStep(StepBase):
    """Step for building FVP executables."""
    
    def __init__(self, config, runtime_env: Optional[RuntimeEnvContext] = None):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.runtime_env = runtime_env

    def _ensure_runtime_env(self) -> RuntimeEnvContext:
        if self.runtime_env is None:
            self.runtime_env = bootstrap_runtime_env(
                downloads_dir=self.config.downloads_dir,
                ensure_setup=True,
            )
        return self.runtime_env

    def _locked_fvp_flags(self) -> list[str]:
        return build_locked_fvp_flags(self.runtime_env, self.config.downloads_dir)
    
    @property
    def name(self) -> str:
        return "build"
    
    def should_skip(self) -> bool:
        """Check if build should be skipped."""
        return self.config.skip_build
    
    def validate(self) -> str | None:
        """Validate prerequisites for build."""
        script_path = find_fvp_script_path(self.config.project_root)
        if not script_path.exists():
            return f"FVP script not found: {script_path}"
        return None

    def _build_commands(self) -> list[list[str]]:
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
                    "--cmake-def",
                    f"CMSIS_OPTIMIZATION_LEVEL={self.config.optimization}",
                    "--no-run",
                    "--no-report",
                ]

                if suite == "float":
                    float_precision = group.get("float_precision")
                    if float_precision in {"both", "f32"}:
                        cmd.extend(["--cmake-def", "ARM_NN_ENABLE_F32=ON"])
                    else:
                        cmd.extend(["--cmake-def", "ARM_NN_ENABLE_F32=OFF"])
                    if float_precision in {"both", "f16"}:
                        cmd.extend(["--cmake-def", "ARM_NN_ENABLE_F16=ON"])
                    else:
                        cmd.extend(["--cmake-def", "ARM_NN_ENABLE_F16=OFF"])
                    if getattr(self.config, "coverage_mve_float", False):
                        cmd.extend(["--cmake-def", "ENABLE_COVERAGE_MVE_FLOAT=ON"])
                else:
                    cmd.extend(["--cmake-def", "ARM_NN_ENABLE_F32=OFF"])
                    cmd.extend(["--cmake-def", "ARM_NN_ENABLE_F16=OFF"])

                if getattr(self.config, "coverage", False):
                    cmd.append("--coverage")
                if self.config.jobs:
                    cmd.extend(["--jobs", str(self.config.jobs)])
                if self.config.verbosity <= 1:
                    cmd.append("--quiet")
                else:
                    cmd.extend(["--verbosity", str(self.config.verbosity)])
                commands.append(cmd)

        return commands

    def _do_execute(self) -> StepResult:
        """Execute CMake build for FVP."""
        if self.config.verbosity >= 1:
            self.logger.info(f"Building for FVP suites={','.join(self.config.suites)} cpus={','.join(self.config.cpus)}")
        commands = self._build_commands()
        
        try:
            runtime_env = self._ensure_runtime_env()
            commands = self._build_commands()
            for cmd in commands:
                if self.config.verbosity >= 2:
                    self.logger.info(f"Running command: {' '.join(cmd)}")
                run_command(
                    cmd,
                    cwd=self.config.project_root,
                    verbosity=self.config.verbosity,
                    env=runtime_env.child_env,
                )
            
            if self.config.verbosity >= 1:
                self.logger.info(f"Successfully built for cpus={','.join(self.config.cpus)}")
            
            return StepResult(
                name=self.name,
                status=StepStatus.SUCCESS,
                message=f"Successfully built suites={','.join(self.config.suites)} for cpus={','.join(self.config.cpus)}",
                outputs={"build_dir": str(self.config.project_root / "artifacts")},
                details={"commands": commands},
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to build for FVP (exit code {e.returncode})"
            self.logger.error(error_msg)
            
            # Try to capture more error details
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.config.project_root,
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.stdout:
                    self.logger.error(f"stdout: {result.stdout}")
                if result.stderr:
                    self.logger.error(f"stderr: {result.stderr}")
            except Exception:
                pass
            
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                message=error_msg,
                error=BuildError(error_msg),
                outputs={"build_dir": str(self.config.project_root / "artifacts")},
                details={"commands": commands},
            )
        except FileNotFoundError as e:
            error_msg = f"Failed to build for FVP: {e}"
            self.logger.error(error_msg)
            build_error = BuildError(error_msg)
            build_error.__cause__ = e
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                message=error_msg,
                error=build_error,
                outputs={"build_dir": str(self.config.project_root / "artifacts")},
                details={"commands": commands},
            )

    def dry_run(self) -> StepResult:
        """Dry run of build step."""
        cmd_previews = self._build_commands()
        return StepResult(
            name=self.name,
            status=StepStatus.SKIPPED,
            message=f"DRY RUN: Would run {len(cmd_previews)} build command(s)",
            outputs={"build_dir": str(self.config.project_root / "artifacts")},
            details={"commands": cmd_previews},
        )

    def _plan_details(self) -> StepPlan:
        commands = self._build_commands()
        return StepPlan(
            name=self.name,
            will_run=True,
            reason="ready",
            commands=commands,
            outputs={"build_dir": str(self.config.project_root / "artifacts")}
        )
