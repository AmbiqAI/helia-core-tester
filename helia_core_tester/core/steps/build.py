"""
CMake build step for FVP.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from helia_core_tester.core.runtime_env import RuntimeEnvContext, bootstrap_runtime_env, build_locked_fvp_flags
from helia_core_tester.core.steps.base import StepBase, StepPlan, StepResult, StepStatus
from helia_core_tester.core.errors import BuildError, ConfigurationError
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
                toolchain=self.config.toolchain,
            )
        return self.runtime_env

    def _locked_fvp_flags(self) -> list[str]:
        return build_locked_fvp_flags(self.runtime_env, self.config.downloads_dir, self.config.toolchain)
    
    @property
    def name(self) -> str:
        return "build"
    
    def should_skip(self) -> bool:
        """Check if build should be skipped."""
        return self.config.skip_build
    
    def _library_build(self):
        """The profile of a library other than ns-cmsis-nn, which builds from its build block; else None
        (ns-cmsis-nn builds from hct's own rules)."""
        from helia_core_tester.core.library import SUPPORTED_LIBRARIES

        try:
            profile = self.config.library_profile
        except ConfigurationError:
            # A --cmsis-nn-root that loads no profile is ns-cmsis-nn's; its CMake checks the checkout.
            if self.config.library is not None:
                raise
            return None
        if profile is None or profile.library in SUPPORTED_LIBRARIES:
            return None
        return profile

    def validate(self) -> str | None:
        """Validate prerequisites for build."""
        script_path = find_fvp_script_path(self.config.project_root)
        if not script_path.exists():
            return f"FVP script not found: {script_path}"
        try:
            profile = self._library_build()
        except ConfigurationError as exc:
            return str(exc)
        if profile is None:
            return self.config.cmsis_nn_root_error()
        if profile.build is None:
            return f"{profile.describe()}: its profile names no build, so hct cannot build its FVP tests"
        if getattr(self.config, "coverage", False):
            return f"--coverage instruments ns-cmsis-nn only; {profile.library} builds without it"
        return None

    def _cmsis5_path(self) -> Path:
        if self.runtime_env is not None:
            return Path(self.runtime_env.cmsis5_path)
        return Path(self.config.downloads_dir).resolve() / "CMSIS_5"

    def _library_definitions(self, profile, cpu: str) -> list[str]:
        build = profile.build
        return [f"HCT_LIBRARY_DIR={build.project}", f"HCT_LIBRARY_TARGET={build.target}",
                f"HCT_LIBRARY_COMPILE_OPTIONS={self.config.optimization}",
                *build.definitions_for(cpu, self._cmsis5_path())]

    def _build_commands(self) -> list[list[str]]:
        profile = self._library_build()
        commands: list[list[str]] = []
        for suite in self.config.suites:
            for group in self.config.cpu_groups_for_suite(suite):
                cpu_sets = [[cpu] for cpu in group["cpus"]] if profile is not None else [group["cpus"]]
                for cpus in cpu_sets:
                    commands.append(self._build_command(suite, group, cpus, profile))
        return commands

    def _build_command(self, suite: str, group: dict, cpus: list[str], profile) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "helia_core_tester.fvp.build_and_run_fvp",
            *self._locked_fvp_flags(),
            "--cpu",
            ",".join(cpus),
            "--suite",
            suite,
            "--cmake-def",
            f"CMSIS_OPTIMIZATION_LEVEL={self.config.optimization}",
            "--no-run",
            "--no-report",
        ]
        if profile is not None:
            for item in self._library_definitions(profile, cpus[0]):
                cmd += ["--cmake-def", item]
        else:
            if getattr(self.config, "cmsis_nn_root", None):
                cmd += ["--cmake-def", f"CMSIS_NN_ROOT={self.config.cmsis_nn_root}"]
            cmd += self._nn_definitions(suite, group)

        if getattr(self.config, "coverage", False):
            cmd.append("--coverage")
        if self.config.jobs:
            cmd.extend(["--jobs", str(self.config.jobs)])
        if self.config.verbosity <= 1:
            cmd.append("--quiet")
        else:
            cmd.extend(["--verbosity", str(self.config.verbosity)])
        return cmd

    def _nn_definitions(self, suite: str, group: dict) -> list[str]:
        cmd: list[str] = []
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
            if getattr(self.config, "coverage_mve_int", False):
                cmd.extend(["--cmake-def", "ENABLE_COVERAGE_MVE_INT=ON"])
        return cmd

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
            
            # Try to capture more error details. Reuse the same runtime
            # environment as the real build invocation so this diagnostic
            # rerun can actually reproduce (and report on) the original
            # failure instead of running under a different environment.
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.config.project_root,
                    capture_output=True,
                    text=True,
                    check=False,
                    env=runtime_env.child_env,
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
