"""
TFLite model generation step.
"""

import subprocess
from pathlib import Path

from helia_core_tester.core.steps.base import StepBase, StepPlan, StepResult, StepStatus
from helia_core_tester.core.errors import GenerationError
from helia_core_tester.core.logging import get_logger
from helia_core_tester.utils.command_runner import run_command


class GenerateStep(StepBase):
    """Step for generating TFLite models."""
    
    def __init__(self, config):
        super().__init__(config)
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "generate"
    
    def should_skip(self) -> bool:
        """Check if generation should be skipped."""
        return self.config.skip_generation
    
    def validate(self) -> str | None:
        """Validate prerequisites for generation."""
        if not self.config.generation_dir.exists():
            return f"Generation directory not found: {self.config.generation_dir}"
        return None

    def _cpu_generated_tests_dir(self, cpu: str) -> Path:
        if len(self.config.cpus) <= 1:
            return self.config.generated_tests_dir
        return self.config.generated_tests_dir / cpu

    def _build_cmd(self, cpu: str, include_seed: bool = True) -> list:
        """Build pytest command list. include_seed=False for dry-run preview."""
        cmd = ["pytest", "test_ops.py::test_generation", "-v"]
        cmd.extend(["--cpu", cpu])
        cmd.extend(["--generated-tests-dir", str(self._cpu_generated_tests_dir(cpu))])
        if self.config.op_filter:
            cmd.extend(["--op", self.config.op_filter])
        if self.config.dtype_filter:
            cmd.extend(["--dtype", self.config.dtype_filter])
        if self.config.name_filter:
            cmd.extend(["--name", self.config.name_filter])
        if self.config.limit:
            cmd.extend(["--limit", str(self.config.limit)])
        if include_seed and self.config.seed is not None:
            cmd.extend(["--seed", str(self.config.seed)])
        return cmd
    
    def _do_execute(self) -> StepResult:
        """Execute TFLite model generation."""
        if self.config.verbosity >= 1:
            self.logger.info("Generating TensorFlow Lite models using pytest")
        try:
            commands = []
            for cpu in self.config.cpus:
                cmd = self._build_cmd(cpu=cpu, include_seed=True)
                commands.append(cmd)
                if self.config.verbosity >= 2:
                    self.logger.info(f"Running command: {' '.join(cmd)}")
                run_command(
                    cmd,
                    cwd=self.config.generation_dir,
                    verbosity=self.config.verbosity
                )
            if self.config.verbosity >= 1:
                self.logger.info(f"TFLite models generated successfully for {', '.join(self.config.cpus)}")
            return StepResult(
                name=self.name,
                status=StepStatus.SUCCESS,
                message="TFLite models generated successfully",
                outputs={
                    "generated_tests_dir": str(self.config.generated_tests_dir)
                },
                details={
                    "commands": commands,
                    "filters": {
                        "op": self.config.op_filter,
                        "dtype": self.config.dtype_filter,
                        "name": self.config.name_filter,
                        "limit": self.config.limit,
                        "seed": self.config.seed,
                    },
                },
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            error_msg = f"Failed to generate TFLite models: {e}"
            self.logger.error(error_msg)
            gen_error = GenerationError(error_msg)
            gen_error.__cause__ = e
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                message=error_msg,
                error=gen_error,
                outputs={
                    "generated_tests_dir": str(self.config.generated_tests_dir)
                },
                details={"cpus": self.config.cpus},
            )
    
    def dry_run(self) -> StepResult:
        """Dry run of generation step."""
        cmd_preview = [self._build_cmd(cpu=cpu, include_seed=False) for cpu in self.config.cpus]
        return StepResult(
            name=self.name,
            status=StepStatus.SKIPPED,
            message=f"DRY RUN: Would run {len(cmd_preview)} generation command(s) in {self.config.generation_dir}",
            outputs={
                "generated_tests_dir": str(self.config.generated_tests_dir)
            },
            details={"commands": cmd_preview},
        )

    def _plan_details(self) -> StepPlan:
        cmd_preview = [self._build_cmd(cpu=cpu, include_seed=True) for cpu in self.config.cpus]
        return StepPlan(
            name=self.name,
            will_run=True,
            reason="ready",
            commands=cmd_preview,
            outputs={"generated_tests_dir": str(self.config.generated_tests_dir)},
            details={"cwd": str(self.config.generation_dir)}
        )
