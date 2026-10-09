"""
TFLite model generation step.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

from helia_core_tester.core.steps.base import StepBase, StepPlan, StepResult, StepStatus
from helia_core_tester.core.errors import GenerationError
from helia_core_tester.core.logging import get_logger
from helia_core_tester.core.path_layout import generated_tests_dir, generated_tests_root
from helia_core_tester.utils.command_runner import run_command


class GenerateStep(StepBase):
    """Step for generating TFLite models."""
    
    _run_seed: Optional[tuple[int, bool]] = None

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
        if self.config.hidden_dir is not None:
            try:
                self._hidden_env()
            except (OSError, ValueError) as exc:
                return str(exc)
        return None

    def _cpu_generated_tests_dir(self, cpu: str, suite: str) -> Path:
        if self.config.hidden_dir is not None:
            return generated_tests_dir(self.config.hidden_dir, cpu, suite=suite)
        return self.config.generated_tests_dir_for(cpu, suite=suite)

    def _output_root(self) -> Path:
        """Generated tests root this run writes."""
        if self.config.hidden_dir is not None:
            return generated_tests_root(self.config.hidden_dir)
        return self.config.generated_tests_root

    def _hidden_env(self) -> dict[str, str]:
        """Environment carrying the checked secret."""
        from helia_core_tester.generation.random_shapes import SECRET_ENV, hidden_secret

        env = dict(os.environ)
        if self.config.hidden_seed_file is not None:
            # Env, not argv: ps shows argv.
            env[SECRET_ENV] = self.config.hidden_seed_file.read_text(encoding="utf-8").strip()
        hidden_secret(env.get(SECRET_ENV, ""))
        return env

    def _build_cmd(
        self,
        cpu: str,
        suite: str,
        include_seed: bool = True,
        float_precision: str | None = None,
    ) -> list:
        """Build pytest command list. include_seed=False for dry-run preview."""
        cmd = ["pytest", "test_ops.py::test_generation", "-v"]
        cmd.extend(["--cpu", cpu])
        cmd.extend(["--suite", suite])
        if self.config.hidden_dir is None:
            cmd.extend(["--generated-tests-dir", str(self._cpu_generated_tests_dir(cpu, suite=suite))])
        if self.config.op_filter:
            cmd.extend(["--op", self.config.op_filter])
        if self.config.dtype_filter:
            cmd.extend(["--dtype", self.config.dtype_filter])
        if self.config.name_filter:
            cmd.extend(["--name", self.config.name_filter])
        if self.config.limit:
            cmd.extend(["--limit", str(self.config.limit)])
        if suite == "float":
            effective_float_precision = float_precision or self.config.effective_float_precision_for_cpu(
                cpu, suite=suite
            )
            if effective_float_precision:
                cmd.extend(["--float-precision", effective_float_precision])
        if include_seed:
            seed, chosen = self.run_seed()
            cmd.extend(["--seed", str(seed)])
            if not chosen:
                cmd.append("--fresh-seed")
        if self.config.force_generate:
            cmd.append("--force-generate")
        if self.config.keep_unselected:
            cmd.append("--keep-unselected")
        if self.config.random_shapes:
            cmd.extend(["--random-shapes", str(self.config.random_shapes)])
        if self.config.hidden_dir is not None:
            # Long tracebacks print the derived seed.
            cmd.extend(["--hidden-dir", str(self.config.hidden_dir), "--tb=native"])
        elif self.config.random_shapes:
            cmd.extend(["--shape-seed", str(self.config.shape_seed)])
        return cmd
    
    def run_seed(self) -> tuple[int, bool]:
        """The run seed and whether it was chosen (--seed / HCT_SEED) rather than drawn.

        Drawn once per step so every generation command of one run (each cpu and suite)
        derives its cases from the same seed, and recorded in the step details."""
        if self._run_seed is None:
            from helia_core_tester.generation.test_ops import resolve_run_seed

            self._run_seed = resolve_run_seed({"seed": self.config.seed})
        return self._run_seed

    def _do_execute(self) -> StepResult:
        """Execute TFLite model generation."""
        seed, chosen = self.run_seed()
        if self.config.verbosity >= 1:
            self.logger.info("Generating reference models and test cases using pytest")
            self.logger.info(
                f"Run seed: {seed} ({'chosen' if chosen else 'fresh draw'}; pass --seed {seed} to reproduce"
                f"{'' if chosen else ' or reuse'})"
            )
        # Propagate an overridden CMSIS-NN root (--cmsis-nn-root) so checkout
        # probes (temp_sizer_probe.py, the s16 activation tables) resolve
        # against it instead of assuming the repo is nested under ns-cmsis-nn/Tests/. Uses
        # CMSIS_NN_ROOT (matching the CMake cache var name), distinct from
        # CMSIS_NN_REPO_ROOT which overrides helia-core-tester's own repo
        # root discovery.
        subprocess_env = self._hidden_env() if self.config.hidden_dir is not None else None
        if self.config.cmsis_nn_root:
            subprocess_env = {**(subprocess_env or os.environ), "CMSIS_NN_ROOT": str(self.config.cmsis_nn_root)}
        try:
            commands = []
            generation_targets = self._targets()
            for cpu, suite, float_precision in generation_targets:
                cmd = self._build_cmd(
                    cpu=cpu,
                    suite=suite,
                    include_seed=True,
                    float_precision=float_precision,
                )
                commands.append(cmd)
                if self.config.verbosity >= 2:
                    self.logger.info(f"Running command: {' '.join(cmd)}")
                run_command(
                    cmd,
                    cwd=self.config.generation_dir,
                    verbosity=self.config.verbosity,
                    env=subprocess_env,
                )
            if self.config.verbosity >= 1:
                self.logger.info(
                    f"TFLite models generated successfully for targets={len(generation_targets)} cpus={','.join(self.config.cpus)}"
                )
            return StepResult(
                name=self.name,
                status=StepStatus.SUCCESS,
                message="TFLite models generated successfully",
                outputs={
                    "generated_tests_root": str(self._output_root())
                },
                details={
                    "commands": commands,
                    "filters": {
                        "op": self.config.op_filter,
                        "dtype": self.config.dtype_filter,
                        "name": self.config.name_filter,
                        "limit": self.config.limit,
                        "seed": seed,
                        "seed_chosen": chosen,
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
                    "generated_tests_root": str(self._output_root())
                },
                details={"cpus": self.config.cpus},
            )
    
    def _targets(self) -> list[tuple[str, str, Optional[str]]]:
        """Targets execute, dry run and plan share."""
        targets = self.config.iter_generation_targets()
        if self.config.random_shapes:
            # Random shapes are int only.
            targets = [target for target in targets if target[1] == "int"]
        return targets

    def dry_run(self) -> StepResult:
        """Dry run of generation step."""
        cmd_preview = [
            self._build_cmd(
                cpu=cpu,
                suite=suite,
                include_seed=False,
                float_precision=float_precision,
            )
            for cpu, suite, float_precision in self._targets()
        ]
        return StepResult(
            name=self.name,
            status=StepStatus.SKIPPED,
            message=f"DRY RUN: Would run {len(cmd_preview)} generation command(s) in {self.config.generation_dir}",
            outputs={
                "generated_tests_root": str(self._output_root())
            },
            details={"commands": cmd_preview},
        )

    def _plan_details(self) -> StepPlan:
        cmd_preview = [
            self._build_cmd(
                cpu=cpu,
                suite=suite,
                include_seed=True,
                float_precision=float_precision,
            )
            for cpu, suite, float_precision in self._targets()
        ]
        return StepPlan(
            name=self.name,
            will_run=True,
            reason="ready",
            commands=cmd_preview,
            outputs={"generated_tests_root": str(self._output_root())},
            details={"cwd": str(self.config.generation_dir)}
        )
