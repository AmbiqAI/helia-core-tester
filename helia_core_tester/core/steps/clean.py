"""
Cleanup step for removing generated tests, reports, and build artifacts.
"""

from __future__ import annotations

import shutil

from helia_core_tester.core.steps.base import StepBase, StepPlan, StepResult, StepStatus
from helia_core_tester.core.errors import StepExecutionError
from helia_core_tester.core.logging import get_logger
from helia_core_tester.core.path_layout import artifacts_root, coverage_report_dir, generation_report_dir, tests_report_dir


class CleanStep(StepBase):
    """Step for cleaning CPU-scoped generated artifacts."""

    def __init__(self, config):
        super().__init__(config)
        self.logger = get_logger(__name__)

    @property
    def name(self) -> str:
        return "clean"

    def should_skip(self) -> bool:
        return False

    def validate(self) -> str | None:
        return None

    def _targets(self) -> list:
        targets = []
        art_root = artifacts_root(self.config.project_root)
        for suite in self.config.suites:
            for cpu in self.config.cpus:
                targets.append(self.config.generated_tests_dir_for(cpu, suite=suite))
                targets.append(generation_report_dir(self.config.project_root, cpu, suite=suite))
                targets.append(tests_report_dir(self.config.project_root, cpu, suite=suite))
                targets.append(coverage_report_dir(self.config.project_root, cpu, suite=suite))
                targets.extend(sorted(p for p in art_root.glob(f"build-{suite}-{cpu}-*") if p.is_dir()))
        return targets

    def _do_execute(self) -> StepResult:
        if self.config.verbosity >= 1:
            self.logger.info("Cleaning generated tests, reports, and build directories")

        cleaned_items = []

        try:
            for path in self._targets():
                if not path.exists():
                    continue
                if self.config.verbosity >= 1:
                    self.logger.info(f"Removing: {path}")
                # Do not use ignore_errors=True: a failed removal (permissions,
                # open file handles, read-only files) must surface as a step
                # failure instead of being silently reported as cleaned, which
                # would let later stages consume stale artifacts.
                shutil.rmtree(path)
                cleaned_items.append(str(path))

            message = (
                f"Cleaned {len(cleaned_items)} item(s): {', '.join(cleaned_items)}"
                if cleaned_items
                else "No artifacts to clean"
            )

            if self.config.verbosity >= 1:
                self.logger.info(message)

            return StepResult(
                name=self.name,
                status=StepStatus.SUCCESS,
                message=message,
                outputs={"reports_root": str(self.config.reports_root)},
            )
        except Exception as e:
            error_msg = f"Failed to clean artifacts: {e}"
            self.logger.error(error_msg)
            exec_error = StepExecutionError(error_msg)
            exec_error.__cause__ = e
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                message=error_msg,
                error=exec_error,
                outputs={"reports_root": str(self.config.reports_root)},
            )

    def dry_run(self) -> StepResult:
        cleaned_items = [str(path) for path in self._targets() if path.exists()]
        message = (
            f"DRY RUN: Would clean {len(cleaned_items)} item(s): {', '.join(cleaned_items)}"
            if cleaned_items
            else "DRY RUN: No artifacts to clean"
        )
        return StepResult(
            name=self.name,
            status=StepStatus.SKIPPED,
            message=message,
            outputs={"reports_root": str(self.config.reports_root)},
        )

    def _plan_details(self) -> StepPlan:
        return StepPlan(
            name=self.name,
            will_run=True,
            reason="ready",
            commands=[],
            outputs={"reports_root": str(self.config.reports_root)},
        )
