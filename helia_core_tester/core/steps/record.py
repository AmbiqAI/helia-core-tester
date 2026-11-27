"""Record step: every generated case recorded on the host against the library's host
reference build and compared with the golden it was generated with (golden parity)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from helia_core_tester.core.logging import get_logger
from helia_core_tester.core.steps.base import StepBase, StepPlan, StepResult, StepStatus

REPORT_NAME = "record_parity.json"
_LOGGED_CASES = 25


class RecordStep(StepBase):
    def __init__(self, config):
        super().__init__(config)
        self.logger = get_logger(__name__)

    @property
    def name(self) -> str:
        return "record"

    def _tree(self) -> Optional[Path]:
        if self.config.cmsis_nn_root is not None:
            return Path(self.config.cmsis_nn_root)
        from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root

        return resolve_cmsis_nn_root()

    def _targets(self) -> List[Tuple[str, str, Path]]:
        return [
            (cpu, suite, self.config.generated_tests_dir_for(cpu, suite=suite))
            for cpu in self.config.cpus
            for suite in self.config.effective_suites_for_cpu(cpu)
        ]

    def _report_path(self, cpu: str, suite: str) -> Path:
        return self.config.generation_report_dir_for(cpu, suite=suite) / REPORT_NAME

    def validate(self) -> Optional[str]:
        from helia_core_tester.generation.reference.host_kernels import checkout_problem

        problem = checkout_problem(self._tree())
        if problem:
            return f"record: {problem}: set --cmsis-nn-root or CMSIS_NN_ROOT"
        return None

    def plan_validate(self) -> Optional[str]:
        return self.validate()

    def _do_execute(self) -> StepResult:
        from helia_core_tester.generation.reference.host_kernels import HostCheckError
        from helia_core_tester.record.runner import RecordError, run_record, write_report
        from helia_core_tester.utils.host_compiler import HostCompilerMissing

        targets = self._targets()
        if not targets:
            return StepResult(self.name, StepStatus.SKIPPED, "no suite selected; nothing to record")
        tree = self._tree()
        if tree is None:
            return StepResult(self.name, StepStatus.FAILED, "record needs an ns-cmsis-nn checkout")
        outputs = {}
        failed = []
        try:
            for cpu, suite, cases_root in targets:
                report = run_record([cases_root], tree, jobs=self.config.jobs)
                path = write_report(report, self._report_path(cpu, suite))
                outputs[f"record_{cpu}_{suite}"] = str(path)
                level = self.logger.info if report.ok else self.logger.error
                level(f"record {cpu}/{suite} [{report.float16_type}]: {report.totals()} ({path})")
                for family, counts in report.families().items():
                    self.logger.info(f"  {family}: {counts}")
                if not report.ok:
                    failed.append(f"{cpu}/{suite}")
                    for case in report.problems()[:_LOGGED_CASES]:
                        self.logger.error(f"  {case.family}/{case.name}: {case.verdict or case.outcome} {case.detail}".rstrip())
        except (RecordError, HostCheckError, HostCompilerMissing) as exc:
            return StepResult(self.name, StepStatus.FAILED, str(exc), outputs=outputs, error=exc)
        if failed:
            return StepResult(
                self.name,
                StepStatus.FAILED,
                f"recorded outputs differ from the goldens, or could not be recorded, in {', '.join(failed)}",
                outputs=outputs,
            )
        return StepResult(self.name, StepStatus.SUCCESS, "every blocking case recorded identical", outputs=outputs)

    def _plan_details(self) -> StepPlan:
        return StepPlan(
            name=self.name,
            will_run=True,
            reason="ready",
            outputs={f"record_{cpu}_{suite}": str(self._report_path(cpu, suite)) for cpu, suite, _ in self._targets()},
            details={"cmsis_nn_root": str(self._tree())},
        )
