"""Host check step: every generated int case, compiled and run on the host
against the ns-cmsis-nn kernels, before any FVP or board build."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from helia_core_tester.core.logging import get_logger
from helia_core_tester.core.path_layout import generated_tests_dir
from helia_core_tester.core.steps.base import StepBase, StepPlan, StepResult, StepStatus

REPORT_NAME = "host_check.json"
# Failures echoed to the log; the report always carries all of them.
_LOGGED_FAILURES = 25


class HostCheckStep(StepBase):
    def __init__(self, config):
        super().__init__(config)
        self.logger = get_logger(__name__)

    @property
    def name(self) -> str:
        return "host-check"

    def should_skip(self) -> bool:
        return bool(self.config.skip_host_check)

    def _tree(self) -> Optional[Path]:
        if self.config.cmsis_nn_root is not None:
            return Path(self.config.cmsis_nn_root)
        from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root

        return resolve_cmsis_nn_root()

    def _targets(self) -> List[Tuple[str, Path]]:
        targets = []
        for cpu in self.config.cpus:
            if "int" not in self.config.effective_suites_for_cpu(cpu):
                continue
            if self.config.hidden_dir is not None:
                cases_root = generated_tests_dir(self.config.hidden_dir, cpu, suite="int")
            else:
                cases_root = self.config.generated_tests_dir_for(cpu, suite="int")
            targets.append((cpu, cases_root))
        return targets

    def validate(self) -> Optional[str]:
        tree = self._tree()
        if tree is None or not (tree / "Source").is_dir():
            return (
                "host check needs an ns-cmsis-nn checkout (Include/ and Source/): set --cmsis-nn-root "
                "or CMSIS_NN_ROOT, or pass --skip-host-check"
            )
        from helia_core_tester.generation.reference.host_kernels import HOST_KERNEL_MODES

        unknown = [m for m in self.config.host_kernels if m not in HOST_KERNEL_MODES]
        if unknown:
            return f"unknown --host-kernels {', '.join(unknown)} (expected {', '.join(HOST_KERNEL_MODES)})"
        return None

    def plan_validate(self) -> Optional[str]:
        return self.validate()

    def _report_path(self, cpu: str) -> Path:
        return self.config.generation_report_dir_for(cpu, suite="int") / REPORT_NAME

    def _do_execute(self) -> StepResult:
        from helia_core_tester.generation.reference.host_kernels import HostCheckError, run_host_check
        from helia_core_tester.utils.host_compiler import HostCompilerMissing

        targets = self._targets()
        if not targets:
            return StepResult(self.name, StepStatus.SKIPPED, "no int suite selected; nothing to host-check")
        tree = self._tree()
        failures_total = 0
        checked_total = 0
        outputs = {}
        try:
            for cpu, cases_root in targets:
                reports = {}
                for mode in self.config.host_kernels:
                    report = run_host_check(
                        [cases_root],
                        tree,
                        mode=mode,
                        seed=self.config.seed,
                        jobs=self.config.jobs,
                    )
                    reports[mode] = report.to_json()
                    checked_total += report.total
                    failures_total += len(report.failures)
                    level = self.logger.info if report.ok else self.logger.error
                    level(
                        f"host-check {cpu} [{mode}]: {report.passed}/{report.total} passed, "
                        f"{len(report.failures)} failed, {len(report.advisory)} advisory "
                        f"(cpu-specific, not blocking), {len(report.not_applicable)} not applicable"
                    )
                    for advisory in report.advisory[:_LOGGED_FAILURES]:
                        self.logger.warning(
                            f"  advisory {advisory['family']}/{advisory['name']}: {advisory['headline']} "
                            f"({advisory['reason']})"
                        )
                    for failure in report.failures[:_LOGGED_FAILURES]:
                        self.logger.error(
                            f"  {failure['family']}/{failure['name']}: {failure['kind']}: "
                            f"{failure['headline']} (reproduce: {failure['repro']})"
                        )
                    if len(report.failures) > _LOGGED_FAILURES:
                        self.logger.error(f"  ... {len(report.failures) - _LOGGED_FAILURES} more in the report")
                    if report.total == 0:
                        failures_total += 1
                        self.logger.error(f"host-check {cpu} [{mode}]: no int cases found under {cases_root}")
                path = self._report_path(cpu)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"cpu": cpu, "modes": reports}, indent=2) + "\n", encoding="utf-8")
                outputs[f"host_check_{cpu}"] = str(path)
        except (HostCheckError, HostCompilerMissing) as exc:
            return StepResult(self.name, StepStatus.FAILED, str(exc), outputs=outputs, error=exc)
        if failures_total:
            return StepResult(
                self.name,
                StepStatus.FAILED,
                f"{failures_total} host-check failure(s) across {checked_total} case run(s); "
                "the FVP/board build is blocked (--skip-host-check to bypass)",
                outputs=outputs,
            )
        return StepResult(
            self.name,
            StepStatus.SUCCESS,
            f"host check passed: {checked_total} case run(s)",
            outputs=outputs,
        )

    def _plan_details(self) -> StepPlan:
        return StepPlan(
            name=self.name,
            will_run=True,
            reason="ready",
            outputs={f"host_check_{cpu}": str(self._report_path(cpu)) for cpu, _ in self._targets()},
            details={"modes": ",".join(self.config.host_kernels), "cmsis_nn_root": str(self._tree())},
        )
