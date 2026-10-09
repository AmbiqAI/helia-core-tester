"""Pre-FVP host check: build, selection, failure reporting, pipeline gating."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from helia_core_tester.core.config import Config
from helia_core_tester.core.errors import ConfigurationError
from helia_core_tester.core.pipeline import FullTestPipeline
from helia_core_tester.core.steps import HostCheckStep
from helia_core_tester.core.steps.base import StepResult, StepStatus
from helia_core_tester.generation.reference import host_kernels as hk
from helia_core_tester.mutation import host_build

TESTER_ROOT = Path(__file__).resolve().parents[2]


def _checkout(tmp_path: Path) -> Path:
    """A fabricated ns-cmsis-nn checkout: one int kernel, one f32 bridge the int
    cases call, and one f16 file that must never be compiled on the host."""
    root = tmp_path / "checkout"
    (root / "Include").mkdir(parents=True)
    (root / "Include" / "arm_nnfunctions.h").write_text(
        "#ifndef ARM_NNFUNCTIONS_H\n#define ARM_NNFUNCTIONS_H\n"
        "typedef enum { ARM_CMSIS_NN_SUCCESS = 0 } arm_cmsis_nn_status;\n#endif\n"
    )
    basic = root / "Source" / "BasicMathFunctions"
    basic.mkdir(parents=True)
    (basic / "kernel_s8.c").write_text("#include <stdint.h>\nint32_t hct_kernel(void) { return 42; }\n")
    quant = root / "Source" / "QuantizationFunctions"
    quant.mkdir(parents=True)
    (quant / "arm_quantize_f32_s8.c").write_text("#include <stdint.h>\nint32_t hct_bridge_f32_s8(float v) { return (int32_t)(v * 2.0f); }\n")
    (quant / "arm_cast_f16.c").write_text("#error f16 sources are not host-checked\n")
    return root


def _case(root: Path, name: str, expr: str, decl: str = "int32_t hct_kernel(void);") -> Path:
    case = root / "Fam" / name
    (case / "includes").mkdir(parents=True)
    (case / f"{name}.c").write_text(
        "#include <stdint.h>\n"
        "extern void helia_test_finish(int32_t failures);\n"
        f"{decl}\n"
        f"int main(void) {{ helia_test_finish(({expr}) ? 0 : 1); return 0; }}\n"
    )
    return case


def _manifest(cases_root: Path, names, run_seed=1234) -> None:
    tests = [{"name": n, "relative_test_dir": f"Fam/{n}", "c_sources": [f"{n}.c"]} for n in names]
    (cases_root / "manifest.json").write_text(json.dumps({"run_seed": run_seed, "tests": tests}))


@pytest.fixture()
def cache(tmp_path) -> Path:
    return tmp_path / "host_kernels"


def test_passing_failing_and_bridge_cases_are_classified(tmp_path, cache) -> None:
    tree = _checkout(tmp_path)
    cases = tmp_path / "gen"
    _case(cases, "case_good", "hct_kernel() == 42")
    _case(cases, "case_bad", "hct_kernel() == 41")
    _case(cases, "case_bridge", "hct_bridge_f32_s8(3.0f) == 6", "int32_t hct_bridge_f32_s8(float v);")
    _case(cases, "case_unlinked", "hct_missing() == 1", "int32_t hct_missing(void);")
    report = hk.run_host_check([cases], tree, mode="m0", cache_root=cache, jobs=2, seed=99)
    assert report.total == 4 and report.passed == 2 and not report.ok
    by_name = {f["name"]: f for f in report.failures}
    assert set(by_name) == {"case_bad", "case_unlinked"}
    assert by_name["case_bad"]["kind"] == host_build.KIND_CASE_FAIL
    assert by_name["case_unlinked"]["kind"] == host_build.KIND_COMPILE_FAILED
    assert "hct_missing" in by_name["case_unlinked"]["headline"]
    assert by_name["case_bad"]["repro"] == "--seed 99 --name case_bad"


def test_manifest_selects_cases_and_supplies_the_seed(tmp_path, cache) -> None:
    tree = _checkout(tmp_path)
    cases = tmp_path / "gen"
    _case(cases, "case_good", "hct_kernel() == 42")
    _case(cases, "stale_unlisted", "hct_kernel() == 0")
    _manifest(cases, ["case_good"], run_seed=777)
    report = hk.run_host_check([cases], tree, cache_root=cache, jobs=2)
    assert report.total == 1 and report.ok and report.seed == 777


def test_manifest_listing_a_missing_case_is_an_error(tmp_path, cache) -> None:
    tree = _checkout(tmp_path)
    cases = tmp_path / "gen"
    cases.mkdir()
    _manifest(cases, ["case_ghost"])
    with pytest.raises(hk.HostCheckError, match="does not exist"):
        hk.run_host_check([cases], tree, cache_root=cache)


def test_corrupt_manifest_is_an_error(tmp_path, cache) -> None:
    cases = tmp_path / "gen"
    cases.mkdir()
    (cases / "manifest.json").write_text("{not json")
    with pytest.raises(hk.HostCheckError, match="unreadable manifest"):
        hk.run_host_check([cases], _checkout(tmp_path), cache_root=cache)


def test_empty_tree_is_not_ok(tmp_path, cache) -> None:
    cases = tmp_path / "gen"
    cases.mkdir()
    report = hk.run_host_check([cases], _checkout(tmp_path), cache_root=cache)
    assert report.total == 0 and not report.ok


def test_bad_inputs_fail_closed(tmp_path, cache) -> None:
    tree = _checkout(tmp_path)
    with pytest.raises(hk.HostCheckError, match="unknown host kernel mode"):
        hk.ensure_host_kernels(tree, mode="mve", cache_root=cache)
    with pytest.raises(hk.HostCheckError, match="not an ns-cmsis-nn checkout"):
        hk.ensure_host_kernels(tmp_path / "nowhere", cache_root=cache)
    with pytest.raises(hk.HostCheckError, match="not found"):
        hk.run_host_check([tmp_path / "no-cases"], tree, cache_root=cache)


def test_kernel_build_failure_is_a_host_check_error(tmp_path, cache) -> None:
    tree = _checkout(tmp_path)
    (tree / "Source" / "BasicMathFunctions" / "broken_s8.c").write_text("this is not C\n")
    with pytest.raises(hk.HostCheckError, match="host kernel build"):
        hk.ensure_host_kernels(tree, cache_root=cache)
    assert not [p for p in cache.iterdir() if p.is_dir()]


def test_library_is_cached_and_rekeyed_on_source_edits(tmp_path, cache, monkeypatch) -> None:
    tree = _checkout(tmp_path)
    first = hk.ensure_host_kernels(tree, cache_root=cache)
    assert first.library.is_file() and first.runtime_obj.is_file()
    flags = json.loads((first.library.parent / "flags.json").read_text())
    assert flags["mode"] == "m0" and "-DARM_MATH_DSP" not in flags["cflags"]

    def no_build(*args, **kwargs):
        raise AssertionError("cached build must be reused")

    monkeypatch.setattr(host_build, "build_kernel_lib", no_build)
    assert hk.ensure_host_kernels(tree, cache_root=cache).key == first.key
    monkeypatch.undo()

    (tree / "Source" / "BasicMathFunctions" / "kernel_s8.c").write_text("#include <stdint.h>\nint32_t hct_kernel(void) { return 41; }\n")
    assert hk.ensure_host_kernels(tree, cache_root=cache).key != first.key


def test_dsp_mode_uses_the_shim(tmp_path, cache) -> None:
    tree = _checkout(tmp_path)
    library = hk.ensure_host_kernels(tree, mode="dsp", cache_root=cache)
    flags = json.loads((library.library.parent / "flags.json").read_text())
    assert "-DARM_MATH_DSP" in flags["cflags"]
    assert library.key != hk.ensure_host_kernels(tree, mode="m0", cache_root=cache).key


def test_report_round_trips(tmp_path, cache) -> None:
    tree = _checkout(tmp_path)
    cases = tmp_path / "gen"
    _case(cases, "case_good", "hct_kernel() == 42")
    report = hk.run_host_check([cases], tree, cache_root=cache)
    path = hk.write_report(report, tmp_path / "out" / "host_check.json")
    data = json.loads(path.read_text())
    assert data["ok"] is True and data["schema"] == hk.REPORT_SCHEMA and data["tree_identity"]["state"] == "content"


def test_headline_prefers_the_first_error() -> None:
    assert hk.headline("a\nfoo.c:1: error: boom\nlast") == "foo.c:1: error: boom"
    assert hk.headline("Undefined symbols for architecture arm64:\n  \"_x\", referenced") == 'Undefined symbols for architecture arm64: "_x", referenced'
    assert hk.headline("FAIL: output mismatch at 3") == "FAIL: output mismatch at 3"
    assert hk.headline("") == ""


# ---- config and pipeline ----


def _config(tmp_path, **kwargs) -> Config:
    return Config(project_root=TESTER_ROOT, generated_tests_root=tmp_path / "generated", reports_root=tmp_path / "reports", **kwargs)


def test_config_parses_and_validates_host_kernels(tmp_path, monkeypatch) -> None:
    assert _config(tmp_path).host_kernels == ["m0"]
    assert _config(tmp_path, host_kernels="dsp, m0,dsp").host_kernels == ["dsp", "m0"]
    with pytest.raises(ConfigurationError):
        _config(tmp_path, host_kernels=["mve"])
    with pytest.raises(ConfigurationError):
        _config(tmp_path, host_kernels=[])
    monkeypatch.setenv("HELIA_CORE_TESTER_HOST_KERNELS", "m0,dsp")
    monkeypatch.setenv("HELIA_CORE_TESTER_SKIP_HOST_CHECK", "true")
    config = _config(tmp_path)
    assert config.host_kernels == ["m0", "dsp"] and config.skip_host_check is True


def test_step_validates_the_checkout(tmp_path) -> None:
    config = _config(tmp_path, cmsis_nn_root=tmp_path / "empty")
    (tmp_path / "empty").mkdir()
    result = HostCheckStep(config).execute()
    assert result.status == StepStatus.FAILED and "--skip-host-check" in result.message


def test_step_skips_without_an_int_suite(tmp_path) -> None:
    config = _config(tmp_path, suite="float", cmsis_nn_root=_checkout(tmp_path))
    assert HostCheckStep(config).execute().status == StepStatus.SKIPPED


def test_step_runs_and_writes_its_report(tmp_path, monkeypatch) -> None:
    tree = _checkout(tmp_path)
    config = _config(tmp_path, cmsis_nn_root=tree, host_kernels=["m0", "dsp"])
    # generated_tests_dir_for() is rooted at the project, never at tmp_path:
    # point the step at a scratch tree instead of the real artifacts.
    cases = tmp_path / "generated" / "int" / "cortex-m55"
    monkeypatch.setattr(HostCheckStep, "_targets", lambda self: [("cortex-m55", cases)])
    _case(cases, "case_good", "hct_kernel() == 42")
    _case(cases, "case_bad", "hct_kernel() == 0")
    monkeypatch.setenv(hk.CACHE_ENV, str(tmp_path / "cache"))
    report_path = tmp_path / "reports" / "host_check.json"
    monkeypatch.setattr(HostCheckStep, "_report_path", lambda self, cpu: report_path)
    result = HostCheckStep(config).execute()
    assert result.status == StepStatus.FAILED
    assert "2 host-check failure(s) across 4 case run(s)" in result.message
    data = json.loads(report_path.read_text())
    assert set(data["modes"]) == {"m0", "dsp"}
    assert [f["name"] for f in data["modes"]["m0"]["failures"]] == ["case_bad"]


def test_failing_host_check_blocks_the_build_even_without_fail_fast(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path, skip_generation=True, fail_fast=False)
    calls = []

    def fake(name, status):
        def execute(self):
            calls.append(name)
            return StepResult(name=name, status=status, message=name)
        return execute

    from helia_core_tester.core import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module.HostCheckStep, "execute", fake("host-check", StepStatus.FAILED))
    monkeypatch.setattr(pipeline_module.BuildStep, "execute", fake("build", StepStatus.SUCCESS))
    monkeypatch.setattr(pipeline_module.RunStep, "execute", fake("run", StepStatus.SUCCESS))
    monkeypatch.setattr(FullTestPipeline, "_ensure_runtime_env", lambda self: None)
    assert FullTestPipeline(config).run() is False
    assert calls == ["host-check"]


def test_host_check_runs_between_generate_and_build(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    calls = []

    def fake(name):
        def execute(self):
            calls.append(name)
            return StepResult(name=name, status=StepStatus.SUCCESS, message=name)
        return execute

    from helia_core_tester.core import pipeline as pipeline_module

    for cls, name in (
        (pipeline_module.GenerateStep, "generate"),
        (pipeline_module.HostCheckStep, "host-check"),
        (pipeline_module.BuildStep, "build"),
        (pipeline_module.RunStep, "run"),
    ):
        monkeypatch.setattr(cls, "execute", fake(name))
    monkeypatch.setattr(FullTestPipeline, "_ensure_runtime_env", lambda self: None)
    assert FullTestPipeline(config).run() is True
    assert calls == ["generate", "host-check", "build", "run"]
    assert [p.name for p in FullTestPipeline(config).build_plan()][:2] == ["generate", "host-check"]


def test_skip_host_check_bypasses_it(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path, skip_generation=True, skip_host_check=True, skip_build=True, skip_run=True)
    from helia_core_tester.core import pipeline as pipeline_module

    def boom(self):
        raise AssertionError("host check must not run")

    monkeypatch.setattr(pipeline_module.HostCheckStep, "execute", boom)
    assert FullTestPipeline(config).run() is True
    plan = {p.name: p for p in FullTestPipeline(config).build_plan()}
    assert plan["host-check"].will_run is False


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
def test_git_identity_ignores_changes_outside_kernel_sources(tmp_path) -> None:
    import subprocess

    tree = _checkout(tmp_path)
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"]):
        subprocess.run(["git", "-C", str(tree), *cmd], check=True)
    clean = hk.kernel_tree_identity(tree)
    assert clean["state"] == "git-clean"
    (tree / "README.md").write_text("docs\n")
    assert hk.kernel_tree_identity(tree) == clean
    (tree / "Source" / "BasicMathFunctions" / "kernel_s8.c").write_text("int x;\n")
    assert hk.kernel_tree_identity(tree)["state"] == "git-dirty"


# ---- case classification: what a host build can judge ----


def _described_case(root: Path, name: str, expr: str, **descriptor) -> Path:
    import yaml

    case = _case(root, name, expr)
    (case / "descriptor.yaml").write_text(yaml.safe_dump({"name": name, **descriptor}))
    return case


@pytest.mark.parametrize(
    "descriptor, cpu, mode, status",
    [
        ({"required_capabilities": ["mve"]}, "cortex-m55", "m0", hk.STATUS_NOT_APPLICABLE),
        ({"required_capabilities": ["dsp"]}, "cortex-m4", "m0", hk.STATUS_NOT_APPLICABLE),
        ({"required_capabilities": ["dsp"]}, "cortex-m4", "dsp", hk.STATUS_BLOCKING),
        # Non-kernel capabilities never gate a host run.
        ({"required_capabilities": ["fp32_execution"]}, "cortex-m55", "m0", hk.STATUS_BLOCKING),
        ({"entry": "arm_convolve_s8_small_cin"}, "cortex-m55", "m0", hk.STATUS_ADVISORY),
        ({"fault": "null_ctx_buf"}, "cortex-m55", "dsp", hk.STATUS_ADVISORY),
        ({"entry": "arm_fully_connected_s8"}, "cortex-m0", "m0", hk.STATUS_BLOCKING),
        ({"fault": "null_ctx_buf"}, "cortex-m4", "dsp", hk.STATUS_BLOCKING),
        ({"operator": "FullyConnected", "activation_dtype": "S8", "weight_dtype": "S8"}, "cortex-m55", "m0", hk.STATUS_ADVISORY),
        ({"operator": "FullyConnected", "activation_dtype": "S8", "weight_dtype": "S8"}, "cortex-m55-dsp", "dsp", hk.STATUS_BLOCKING),
        ({"operator": "FullyConnected", "activation_dtype": "S16", "weight_dtype": "S8"}, "cortex-m55", "m0", hk.STATUS_BLOCKING),
        ({"operator": "FullyConnected", "activation_dtype": "S8", "weight_dtype": "S4"}, "cortex-m55", "m0", hk.STATUS_BLOCKING),
        ({"operator": "Convolve"}, "cortex-m55", "m0", hk.STATUS_BLOCKING),
        # Unknown target: nothing is relaxed.
        ({"entry": "arm_convolve_s8_small_cin"}, None, "m0", hk.STATUS_BLOCKING),
        ({"entry": "arm_convolve_s8_small_cin"}, "cortex-m99", "m0", hk.STATUS_BLOCKING),
    ],
)
def test_classify_case(tmp_path, descriptor, cpu, mode, status) -> None:
    case = _described_case(tmp_path, "case_x", "1", **descriptor)
    assert hk.classify_case(case, cpu, mode).status == status


def test_classify_without_descriptor_is_blocking(tmp_path) -> None:
    case = _case(tmp_path, "case_bare", "1")
    assert hk.classify_case(case, "cortex-m55", "m0").status == hk.STATUS_BLOCKING


def test_corrupt_case_descriptor_is_an_error(tmp_path) -> None:
    case = _case(tmp_path, "case_bad_yaml", "1")
    (case / "descriptor.yaml").write_text("name: [unclosed\n")
    with pytest.raises(hk.HostCheckError, match="unreadable case descriptor"):
        hk.classify_case(case, "cortex-m55", "m0")


def test_advisory_failures_do_not_block_and_na_cases_do_not_run(tmp_path, cache) -> None:
    tree = _checkout(tmp_path)
    cases = tmp_path / "gen" / "cortex-m55"
    _described_case(cases, "case_neutral", "hct_kernel() == 42", operator="Convolve")
    _described_case(cases, "case_entry_bad", "hct_kernel() == 0", entry="arm_x_s8")
    _described_case(cases, "case_needs_mve", "hct_missing() == 0", required_capabilities=["mve"])
    report = hk.run_host_check([cases], tree, mode="m0", cache_root=cache)
    assert report.ok and report.total == 2 and report.passed == 1
    assert [a["name"] for a in report.advisory] == ["case_entry_bad"]
    assert "generated for cortex-m55" in report.advisory[0]["reason"]
    assert report.not_applicable == [{"family": "Fam", "name": "case_needs_mve", "reason": "requires mve"}]


def test_target_cpu_comes_from_the_manifest(tmp_path, cache) -> None:
    tree = _checkout(tmp_path)
    cases = tmp_path / "gen" / "unnamed-tree"
    _described_case(cases, "case_entry_bad", "hct_kernel() == 0", entry="arm_x_s8")
    tests = [{"name": "case_entry_bad", "relative_test_dir": "Fam/case_entry_bad", "c_sources": ["case_entry_bad.c"], "cpu": "cortex-m55"}]
    (cases / "manifest.json").write_text(json.dumps({"run_seed": 3, "tests": tests}))
    report = hk.run_host_check([cases], tree, mode="m0", cache_root=cache)
    assert report.ok and len(report.advisory) == 1
    # Same failing entry case generated for cortex-m0: nothing to excuse it.
    tests[0]["cpu"] = "cortex-m0"
    (cases / "manifest.json").write_text(json.dumps({"run_seed": 3, "tests": tests}))
    report = hk.run_host_check([cases], tree, mode="m0", cache_root=cache)
    assert not report.ok and [f["name"] for f in report.failures] == ["case_entry_bad"]
