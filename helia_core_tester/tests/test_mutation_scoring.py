"""
Tests for the mutation-scoring machinery (issue #76): catalog integrity,
patch application + tree restoration, case discovery, and report format.

These tests exercise the pure-python machinery on fabricated trees; they do
not compile kernels and need neither gcc nor an ns-cmsis-nn checkout.
"""

import json
import os
import re
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helia_core_tester.core.cpu_targets import get_cpu_profile, known_capabilities
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.mutation import cli as mutation_cli
from helia_core_tester.mutation.catalog import MUTANTS_V1, Edit, Mutant, get_mutants
from helia_core_tester.mutation.host_build import (
    KERNEL_SOURCE_DIRS,
    KIND_CASE_FAIL,
    KIND_COMPILE_FAILED,
    KIND_NO_SOURCE,
    KIND_PASS,
    KIND_TIMEOUT,
    CaseResult,
    build_and_run_case,
    discover_cases,
)
from helia_core_tester.mutation.patching import AppliedMutant, MutantApplyError, verify_pristine
from helia_core_tester.mutation.runner import (
    STATUS_APPLY_FAILED,
    STATUS_BUILD_FAILED,
    STATUS_KILLED,
    STATUS_NOT_APPLICABLE,
    STATUS_SURVIVED,
    MutantOutcome,
    MutationReport,
    prepare_tree,
    run_mutation_scoring,
)

TESTER_ROOT = Path(__file__).resolve().parents[2]


# Dedicated to this test on purpose. HELIA_CORE_TESTER_CMSIS_NN_ROOT is the
# pipeline Config override, so reusing it here would mean that enabling the
# catalog check also redirects the pipeline tests' notion of the checkout.
MUTATION_CHECKOUT_ENV = "HELIA_CORE_TESTER_MUTATION_CHECKOUT"


def _kernel_checkout():
    """An ns-cmsis-nn checkout to patch against, or None when there is none.

    The env var is the explicit handle; the fallback is the conventional
    nested layout (<ns-cmsis-nn>/Tests/helia-core-tester).
    """
    def looks_like_a_checkout(path: Path) -> bool:
        return (path / "Source" / "BasicMathFunctions").is_dir() and (path / "Include").is_dir()

    env = os.environ.get(MUTATION_CHECKOUT_ENV)
    if env:
        # Setting the variable is a claim that the check can run. Skipping on a
        # bad path would let CI report green having exercised nothing.
        assert looks_like_a_checkout(Path(env)), (
            f"{MUTATION_CHECKOUT_ENV}={env} is not an ns-cmsis-nn checkout "
            f"(no Source/BasicMathFunctions and Include)"
        )
        return Path(env)
    nested = TESTER_ROOT.parents[1]
    return nested if looks_like_a_checkout(nested) else None


def _mutant(edits) -> Mutant:
    return Mutant(
        mutant_id="test_mutant",
        description="test",
        family="Test",
        edits=tuple(edits),
        expected_detected_by="test cases",
    )


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    (root / "Source" / "BasicMathFunctions").mkdir(parents=True)
    (root / "Source" / "BasicMathFunctions" / "kernel.c").write_text(
        "int f(void) { return a - b; }\nint g(void) { return a - b; }\n"
    )
    return root


class TestCatalogIntegrity:
    def test_ids_unique_and_nonempty(self):
        ids = [m.mutant_id for m in MUTANTS_V1]
        assert len(ids) == len(set(ids))
        assert all(ids)

    def test_every_mutant_has_edits_with_positive_counts(self):
        for mutant in MUTANTS_V1:
            assert mutant.edits, mutant.mutant_id
            for edit in mutant.edits:
                assert edit.count > 0
                assert edit.relpath.startswith(("Source/", "Include/"))

    def test_v1_covers_the_grounding_bug_classes(self):
        ids = {m.mutant_id for m in MUTANTS_V1}
        assert {"drop_conv_bias", "packed_sign_mask_343", "tail_loop_off_by_one",
                "sub_operand_swap", "requantize_shift_off_by_one", "broadcast_row_reuse"} <= ids

    def test_get_mutants_filters_and_rejects_unknown(self):
        assert [m.mutant_id for m in get_mutants(["sub_operand_swap"])] == ["sub_operand_swap"]
        with pytest.raises(KeyError, match="no_such_mutant"):
            get_mutants(["no_such_mutant"])


class TestPatchApplication:
    def test_apply_and_restore_roundtrip(self, tree: Path):
        target = tree / "Source" / "BasicMathFunctions" / "kernel.c"
        original = target.read_text()
        mutant = _mutant([
            Edit("Source/BasicMathFunctions/kernel.c", "a - b", "b - a", count=2)
        ])
        with AppliedMutant(tree, mutant):
            assert target.read_text().count("b - a") == 2
        assert target.read_text() == original

    def test_restore_happens_even_when_body_raises(self, tree: Path):
        target = tree / "Source" / "BasicMathFunctions" / "kernel.c"
        original = target.read_text()
        mutant = _mutant([
            Edit("Source/BasicMathFunctions/kernel.c", "a - b", "b - a", count=2)
        ])
        with pytest.raises(RuntimeError, match="boom"):
            with AppliedMutant(tree, mutant):
                raise RuntimeError("boom")
        assert target.read_text() == original

    def test_count_mismatch_is_loud_and_restores(self, tree: Path):
        target = tree / "Source" / "BasicMathFunctions" / "kernel.c"
        original = target.read_text()
        mutant = _mutant([
            Edit("Source/BasicMathFunctions/kernel.c", "a - b", "b - a", count=3)
        ])
        with pytest.raises(MutantApplyError, match="matched 2 time"):
            with AppliedMutant(tree, mutant):
                pass  # pragma: no cover
        assert target.read_text() == original

    def test_partial_apply_is_rolled_back(self, tree: Path):
        target = tree / "Source" / "BasicMathFunctions" / "kernel.c"
        original = target.read_text()
        mutant = _mutant([
            Edit("Source/BasicMathFunctions/kernel.c", "a - b", "b - a", count=2),
            Edit("Source/BasicMathFunctions/missing.c", "x", "y", count=1),
        ])
        with pytest.raises(MutantApplyError, match="missing.c"):
            with AppliedMutant(tree, mutant):
                pass  # pragma: no cover
        assert target.read_text() == original

    def test_regex_edit_with_named_group(self, tree: Path):
        target = tree / "Source" / "BasicMathFunctions" / "kernel.c"
        mutant = _mutant([
            Edit(
                "Source/BasicMathFunctions/kernel.c",
                r"return (?P<x>[ab]) - b;",
                r"return b - \g<x>;",
                count=2,
                regex=True,
            )
        ])
        with AppliedMutant(tree, mutant):
            assert target.read_text().count("return b - a;") == 2


class TestPrepareTree:
    def test_copies_source_and_include_only(self, tmp_path: Path):
        checkout = tmp_path / "checkout"
        (checkout / "Source").mkdir(parents=True)
        (checkout / "Include").mkdir()
        (checkout / "Source" / "a.c").write_text("int a;\n")
        (checkout / ".git").mkdir()
        tree = prepare_tree(checkout, tmp_path / "work")
        assert (tree / "Source" / "a.c").is_file()
        assert not (tree / ".git").exists()
        # The user's checkout is untouched by later edits to the copy.
        (tree / "Source" / "a.c").write_text("int mutated;\n")
        assert (checkout / "Source" / "a.c").read_text() == "int a;\n"

    def test_rejects_non_checkout(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="ns-cmsis-nn"):
            prepare_tree(tmp_path / "empty", tmp_path / "work")


class TestCaseDiscovery:
    def test_finds_case_dirs_with_includes(self, tmp_path: Path):
        case = tmp_path / "gen" / "BasicMathFunctions" / "add_default_s8"
        (case / "includes").mkdir(parents=True)
        (case / "add_default_s8_add.c").write_text("int main(void){return 0;}\n")
        not_a_case = tmp_path / "gen" / "BasicMathFunctions" / "stray_file.c"
        not_a_case.write_text("int x;\n")
        found = discover_cases([tmp_path / "gen"])
        assert found == [case]


class TestReportFormat:
    # Named rather than indexed: the catalog grows in place, and these
    # assertions are about the report, not about catalog order.
    def _mutant(self, mutant_id: str) -> Mutant:
        return next(m for m in MUTANTS_V1 if m.mutant_id == mutant_id)

    def _report(self) -> MutationReport:
        killed = MutantOutcome(
            self._mutant("packed_sign_mask_343"), STATUS_KILLED, killed_by=["chunked_add_s8"]
        )
        survived = MutantOutcome(self._mutant("drop_conv_bias"), STATUS_SURVIVED)
        return MutationReport(
            baseline_total=3,
            baseline_failed=[CaseResult("broken_case", "Fam", False, "exit 1")],
            scored_cases=["chunked_add_s8", "conv_case"],
            outcomes=[survived, killed],
            wall_time_s=12.3,
        )

    def test_headline_and_vacuous_candidates(self):
        report = self._report()
        assert report.killed_count == 1
        assert report.vacuous_case_candidates() == ["conv_case"]

    def test_to_dict_schema(self):
        d = self._report().to_dict()
        assert d["schema"] == "helia-core-tester/mutation-report/v1"
        assert d["headline"]["mutants_killed"] == 1
        assert d["headline"]["mutants_total"] == 2
        assert d["headline"]["vacuous_case_candidates"] == ["conv_case"]
        assert d["baseline"]["failed_cases"][0]["name"] == "broken_case"
        statuses = {m["id"]: m["status"] for m in d["mutants"]}
        assert statuses["drop_conv_bias"] == STATUS_SURVIVED
        json.dumps(d)  # must be serializable

    def test_render_text_mentions_survivors_and_vacuous(self):
        text = self._report().render_text()
        assert "SURVIVED  drop_conv_bias" in text
        assert "Vacuous-case candidates" in text
        assert "1/2 mutants killed" in text

    def test_apply_failures_surface(self):
        report = self._report()
        report.outcomes.append(
            MutantOutcome(
                self._mutant("tail_loop_off_by_one"),
                STATUS_APPLY_FAILED,
                detail="pattern matched 0 times",
            )
        )
        assert [o.mutant.mutant_id for o in report.apply_failures] == ["tail_loop_off_by_one"]
        assert "APPLY_FAILED" in report.render_text()


class TestFailureKinds:
    def test_killed_only_for_behavioural_failures(self):
        assert CaseResult("c", "F", False, kind=KIND_CASE_FAIL).killed
        assert CaseResult("c", "F", False, kind=KIND_TIMEOUT).killed
        assert not CaseResult("c", "F", False, kind=KIND_COMPILE_FAILED).killed
        assert not CaseResult("c", "F", False, kind=KIND_NO_SOURCE).killed
        assert not CaseResult("c", "F", True, kind=KIND_PASS).killed

    def test_compile_failure_is_tagged_not_killed(self, tmp_path: Path):
        """A case binary whose compile fails must come back KIND_COMPILE_FAILED."""
        case = tmp_path / "Fam" / "case_a"
        (case / "includes").mkdir(parents=True)
        (case / "case_a.c").write_text("int main(void){return 0;}\n")
        (tmp_path / "tree" / "Include").mkdir(parents=True)
        result = build_and_run_case(
            case,
            tmp_path / "tree",
            tmp_path / "lib.a",
            tmp_path / "runtime.o",
            TESTER_ROOT,
            tmp_path / "bin",
            cc="/bin/false",  # every compile invocation fails
        )
        assert not result.passed
        assert result.kind == KIND_COMPILE_FAILED
        assert not result.killed

    def test_missing_source_is_tagged(self, tmp_path: Path):
        case = tmp_path / "Fam" / "empty_case"
        (case / "includes").mkdir(parents=True)
        result = build_and_run_case(
            case, tmp_path, tmp_path / "lib.a", tmp_path / "rt.o", TESTER_ROOT, tmp_path / "bin"
        )
        assert result.kind == KIND_NO_SOURCE
        assert not result.killed


def _crafted_checkout(tmp_path: Path) -> Path:
    """A minimal fake ns-cmsis-nn checkout with one host-compilable kernel."""
    checkout = tmp_path / "checkout"
    # Driven off KERNEL_SOURCE_DIRS so a new kernel family in the host build
    # cannot silently leave this fabricated checkout incomplete.
    for sub in KERNEL_SOURCE_DIRS:
        (checkout / sub).mkdir(parents=True)
    (checkout / "Include").mkdir()
    # helia_test_runtime.h includes arm_nnfunctions.h for ARM_CMSIS_NN_SUCCESS;
    # stub just that out rather than depending on a real ns-cmsis-nn checkout.
    (checkout / "Include" / "arm_nnfunctions.h").write_text(
        "#ifndef ARM_NNFUNCTIONS_H\n"
        "#define ARM_NNFUNCTIONS_H\n"
        "typedef enum { ARM_CMSIS_NN_SUCCESS = 0 } arm_cmsis_nn_status;\n"
        "#endif\n"
    )
    (checkout / "Source" / "BasicMathFunctions" / "kernel.c").write_text(
        "#include <stdint.h>\n"
        "int32_t helia_mut_test_kernel(void) { return 42; }\n"
    )
    return checkout


def _crafted_case(tmp_path: Path) -> Path:
    case = tmp_path / "cases" / "CraftedFamily" / "crafted_case"
    (case / "includes").mkdir(parents=True)
    (case / "crafted_case.c").write_text(
        "#include <stdint.h>\n"
        "extern void helia_test_finish(int32_t failures);\n"
        "int32_t helia_mut_test_kernel(void);\n"
        "int main(void) { helia_test_finish(helia_mut_test_kernel() == 42 ? 0 : 1); return 0; }\n"
    )
    return case


_LINK_BREAKER = Mutant(
    mutant_id="link_breaker",
    description="renames the kernel symbol so every case binary fails to link",
    family="Test",
    edits=(
        Edit(
            "Source/BasicMathFunctions/kernel.c",
            "helia_mut_test_kernel(void) { return 42; }",
            "helia_mut_test_kernel_gone(void) { return 42; } /* MUTANT link_breaker */",
            count=1,
        ),
    ),
    expected_detected_by="nothing: this must be BUILD_FAILED, not a kill",
)

_WRONG_VALUE = Mutant(
    mutant_id="wrong_value",
    description="kernel returns the wrong value",
    family="Test",
    edits=(
        Edit(
            "Source/BasicMathFunctions/kernel.c",
            "return 42;",
            "return 41; /* MUTANT wrong_value */",
            count=1,
        ),
    ),
    expected_detected_by="the crafted case",
)

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="host gcc required")


@needs_gcc
class TestRunnerFailureClassification:
    """End-to-end crafted-mutant runs on a fabricated checkout (host gcc)."""

    def test_all_cases_compile_failing_is_build_failed_not_killed(self, tmp_path: Path):
        checkout = _crafted_checkout(tmp_path)
        case = _crafted_case(tmp_path)
        report = run_mutation_scoring(
            cmsis_nn_root=checkout,
            case_dirs=[case],
            mutants=[_LINK_BREAKER, _WRONG_VALUE],
            tester_root=TESTER_ROOT,
            workdir=tmp_path / "work",
            log=lambda *_: None,
        )
        by_id = {o.mutant.mutant_id: o for o in report.outcomes}
        broken = by_id["link_breaker"]
        assert broken.status == STATUS_BUILD_FAILED
        assert broken.killed_by == []
        assert "failed to compile/link" in broken.detail
        # BUILD_FAILED surfaces alongside APPLY_FAILED so the run exits nonzero.
        assert broken in report.apply_failures
        # The genuine behavioural mutant is still killed by the same case.
        assert by_id["wrong_value"].status == STATUS_KILLED
        assert by_id["wrong_value"].killed_by == ["crafted_case"]

    def test_mutant_needing_an_absent_capability_is_not_applicable_not_survived(
        self, tmp_path: Path
    ):
        # SURVIVED is a claim about the suite; a corpus that cannot contain a
        # killer has not made that claim either way.
        checkout = _crafted_checkout(tmp_path)
        case = _crafted_case(tmp_path)
        gated = replace(_WRONG_VALUE, mutant_id="gated", requires_capabilities=("mve",))
        report = run_mutation_scoring(
            cmsis_nn_root=checkout,
            case_dirs=[case],
            mutants=[gated, _WRONG_VALUE],
            tester_root=TESTER_ROOT,
            workdir=tmp_path / "work",
            capabilities=get_cpu_profile("cortex-m4").capabilities,
            log=lambda *_: None,
        )
        by_id = {o.mutant.mutant_id: o for o in report.outcomes}
        assert by_id["gated"].status == STATUS_NOT_APPLICABLE
        assert "mve" in by_id["gated"].detail
        assert report.survivors == []
        # The ungated control still scores, and the headline counts only what
        # the corpus could reach.
        assert by_id["wrong_value"].status == STATUS_KILLED
        assert report.to_dict()["headline"]["mutants_total"] == 1
        assert report.to_dict()["headline"]["mutants_not_applicable"] == 1
        assert "not applicable" in report.render_text()

    def test_a_present_capability_leaves_the_mutant_scored(self, tmp_path: Path):
        checkout = _crafted_checkout(tmp_path)
        case = _crafted_case(tmp_path)
        gated = replace(_WRONG_VALUE, mutant_id="gated", requires_capabilities=("mve",))
        report = run_mutation_scoring(
            cmsis_nn_root=checkout,
            case_dirs=[case],
            mutants=[gated],
            tester_root=TESTER_ROOT,
            workdir=tmp_path / "work",
            capabilities=get_cpu_profile("cortex-m55").capabilities,
            log=lambda *_: None,
        )
        assert report.outcomes[0].status == STATUS_KILLED

    def test_poisoned_restore_raises(self, tmp_path: Path, monkeypatch):
        """verify_pristine must abort the run when a restore leaves mutant markers."""
        checkout = _crafted_checkout(tmp_path)
        case = _crafted_case(tmp_path)
        monkeypatch.setattr(AppliedMutant, "_restore", lambda self: None)
        with pytest.raises(RuntimeError, match="not pristine"):
            run_mutation_scoring(
                cmsis_nn_root=checkout,
                case_dirs=[case],
                mutants=[_WRONG_VALUE],
                tester_root=TESTER_ROOT,
                workdir=tmp_path / "work",
                log=lambda *_: None,
            )




def _shipped_descriptors():
    return load_all_descriptors(str(TESTER_ROOT / "assets" / "descriptors"))


def _descriptors_named_by(descriptors, expected_detected_by: str):
    """Shipped descriptors a mutant's expected-killer prose names by kernel or case name."""
    words = set(re.findall(r"[a-z_0-9]+", expected_detected_by.lower()))
    return [
        d for d in descriptors
        if str(d.get("kernel") or "").lower() in words or str(d.get("name") or "").lower() in words
    ]


class TestVerifyPristine:
    def test_raises_on_leftover_marker(self, tree: Path):
        target = tree / "Source" / "BasicMathFunctions" / "kernel.c"
        target.write_text(target.read_text() + "/* MUTANT leftover */\n")
        with pytest.raises(RuntimeError, match="not pristine"):
            verify_pristine(tree, MUTANTS_V1)

    def test_passes_on_clean_tree(self, tree: Path):
        verify_pristine(tree, MUTANTS_V1)

    def test_v1_covers_the_issue_81_chunked_families(self):
        ids = {m.mutant_id for m in MUTANTS_V1}
        assert {"squared_difference_tail_drop", "minmax_no_broadcast_tail_drop",
                "requantize_tail_drop"} <= ids

    def test_capability_requirements_are_real_capabilities(self):
        # A typo would be unsatisfiable everywhere and turn the mutant into a
        # permanent NOT_APPLICABLE that no run ever scores.
        known = known_capabilities()
        for mutant in MUTANTS_V1:
            assert set(mutant.requires_capabilities) <= known, mutant.mutant_id

    def test_the_requantize_mutant_declares_the_capability_its_killers_need(self):
        # Its only killers are the MVE-gated chunked requantize cases, so on a
        # non-MVE CPU the corpus provably contains no case that could kill it.
        (mutant,) = [m for m in MUTANTS_V1 if m.mutant_id == "requantize_tail_drop"]
        assert mutant.requires_capabilities == ("mve",)
        assert get_cpu_profile("cortex-m55").capabilities >= set(mutant.requires_capabilities)
        assert not get_cpu_profile("cortex-m4").capabilities >= set(mutant.requires_capabilities)

    def test_the_shipped_requantize_descriptors_carry_the_mutant_gate(self):
        # The mutant's gate and its killers' gate are one fact stored twice:
        # loosening the descriptors without the mutant turns a real survivor
        # into a silent NOT_APPLICABLE, and the reverse hides a scoreable bug.
        descriptors = _shipped_descriptors()
        (mutant,) = [m for m in MUTANTS_V1 if m.mutant_id == "requantize_tail_drop"]
        killers = [
            d for d in descriptors
            if d.get("operator") == "ChunkedEquivalence" and d.get("kernel") == "requantize"
        ]
        assert killers
        for desc in killers:
            assert tuple(desc.get("required_capabilities") or ()) == mutant.requires_capabilities, desc["name"]

    def test_every_gated_mutant_has_a_killer_descriptor_with_exactly_that_gate(self):
        descriptors = _shipped_descriptors()
        for mutant in MUTANTS_V1:
            if not mutant.requires_capabilities:
                continue
            named = _descriptors_named_by(descriptors, mutant.expected_detected_by)
            assert named, f"{mutant.mutant_id}: expected_detected_by names no shipped descriptor"
            gates = {tuple(d.get("required_capabilities") or ()) for d in named}
            assert mutant.requires_capabilities in gates, mutant.mutant_id

    def test_the_default_cpu_covers_every_catalogued_gate(self):
        # The CLI default exists so one run can score the whole catalog; a
        # mutant needing a capability it lacks would be permanently excused.
        required = {c for m in MUTANTS_V1 for c in m.requires_capabilities}
        assert required <= get_cpu_profile(mutation_cli.DEFAULT_CPU).capabilities

    def test_every_v1_edit_applies_to_a_real_checkout(self, tmp_path: Path):
        """
        A catalogued edit is only worth anything if it still matches the
        kernel source. Applying the whole catalog to a copy of a real checkout
        is the only thing that proves the patterns and their exact replacement
        counts have not drifted; the fabricated-tree tests above cannot.
        """
        checkout = _kernel_checkout()
        if checkout is None:
            pytest.skip(f"no ns-cmsis-nn checkout: set {MUTATION_CHECKOUT_ENV} to one")
        tree = prepare_tree(checkout, tmp_path / "tree")
        for mutant in MUTANTS_V1:
            with AppliedMutant(tree, mutant):
                for edit in mutant.edits:
                    assert "/* MUTANT " in (tree / edit.relpath).read_text(), (
                        f"{mutant.mutant_id}: {edit.relpath} was not patched"
                    )
            verify_pristine(tree, [mutant])

    def test_every_v1_edit_leaves_a_marker_for_verify_pristine(self):
        """verify_pristine can only catch a poisoned restore if every
        replacement carries the MUTANT marker it scans for."""
        for mutant in MUTANTS_V1:
            for edit in mutant.edits:
                assert "/* MUTANT " in edit.replacement, (
                    f"{mutant.mutant_id}: replacement for {edit.relpath} has no marker"
                )


class _StubReport:
    """Stand-in for a scoring report: the CLI only renders it and reads its lists."""

    apply_failures = ()
    survivors = ()

    def render_text(self) -> str:
        return "stub report"


def _corpus_tree(tmp_path: Path, cpu: str = None, *, manifest_cpu: str = None) -> Path:
    """A minimal discoverable corpus, optionally placed in the layout for ``cpu``."""
    root = (tmp_path / "artifacts" / "generated_tests" / "int" / cpu) if cpu else (tmp_path / "cases")
    case_dir = root / "BasicMathFunctions" / "chunked_equivalence_requantize_singles_s8"
    (case_dir / "includes").mkdir(parents=True)
    (case_dir / "chunked_equivalence_requantize_singles_s8.c").write_text("int main(void){return 0;}\n")
    if manifest_cpu:
        (root / "manifest.json").write_text(json.dumps({"filters": {"cpu": manifest_cpu}, "tests": []}))
    return root


def _invoke_run(tmp_path: Path, cases_root: Path, extra=()):
    return CliRunner().invoke(
        mutation_cli.app,
        [
            "run",
            "--cmsis-nn-root", str(tmp_path / "checkout"),
            "--cases-root", str(cases_root),
            "--cc", "sh",
            *extra,
        ],
    )


class TestCorpusCapabilityDerivation:
    """--cases-root takes its cases from disk, so its capabilities must come
    from the same place: a corpus CPU inferred from --cpu instead can excuse a
    mutant as NOT_APPLICABLE while its killers sit in the tree (issue #81)."""

    def test_the_layout_path_names_the_cpu(self, tmp_path: Path):
        root = _corpus_tree(tmp_path, "cortex-m55")
        assert mutation_cli.derive_corpus_cpu([root], discover_cases([root])) == "cortex-m55"

    def test_the_manifest_names_the_cpu(self, tmp_path: Path):
        root = _corpus_tree(tmp_path, manifest_cpu="cortex-m4")
        assert mutation_cli.derive_corpus_cpu([root], discover_cases([root])) == "cortex-m4"

    def test_an_unlabelled_tree_derives_nothing(self, tmp_path: Path):
        root = _corpus_tree(tmp_path)
        assert mutation_cli.derive_corpus_cpu([root], discover_cases([root])) is None

    def test_a_mixed_tree_is_an_error(self, tmp_path: Path):
        root = _corpus_tree(tmp_path, "cortex-m55", manifest_cpu="cortex-m4")
        with pytest.raises(ValueError, match="more than one CPU"):
            mutation_cli.derive_corpus_cpu([root], discover_cases([root]))

    def test_the_tree_cpu_is_scored_not_the_default_cpu(self, tmp_path: Path, monkeypatch):
        captured = {}

        def fake_scoring(**kwargs):
            captured.update(kwargs)
            return _StubReport()

        monkeypatch.setattr(mutation_cli, "run_mutation_scoring", fake_scoring)
        result = _invoke_run(tmp_path, _corpus_tree(tmp_path, "cortex-m4"))
        assert result.exit_code == 0, result.output
        assert captured["capabilities"] == get_cpu_profile("cortex-m4").capabilities

    def test_a_cpu_contradicting_the_tree_is_refused(self, tmp_path: Path, monkeypatch):
        # The false-excuse scenario: cortex-m4 capabilities over a cortex-m55
        # corpus would report requantize_tail_drop NOT_APPLICABLE even though
        # its MVE-gated killers were generated into this very tree.
        def refuse_scoring(**kwargs):
            raise AssertionError("scoring must not start on a contradicted corpus CPU")

        monkeypatch.setattr(mutation_cli, "run_mutation_scoring", refuse_scoring)
        result = _invoke_run(tmp_path, _corpus_tree(tmp_path, "cortex-m55"), ["--cpu", "cortex-m4"])
        assert result.exit_code == 1
        assert "cortex-m55" in result.output

    def test_an_unlabelled_tree_requires_an_explicit_cpu(self, tmp_path: Path, monkeypatch):
        captured = {}

        def fake_scoring(**kwargs):
            captured.update(kwargs)
            return _StubReport()

        monkeypatch.setattr(mutation_cli, "run_mutation_scoring", fake_scoring)
        root = _corpus_tree(tmp_path)
        defaulted = _invoke_run(tmp_path, root)
        assert defaulted.exit_code == 1
        assert "--cpu" in defaulted.output
        assert not captured

        explicit = _invoke_run(tmp_path, root, ["--cpu", "cortex-m4"])
        assert explicit.exit_code == 0, explicit.output
        assert captured["capabilities"] == get_cpu_profile("cortex-m4").capabilities
