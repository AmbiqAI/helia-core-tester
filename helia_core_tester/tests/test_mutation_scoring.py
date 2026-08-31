"""
Tests for the mutation-scoring machinery (issue #76): catalog integrity,
patch application + tree restoration, case discovery, and report format.

These tests exercise the pure-python machinery on fabricated trees; they do
not compile kernels and need neither gcc nor an ns-cmsis-nn checkout.
"""

import json
from pathlib import Path

import pytest

from helia_core_tester.mutation.catalog import MUTANTS_V1, Edit, Mutant, get_mutants
from helia_core_tester.mutation.host_build import CaseResult, discover_cases
from helia_core_tester.mutation.patching import AppliedMutant, MutantApplyError
from helia_core_tester.mutation.runner import (
    STATUS_APPLY_FAILED,
    STATUS_KILLED,
    STATUS_SURVIVED,
    MutantOutcome,
    MutationReport,
    prepare_tree,
)


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
    def _report(self) -> MutationReport:
        killed = MutantOutcome(MUTANTS_V1[1], STATUS_KILLED, killed_by=["chunked_add_s8"])
        survived = MutantOutcome(MUTANTS_V1[0], STATUS_SURVIVED)
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
            MutantOutcome(MUTANTS_V1[2], STATUS_APPLY_FAILED, detail="pattern matched 0 times")
        )
        assert [o.mutant.mutant_id for o in report.apply_failures] == ["tail_loop_off_by_one"]
        assert "APPLY_FAILED" in report.render_text()
