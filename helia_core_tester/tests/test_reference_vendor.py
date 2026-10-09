"""The vendored TFLM reference tree: pinned, hashed, and nothing but the closure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from helia_core_tester.generation.reference import host_build

REPO_ROOT = Path(__file__).resolve().parents[2]


def _vendor_script():
    spec = importlib.util.spec_from_file_location("vendor_tflm_reference", REPO_ROOT / "scripts" / "vendor_tflm_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_vendored_file_matches_the_manifest() -> None:
    manifest = host_build.verify_vendored_tree()
    assert manifest["schema"] == 1
    assert len(manifest["files"]) > 20


def test_manifest_pins_both_upstreams_by_full_commit() -> None:
    manifest = host_build.load_manifest()
    for name in ("tflite_micro", "gemmlowp"):
        commit = manifest["upstream"][name]["commit"]
        assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), name
        assert manifest["upstream"][name]["repo"].startswith("https://github.com/")


def test_both_licenses_ship_with_the_code() -> None:
    files = host_build.load_manifest()["files"]
    assert "third_party/tflite_micro/LICENSE" in files
    assert "third_party/gemmlowp/LICENSE" in files
    assert "Apache License" in (host_build.THIRD_PARTY / "tflite_micro" / "LICENSE").read_text()


# The only micro files vendored: TFLM's integer LSTM, driven directly by shim/hct_ref_lstm.cc.
_MICRO_LSTM = {
    "third_party/tflite_micro/tensorflow/lite/micro/kernels/lstm_eval.h",
    "third_party/tflite_micro/tensorflow/lite/micro/kernels/lstm_eval.cc",
    "third_party/tflite_micro/tensorflow/lite/micro/kernels/lstm_shared.h",
}


def test_nothing_from_the_micro_runtime_is_vendored() -> None:
    files = host_build.load_manifest()["files"]
    assert {f for f in files if "/tensorflow/lite/micro/" in f} == _MICRO_LSTM
    assert not [f for f in files if "/kernels/internal/optimized/" in f and not f.endswith("neon_check.h")]
    assert not [f for f in files if "flatbuffers" in f or "/schema/" in f]


def test_compiled_sources_are_manifested_kernel_files() -> None:
    manifest = host_build.load_manifest()
    assert manifest["sources"]
    for source in manifest["sources"]:
        assert source in manifest["files"]
        assert source.endswith(".cc")
        assert "/tensorflow/lite/kernels/internal/" in source or source in _MICRO_LSTM


def test_vendor_md_names_the_pinned_commits() -> None:
    manifest = host_build.load_manifest()
    text = (host_build.THIRD_PARTY / "VENDOR.md").read_text()
    for name in ("tflite_micro", "gemmlowp"):
        assert manifest["upstream"][name]["commit"] in text
    assert "690a2d72" in text


def test_allow_list_refuses_runtime_and_optimized_headers() -> None:
    vendor = _vendor_script()
    allowed = lambda rel: vendor._allowed(rel, vendor.TFLM_ALLOWED, vendor.TFLM_DENIED, vendor.TFLM_DENIED_EXCEPTIONS)  # noqa: E731
    assert allowed("tensorflow/lite/kernels/internal/reference/conv.h")
    assert allowed("tensorflow/lite/kernels/internal/optimized/neon_check.h")
    assert not allowed("tensorflow/lite/kernels/internal/optimized/optimized_ops.h")
    assert not allowed("tensorflow/lite/micro/micro_log.h")
    assert not allowed("tensorflow/lite/micro/kernels/kernel_util.h")
    assert allowed("tensorflow/lite/micro/kernels/lstm_eval.h")
    assert not allowed("tensorflow/lite/schema/schema_generated.h")
    assert not allowed("tensorflow/lite/kernels/kernel_util.h")


def test_closure_walk_fails_on_a_disallowed_include(tmp_path, monkeypatch) -> None:
    vendor = _vendor_script()
    tflm = tmp_path / "tflm"
    (tflm / "tensorflow" / "lite" / "micro").mkdir(parents=True)
    (tflm / "tensorflow" / "lite" / "micro" / "micro_interpreter.h").write_text("// runtime\n")
    gemmlowp = tmp_path / "gemmlowp"
    gemmlowp.mkdir()
    shim = tmp_path / "shim"
    (shim / "stubs").mkdir(parents=True)
    (shim / "bad.cc").write_text('#include "tensorflow/lite/micro/micro_interpreter.h"\n')
    monkeypatch.setattr(vendor, "SHIM_DIR", shim)
    monkeypatch.setattr(vendor, "STUBS_DIR", shim / "stubs")
    with pytest.raises(vendor.VendorError, match="outside the reference-kernel allow-list"):
        vendor.collect_closure(tflm, gemmlowp)


def test_closure_walk_fails_on_an_unresolvable_include(tmp_path, monkeypatch) -> None:
    vendor = _vendor_script()
    shim = tmp_path / "shim"
    (shim / "stubs").mkdir(parents=True)
    (shim / "bad.cc").write_text('#include "tensorflow/lite/kernels/internal/nowhere.h"\n')
    monkeypatch.setattr(vendor, "SHIM_DIR", shim)
    monkeypatch.setattr(vendor, "STUBS_DIR", shim / "stubs")
    (tmp_path / "tflm").mkdir()
    (tmp_path / "gemmlowp").mkdir()
    with pytest.raises(vendor.VendorError, match="unresolvable include"):
        vendor.collect_closure(tmp_path / "tflm", tmp_path / "gemmlowp")


def test_stubs_shadow_the_two_out_of_closure_headers() -> None:
    stubs = host_build.STUBS_DIR
    assert (stubs / "ruy" / "profiler" / "instrumentation.h").is_file()
    micro_log = (stubs / "tensorflow" / "lite" / "micro" / "micro_log.h").read_text()
    assert "TF_LITE_STRIP_ERROR_STRINGS" in micro_log
    assert "-DTF_LITE_STRIP_ERROR_STRINGS" in host_build.CXXFLAGS


def test_manifest_json_is_stable_and_sorted() -> None:
    manifest = json.loads(host_build.MANIFEST.read_text())
    assert list(manifest["files"]) == sorted(manifest["files"])
    assert manifest["sources"] == sorted(manifest["sources"])
