"""
Main TFLite model generation for Helia-Core Tester.
Thin generator that discovers YAML descriptors and generates TFLite models.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

from helia_core_tester.core.discovery import find_descriptors_dir, find_generated_tests_dir, find_repo_root
from helia_core_tester.core.cpu_targets import missing_required_capabilities, normalize_cpu
from helia_core_tester.generation.io.dtypes import descriptor_matches_dtype_filter, resolve_comparison, resolve_tensor_dtypes
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.core.path_layout import generation_report_dir
from helia_core_tester.generation.ops import get_op_map, get_operator_spec
from helia_core_tester.generation.reuse import (
    case_artifacts_present,
    case_stamp,
    clear_stamp,
    generator_version_hash,
    prune_unlisted_cases,
    read_stamp,
    write_stamp,
)
from helia_core_tester.generation.utils.temp_sizer_probe import kernel_source_exists, missing_header_symbols


def default_seed_for_case(name: str) -> int:
    """Deterministic per-case seed, independent of PYTHONHASHSEED."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


def _descriptor_family(desc: Dict[str, Any]) -> str:
    return str(desc.get("_family") or get_operator_spec(str(desc["operator"])).artifact_family_dir)


def _descriptor_parity_kind(desc: Dict[str, Any]) -> str:
    return str(desc.get("_parity_kind") or get_operator_spec(str(desc["operator"])).parity_kind)


def _descriptor_test_dir(root_dir: Path, desc: Dict[str, Any]) -> Path:
    return root_dir / _descriptor_family(desc) / str(desc["name"])


def _descriptor_suite(desc: Dict[str, Any]) -> str:
    return str(desc.get("_descriptor_suite") or desc.get("suite") or "default")


def _suite_mode(filters: Dict[str, Any]) -> str:
    return str(filters.get("suite") or "int").strip().lower()


def _float_precision_mode(filters: Dict[str, Any]) -> str:
    return str(filters.get("float_precision") or "both").strip().lower()


def should_run_test(desc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Determine if test should run based on filters.
    
    Args:
        desc: Test descriptor
        filters: Filter dictionary from command line
        
    Returns:
        True if test should run
    """
    if filters.get('name'):
        if desc['name'] != filters['name']:
            return False

    if filters.get('op'):
        filter_op = filters['op']
        desc_name = desc['name']
        base_name = desc.get('_base_name', None)
        source_stem = desc.get('_source_stem', None)
        source_relpath = desc.get('_source_relpath', None)
        desc_operator = desc.get('operator', None)
        
        name_matches = desc_name == filter_op or desc_name.startswith(filter_op + '_')
        base_matches = base_name == filter_op if base_name else False
        stem_matches = source_stem == filter_op if source_stem else False
        relpath_matches = source_relpath == filter_op if source_relpath else False
        operator_matches = desc_operator == filter_op if desc_operator else False
        
        if not name_matches and not base_matches and not stem_matches and not relpath_matches and not operator_matches:
            return False
        
    # Filter by activation dtype
    if filters.get('dtype') and not descriptor_matches_dtype_filter(desc, str(filters['dtype'])):
        return False

    descriptor_suite = _descriptor_suite(desc).strip().lower()
    suite_mode = _suite_mode(filters)
    is_float_descriptor = descriptor_suite == "float"
    if suite_mode == "int" and is_float_descriptor:
        return False
    if suite_mode == "float" and not is_float_descriptor:
        return False

    if suite_mode == "float":
        float_precision = _float_precision_mode(filters)
        resolved_tensor_dtypes = desc.get("resolved_tensor_dtypes") or resolve_tensor_dtypes(desc)
        has_f16 = any(dtype == "FP16" for dtype in resolved_tensor_dtypes.values())
        has_f32 = any(dtype == "FP32" for dtype in resolved_tensor_dtypes.values())
        if float_precision == "f16" and not has_f16:
            return False
        if float_precision == "f32" and not has_f32:
            return False
        
    # Filter by weight dtype
    resolved_tensor_dtypes = desc.get("resolved_tensor_dtypes") or resolve_tensor_dtypes(desc)
    if filters.get('wtype') and resolved_tensor_dtypes.get('weights') != str(filters['wtype']).upper():
        return False
            
    return True


def _required_capabilities(desc: Dict[str, Any]) -> list[str]:
    required: list[str] = []
    raw = desc.get("required_capabilities")
    if raw is not None:
        if isinstance(raw, str):
            required.append(raw)
        else:
            required.extend(str(capability) for capability in raw if str(capability).strip())

    resolved = _resolved_tensor_dtypes(desc)
    if any(dtype == "FP32" for dtype in resolved.values()):
        required.append("fp32_execution")
    if any(dtype == "FP16" for dtype in resolved.values()):
        required.append("fp16_execution")

    normalized: list[str] = []
    seen = set()
    for capability in required:
        capability_name = str(capability).strip()
        if capability_name and capability_name not in seen:
            normalized.append(capability_name)
            seen.add(capability_name)
    return normalized


def _required_kernel_symbols(desc: Dict[str, Any]) -> list[str]:
    """Public ns-cmsis-nn kernel symbols a descriptor's generated test calls.

    Descriptors listing ``required_kernel_symbols`` are generated only when
    every symbol is declared in the target ns-cmsis-nn checkout's public
    headers (per-symbol codegen probe, see temp_sizer_probe). This lets one
    tester pin serve several in-flight ns-cmsis-nn kernel branches: each
    branch declares only its own kernels and the rest skip cleanly instead
    of failing the build with references to missing symbols.
    """
    raw = desc.get("required_kernel_symbols")
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    ordered: list[str] = []
    seen = set()
    for symbol in raw:
        name = str(symbol).strip()
        if name and name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _resolved_tensor_dtypes(desc: Dict[str, Any]) -> Dict[str, str]:
    return dict(desc.get("resolved_tensor_dtypes") or resolve_tensor_dtypes(desc))


def _resolved_comparison(desc: Dict[str, Any]) -> Dict[str, Any]:
    return dict(desc.get("resolved_comparison") or resolve_comparison(desc, _resolved_tensor_dtypes(desc)))


def _skip_manifest_entry(
    desc: Dict[str, Any],
    *,
    cpu: str,
    missing_capabilities: list[str],
    status: str = "skipped_capability",
    missing_kernel_symbols: Optional[list[str]] = None,
) -> Dict[str, Any]:
    entry = {
        "name": desc.get("name"),
        "operator": desc.get("operator"),
        "family": _descriptor_family(desc),
        "parity_kind": _descriptor_parity_kind(desc),
        "activation_dtype": desc.get("activation_dtype"),
        "weight_dtype": desc.get("weight_dtype"),
        "resolved_tensor_dtypes": _resolved_tensor_dtypes(desc),
        "resolved_comparison": _resolved_comparison(desc),
        "descriptor_relpath": desc.get("_source_relpath"),
        "suite": _descriptor_suite(desc),
        "cpu": cpu,
        "status": status,
        "missing_capabilities": missing_capabilities,
        "required_capabilities": _required_capabilities(desc),
        "required_kernel_symbols": _required_kernel_symbols(desc),
    }
    if missing_kernel_symbols is not None:
        entry["missing_kernel_symbols"] = missing_kernel_symbols
    return entry


def _manifest_entry(
    desc: Dict[str, Any],
    *,
    test_dir: Path,
    generated_tests_dir: Path,
    cpu: str,
    reused: bool,
) -> Dict[str, Any]:
    return {
        "name": desc.get("name"),
        "operator": desc.get("operator"),
        "family": _descriptor_family(desc),
        "parity_kind": _descriptor_parity_kind(desc),
        "activation_dtype": desc.get("activation_dtype"),
        "weight_dtype": desc.get("weight_dtype"),
        "resolved_tensor_dtypes": _resolved_tensor_dtypes(desc),
        "resolved_comparison": _resolved_comparison(desc),
        "descriptor_relpath": desc.get("_source_relpath"),
        "suite": _descriptor_suite(desc),
        "path": str(test_dir),
        "relative_test_dir": str(test_dir.relative_to(generated_tests_dir)),
        "tflite": str(test_dir / f"{desc['name']}.tflite"),
        "c_sources": sorted(p.name for p in test_dir.glob("*.c")),
        "cpu": cpu,
        "reused": reused,
    }


def _reused_manifest_entry(
    test_dir: Path,
    *,
    generated_tests_dir: Path,
    cpu: str,
) -> Optional[Dict[str, Any]]:
    """Rebuild a reused case's manifest entry from what is on disk.

    Reading the sidecar back rather than reusing the in-memory descriptor keeps
    the entry a statement about the tree build and run will actually consume,
    and fails the reuse closed if the sidecar is unreadable.
    """
    try:
        sidecar = yaml.safe_load((test_dir / "descriptor.yaml").read_text())
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(sidecar, dict) or not sidecar.get("name") or not sidecar.get("operator"):
        return None
    return _manifest_entry(
        sidecar,
        test_dir=test_dir,
        generated_tests_dir=generated_tests_dir,
        cpu=cpu,
        reused=True,
    )


def generate_test(
    desc: Dict[str, Any],
    out_dir: str,
    seed: Optional[int] = None,
    cpu: str = "cortex-m55",
    conversion_failures: Optional[List[Dict[str, Any]]] = None,
    generation_failures: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Generate TFLite model for a descriptor.
    
    Args:
        desc: YAML descriptor
        out_dir: Output directory for generated files
        seed: Optional random seed (if None, uses hash of test name)
    """
    name = desc['name']
    operator = desc['operator']
    print(f"Generating test: {name} ({operator})")
    
    # Create output directory
    test_dir = _descriptor_test_dir(Path(out_dir), desc)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the complete descriptor as YAML in the test directory
    descriptor_path = test_dir / "descriptor.yaml"
    with open(descriptor_path, 'w') as f:
        yaml.dump(desc, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Get operation class
    op_map = get_op_map()
    if operator not in op_map:
        raise ValueError(f"Unsupported operator: {operator}")
        
    op_class = op_map[operator]
    
    # Initialize operation with deterministic seed
    if seed is None:
        seed = default_seed_for_case(name)
    op = op_class(desc, seed, target_cpu=cpu)
    
    # Build Keras model (skip for ops that generate LiteRT models directly)
    if op.needs_keras_model():
        try:
            model = op.build_keras_model()
        except Exception as e:
            if generation_failures is not None:
                import traceback
                generation_failures.append({
                    "name": name,
                    "operator": operator,
                    "family": _descriptor_family(desc),
                    "parity_kind": _descriptor_parity_kind(desc),
                    "stage": "build_model",
                    "exception": repr(e),
                    "traceback": traceback.format_exc(),
                })
            print(f"ERROR: Model build failed for {name} ({operator}): {e}")
            raise
    else:
        model = None
    
    # Convert to TFLite (some ops allow no-tflite fallback)
    tflite_path = test_dir / f"{name}.tflite"
    try:
        op.convert_to_tflite(model, str(tflite_path), seed)
        print(f"Generated TFLite model: {name}")
    except Exception as e:
        if op.allow_no_tflite():
            print(f"INFO: Skipping TFLite generation for {name}: {e}")
        else:
            if conversion_failures is not None:
                import traceback
                conversion_failures.append({
                    "name": name,
                    "operator": operator,
                    "family": _descriptor_family(desc),
                    "parity_kind": _descriptor_parity_kind(desc),
                    "exception": repr(e),
                    "traceback": traceback.format_exc(),
                })
            if generation_failures is not None:
                import traceback
                generation_failures.append({
                    "name": name,
                    "operator": operator,
                    "family": _descriptor_family(desc),
                    "parity_kind": _descriptor_parity_kind(desc),
                    "stage": "conversion",
                    "exception": repr(e),
                    "traceback": traceback.format_exc(),
                })
            print(f"ERROR: TFLite conversion failed for {name} ({operator}): {e}")
            raise
    
    # Generate C/H files from templates
    try:
        op.generate_c_files(test_dir)
        op.assert_input_mode_consumed()
    except NotImplementedError:
        # Operator doesn't support C file generation yet
        print(f"INFO: {name} - C file generation not implemented")
    except Exception as e:
        import traceback
        print(f"ERROR: Failed to generate C/H files for {name}: {e}")
        print(f"ERROR: Traceback:")
        traceback.print_exc()
        if generation_failures is not None:
            generation_failures.append({
                "name": name,
                "operator": operator,
                "family": _descriptor_family(desc),
                "parity_kind": _descriptor_parity_kind(desc),
                "stage": "c_files",
                "exception": repr(e),
                "traceback": traceback.format_exc(),
            })
        # Do not silently continue: an unexpected failure to generate C/H files
        # must fail this descriptor's generation (and therefore the overall
        # generation run) rather than being counted as successfully generated.
        raise


def test_generation(test_filters):
    """
    Generate TFLite models for all descriptors.
    """
    # Load all descriptors using discovery
    descriptors_dir = find_descriptors_dir()
    descriptors = load_all_descriptors(str(descriptors_dir))

    # Apply filters
    filtered_descriptors = []
    for desc in descriptors:
        if should_run_test(desc, test_filters):
            filtered_descriptors.append(desc)

    # Apply limit
    if test_filters.get('limit'):
        filtered_descriptors = filtered_descriptors[:test_filters['limit']]

    start_time = datetime.now(timezone.utc)
    target_cpu = normalize_cpu(test_filters.get('cpu') or "cortex-m55")
    suite_mode = _suite_mode(test_filters)
        
    # Generate TFLite models for each descriptor.
    generated_override = test_filters.get("generated_tests_dir")
    top_generated = (
        Path(generated_override).resolve()
        if generated_override
        else find_generated_tests_dir(cpu=target_cpu, suite=suite_mode, create=True)
    )
    top_generated.mkdir(parents=True, exist_ok=True)
    print(f"Generated tests output dir: {top_generated}")
    repo_root = find_repo_root()
    report_dir = generation_report_dir(repo_root, target_cpu, suite=suite_mode)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Place models in generated tests root
    generated_count = 0
    reused_count = 0
    force_generate = bool(test_filters.get("force_generate"))
    float_precision_mode = _float_precision_mode(test_filters)
    version_hash = generator_version_hash()
    manifest_entries: List[Dict[str, Any]] = []
    skipped_entries: List[Dict[str, Any]] = []
    conversion_failures: List[Dict[str, Any]] = []
    generation_failures: List[Dict[str, Any]] = []
    # Per-run cache for the per-symbol codegen probe (one header scan per
    # distinct kernel symbol, however many descriptors require it).
    symbol_declared_cache: Dict[str, bool] = {}
    symbol_source_cache: Dict[str, bool] = {}
    for desc in filtered_descriptors:
        missing_capabilities = missing_required_capabilities(target_cpu, _required_capabilities(desc))
        if missing_capabilities:
            skipped_entries.append(
                _skip_manifest_entry(
                    desc,
                    cpu=target_cpu,
                    missing_capabilities=missing_capabilities,
                )
            )
            print(
                f"Skipping {desc['name']} on {target_cpu}: missing capabilities "
                f"{', '.join(missing_capabilities)}"
            )
            continue
        missing_symbols: List[str] = []
        for symbol in _required_kernel_symbols(desc):
            if symbol not in symbol_declared_cache:
                symbol_declared_cache[symbol] = not missing_header_symbols([symbol])
            if not symbol_declared_cache[symbol]:
                missing_symbols.append(symbol)
        # Backstop (hct#92 review): a symbol the header probe reports as
        # undeclared while its kernel source ships in the checkout means the
        # probe missed a declaration (renamed symbol, or a public header
        # outside the probed set). Silently skipping would delete every
        # dependent case with exit 0 and green CI, so fail generation loudly
        # instead. Genuinely absent kernels (no source file) keep the skip.
        contradicted_symbols: List[str] = []
        for symbol in missing_symbols:
            if symbol not in symbol_source_cache:
                symbol_source_cache[symbol] = kernel_source_exists(symbol)
            if symbol_source_cache[symbol]:
                contradicted_symbols.append(symbol)
        if contradicted_symbols:
            message = (
                f"Kernel symbol probe contradiction for {desc['name']}: "
                f"{', '.join(contradicted_symbols)} declared in no probed public header, "
                f"but the checkout ships Source/**/<symbol>.c. The probe missed a "
                f"declaration (renamed symbol or unprobed public header); refusing to "
                f"silently skip. Update _PUBLIC_HEADER_NAMES / required_kernel_symbols."
            )
            print(f"ERROR: {message}")
            generation_failures.append({
                "name": desc.get("name"),
                "operator": desc.get("operator"),
                "family": _descriptor_family(desc),
                "parity_kind": _descriptor_parity_kind(desc),
                "stage": "kernel_symbol_probe",
                "exception": message,
            })
            continue
        if missing_symbols:
            skipped_entries.append(
                _skip_manifest_entry(
                    desc,
                    cpu=target_cpu,
                    missing_capabilities=[],
                    status="skipped_kernel_symbol",
                    missing_kernel_symbols=missing_symbols,
                )
            )
            print(
                f"Skipping {desc['name']}: ns-cmsis-nn checkout does not declare "
                f"{', '.join(missing_symbols)}"
            )
            continue
        case_name = str(desc["name"])
        test_dir = _descriptor_test_dir(Path(top_generated), desc)
        stamp = case_stamp(
            desc,
            case_name=case_name,
            cpu=target_cpu,
            suite=suite_mode,
            float_precision=float_precision_mode,
            seed=test_filters.get("seed") if test_filters.get("seed") is not None
            else default_seed_for_case(case_name),
            version_hash=version_hash,
        )
        if (
            not force_generate
            and read_stamp(test_dir) == stamp
            and case_artifacts_present(test_dir, case_name)
        ):
            reused_entry = _reused_manifest_entry(
                test_dir, generated_tests_dir=Path(top_generated), cpu=target_cpu
            )
            if reused_entry is not None:
                manifest_entries.append(reused_entry)
                reused_count += 1
                continue
        try:
            # The stamp goes away before any file is touched and comes back only
            # once the case is whole, so an interrupted run can never leave a
            # stamp standing over half-written output.
            clear_stamp(test_dir)
            generate_test(
                desc,
                str(top_generated),
                seed=test_filters.get('seed'),
                cpu=target_cpu,
                conversion_failures=conversion_failures,
                generation_failures=generation_failures,
            )
            manifest_entries.append(
                _manifest_entry(
                    desc,
                    test_dir=test_dir,
                    generated_tests_dir=Path(top_generated),
                    cpu=target_cpu,
                    reused=False,
                )
            )
            write_stamp(test_dir, stamp)
            generated_count += 1
        except Exception as e:
            print(f"Failed to generate TFLite model for {desc['name']}: {e}")
            # Continue with other models
            continue
            
    conversion_failures = sorted(
        conversion_failures,
        key=lambda item: (str(item.get("name", "")), str(item.get("operator", "")), str(item.get("exception", ""))),
    )
    generation_failures = sorted(
        generation_failures,
        key=lambda item: (
            str(item.get("name", "")),
            str(item.get("operator", "")),
            str(item.get("stage", "")),
            str(item.get("exception", "")),
        ),
    )
    skipped_entries = sorted(
        skipped_entries,
        key=lambda item: (
            str(item.get("name", "")),
            str(item.get("operator", "")),
        ),
    )

    produced_count = generated_count + reused_count
    pruned_count = prune_unlisted_cases(
        top_generated,
        {str(entry["relative_test_dir"]) for entry in manifest_entries},
    )
    if pruned_count:
        print(f"Pruned {pruned_count} case director(ies) outside the active filter")

    print(
        f"Successfully produced {produced_count} test case(s): "
        f"{generated_count} generated, {reused_count} reused"
    )
    conversion_failures_path = report_dir / "conversion_failures.json"
    conversion_failures_path.write_text(json.dumps(conversion_failures, indent=2))
    if conversion_failures:
        failed_names = ", ".join(f["name"] for f in conversion_failures)
        print(f"Conversion failures ({len(conversion_failures)}): {failed_names}")
    else:
        print("Conversion failures (0)")
    generation_failures_path = report_dir / "generation_failures.json"
    generation_failures_path.write_text(json.dumps(generation_failures, indent=2))
    if generation_failures:
        failed_names = ", ".join(f["name"] for f in generation_failures)
        print(f"Generation failures ({len(generation_failures)}): {failed_names}")
    else:
        print("Generation failures (0)")
    capability_skips_path = report_dir / "capability_skips.json"
    capability_skips_path.write_text(json.dumps(skipped_entries, indent=2))
    if skipped_entries:
        skipped_names = ", ".join(f["name"] for f in skipped_entries)
        print(f"Capability skips ({len(skipped_entries)}): {skipped_names}")
    else:
        print("Capability skips (0)")

    manifest_path: Optional[Path] = None
    if produced_count > 0 or skipped_entries:
        manifest_path = _write_manifest_and_cmake(
            manifest_entries,
            skipped_entries,
            test_filters,
            generated_tests_dir=top_generated,
            cpu=target_cpu,
        )

    manifest_pointer_path = report_dir / "manifest_pointer.json"
    manifest_pointer = {
        "cpu": target_cpu,
        "generated_tests_dir": str(top_generated),
        "manifest_path": str(manifest_path) if manifest_path else None,
    }
    if manifest_path:
        try:
            manifest_pointer["manifest_relative_path"] = str(manifest_path.relative_to(repo_root))
        except ValueError:
            manifest_pointer["manifest_relative_path"] = str(manifest_path)
    manifest_pointer_path.write_text(json.dumps(manifest_pointer, indent=2))

    end_time = datetime.now(timezone.utc)
    status = "success"
    if conversion_failures or generation_failures:
        status = "partial_failure" if produced_count > 0 or skipped_entries else "failed_no_generated_tests"
    elif produced_count == 0 and skipped_entries:
        status = "skipped_only"
    elif produced_count == 0:
        status = "failed_no_generated_tests"

    summary = {
        "status": status,
        "cpu": target_cpu,
        "families": sorted({entry["family"] for entry in [*manifest_entries, *skipped_entries]}),
        "parity_kind_counts": {
            parity_kind: sum(
                1
                for entry in [*manifest_entries, *skipped_entries]
                if entry["parity_kind"] == parity_kind
            )
            for parity_kind in sorted({entry["parity_kind"] for entry in [*manifest_entries, *skipped_entries]})
        },
        "timestamps": {
            "started_utc": start_time.isoformat(),
            "ended_utc": end_time.isoformat(),
            "duration_seconds": round((end_time - start_time).total_seconds(), 3),
        },
        "filters": {
            "op": test_filters.get("op"),
            "dtype": test_filters.get("dtype"),
            "wtype": test_filters.get("wtype"),
            "name": test_filters.get("name"),
            "limit": test_filters.get("limit"),
            "seed": test_filters.get("seed"),
            "suite": suite_mode,
            "float_precision": float_precision_mode,
            "force_generate": force_generate,
        },
        "counts": {
            "descriptors_total": len(descriptors),
            "descriptors_after_filters": len(filtered_descriptors),
            "generated": generated_count,
            "reused": reused_count,
            "cases_total": produced_count,
            "pruned": pruned_count,
            "skipped_capability": sum(
                1 for entry in skipped_entries if entry.get("status") == "skipped_capability"
            ),
            "skipped_kernel_symbol": sum(
                1 for entry in skipped_entries if entry.get("status") == "skipped_kernel_symbol"
            ),
            "conversion_failures": len(conversion_failures),
            "generation_failures": len(generation_failures),
        },
        "outputs": {
            "generated_tests_dir": str(top_generated),
            "conversion_failures": str(conversion_failures_path),
            "generation_failures": str(generation_failures_path),
            "capability_skips": str(capability_skips_path),
            "manifest_pointer": str(manifest_pointer_path),
        },
    }
    (report_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    assert produced_count > 0 or skipped_entries, "No TFLite models were generated"
    assert not conversion_failures, (
        f"Conversion failures occurred for {len(conversion_failures)} descriptor(s): "
        f"{', '.join(str(f.get('name')) for f in conversion_failures)}"
    )
    assert not generation_failures, (
        f"Generation failures occurred for {len(generation_failures)} descriptor(s): "
        f"{', '.join(str(f.get('name')) for f in generation_failures)}"
    )


def _write_manifest_and_cmake(
    entries: List[Dict[str, Any]],
    skipped_entries: List[Dict[str, Any]],
    test_filters: Dict[str, Any],
    generated_tests_dir: Path,
    cpu: str
) -> Path:
    repo_root = find_repo_root()

    manifest = {
        "generated_count": len(entries),
        "regenerated_count": sum(1 for entry in entries if not entry.get("reused")),
        "reused_count": sum(1 for entry in entries if entry.get("reused")),
        "skipped_count": len(skipped_entries),
        "families": sorted({str(entry["family"]) for entry in [*entries, *skipped_entries]}),
        "parity_kinds": sorted({str(entry["parity_kind"]) for entry in [*entries, *skipped_entries]}),
        "filters": {
            "op": test_filters.get("op"),
            "dtype": test_filters.get("dtype"),
            "wtype": test_filters.get("wtype"),
            "name": test_filters.get("name"),
            "limit": test_filters.get("limit"),
            "seed": test_filters.get("seed"),
            "cpu": cpu,
            "suite": _suite_mode(test_filters),
            "float_precision": _float_precision_mode(test_filters),
        },
        "tests": entries,
        "skipped": skipped_entries,
    }
    manifest_path = generated_tests_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    try:
        rel_root = generated_tests_dir.relative_to(repo_root)
    except ValueError:
        rel_root = generated_tests_dir
    runnable_entries = [entry for entry in entries if entry.get("c_sources")]
    test_dirs = sorted(
        {
            str(Path(rel_root) / Path(str(entry["relative_test_dir"])))
            for entry in runnable_entries
        }
    )
    cmake_lines = ["set(GENERATED_TEST_DIRS"]
    for d in test_dirs:
        cmake_lines.append(f"  \"{d}\"")
    cmake_lines.append(")")
    cmake_path = generated_tests_dir / "tests.cmake"
    cmake_path.write_text("\n".join(cmake_lines) + "\n")
    return manifest_path


def test_generated_files_exist(test_filters):
    """
    Verify that generated TFLite files exist and are valid.
    This should run AFTER test_generation().
    """
    # Don't generate, just validate what test_generation() created
    generated_override = test_filters.get("generated_tests_dir")
    target_cpu = normalize_cpu(test_filters.get('cpu') or "cortex-m55")
    suite_mode = _suite_mode(test_filters)
    generated_tests_dir = (
        Path(generated_override).resolve()
        if generated_override
        else find_generated_tests_dir(cpu=target_cpu, suite=suite_mode, create=False)
    )
    if not generated_tests_dir.exists():
        pytest.skip("No generated tests found")
        
    # Check that we have some generated tests
    test_dirs = sorted(descriptor_file.parent for descriptor_file in generated_tests_dir.rglob("descriptor.yaml"))
    if not test_dirs:
        manifest_path = generated_tests_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("skipped_count", 0) > 0 and manifest.get("generated_count", 0) == 0:
                pytest.skip("Only capability-skipped descriptors are present")
        assert len(test_dirs) > 0, "No test directories found"
    
    # Check that each test has TFLite file or generated headers
    for test_dir in test_dirs:
        name = test_dir.name
        tflite_file = test_dir / f"{name}.tflite"
        if tflite_file.exists():
            # Check that file is not empty
            assert tflite_file.stat().st_size > 0, f"{name}.tflite is empty"
            continue

        includes_dir = test_dir / "includes"
        headers = list(includes_dir.glob("*.h")) if includes_dir.exists() else []
        assert headers, f"Missing {name}.tflite and no generated headers"
