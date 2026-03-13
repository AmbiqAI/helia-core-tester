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
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.core.cpu_targets import normalize_cpu
from helia_core_tester.core.path_layout import generation_report_dir
from helia_core_tester.generation.ops import get_op_map, get_operator_spec


def _descriptor_family(desc: Dict[str, Any]) -> str:
    return str(desc.get("_family") or get_operator_spec(str(desc["operator"])).artifact_family_dir)


def _descriptor_parity_kind(desc: Dict[str, Any]) -> str:
    return str(desc.get("_parity_kind") or get_operator_spec(str(desc["operator"])).parity_kind)


def _descriptor_test_dir(root_dir: Path, desc: Dict[str, Any]) -> Path:
    return root_dir / _descriptor_family(desc) / str(desc["name"])


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
    if filters.get('dtype') and desc['activation_dtype'] != filters['dtype']:
        return False
        
    # Filter by weight dtype
    if filters.get('wtype') and desc['weight_dtype'] != filters['wtype']:
        return False
            
    return True


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
        # Stable deterministic seed from name (independent of PYTHONHASHSEED)
        seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")
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
        # Continue anyway - C generation is optional during transition


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
        
    # Generate TFLite models for each descriptor.
    generated_override = test_filters.get("generated_tests_dir")
    top_generated = (
        Path(generated_override).resolve()
        if generated_override
        else find_generated_tests_dir(cpu=target_cpu, create=True)
    )
    top_generated.mkdir(parents=True, exist_ok=True)
    print(f"Generated tests output dir: {top_generated}")
    repo_root = find_repo_root()
    report_dir = generation_report_dir(repo_root, target_cpu)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Place models in generated tests root
    generated_count = 0
    manifest_entries: List[Dict[str, Any]] = []
    conversion_failures: List[Dict[str, Any]] = []
    generation_failures: List[Dict[str, Any]] = []
    for desc in filtered_descriptors:
        try:
            generate_test(
                desc,
                str(top_generated),
                seed=test_filters.get('seed'),
                cpu=target_cpu,
                conversion_failures=conversion_failures,
                generation_failures=generation_failures,
            )
            test_dir = _descriptor_test_dir(Path(top_generated), desc)
            tflite_path = test_dir / f"{desc['name']}.tflite"
            c_sources = sorted([str(p.name) for p in test_dir.glob("*.c")])
            relative_test_dir = str(test_dir.relative_to(top_generated))
            manifest_entries.append({
                "name": desc.get("name"),
                "operator": desc.get("operator"),
                "family": _descriptor_family(desc),
                "parity_kind": _descriptor_parity_kind(desc),
                "activation_dtype": desc.get("activation_dtype"),
                "weight_dtype": desc.get("weight_dtype"),
                "descriptor_relpath": desc.get("_source_relpath"),
                "path": str(test_dir),
                "relative_test_dir": relative_test_dir,
                "tflite": str(tflite_path),
                "c_sources": c_sources,
                "cpu": target_cpu,
            })
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

    print(f"Successfully generated {generated_count} TFLite models")
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

    manifest_path: Optional[Path] = None
    if generated_count > 0:
        manifest_path = _write_manifest_and_cmake(
            manifest_entries,
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
    if generated_count == 0:
        status = "failed_no_generated_tests"
    elif conversion_failures or generation_failures:
        status = "partial_failure"

    summary = {
        "status": status,
        "cpu": target_cpu,
        "families": sorted({entry["family"] for entry in manifest_entries}),
        "parity_kind_counts": {
            parity_kind: sum(1 for entry in manifest_entries if entry["parity_kind"] == parity_kind)
            for parity_kind in sorted({entry["parity_kind"] for entry in manifest_entries})
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
        },
        "counts": {
            "descriptors_total": len(descriptors),
            "descriptors_after_filters": len(filtered_descriptors),
            "generated": generated_count,
            "conversion_failures": len(conversion_failures),
            "generation_failures": len(generation_failures),
        },
        "outputs": {
            "generated_tests_dir": str(top_generated),
            "conversion_failures": str(conversion_failures_path),
            "generation_failures": str(generation_failures_path),
            "manifest_pointer": str(manifest_pointer_path),
        },
    }
    (report_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    assert generated_count > 0, "No TFLite models were generated"


def _write_manifest_and_cmake(
    entries: List[Dict[str, Any]],
    test_filters: Dict[str, Any],
    generated_tests_dir: Path,
    cpu: str
) -> Path:
    repo_root = find_repo_root()

    manifest = {
        "generated_count": len(entries),
        "families": sorted({str(entry["family"]) for entry in entries}),
        "parity_kinds": sorted({str(entry["parity_kind"]) for entry in entries}),
        "filters": {
            "op": test_filters.get("op"),
            "dtype": test_filters.get("dtype"),
            "wtype": test_filters.get("wtype"),
            "name": test_filters.get("name"),
            "limit": test_filters.get("limit"),
            "seed": test_filters.get("seed"),
            "cpu": cpu,
        },
        "tests": entries,
    }
    manifest_path = generated_tests_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    try:
        rel_root = generated_tests_dir.relative_to(repo_root)
    except ValueError:
        rel_root = generated_tests_dir
    test_dirs = sorted({str(Path(rel_root) / Path(str(e["relative_test_dir"]))) for e in entries})
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
    generated_tests_dir = (
        Path(generated_override).resolve()
        if generated_override
        else find_generated_tests_dir(cpu=target_cpu, create=False)
    )
    if not generated_tests_dir.exists():
        pytest.skip("No generated tests found")
        
    # Check that we have some generated tests
    test_dirs = sorted(descriptor_file.parent for descriptor_file in generated_tests_dir.rglob("descriptor.yaml"))
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
