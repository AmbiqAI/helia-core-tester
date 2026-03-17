"""
Descriptor tracking for test reporting.

This module provides functionality to load descriptors, map tests to descriptors,
and determine descriptor status based on test results and artifact presence.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.reporting.models import TestStatus, TestResult


class DescriptorTracker:
    """Track descriptors and map them to test results."""
    
    def __init__(self, descriptors_dir: Path):
        """
        Initialize descriptor tracker.
        
        Args:
            descriptors_dir: Path to directory containing YAML descriptors
        """
        self.descriptors_dir = descriptors_dir
        self._descriptors: Dict[str, Dict] = {}
        self._descriptor_paths: Dict[str, Path] = {}
    
    def load_all_descriptors(self) -> Dict[str, Dict]:
        """
        Load all descriptors from the descriptors directory.
        
        Returns:
            Dictionary mapping descriptor name to descriptor content
        """
        if not self.descriptors_dir.exists():
            return {}
        
        try:
            descriptors_list = load_all_descriptors(str(self.descriptors_dir))
            
            for desc in descriptors_list:
                name = desc.get('name')
                if name:
                    self._descriptors[name] = desc
                    source_relpath = desc.get("_source_relpath")
                    if source_relpath:
                        self._descriptor_paths[name] = self.descriptors_dir / str(source_relpath)
                    else:
                        self._descriptor_paths[name] = self._find_descriptor_file(name)
            
            return self._descriptors
        except Exception as e:
            print(f"Warning: Failed to load descriptors from {self.descriptors_dir}: {e}")
            return {}
    
    def _find_descriptor_file(self, descriptor_name: str) -> Path:
        """Find the YAML file containing this descriptor."""
        for yaml_file in self.descriptors_dir.rglob("*.yaml"):
            stem = yaml_file.stem
            if descriptor_name == stem or descriptor_name.startswith(f"{stem}_"):
                return yaml_file
        return self.descriptors_dir / f"{descriptor_name}.yaml"

    def _descriptor_family(self, descriptor_name: str) -> Optional[str]:
        descriptor = self._descriptors.get(descriptor_name)
        if not descriptor:
            return None
        family = descriptor.get("_family")
        return str(family) if family else None

    def generated_test_dir_for(self, descriptor_name: str, generated_tests_dir: Path) -> Path:
        family = self._descriptor_family(descriptor_name)
        if family:
            return generated_tests_dir / family / descriptor_name
        matches = list(generated_tests_dir.rglob(descriptor_name))
        for match in matches:
            if match.is_dir():
                return match
        return generated_tests_dir / descriptor_name

    def elf_path_for(self, descriptor_name: str, build_dir: Path) -> Path:
        family = self._descriptor_family(descriptor_name)
        if family:
            return build_dir / "tests" / family / f"{descriptor_name}.elf"
        matches = list((build_dir / "tests").rglob(f"{descriptor_name}.elf"))
        if matches:
            return matches[0]
        return build_dir / "tests" / f"{descriptor_name}.elf"
    
    def map_test_to_descriptor(self, test_name: str, descriptors: Optional[Dict[str, Dict]] = None) -> Optional[Dict]:
        """
        Find descriptor for a given test name.
        
        Args:
            test_name: Name of the test (from ELF file stem)
            descriptors: Optional descriptor dictionary (uses loaded if not provided)
            
        Returns:
            Descriptor dictionary if found, None otherwise
        """
        if descriptors is None:
            descriptors = self._descriptors
        
        if test_name in descriptors:
            return descriptors[test_name]
        
        for desc_name, desc in descriptors.items():
            if test_name.startswith(f"{desc_name}_") or desc_name.startswith(f"{test_name}_"):
                return desc
        
        return None
    
    def find_missing_tests(self, descriptors: Dict[str, Dict], test_results: List[TestResult]) -> List[str]:
        """
        Find descriptors that don't have corresponding test results.
        
        Args:
            descriptors: Dictionary of all descriptors
            test_results: List of test results
            
        Returns:
            List of descriptor names without test results
        """
        test_names = {result.test_name for result in test_results}
        missing = []
        
        for desc_name in descriptors.keys():
            found = False
            for test_name in test_names:
                if (
                    desc_name == test_name
                    or desc_name.startswith(f"{test_name}_")
                    or test_name.startswith(f"{desc_name}_")
                ):
                    found = True
                    break
            if not found:
                missing.append(desc_name)
        
        return missing
    
    def determine_descriptor_status(
        self, 
        descriptor_name: str, 
        test_result: Optional[TestResult],
        build_dir: Path,
        generated_tests_dir: Path
    ) -> Tuple[TestStatus, Optional[str], Optional[str]]:
        """
        Determine descriptor status based on test result and artifact presence.
        
        Args:
            descriptor_name: Name of the descriptor
            test_result: Test result if test was executed
            build_dir: Build directory to check for ELF files
            generated_tests_dir: Generated tests directory to check for artifacts
            
        Returns:
            Tuple of (status, failure_stage, failure_reason)
        """
        if test_result:
            if test_result.status == TestStatus.PASS:
                return TestStatus.PASS, None, None
            elif test_result.status == TestStatus.FAIL:
                return TestStatus.FAIL, "execution", test_result.failure_reason
            elif test_result.status == TestStatus.TIMEOUT:
                return TestStatus.TIMEOUT, "execution", "Test execution timed out"
            elif test_result.status == TestStatus.ERROR:
                return TestStatus.ERROR, "execution", test_result.failure_reason or "Test execution error"
            elif test_result.status == TestStatus.SKIP:
                return TestStatus.SKIP, None, test_result.skip_reason
            else:
                return test_result.status, "execution", test_result.failure_reason
        
        elf_path = self.elf_path_for(descriptor_name, build_dir)
        if not elf_path.exists():
            # Check for generated header files (new template system uses operator-specific names)
            # Look for any header file matching the pattern: {descriptor_name}_*.h
            generated_test_dir = self.generated_test_dir_for(descriptor_name, generated_tests_dir)
            includes_api_dir = generated_test_dir / "includes"
            model_headers = list(includes_api_dir.glob(f"{descriptor_name}_*.h")) if includes_api_dir.exists() else []
            model_header_old = generated_test_dir / "includes" / f"{descriptor_name}_model.h"
            
            if not model_headers and not model_header_old.exists():
                tflite_file = generated_test_dir / f"{descriptor_name}.tflite"
                if not tflite_file.exists():
                    return TestStatus.GENERATION_FAILED, "generation", "TFLite model not generated"
                else:
                    return TestStatus.CONVERSION_FAILED, "conversion", "Model header not found"
            else:
                return TestStatus.BUILD_FAILED, "build", "ELF file not found in build directory"
        else:
            return TestStatus.NOT_RUN, None, "Test not executed"
    
    def get_descriptor_path(self, descriptor_name: str) -> Path:
        """Get the file path for a descriptor."""
        return self._descriptor_paths.get(descriptor_name, 
                                          self.descriptors_dir / f"{descriptor_name}.yaml")
