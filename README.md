# CMSIS-NN Tools

A comprehensive toolkit for CMSIS-NN testing, model generation, and FVP simulation.

## Features

- **Complete Test Pipeline**: End-to-end workflow from model generation to FVP execution
- **TFLite Model Generation**: Generate TensorFlow Lite models using pytest
- **AOT Conversion**: Convert TFLite models to C modules using helia-aot
- **Test Runner Generation**: Automatically generate Unity test runners
- **FVP Building**: Build test executables for FVP Corstone-300 simulator
- **FVP Execution**: Run tests on FVP simulator with comprehensive reporting

## Getting Started (no installation required)

### Prerequisites

1. **Initialize submodules**:
   ```bash
   git submodule update --init --recursive
   ```

2. **Run the tool**:
   ```bash
   # From the repository root
   python3 cmsis_nn_tools/cli.py --help
   ```

## Usage

### Basic Usage

```bash
# Run complete pipeline (default: cortex-m55)
python3 cmsis_nn_tools/cli.py

# Specify CPU
python3 cmsis_nn_tools/cli.py --cpu cortex-m4
```

### Advanced Usage

```bash
# Run with specific filters
python3 cmsis_nn_tools/cli.py --op Conv2D --dtype S8 --limit 5

# Skip certain steps
python3 cmsis_nn_tools/cli.py --skip-generation --skip-conversion

# Dry run to see what would be done
python3 cmsis_nn_tools/cli.py --dry-run

# Verbose output with custom CPU
python3 cmsis_nn_tools/cli.py --cpu cortex-m3 --verbose

# Custom optimization level and jobs
python3 cmsis_nn_tools/cli.py --opt "-O2" --jobs 8
```

### Command Line Options

#### Pipeline Control
- `--skip-generation`: Skip TFLite model generation
- `--skip-conversion`: Skip TFLite to C conversion
- `--skip-runners`: Skip test runner generation
- `--skip-build`: Skip FVP build
- `--skip-run`: Skip FVP test execution

#### Generation Filters
- `--op OPERATOR`: Generate only specific operator (e.g., 'Conv2D')
- `--dtype DTYPE`: Generate only specific dtype (e.g., 'S8', 'S16')
- `--limit N`: Limit number of models to generate

#### Build Options
- `--cpu CPU`: Target CPU (default: cortex-m55)
- `--opt LEVEL`: Optimization level (default: -Ofast)
- `--jobs N`: Parallel build jobs (default: auto)

#### Run Options
- `--timeout SECONDS`: Per-test timeout in seconds (0 = none)
- `--no-fail-fast`: Don't stop on first test failure

#### General Options
- `--verbose, -v`: Show detailed output
- `--dry-run`: Show what would be done without actually doing it
- `--quiet, -q`: Reduce output verbosity
- `--log-file PATH`: Log file path

## Architecture

All Python lives under `cmsis_nn_tools/`:

### Core Modules
- `cmsis_nn_tools.core.pipeline`: Main pipeline orchestration
- `cmsis_nn_tools.core.config`: Configuration management
- `cmsis_nn_tools.core.logger`: Logging configuration

### Generator Modules
- `cmsis_nn_tools/generators/tflite_generator.py`: TFLite model generation and conversion
- `cmsis_nn_tools/generators/test_runner_generator.py`: Unity test runner generation

### Builder Modules
- `cmsis_nn_tools/builders/fvp_builder.py`: FVP build operations

### Runner Modules
- `cmsis_nn_tools/runners/fvp_runner.py`: FVP test execution

### Utility Modules
- `cmsis_nn_tools/utils/command_runner.py`: Command execution utilities

### TFLite Generator Tree (moved in-repo)
- `cmsis_nn_tools/tflite_generator/` contains:
  - `test_ops.py`, `conftest.py`
  - `tester/ops/`, `tester/io/`, `tester/descriptors/`

## Development

### Generate TFLite models directly (pytest)

```bash
cd cmsis_nn_tools/tflite_generator
pytest test_ops.py::test_generation -v --op mean_int16
```

### Code Quality (optional)

```bash
# Format
black cmsis_nn_tools/

# Lint
flake8 cmsis_nn_tools/

# Type check
mypy cmsis_nn_tools/
```

## Requirements

- Python 3.8+
- pytest
- TensorFlow Lite
- helia-aot
- CMake
- ARM GCC toolchain
- FVP Corstone-300 simulator