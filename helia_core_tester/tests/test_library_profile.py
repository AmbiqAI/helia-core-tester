"""hct-library.yaml, --library and its interplay with --cmsis-nn-root (plan E1)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helia_core_tester import cli as cli_module
from helia_core_tester.contract import cli as contract_cli
from helia_core_tester.core import library as library_module
from helia_core_tester.core.config import Config
from helia_core_tester.core.errors import ConfigurationError
from helia_core_tester.core.library import (
    PROFILE_FILE,
    LibraryProfileError,
    load_profile,
    require_supported,
    resolve_library,
)

LIBRARY_ENV = "HELIA_CORE_TESTER_LIBRARY"
ROOT_ENV = "HELIA_CORE_TESTER_CMSIS_NN_ROOT"


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "tester"
    (root / "helia_core_tester" / "generation").mkdir(parents=True)
    (root / "helia_core_tester" / "fvp").mkdir(parents=True)
    (root / "helia_core_tester" / "fvp" / "build_and_run_fvp.py").write_text("")
    return root


def _nn_checkout(tmp_path: Path, name: str = "ns-cmsis-nn") -> Path:
    root = tmp_path / name
    (root / "Include").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("project(cmsis-nn)\n")
    (root / "Include" / "arm_nnfunctions.h").write_text("")
    return root


def _profile(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PROFILE_FILE
    path.write_text(text)
    return path


def _nn_profile(checkout: Path, pack: str = ".", source_root: str = ".") -> Path:
    return _profile(checkout / pack, f"schema: hct/library-profile/1\nlibrary: ns-cmsis-nn\nsource_root: {source_root}\n")


def _config(repo: Path, **overrides) -> Config:
    return Config(project_root=repo, _explicit_overrides={"project_root", *overrides}, **overrides)


def _text(result) -> str:
    return "".join(getattr(result, attr, "") or "" for attr in ("output", "stdout", "stderr"))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (LIBRARY_ENV, ROOT_ENV, "CMSIS_NN_ROOT", "HELIA_CORE_TESTER_CONFIG"):
        monkeypatch.delenv(name, raising=False)


# --- the loader ---------------------------------------------------------------------------------


def test_minimal_profile(tmp_path) -> None:
    path = _profile(tmp_path / "lib", "schema: hct/library-profile/1\nlibrary: cmsis-dsp\n")
    profile = load_profile(path)
    assert (profile.library, profile.source_root, profile.profile_path) == ("cmsis-dsp", path.parent.resolve(), path.resolve())
    assert not profile.legacy
    assert "cmsis-dsp" in profile.describe() and str(path.resolve()) in profile.describe()


def test_pack_below_the_library(tmp_path) -> None:
    checkout = _nn_checkout(tmp_path)
    profile = resolve_library(_nn_profile(checkout, pack="Testing/hct", source_root="../..").parent)
    assert profile.source_root == checkout.resolve()


def test_legacy_ns_cmsis_nn_checkout(tmp_path) -> None:
    checkout = _nn_checkout(tmp_path)
    profile = resolve_library(checkout)
    assert (profile.library, profile.source_root, profile.legacy) == ("ns-cmsis-nn", checkout.resolve(), True)
    assert "legacy" in profile.describe()


def test_profile_wins_over_the_legacy_layout(tmp_path) -> None:
    checkout = _nn_checkout(tmp_path)
    _nn_profile(checkout)
    assert not resolve_library(checkout).legacy


def test_legacy_can_be_refused(tmp_path) -> None:
    with pytest.raises(LibraryProfileError, match="has no hct-library.yaml$"):
        resolve_library(_nn_checkout(tmp_path), allow_legacy=False)


@pytest.mark.parametrize(
    "text, said",
    [
        ("", "(top level): None is not of type 'object'"),
        ("- a\n- b\n", "is not of type 'object'"),
        ("schema: [\n", "not valid YAML"),
        ("schema: hct/library-profile/1\nlibrary: a\nlibrary: b\n", "duplicate key 'library'"),
        ("? [a, b]\n: 1\nschema: hct/library-profile/1\nlibrary: a\n", "unhashable key ['a', 'b']"),
        ("? {a: 1}\n: 1\n", "unhashable key {'a': 1}"),
        ("schema: hct/library-profile/1\nlibrary: a\nextra: 1\n", "Additional properties are not allowed ('extra'"),
        ("schema: hct/library-profile/2\nlibrary: a\n", "schema: 'hct/library-profile/1' was expected"),
        ("library: a\n", "'schema' is a required property"),
        ("schema: hct/library-profile/1\n", "'library' is a required property"),
        ("schema: hct/library-profile/1\nlibrary: CMSIS-DSP\n", "library: 'CMSIS-DSP' does not match"),
        ("schema: hct/library-profile/1\nlibrary: ''\n", "library: '' does not match"),
        ("schema: hct/library-profile/1\nlibrary: -x\n", "does not match"),
        (f"schema: hct/library-profile/1\nlibrary: {'a' * 65}\n", "does not match"),
        ("schema: hct/library-profile/1\nlibrary: 1.5\n", "library: 1.5 is not of type 'string'"),
        ("schema: hct/library-profile/1\nlibrary: a\nsource_root: ''\n", "source_root: '' should be non-empty"),
        ("schema: hct/library-profile/1\nlibrary: a\nsource_root: 3\n", "source_root: 3 is not of type 'string'"),
    ],
)
def test_bad_profiles(tmp_path, text, said) -> None:
    path = _profile(tmp_path / "lib", text)
    with pytest.raises(LibraryProfileError) as caught:
        load_profile(path)
    assert said in str(caught.value) and str(path) in str(caught.value)


def test_every_schema_error_is_counted(tmp_path) -> None:
    path = _profile(tmp_path / "lib", "schema: x\nlibrary: Y\nother: 1\n")
    with pytest.raises(LibraryProfileError, match=r"\(and 2 more\)"):
        load_profile(path)


@pytest.mark.parametrize(
    "value, said",
    [
        ("/abs", "must be a relative POSIX path"),
        ("..\\\\x", "must be a relative POSIX path"),
        ("missing", "is not a directory"),
        ("afile", "is not a directory"),
        ("../sibling", "does not contain the profile"),
        ("inner", "does not contain the profile"),
    ],
)
def test_bad_source_roots(tmp_path, value, said) -> None:
    (tmp_path / "sibling").mkdir()
    pack = tmp_path / "lib"
    (pack / "inner").mkdir(parents=True)
    (pack / "afile").write_text("")
    path = _profile(pack, f"schema: hct/library-profile/1\nlibrary: a\nsource_root: '{value}'\n")
    with pytest.raises(LibraryProfileError, match=said):
        load_profile(path)


def test_oversized_profile(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(library_module, "MAX_PROFILE_BYTES", 64)
    path = _profile(tmp_path / "lib", "schema: hct/library-profile/1\nlibrary: a\n" + "#" * 64 + "\n")
    with pytest.raises(LibraryProfileError, match="a profile is at most 64"):
        load_profile(path)


def test_profile_at_the_size_limit_loads(tmp_path, monkeypatch) -> None:
    text = "schema: hct/library-profile/1\nlibrary: a\n"
    monkeypatch.setattr(library_module, "MAX_PROFILE_BYTES", len(text))
    assert load_profile(_profile(tmp_path / "lib", text)).library == "a"


def test_non_utf8_profile(tmp_path) -> None:
    pack = tmp_path / "lib"
    pack.mkdir()
    (pack / PROFILE_FILE).write_bytes(b"library: \xff\n")
    with pytest.raises(LibraryProfileError, match="not UTF-8"):
        load_profile(pack / PROFILE_FILE)


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions and a non-root user")
def test_unreadable_profile(tmp_path) -> None:
    path = _profile(tmp_path / "lib", "schema: hct/library-profile/1\nlibrary: a\n")
    path.chmod(0)
    try:
        with pytest.raises(LibraryProfileError, match="cannot read"):
            load_profile(path)
    finally:
        path.chmod(0o644)


@pytest.mark.parametrize("layout", ["missing", "file", "profile_is_dir", "dangling_link", "empty_dir"])
def test_bad_library_directories(tmp_path, layout) -> None:
    target = tmp_path / "lib"
    if layout == "file":
        target.write_text("")
    elif layout == "profile_is_dir":
        (target / PROFILE_FILE).mkdir(parents=True)
    elif layout == "dangling_link":
        target.mkdir()
        (target / PROFILE_FILE).symlink_to(tmp_path / "nowhere.yaml")
    elif layout == "empty_dir":
        target.mkdir()
    said = {"missing": "library directory not found", "file": "library directory not found",
            "profile_is_dir": "not a regular file", "dangling_link": "not a regular file",
            "empty_dir": "has no hct-library.yaml and is not an ns-cmsis-nn checkout"}[layout]
    with pytest.raises(LibraryProfileError, match=said):
        resolve_library(target)


def test_unsupported_library(tmp_path) -> None:
    profile = load_profile(_profile(tmp_path / "lib", "schema: hct/library-profile/1\nlibrary: cmsis-dsp\n"))
    with pytest.raises(LibraryProfileError, match="'cmsis-dsp'.*not supported by this hct; it runs: ns-cmsis-nn"):
        require_supported(profile)


def test_profile_error_is_a_configuration_error() -> None:
    assert issubclass(LibraryProfileError, ConfigurationError)


# --- Config ------------------------------------------------------------------------------------


def test_library_sets_the_checkout(tmp_path) -> None:
    checkout = _nn_checkout(tmp_path)
    _nn_profile(checkout, pack="Tests/hct", source_root="../..")
    config = _config(_repo(tmp_path), library=checkout / "Tests" / "hct")
    assert config.cmsis_nn_root == checkout.resolve()
    assert config.library_profile.source_root == checkout.resolve()
    assert config.cmsis_nn_root_error() is None


def test_library_env_and_tilde(tmp_path, monkeypatch) -> None:
    checkout = _nn_checkout(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(LIBRARY_ENV, f"~/{checkout.name}")
    config = Config(project_root=_repo(tmp_path), _explicit_overrides={"project_root"})
    assert config.library == checkout.resolve() and config.cmsis_nn_root == checkout.resolve()


def test_library_from_toml(tmp_path) -> None:
    checkout, repo = _nn_checkout(tmp_path), _repo(tmp_path)
    (repo / "helia_core_tester.toml").write_text(f'[helia_core_tester]\nlibrary = "{checkout}"\n')
    assert Config(project_root=repo, _explicit_overrides={"project_root"}).cmsis_nn_root == checkout.resolve()


def test_toml_cannot_set_private_fields(tmp_path) -> None:
    repo = _repo(tmp_path)
    (repo / "helia_core_tester.toml").write_text('[helia_core_tester]\n_library_profile = "x"\n_frozen = true\n')
    config = Config(project_root=repo, _explicit_overrides={"project_root"})
    assert config.library_profile is None


def test_toml_cannot_set_properties(tmp_path) -> None:
    repo = _repo(tmp_path)
    (repo / "helia_core_tester.toml").write_text('[helia_core_tester]\nlibrary_profile = "x"\n')
    assert Config(project_root=repo, _explicit_overrides={"project_root"}).library_profile is None


@pytest.mark.parametrize("key, value", [("library", "3"), ("library", "true"), ("cmsis_nn_root", "[1]"),
                                        ("library", '{a = 1}')])
def test_toml_path_of_the_wrong_type_is_refused(tmp_path, key, value) -> None:
    repo = _repo(tmp_path)
    (repo / "helia_core_tester.toml").write_text(f"[helia_core_tester]\n{key} = {value}\n")
    with pytest.raises(ConfigurationError, match=f"{key} in .*helia_core_tester.toml must be a path string"):
        Config(project_root=repo, _explicit_overrides={"project_root"})


def test_cli_refuses_a_toml_library_of_the_wrong_type(tmp_path) -> None:
    repo = _repo(tmp_path)
    (repo / "helia_core_tester.toml").write_text("[helia_core_tester]\nlibrary = 3\n")
    result = CliRunner().invoke(cli_module.app, ["generate", "--repo-root", str(repo), "--dry-run"])
    assert isinstance(result.exception, ConfigurationError), result.exception
    assert "library in" in str(result.exception) and "must be a path string" in str(result.exception)


def test_same_checkout_twice_is_fine(tmp_path) -> None:
    checkout = _nn_checkout(tmp_path)
    config = _config(_repo(tmp_path), library=checkout, cmsis_nn_root=checkout)
    assert config.cmsis_nn_root == checkout.resolve()


def test_two_checkouts_conflict(tmp_path, monkeypatch) -> None:
    one, other = _nn_checkout(tmp_path, "one"), _nn_checkout(tmp_path, "other")
    monkeypatch.setenv(ROOT_ENV, str(other))
    with pytest.raises(LibraryProfileError) as caught:
        _config(_repo(tmp_path), library=one)
    message = str(caught.value)
    assert "--library names ns-cmsis-nn at" in message and f"{ROOT_ENV} names {other.resolve()}" in message
    assert "pass one of them" in message


def test_bad_library_fails_at_init(tmp_path) -> None:
    with pytest.raises(LibraryProfileError, match="library directory not found"):
        _config(_repo(tmp_path), library=tmp_path / "missing")


def test_unsupported_library_fails_at_init(tmp_path) -> None:
    pack = tmp_path / "dsp"
    _profile(pack, "schema: hct/library-profile/1\nlibrary: cmsis-dsp\n")
    with pytest.raises(LibraryProfileError, match="not supported"):
        _config(_repo(tmp_path), library=pack)


def test_root_error_names_library_as_the_source(tmp_path) -> None:
    pack = tmp_path / "pack"
    _profile(pack, "schema: hct/library-profile/1\nlibrary: ns-cmsis-nn\n")
    error = _config(_repo(tmp_path), library=pack).cmsis_nn_root_error()
    assert "(from --library)" in error and "has no CMakeLists.txt" in error


def test_alias_resolves_the_profile_lazily(tmp_path) -> None:
    checkout = _nn_checkout(tmp_path)
    config = _config(_repo(tmp_path), cmsis_nn_root=checkout)
    assert config.library_profile.legacy and config.library_profile.source_root == checkout.resolve()


def test_alias_with_a_bad_profile_raises_when_asked(tmp_path) -> None:
    checkout = _nn_checkout(tmp_path)
    _profile(checkout, "schema: hct/library-profile/1\n")
    config = _config(_repo(tmp_path), cmsis_nn_root=checkout)
    with pytest.raises(LibraryProfileError, match="'library' is a required property"):
        config.library_profile


def test_no_library_and_no_root(tmp_path) -> None:
    config = _config(_repo(tmp_path))
    assert config.library is None and config.cmsis_nn_root is None and config.library_profile is None


def test_alias_to_a_missing_root_has_no_profile(tmp_path) -> None:
    assert _config(_repo(tmp_path), cmsis_nn_root=tmp_path / "missing").library_profile is None


# --- CLI ---------------------------------------------------------------------------------------


@pytest.fixture
def captured(monkeypatch) -> list[Config]:
    seen: list[Config] = []

    class _FakeStep:
        def __init__(self, config, *args, **kwargs):
            seen.append(config)

        def plan(self):
            return None

    class _FakePipeline:
        def __init__(self, config):
            seen.append(config)

        def print_plan(self):
            pass

    for name in ("GenerateStep", "BuildStep", "HostCheckStep", "RecordStep"):
        monkeypatch.setattr(cli_module, name, _FakeStep)
    monkeypatch.setattr(cli_module, "FullTestPipeline", _FakePipeline)
    monkeypatch.setattr(cli_module, "_print_plan_item", lambda item: None)
    return seen


COMMANDS = ["generate", "build", "host-check", "record", "full"]


@pytest.mark.parametrize("command", COMMANDS)
def test_option_reaches_config(tmp_path, captured, command) -> None:
    checkout = _nn_checkout(tmp_path)
    result = CliRunner().invoke(
        cli_module.app, [command, "--plan", "--library", str(checkout), "--repo-root", str(_repo(tmp_path))])
    assert result.exit_code == 0, _text(result)
    assert captured[0].cmsis_nn_root == checkout.resolve()
    assert captured[0].library_profile.library == "ns-cmsis-nn"
    assert "library" in captured[0]._explicit_overrides


@pytest.mark.parametrize("command", COMMANDS)
def test_bad_library_exits_2_with_the_reason(tmp_path, captured, command) -> None:
    result = CliRunner().invoke(
        cli_module.app, [command, "--plan", "--library", str(tmp_path / "missing"), "--repo-root", str(_repo(tmp_path))])
    assert result.exit_code == 2, _text(result)
    assert "library directory not found" in _text(result)
    assert captured == []


def test_conflicting_flags_exit_2(tmp_path, captured) -> None:
    one, other = _nn_checkout(tmp_path, "one"), _nn_checkout(tmp_path, "other")
    result = CliRunner().invoke(cli_module.app, [
        "build", "--plan", "--library", str(one), "--cmsis-nn-root", str(other), "--repo-root", str(_repo(tmp_path))])
    assert result.exit_code == 2, _text(result)
    assert "--library names" in _text(result) and "--cmsis-nn-root names" in _text(result)


def test_doctor_checks_the_library(tmp_path) -> None:
    repo = _repo(tmp_path)
    good = CliRunner().invoke(cli_module.app, ["doctor", "--repo-root", str(repo), "--library", str(_nn_checkout(tmp_path))])
    assert "✓ Library (from --library): ns-cmsis-nn at" in _text(good)
    bad = CliRunner().invoke(cli_module.app, ["doctor", "--repo-root", str(repo), "--library", str(tmp_path / "missing")])
    assert bad.exit_code == 2 and "✗ Library: library directory not found" in _text(bad)


@pytest.mark.parametrize("where", ["env", "toml"])
def test_doctor_checks_the_configured_library(tmp_path, monkeypatch, where) -> None:
    repo = _repo(tmp_path)
    if where == "env":
        monkeypatch.setenv(LIBRARY_ENV, str(tmp_path / "missing"))
        source = LIBRARY_ENV
    else:
        (repo / "helia_core_tester.toml").write_text(f'[helia_core_tester]\nlibrary = "{tmp_path / "missing"}"\n')
        source = None
    result = CliRunner().invoke(cli_module.app, ["doctor", "--repo-root", str(repo)])
    assert result.exit_code == 2 and "✗ Library: library directory not found" in _text(result)
    good = _nn_checkout(tmp_path)
    if where == "env":
        monkeypatch.setenv(LIBRARY_ENV, str(good))
    else:
        (repo / "helia_core_tester.toml").write_text(f'[helia_core_tester]\nlibrary = "{good}"\n')
    result = CliRunner().invoke(cli_module.app, ["doctor", "--repo-root", str(repo)])
    assert f"✓ Library (from {source or 'library in ' + str(repo / 'helia_core_tester.toml')}): ns-cmsis-nn" in _text(result)


def test_doctor_without_a_library_says_nothing_about_one(tmp_path) -> None:
    assert "Library" not in _text(CliRunner().invoke(cli_module.app, ["doctor", "--repo-root", str(_repo(tmp_path))]))


def test_configured_library_precedence(tmp_path, monkeypatch) -> None:
    from helia_core_tester.core.config import configured_library

    repo = _repo(tmp_path)
    (repo / "helia_core_tester.toml").write_text(f'[helia_core_tester]\nlibrary = "{tmp_path / "toml"}"\n')
    assert configured_library(repo, None) == ((tmp_path / "toml").resolve(), f"library in {repo / 'helia_core_tester.toml'}")
    monkeypatch.setenv(LIBRARY_ENV, str(tmp_path / "env"))
    assert configured_library(repo, None) == ((tmp_path / "env").resolve(), LIBRARY_ENV)
    assert configured_library(repo, tmp_path / "cli") == ((tmp_path / "cli").resolve(), "--library")
    monkeypatch.delenv(LIBRARY_ENV)
    (repo / "helia_core_tester.toml").write_text("[helia_core_tester]\nlibrary = 3\n")
    with pytest.raises(ConfigurationError, match="must be a path string"):
        configured_library(repo, None)
    (repo / "helia_core_tester.toml").unlink()
    assert configured_library(repo, None) == (None, "")


def _stop_at_contract(monkeypatch) -> list:
    seen: list = []

    def _load(root):
        seen.append(root)
        raise contract_cli.ContractError("stop here")

    monkeypatch.setattr(contract_cli, "load_contract_set", _load)
    return seen


def test_contract_inventory_accepts_the_same_checkout_twice(tmp_path, monkeypatch) -> None:
    checkout = _nn_checkout(tmp_path)
    seen = _stop_at_contract(monkeypatch)
    result = CliRunner().invoke(cli_module.app, ["contract",
        "inventory", "--library", str(checkout), "--cmsis-nn-root", str(checkout)])
    assert "stop here" in _text(result), _text(result)
    assert seen == [checkout.resolve()]


def test_contract_inventory_refuses_two_checkouts(tmp_path, monkeypatch) -> None:
    one, other = _nn_checkout(tmp_path, "one"), _nn_checkout(tmp_path, "other")
    seen = _stop_at_contract(monkeypatch)
    result = CliRunner().invoke(cli_module.app, ["contract",
        "inventory", "--library", str(one), "--cmsis-nn-root", str(other)])
    assert result.exit_code == contract_cli.EXIT_UNAVAILABLE
    assert "--library names ns-cmsis-nn at" in _text(result) and "pass one of them" in _text(result)
    assert seen == []


@pytest.mark.parametrize("where", ["env", "toml"])
def test_contract_inventory_reads_the_configured_library(tmp_path, monkeypatch, where) -> None:
    checkout, repo = _nn_checkout(tmp_path), _repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(contract_cli, "find_repo_root_or_cwd", lambda: repo)
    if where == "env":
        monkeypatch.setenv(LIBRARY_ENV, str(checkout))
    else:
        (repo / "helia_core_tester.toml").write_text(f'[helia_core_tester]\nlibrary = "{checkout}"\n')
    seen = _stop_at_contract(monkeypatch)
    CliRunner().invoke(cli_module.app, ["contract", "inventory"])
    assert seen == [checkout.resolve()]


def test_contract_inventory_conflicts_with_the_env_library(tmp_path, monkeypatch) -> None:
    one, other = _nn_checkout(tmp_path, "one"), _nn_checkout(tmp_path, "other")
    monkeypatch.setenv(LIBRARY_ENV, str(one))
    result = CliRunner().invoke(cli_module.app, ["contract", "inventory", "--cmsis-nn-root", str(other)])
    assert result.exit_code == contract_cli.EXIT_UNAVAILABLE
    assert f"{LIBRARY_ENV} names" in _text(result)


def test_contract_inventory_refuses_a_bad_library(tmp_path) -> None:
    result = CliRunner().invoke(cli_module.app, ["contract", "inventory", "--library", str(tmp_path / "missing")])
    assert result.exit_code == contract_cli.EXIT_UNAVAILABLE
    assert "library directory not found" in _text(result)


def test_contract_inventory_reads_the_library_checkout(tmp_path, monkeypatch) -> None:
    checkout = _nn_checkout(tmp_path)
    _nn_profile(checkout, pack="Tests/hct", source_root="../..")
    seen = _stop_at_contract(monkeypatch)
    result = CliRunner().invoke(cli_module.app, ["contract", "inventory", "--library", str(checkout / "Tests" / "hct")])
    assert result.exit_code == contract_cli.EXIT_UNAVAILABLE and "stop here" in _text(result)
    assert seen == [checkout.resolve()]


def test_a_nul_in_source_root_is_refused(tmp_path) -> None:
    path = _profile(tmp_path / "lib", 'schema: hct/library-profile/1\nlibrary: a\nsource_root: "a\\0b"\n')
    with pytest.raises(LibraryProfileError, match="source_root: .* does not match"):
        load_profile(path)


def test_a_source_root_that_cannot_be_resolved_is_refused(tmp_path) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "loop_a").symlink_to(lib / "loop_b")
    (lib / "loop_b").symlink_to(lib / "loop_a")
    path = _profile(lib, "schema: hct/library-profile/1\nlibrary: a\nsource_root: loop_a\n")
    with pytest.raises(LibraryProfileError, match="source_root 'loop_a' (cannot be resolved|is not a directory)"):
        load_profile(path)
