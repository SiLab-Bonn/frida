"""Remote CLI contracts, with Git/rsync/shell tests but no SSH or EDA jobs."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from . import remote


@pytest.fixture
def repository(tmp_path):
    for name in (
        "flow/adc/sim.py",
        "pyproject.toml",
        "uv.lock",
        "pdk/__init__.py",
        "pdk/tsmc65/__init__.py",
        "libs/Hdl21/pyproject.toml",
        "libs/Vlsir/VlsirTools/setup.py",
        "libs/basil/pyproject.toml",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    (tmp_path / ".gitignore").write_text("build/\n.venv/\n__pycache__/\n")
    for args in (
        ("init", "-q"),
        ("add", "."),
        ("-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "fixture"),
    ):
        subprocess.run(["git", *args], cwd=tmp_path, check=True)
    return tmp_path


def test_snapshot_includes_only_tracked_source_and_selected_inputs(repository):
    for name in (
        "build/layout/adc/example/pex.netlist",
        "build/sim/old/result.h5",
        ".venv/cache",
        "flow/__pycache__/cache",
    ):
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    files = remote._snapshot_files(repository, ["build/layout/adc/example"], "tsmc65")
    assert "uv.lock" in files
    assert "libs/Vlsir/VlsirTools/setup.py" in files
    assert "pdk/tsmc65/__init__.py" in files
    assert "build/layout/adc/example/pex.netlist" in files
    assert not any("cache" in name or "build/sim" in name or ".git" in name for name in files)


@pytest.mark.parametrize("name", ("flow/new.py", "flow/adc/sim.py"))
def test_snapshot_rejects_dirty_or_untracked_source(repository, name):
    (repository / name).write_text("changed")
    with pytest.raises(ValueError, match="Commit"):
        remote._snapshot_files(repository, [], "tsmc65")


@pytest.mark.parametrize("name", ("build", "build/remote/old", "build/sim/old", "flow", "../outside"))
def test_snapshot_rejects_broad_or_external_inputs(repository, name):
    with pytest.raises(ValueError):
        remote._snapshot_files(repository, [name], "tsmc65")


def test_snapshot_requires_configured_inputs(repository):
    with pytest.raises(FileNotFoundError):
        remote._snapshot_files(repository, ["build/layout/missing"], "tsmc65")


def test_snapshot_rejects_input_symlinks(repository):
    directory = repository / "build/layout/example"
    directory.mkdir(parents=True)
    (directory / "input").symlink_to(repository / "uv.lock")
    with pytest.raises(ValueError, match="symlinks"):
        remote._snapshot_files(repository, ["build/layout/example"], "tsmc65")
    with pytest.raises(ValueError, match="symlinks"):
        remote._snapshot_files(repository, ["build/layout/example/input"], "tsmc65")


@pytest.mark.parametrize(
    "args",
    (
        ["--host", "-oProxyCommand=bad", "--work-root", "/local/user"],
        ["--host", "worker;bad", "--work-root", "/local/user"],
        ["--host", "worker", "--work-root", "~/work"],
        ["--host", "worker", "--work-root", "/"],
        ["--host", "worker", "--work-root", "/local/../tmp"],
    ),
)
def test_cli_rejects_unsafe_arguments(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["remote", "launch", "frida2_fixed_input_noise", *args])
    with pytest.raises(SystemExit) as error:
        remote.main()
    assert error.value.code == 2


def test_ssh_quotes_one_remote_bash_command(monkeypatch):
    captured = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: captured.append(argv) or SimpleNamespace(stdout="ok\n"))
    script = "cd '/local/a b' && printf '%s' '$literal'"
    assert remote._ssh("worker", script) == "ok"
    assert shlex.split(captured[0][-1]) == ["bash", "-lc", script]
    assert "BatchMode=yes" in captured[0]


def test_launch_uses_existing_target_and_detached_collector(repository, monkeypatch):
    calls = []
    original = subprocess.run
    monkeypatch.setattr(remote.shutil, "which", lambda tool: f"/usr/bin/{tool}")

    def run(argv, **kwargs):
        if argv[0] == "git":
            return original(argv, **kwargs)
        calls.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(remote, "_ssh", lambda host, command: calls.append([host, command]))
    args = argparse.Namespace(
        host="worker",
        work_root="/local/user",
        setup="/eda/setup.sh",
        block="adc",
        target="frida2_fixed_input_noise",
        input=[],
        pdk="tsmc65",
        analysis=None,
    )
    campaign = remote.launch(args, repository)
    record = json.loads((campaign / "manifest.json").read_text())
    assert (
        record["revision"] == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    )
    assert record["remote_dir"].startswith("/local/user/frida-")
    assert "uv run --frozen python -m flow.adc.sim frida2_fixed_input_noise" in (campaign / "run.sh").read_text()
    assert any(call[0] == "tmux" and "collect" in call[-1] and "--watch" in call[-1] for call in calls)
    assert any(call[0] == "worker" and "tmux new-session -d" in call[1] for call in calls)
    assert not any("--delete" in call for call in calls)


def test_resource_status_is_read_only_and_works_before_launch(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(sys, "argv", ["remote", "status", "--host", "worker", "--setup", "/eda/setup.sh"])
    monkeypatch.setattr(remote, "_ssh", lambda host, command: calls.append(command) or "resource report")
    remote.main()
    script = calls[0]
    assert "source /eda/setup.sh" in script
    assert "lscpu" in script and "ps -eo" in script
    assert "lmutil lmstat -f Virtuoso_Multi_mode_Simulation" in script
    assert "lmutil lmstat -f Spectre_XPS" in script
    assert "${CDS_LIC_FILE:-${LM_LICENSE_FILE:-}}" in script
    assert "mkdir" not in script and "new-session" not in script
    assert "resource report" in capsys.readouterr().out


@pytest.mark.parametrize(
    "scenario,code",
    (("pass", 0), ("skip", 1), ("empty", 1), ("diagnostic_error", 4), ("focused_error", 3), ("simulation_error", 7)),
)
def test_worker_shell_gates_simulation_and_records_exit(tmp_path, scenario, code):
    if shutil.which("bash") is None:
        pytest.skip("bash unavailable")
    work = tmp_path / "worker's snapshot"
    (work / "build/remote").mkdir(parents=True)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    uv = binaries / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        + """import os, pathlib, sys
args = sys.argv[1:]
case = os.environ['SCENARIO']
if 'pytest' in args:
    if '-m' in args:
        if case == 'diagnostic_error': sys.exit(4)
        tests, skipped = (0 if case == 'empty' else 1), int(case == 'skip')
        pathlib.Path('build/remote/preflight.xml').write_text(f'<testsuites><testsuite tests="{tests}" skipped="{skipped}" errors="0" failures="0"/></testsuites>')
    elif case == 'focused_error': sys.exit(3)
elif 'python' in args:
    args = args[args.index('python')+1:]
    if '-c' in args: os.execv(sys.executable, [sys.executable, *args])
    pathlib.Path('simulation-started').touch()
    if case == 'simulation_error': sys.exit(7)
"""
    )
    uv.chmod(0o755)
    for name in ("spectre", "tmux", "rsync", "pgrep"):
        path = binaries / name
        path.write_text("#!/bin/sh\nexit 1\n")
        path.chmod(0o755)
    setup = tmp_path / "vendor's setup.sh"
    setup.write_text("false\nexport VENDOR_ENVIRONMENT=ready\n")
    script = remote._worker_script(str(work), "adc", "frida2_fixed_input_noise", str(setup))
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}", "SCENARIO": scenario},
        check=False,
    )
    assert result.returncode == code
    assert (work / "build/remote/exit-code").read_text().strip() == str(code)
    assert (work / "simulation-started").exists() == (scenario in ("pass", "simulation_error"))


@pytest.fixture
def campaign(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "host": "worker",
                "remote_dir": "/local/user/snapshot",
                "session": "test",
                "analysis": "adc_pex_flavor_paths",
            }
        )
    )
    return tmp_path


@pytest.mark.parametrize("state,analysis", (("0", True), ("7", False), ("running", False)))
def test_collection_waits_for_exit_and_only_analyzes_success(campaign, monkeypatch, state, analysis):
    calls = []
    monkeypatch.setattr(remote, "_status", lambda _: state)
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append(argv) or SimpleNamespace(returncode=0))
    assert remote.collect(campaign) == (7 if state == "7" else 0)
    assert any("flow.analysis.runner" in call for call in calls) == analysis
    assert any(call[0] == "rsync" for call in calls) == (state != "running")
    if state != "running":
        count = len(calls)
        remote.collect(campaign)
        assert len(calls) == count


def test_collector_retries_transport_failure_without_relaunching(campaign, monkeypatch):
    states = iter((subprocess.CalledProcessError(255, ["ssh"]), "running", "0"))
    sleeps = []

    def status(_):
        value = next(states)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(remote, "_status", status)
    monkeypatch.setattr(remote.time, "sleep", sleeps.append)
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: SimpleNamespace(returncode=0))
    assert remote.collect(campaign, watch=True) == 0
    assert sleeps == [1800, 1800]


def test_missing_worker_never_looks_successful(campaign, monkeypatch):
    monkeypatch.setattr(remote, "_status", lambda _: "missing")
    with pytest.raises(RuntimeError, match="disappeared"):
        remote.collect(campaign)
    assert not (campaign / "collected.json").exists()


def test_failed_copy_is_not_marked_collected(campaign, monkeypatch):
    monkeypatch.setattr(remote, "_status", lambda _: "0")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(23, ["rsync"])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        remote.collect(campaign)
    assert not (campaign / "collected.json").exists()


def test_analysis_failure_is_reported_separately(campaign, monkeypatch):
    monkeypatch.setattr(remote, "_status", lambda _: "0")
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **kw: SimpleNamespace(returncode=9 if "flow.analysis.runner" in argv else 0)
    )
    assert remote.collect(campaign) == 9
    assert json.loads((campaign / "collected.json").read_text()) == {"worker_exit_code": 0, "exit_code": 9}


def test_rsync_result_filter_preserves_raw_and_hdf5_but_omits_inputs(tmp_path):
    if shutil.which("rsync") is None:
        pytest.skip("rsync unavailable")
    source, dest = tmp_path / "source", tmp_path / "dest"
    for name in (
        "sim/target/stamp/case/raw/data",
        "sim/target/stamp/case/result.h5",
        "diagnostics/log",
        "remote/run.log",
        "layout/large-input.gds",
    ):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    subprocess.run(
        [
            "rsync",
            "-a",
            "--include=/sim/***",
            "--include=/diagnostics/***",
            "--include=/remote/***",
            "--exclude=*",
            f"{source}/",
            f"{dest}/",
        ],
        check=True,
    )
    assert (dest / "sim/target/stamp/case/raw/data").exists()
    assert (dest / "sim/target/stamp/case/result.h5").exists()
    assert (dest / "remote/run.log").exists()
    assert not (dest / "layout").exists()
