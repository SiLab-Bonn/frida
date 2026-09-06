"""Deploy existing simulation targets with SSH/rsync/tmux; collect with this CLI.

FTD infrastructure (examples, not defaults): launch from /local/frida on
asiclab003 to SSH aliases juno or jupiter, using --work-root /local/kcaisley and
--setup /eda/local/scripts/cadence_2024-25.sh. Both workers have 28 physical
cores / 56 hardware threads; existing runners budget about 24 compute threads.
Before launch, review uptime, free -h, running Spectre processes, and lmutil
lmstat for Virtuoso_Multi_mode_Simulation and Spectre_XPS after sourcing setup.
Account for other users and multicore license tiers; this is not a scheduler.

Use only trusted private workers: snapshots include the selected private PDK
code, but not the foundry installation. Models and any absolute site inputs
must already be accessible there. SSH authentication, uv, rsync and tmux must
be configured on the worker; rsync, SSH and tmux are also needed locally.

Each launch gets a fresh revision-pinned snapshot under the supplied work root
and a local build/remote/<session>/ record. --input adds explicit build inputs
(e.g. signed-off PEX directories), never an old simulation campaign or a venv.
The detached worker runs focused tests, the target's Spectre diagnostic (skips
are failures), then the unchanged runner CLI. A detached *local* tmux collector
checks every 30 minutes and copies raw/HDF5 results and logs when the whole
target exits, including failures. Optional analysis uses the existing CLI.
Neither host may power off; SSH outages are retried by the collector. Resume
collection with `collect <campaign> --watch` if the local session was lost.
No simulation is automatically restarted and no remote files are deleted.
"""

import argparse
import fcntl
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _ssh(host: str, script: str) -> str:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, shlex.join(["bash", "-lc", script])],
        check=True,
        stdout=subprocess.PIPE,
        timeout=60,
        text=True,
    ).stdout.strip()


def _snapshot_files(root: Path, inputs: list[str], pdk: str) -> list[str]:
    """Use Git's tracked-file list; only explicitly selected build inputs join it."""
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"], cwd=root, text=True
    )
    if dirty:
        raise ValueError("Commit or set aside local changes before deploying a revision-pinned snapshot")
    for dependency in (
        "libs/Hdl21/pyproject.toml",
        "libs/Vlsir/VlsirTools/setup.py",
        "libs/basil/pyproject.toml",
        f"pdk/{pdk}/__init__.py",
    ):
        if not (root / dependency).is_file():
            raise FileNotFoundError(f"Required dependency is not initialized: {dependency}")
    files = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--recurse-submodules",
                "-z",
                "--",
                "flow",
                "design",
                "test",
                "pyproject.toml",
                "uv.lock",
                "libs/Hdl21",
                "libs/Vlsir",
                "libs/basil",
                "pdk/__init__.py",
                f"pdk/{pdk}",
            ],
            cwd=root,
        )
        .decode()
        .split("\0")[:-1]
    )
    if "uv.lock" not in files:
        raise ValueError("A tracked uv.lock is required")
    for name in inputs:
        if (root / name).is_symlink():
            raise ValueError(f"Build inputs must be real files, not symlinks: {name}")
        path = (root / name).resolve()
        relative = path.relative_to(root)
        if len(relative.parts) < 3 or relative.parts[0] != "build" or relative.parts[1] in ("remote", "sim"):
            raise ValueError(f"Select a specific build input, not a campaign or source tree: {name}")
        if not path.exists():
            raise FileNotFoundError(path)
        selected = [path] if path.is_file() else list(path.rglob("*"))
        for item in selected:
            if item.is_symlink():
                raise ValueError(f"Build inputs must be real files, not symlinks: {item}")
            if item.is_file():
                files.append(str(item.relative_to(root)))
    return sorted(set(files))


def _worker_script(remote_dir: str, block: str, target: str, setup: str | None) -> str:
    """Only shell plumbing: experiment/diagnostic definitions stay in the runners/tests."""
    node = f"flow/circuit/test_spectre.py::test_target_diagnostic[{block}-{target}]"
    # pytest succeeds on skipped tests; require a real passing diagnostic before a campaign.
    gate = (
        "import xml.etree.ElementTree as E; "
        "s=list(E.parse('build/remote/preflight.xml').iter('testsuite')); "
        "assert sum(int(x.get('tests',0)) for x in s)>0, 'no diagnostic tests ran'; "
        "assert not any(int(x.get(k,0)) for x in s for k in ('skipped','failures','errors')), "
        "'diagnostic must pass without skips'"
    )
    return "\n".join(
        [
            "#!/bin/bash",
            "set -e",
            f"cd {shlex.quote(remote_dir)}",
            "exec > build/remote/run.log 2>&1",
            (
                'trap \'code=$?; printf "%s\\n" "$code" > build/remote/exit-code.tmp; '
                "mv build/remote/exit-code.tmp build/remote/exit-code' EXIT"
            ),
            # Vendor setup scripts contain nonfatal probes; don't inherit errexit into them.
            f"source {shlex.quote(setup)} || exit $?" if setup else ":",
            "command -v uv spectre rsync tmux",
            "uptime",
            "free -h",
            'if pgrep -u "$(id -u)" -x spectre; then echo "Existing user Spectre jobs: review resources first"; exit 1; fi',
            "uv sync --frozen --dev",
            f"uv run --frozen pytest flow/{block}/test_sim.py",
            f"uv run --frozen pytest -m spectre {shlex.quote(node)} --junitxml=build/remote/preflight.xml",
            f"uv run --frozen python -c {shlex.quote(gate)}",
            shlex.join(["uv", "run", "--frozen", "python", "-m", f"flow.{block}.sim", target]),
            "",
        ]
    )


def launch(args: argparse.Namespace, root: Path) -> Path:
    for tool in ("ssh", "rsync", "tmux"):
        if shutil.which(tool) is None:
            raise FileNotFoundError(f"Required local tool is missing: {tool}")
    files = _snapshot_files(root, args.input, args.pdk)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    session = f"frida-{stamp}"
    campaign = root / "build/remote" / session
    remote_dir = f"{args.work_root.rstrip('/')}/{session}"
    campaign.mkdir(parents=True)
    record = {
        "host": args.host,
        "remote_dir": remote_dir,
        "session": session,
        "block": args.block,
        "target": args.target,
        "setup": args.setup,
        "analysis": args.analysis,
        "inputs": args.input,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "submodules": subprocess.check_output(["git", "submodule", "status", "--recursive"], cwd=root, text=True),
    }
    (campaign / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    (campaign / "source-files.txt").write_bytes("\0".join(files).encode() + b"\0")
    (campaign / "run.sh").write_text(_worker_script(remote_dir, args.block, args.target, args.setup))
    # No mkdir -p for the snapshot itself: an existing destination must never be overwritten.
    _ssh(
        args.host,
        f"set -e; command -v rsync tmux; mkdir -- {shlex.quote(remote_dir)}; "
        f"mkdir -p -- {shlex.quote(remote_dir + '/build/remote')}",
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            "--protect-args",
            "--timeout=120",
            "--rsh=ssh -o BatchMode=yes -o ConnectTimeout=10",
            "--from0",
            f"--files-from={campaign / 'source-files.txt'}",
            f"{root}/",
            f"{args.host}:{remote_dir}/",
        ],
        check=True,
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            "--protect-args",
            "--timeout=120",
            "--rsh=ssh -o BatchMode=yes -o ConnectTimeout=10",
            f"{campaign}/",
            f"{args.host}:{remote_dir}/build/remote/",
        ],
        check=True,
    )
    command = shlex.join(["bash", f"{remote_dir}/build/remote/run.sh"])
    _ssh(args.host, shlex.join(["tmux", "new-session", "-d", "-s", session, command]))
    print(f"Campaign: {campaign}\nWorker: {args.host}:{remote_dir}\nSession: {session}", flush=True)
    collector = shlex.join([sys.executable, "-m", "flow.circuit.remote", "collect", str(campaign), "--watch"])
    collector = f"cd {shlex.quote(str(root))} && {collector} > {shlex.quote(str(campaign / 'collector.log'))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", f"{session}-collect", collector], check=True)
    return campaign


def _status(record: dict[str, str]) -> str:
    path = shlex.quote(f"{record['remote_dir']}/build/remote/exit-code")
    session = shlex.quote("=" + record["session"])
    return _ssh(
        record["host"],
        f"if test -f {path}; then cat {path}; "
        f"elif tmux has-session -t {session} 2>/dev/null; then echo running; else echo missing; fi",
    )


def collect(campaign: Path, *, watch: bool = False) -> int:
    """Copy only after exit; flock excludes competing collectors, rsync makes retries safe."""
    record = json.loads((campaign / "manifest.json").read_text())
    with (campaign / "collect.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finished = campaign / "collected.json"
        if finished.exists():
            return json.loads(finished.read_text())["exit_code"]
        while True:
            try:
                state = _status(record)
                print(f"{datetime.now().astimezone().isoformat()} {record['host']}: {state}", flush=True)
                if state == "missing":
                    raise RuntimeError(
                        "Worker session disappeared without an exit record; inspect the worker, do not restart blindly"
                    )
                if state != "running":
                    code = int(state)
                    results = campaign / "results"
                    results.mkdir(exist_ok=True)
                    subprocess.run(
                        [
                            "rsync",
                            "-a",
                            "--protect-args",
                            "--timeout=120",
                            "--rsh=ssh -o BatchMode=yes -o ConnectTimeout=10",
                            "--include=/sim/***",
                            "--include=/diagnostics/***",
                            "--include=/remote/***",
                            "--exclude=*",
                            f"{record['host']}:{record['remote_dir']}/build/",
                            f"{results}/",
                        ],
                        check=True,
                    )
                    if code == 0 and record["analysis"]:
                        with (campaign / "analysis.log").open("w") as log:
                            code = subprocess.run(
                                [
                                    sys.executable,
                                    "-m",
                                    "flow.analysis.runner",
                                    record["analysis"],
                                    "--inputs",
                                    str(results),
                                ],
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=False,
                            ).returncode
                    finished.write_text(json.dumps({"worker_exit_code": int(state), "exit_code": code}) + "\n")
                    return code
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                if not watch:
                    raise
                print(f"Transport/copy failed; retry in 30 minutes: {error}", flush=True)
            if not watch:
                return 0
            time.sleep(1800)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("launch", help="Snapshot, diagnose, run detached, and collect automatically")
    start.add_argument("target", help="Existing named simulation target")
    start.add_argument("--block", choices=("adc", "comp", "samp", "cdac"), default="adc")
    start.add_argument("--host", required=True, help="Trusted SSH host/alias (FTD: juno or jupiter)")
    start.add_argument("--work-root", required=True, help="Existing absolute worker parent directory")
    start.add_argument("--setup", help="Absolute shell environment script on the worker")
    start.add_argument("--pdk", default="tsmc65", help="PDK submodule to include, not a foundry model installation")
    start.add_argument("--input", action="append", default=[], help="Repository-relative build input; repeat as needed")
    start.add_argument(
        "--analysis", choices=("adc_pex_flavor_paths",), help="Run existing analysis after successful collection"
    )
    status = commands.add_parser("status", help="Check worker CPU/memory use and available Spectre licenses")
    status.add_argument("--host", required=True)
    status.add_argument("--setup", help="Absolute shell environment script on the worker")
    status.add_argument("--license-server", help="FlexNet server; default: worker CDS_LIC_FILE, then LM_LICENSE_FILE")
    status.add_argument(
        "--license-feature", action="append", help="FlexNet feature; repeat to override the usual Spectre features"
    )
    copy = commands.add_parser("collect", help="Collect or resume watching a campaign")
    copy.add_argument("campaign", type=Path, help="Local build/remote/<session> directory")
    copy.add_argument("--watch", action="store_true", help="Wait/retry every 30 minutes (foreground)")
    args = parser.parse_args()
    if args.command in ("launch", "status"):
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", args.host):
            parser.error("--host must be a plain SSH alias or user@hostname")
        for path in (getattr(args, "work_root", None), args.setup):
            if path is not None and (not Path(path).is_absolute() or ".." in Path(path).parts or path == "/"):
                parser.error("worker paths must be absolute, non-root paths without '..'")
    if args.command == "launch":
        if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) for value in (args.target, args.pdk)):
            parser.error("target and PDK must be identifiers")
        launch(args, Path(__file__).resolve().parents[2])
    elif args.command == "collect":
        raise SystemExit(collect(args.campaign.resolve(), watch=args.watch))
    else:
        script = [
            "set -e",
            f"source {shlex.quote(args.setup)} || exit $?" if args.setup else ":",
            "hostname",
            "uptime",
            "LC_ALL=C lscpu | grep -E '^(CPU\\(s\\)|On-line|Thread|Core|Socket)'",
            "free -h",
            "ps -eo user,pid,comm,nlwp,pcpu,etime --sort=-pcpu | head -n 12",
        ]
        server = shlex.quote(args.license_server) if args.license_server else '"${CDS_LIC_FILE:-${LM_LICENSE_FILE:-}}"'
        for feature in args.license_feature or ("Virtuoso_Multi_mode_Simulation", "Spectre_XPS"):
            script.append(shlex.join(["lmutil", "lmstat", "-f", feature]) + f" -c {server}")
        try:
            print(_ssh(args.host, "\n".join(script)))
        except subprocess.CalledProcessError as error:
            print(error.stdout or "Remote status check failed", file=sys.stderr)
            raise SystemExit(error.returncode) from error


if __name__ == "__main__":
    main()
