"""Opt-in diagnostics: reuse runner recipes, not separate testbench definitions.

Allow roughly one hour for the full suite on asiclab003 (host/load dependent).
Run it occasionally or after broad simulation-flow changes, not after every edit.
For routine changes, select affected blocks/experiments with pytest's -k option,
for example: uv run pytest -m spectre -k 'adc-frida2_fixed_input_noise'.
The short simulated durations do not imply short wall-clock runtimes.
"""

import importlib
import importlib.util
import shutil
from datetime import datetime
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def spectre_installation():
    """Skip absent site installations, never broken inputs or simulator failures."""

    if shutil.which("spectre") is None:
        pytest.skip("Spectre binary is not on PATH; source the Cadence environment")
    spec = importlib.util.find_spec("pdk.tsmc65")
    if spec is None or spec.origin is None:
        pytest.skip("TSMC65 PDK Python package is not installed")
    from pdk.tsmc65 import site

    if not site.install.pdk_path.is_dir():
        pytest.skip(f"TSMC65 PDK installation is unavailable: {site.install.pdk_path}")
    # The installation exists: missing configured collateral is an error.
    for path in site.MODEL_FILES:
        if not path.is_file():
            raise FileNotFoundError(path)


@pytest.fixture(scope="module")
def diagnostic_root(spectre_installation):
    root = (
        Path(__file__).resolve().parents[2]
        / "build/diagnostics"
        / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    root.mkdir(parents=True, exist_ok=False)
    return root


@pytest.mark.spectre
@pytest.mark.parametrize(
    "block,target",
    (
        ("adc", "hdl21_fixed_input_noise_vs_rate"),
        ("adc", "hdl21_transfer_curve"),
        ("adc", "frida1_fixed_input_noise"),
        ("adc", "frida1_fixed_input_noise_vs_rate"),
        ("adc", "frida1_transfer_curve"),
        ("adc", "frida1_supply_noise_vs_rate"),
        ("adc", "frida2_fixed_input_noise"),
        ("comp", "hdl21_comp_perf_vs_size"),
        ("comp", "frida1_fixed_input_noise"),
        ("samp", "frida1_transient"),
        ("cdac", "frida1_transfer_curve"),
    ),
    ids=lambda value: value,
)
def test_target_diagnostic(block, target, diagnostic_root):
    runner = getattr(importlib.import_module(f"flow.{block}.sim"), target)
    run_dir = diagnostic_root / block / target
    assert runner(run_dir, check=True) == run_dir
    assert list(run_dir.rglob("netlist.scs")), f"no generated decks beneath {run_dir}"
    assert list(run_dir.rglob("spectre.log")), f"no simulator reports beneath {run_dir}"
