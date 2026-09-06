"""PDK-independent runner contracts and diagnostic prerequisite handling."""

import importlib
import inspect
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import hdl21 as h
import hdl21.sim as hs
import pytest
from vlsirtools.spice import ResultFormat
from vlsirtools.spice.sim_data import AnalysisType

from flow.comp import sim as comp

from . import test_spectre


@pytest.mark.parametrize("block", ("adc", "comp", "samp", "cdac"))
def test_runner_cli_and_python_contract(block, monkeypatch, capsys):
    module = importlib.import_module(f"flow.{block}.sim")
    monkeypatch.setattr(sys, "argv", ["sim"])
    module.main()
    names = capsys.readouterr().out.splitlines()[1:]
    assert names
    targets = [name.strip() for name in names]
    assert targets == sorted(targets, key=lambda name: (not name.startswith("hdl21"), name.startswith("frida2")))
    for name in targets:
        signature = inspect.signature(getattr(module, name))
        assert list(signature.parameters) == ["run_dir", "check"]
        assert signature.parameters["check"].default is False
        assert signature.parameters["check"].kind is inspect.Parameter.KEYWORD_ONLY
        for flag in ("--check", "--netlist-only"):
            monkeypatch.setattr(sys, "argv", ["sim", name, flag])
            with pytest.raises(SystemExit) as error:
                module.main()
            assert error.value.code == 2
    assert "netlist_only" not in inspect.getsource(module)
    assert not any(name.startswith("frida65") for name in vars(module))


@pytest.fixture
def capture_sim(monkeypatch):
    """Replace site dependencies and execution, keeping real testbench generators."""

    site = SimpleNamespace(
        install=SimpleNamespace(
            include=lambda *_: h.Literal("models"),
            include_pre_simulation=lambda: h.Literal("pre"),
        )
    )
    monkeypatch.setitem(sys.modules, "pdk.tsmc65", SimpleNamespace(site=site, pdk_logic=object()))
    monkeypatch.setitem(sys.modules, "pdk.tsmc65.site", site)
    monkeypatch.setattr(h.pdk, "set_default", lambda *_: None)
    monkeypatch.setattr(h.pdk, "compile", lambda *_: None)
    captured = {}

    def build(**kwargs):
        captured.update(kwargs)

        def run(options):
            captured["options"] = options
            if captured.get("fail"):
                raise RuntimeError("simulator/license failure")
            return {AnalysisType.TRAN: SimpleNamespace(data={})}

        return SimpleNamespace(run=run)

    monkeypatch.setattr(hs, "Sim", build)
    monkeypatch.setattr(comp, "convert_spectre_comp_to_measurement", lambda *a, **kw: captured.update(measurement=kw))
    monkeypatch.setattr(comp, "write_measurement", lambda path, _: captured.update(hdf5=path))
    return captured


@pytest.mark.parametrize(
    "block,target,short_stop,full_stop,threads",
    (
        ("comp", "frida1_fixed_input_noise", 40e-9, 61 * 100 * 40e-9, 1),
        ("samp", "frida1_transient", 100e-9, 500e-9, 4),
        ("cdac", "frida1_transfer_curve", 200.1e-9, 2048 * 200e-9 + 2047 * 100e-12, 4),
    ),
)
@pytest.mark.parametrize("check", (False, True))
def test_block_simulation_recipe(block, target, short_stop, full_stop, threads, check, capture_sim, tmp_path):
    runner = getattr(importlib.import_module(f"flow.{block}.sim"), target)
    assert runner(tmp_path, check=check) == tmp_path
    tran = next(attr for attr in capture_sim["attrs"] if isinstance(attr, hs.Tran))
    assert float(tran.tstop) == pytest.approx(short_stop if check else full_stop)
    assert bool(tran.noise) == (block == "comp" and not check)
    assert float(tran.options["strobeperiod"]) == pytest.approx(500e-12 if block == "comp" else 100e-12)
    assert tran.options["strobeoutput"].text == "strobeonly"
    options = capture_sim["options"]
    assert options.fmt == (ResultFormat.NONE if check else ResultFormat.SIM_DATA)
    assert f"+mt={threads}" in options.simulator_args
    assert ("-ahdllint=warn" in options.simulator_args) == check
    assert options.rundir == tmp_path
    literals = "\n".join(attr.text for attr in capture_sim["attrs"] if isinstance(attr, h.Literal))
    assert ("static_erc" in literals) == check
    assert ("dyn_nodecap" in literals) == check
    if block == "comp":
        assert ("hdf5" in capture_sim) == (not check)
        if not check:
            assert float(tran.options["noisefmin"]) == pytest.approx(1 / full_stop)
            assert tran.options["noisefmax"].text == "25G"
            assert float(tran.options["noiseseed"]) == 1
            assert capture_sim["measurement"]["candidate_id"] == "frida1_fabricated_baseline"
    capture_sim["fail"] = True
    with pytest.raises(RuntimeError, match="simulator/license failure"):
        runner(tmp_path / "failure", check=check)


@pytest.mark.parametrize("check,count", ((False, 297), (True, 6)))
def test_comparator_coverage_and_concurrency(check, count, monkeypatch, tmp_path):
    calls = []
    barrier = Barrier(min(count, 24))

    def pool(*, max_workers, mp_context):
        assert max_workers == 24
        assert mp_context.get_start_method() == "spawn"
        return ThreadPoolExecutor(max_workers=max_workers)

    def capture(directory, params, **options):
        calls.append((directory, params, options))
        if len(calls) <= min(count, 24):
            barrier.wait(timeout=10)
        return directory

    monkeypatch.setattr(comp, "ProcessPoolExecutor", pool)
    monkeypatch.setattr(comp, "_run_comp_sim", capture)
    comp.hdl21_comp_perf_vs_size(tmp_path, check=check)
    assert len(calls) == count
    assert len({directory.name for directory, *_ in calls}) == count
    assert all(comp.is_valid_comp_params(params.comp) for _, params, _ in calls)
    for directory, params, options in calls:
        assert options["check"] == check
        assert options["candidate_id"] == directory.name
        assert params.conversions == (1 if check else 100)
        assert len(params.vin_diff_values_v) == (1 if check else 61)
        size = directory.name.split("_")[0] if check else options["size_profile"]
        expected_widths = {
            "half": (19, 3, 4, 13, 17, 2, 4),
            "double": (74, 10, 16, 50, 66, 8, 16),
            "fabricated": (37, 5, 8, 25, 33, 4, 8),
        }[size]
        assert (
            tuple(
                getattr(params.comp, name)
                for name in (
                    "diffpair_w",
                    "tail_w",
                    "rst_w",
                    "latch_on_w",
                    "latch_init_w",
                    "srlatch_n_w",
                    "srlatch_p_w",
                )
            )
            == expected_widths
        )
        assert tuple(
            getattr(params.comp, name) for name in ("diffpair_l", "tail_l", "rst_l", "latch_on_l", "latch_init_l")
        ) == (5, 13, 1, 6, 17)
    if check:
        assert {path.name for path, *_ in calls} == {
            "fabricated_single_nmos_switched",
            "half_single_pmos_dynamic",
            "double_double_nmos_switched",
            "fabricated_double_pmos_dynamic",
            "half_double_nmos_dynamic",
            "double_double_pmos_switched",
        }
    else:
        assert {options["topology_index"] for _, _, options in calls} == set(range(148))
        assert sum(options["size_profile"] == "half" for _, _, options in calls) == 148
        assert sum(options["size_profile"] == "double" for _, _, options in calls) == 148
        assert sum(options["size_profile"] == "fabricated" for _, _, options in calls) == 1


def test_comparator_reports_all_case_failures(monkeypatch, tmp_path):
    import json

    calls = []
    monkeypatch.setattr(comp, "ProcessPoolExecutor", lambda **kw: ThreadPoolExecutor(max_workers=kw["max_workers"]))

    def fail(directory, *_args, **_kwargs):
        calls.append(directory.name)
        raise RuntimeError("broken deck")

    monkeypatch.setattr(comp, "_run_comp_sim", fail)
    with pytest.raises(RuntimeError, match="6 comparator cases failed"):
        comp.hdl21_comp_perf_vs_size(tmp_path, check=True)
    failures = json.loads((tmp_path / "failures.json").read_text())
    assert set(failures) == set(calls)
    assert all("broken deck" in error for error in failures.values())


def test_missing_spectre_skips_before_pdk_import(monkeypatch):
    monkeypatch.setattr(test_spectre.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        test_spectre.importlib.util, "find_spec", lambda _: pytest.fail("PDK imported before binary check")
    )
    with pytest.raises(pytest.skip.Exception, match="Spectre binary"):
        inspect.unwrap(test_spectre.spectre_installation)()


@pytest.mark.parametrize("spec", (None, SimpleNamespace(origin=None)))
def test_missing_pdk_package_skips(spec, monkeypatch):
    monkeypatch.setattr(test_spectre.shutil, "which", lambda _: "/mock/spectre")
    monkeypatch.setattr(test_spectre.importlib.util, "find_spec", lambda _: spec)
    with pytest.raises(pytest.skip.Exception, match="PDK Python package"):
        inspect.unwrap(test_spectre.spectre_installation)()


def test_missing_installation_skips_but_missing_collateral_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(test_spectre.shutil, "which", lambda _: "/mock/spectre")
    monkeypatch.setattr(test_spectre.importlib.util, "find_spec", lambda _: SimpleNamespace(origin="site.py"))
    site = SimpleNamespace(install=SimpleNamespace(pdk_path=tmp_path / "missing"), MODEL_FILES=(tmp_path / "model",))
    monkeypatch.setitem(sys.modules, "pdk.tsmc65", SimpleNamespace(site=site))
    with pytest.raises(pytest.skip.Exception, match="installation is unavailable"):
        inspect.unwrap(test_spectre.spectre_installation)()
    site.install.pdk_path = tmp_path
    with pytest.raises(FileNotFoundError):
        inspect.unwrap(test_spectre.spectre_installation)()


def test_spectre_marker_is_opt_in():
    import subprocess

    for selection, expected in (([], "11 deselected"), (["-m", "spectre"], "11 tests collected")):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", str(test_spectre.__file__), *selection],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == (0 if selection else 5), result.stdout + result.stderr
        assert expected in result.stdout
