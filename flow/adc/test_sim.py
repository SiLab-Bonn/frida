"""Software-only checks for the native HDL21 ADC simulation interface."""

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from threading import Barrier

import hdl21 as h
import hdl21.sim as hs
import numpy as np
import pytest

from flow.adc.subckt import (
    Frida1_1LayerRadix17PexAdc,
    Frida1_1LayerRadix20PexAdc,
    Frida1_2LayerRadix17PexAdc,
    Frida1_2LayerRadix20PexAdc,
    Frida1PexAdc,
    Frida2PexAdc,
)

from . import sim


def test_adc_testbench_parameters_are_simulation_only() -> None:
    params = sim.AdcTbParams()

    assert params.view == "hdl21gen"
    assert params.pex_cell == ""
    assert float(params.vin_cm.dc) == pytest.approx(0.7)
    assert not hasattr(params, "temperature_c")
    assert not hasattr(params, "board_id")
    assert not hasattr(params, "campaign")
    assert not hasattr(params, "vdd_io")


@pytest.mark.parametrize("view", ("frida1", "hdl21gen"))
def test_adc_testbench_generates_each_view(view: str) -> None:
    tb = sim.AdcTb(sim.AdcTbParams(view=view, conversions=1))

    assert isinstance(tb, h.Module)
    assert tb.xadc is not None
    assert tb.dac_astate_p.width == 16
    assert tb.vin.p is not None


def test_adc_transfer_staircase_has_151_codes() -> None:
    params = sim.AdcTbParams(
        symbol_rate=1.6e9,
        conversions=151,
        vin_diff=hs.LinearSweep(start=-0.75, stop=0.75, step=0.01),
    )
    tb = sim.AdcTb(params)
    wave = tb.vvin_diff.of.params.wave

    assert isinstance(wave, h.Pwl)
    assert len(wave.points) == 302
    assert float(wave.points[0][1]) == pytest.approx(-0.75)
    assert float(wave.points[1][0]) == pytest.approx(len(params.seq_init_pattern) / float(params.symbol_rate) - 100e-12)
    assert float(wave.points[-1][1]) == pytest.approx(0.75)
    assert float(wave.points[-1][0]) == pytest.approx(
        params.conversions * len(params.seq_init_pattern) / float(params.symbol_rate)
    )


def test_adc_transfer_sweep_must_match_conversion_count() -> None:
    params = sim.AdcTbParams(
        conversions=2,
        vin_diff=hs.LinearSweep(start=-0.75, stop=0.75, step=0.01),
    )

    with pytest.raises(ValueError, match="151 values, but conversions=2"):
        sim.AdcTb(params)


def test_extracted_adc_keeps_calibre_port_order() -> None:
    modules = (
        Frida1_1LayerRadix17PexAdc,
        Frida1_1LayerRadix20PexAdc,
        Frida1_2LayerRadix17PexAdc,
        Frida1_2LayerRadix20PexAdc,
        Frida2PexAdc,
        Frida1PexAdc,
    )
    for module in modules:
        names = tuple(port.name for port in module.port_list)
        assert len(names) == 84
        assert len(set(names)) == 84
        assert names[:5] == ("vdd_a", "vin_p", "vss_a", "dac_mode", "dac_diffcaps")


@pytest.mark.parametrize(
    "pex_cell",
    (
        "adc_1layer_radix17",
        "adc_1layer_radix20",
        "adc_2layer_radix17",
        "adc_2layer_radix20",
        "adc_12b_17step",
    ),
)
def test_extracted_adc_selects_requested_pex_cell(pex_cell: str) -> None:
    tb = sim.AdcTb(sim.AdcTbParams(view="frida1", pex_cell=pex_cell, conversions=1))

    assert tb.xadc.of.module.name == pex_cell


def test_pex_cell_rejects_unknown_and_generated_views() -> None:
    with pytest.raises(ValueError, match="unsupported FRIDA-1 PEX cell"):
        sim.AdcTb(sim.AdcTbParams(view="frida1", pex_cell="adc_unknown", conversions=1))
    with pytest.raises(ValueError, match="applies only to extracted views"):
        sim.AdcTb(sim.AdcTbParams(view="hdl21gen", pex_cell="adc_1layer_radix17", conversions=1))


def test_c0_rename_preserves_the_physical_initialization_voltages() -> None:
    for bus in ("dac_astate_p", "dac_bstate_p", "dac_astate_n", "dac_bstate_n"):
        pattern = tuple(int(stage in (0, 3, 8, 14)) for stage in range(16))
        old = sim.AdcTb(sim.AdcTbParams(view="frida1", pex_cell="adc_12b_17step", **{bus: pattern}))
        new = sim.AdcTb(sim.AdcTbParams(view="frida2", pex_cell="adc_12b_17step", **{bus: pattern}))
        assert old.xadc.of.module is Frida1PexAdc
        assert new.xadc.of.module is Frida2PexAdc
        for stage, state in enumerate(pattern):
            assert float(getattr(old, f"v{bus}_{15 - stage}").of.params.dc) == pytest.approx(1.2 * state)
            assert float(getattr(new, f"v{bus}_{stage}").of.params.dc) == pytest.approx(1.2 * state)


def test_supply_noise_testbench_repeats_independent_rail_networks() -> None:
    params = sim.AdcTbParams(
        view="frida1",
        conversions=1,
        supply_series_resistance_ohm=1.0,
        supply_series_inductance_h=1e-9,
        supply_decoupling_capacitance_f=1e-12,
        supply_noise_rms_v=(1e-3, 0.0, 0.0),
        supply_noise_bandwidth_hz=25e9,
    )
    tb = sim.AdcTb(params)
    netlist = StringIO()

    h.netlist(tb, netlist, fmt="spectre")
    text = netlist.getvalue()

    for rail in ("vdd_a", "vdd_d", "vdd_dac"):
        assert float(getattr(tb, f"r{rail}").of.params.r) == pytest.approx(1.0)
        assert float(getattr(tb, f"l{rail}").of.params.l) == pytest.approx(1e-9)
        assert float(getattr(tb, f"c{rail}").of.params.c) == pytest.approx(1e-12)
    assert "vvdd_a (vdd_a_source vss) vsource dc=1.2 noisevec=[0 4e-17 25000000000 4e-17]" in text
    assert tb.vvdd_d.conns["p"] is tb.vdd_d_source
    assert tb.vvdd_dac.conns["p"] is tb.vdd_dac_source
    assert text.count("noisevec=") == 1


@pytest.mark.parametrize("family,count", (("frida1", 4), ("frida2", 3)))
@pytest.mark.parametrize("check", (False, True))
def test_fixed_input_campaigns(family, count, check, tmp_path, monkeypatch):
    calls = []
    started = Barrier(count)

    def pool(*, max_workers, mp_context):
        assert max_workers == count
        assert mp_context.get_start_method() == "spawn"
        return ThreadPoolExecutor(max_workers=max_workers)

    def capture(directory, params, **options):
        calls.append((directory, params, options))
        started.wait(timeout=10)
        return directory

    monkeypatch.setattr(sim, "ProcessPoolExecutor", pool)
    monkeypatch.setattr(sim, "_run_adc_sim", capture)
    root = tmp_path / "campaign"
    assert getattr(sim, f"{family}_fixed_input_noise")(root, check=check) == root
    assert len(calls) == count
    assert len({directory.name for directory, *_ in calls}) == count
    for directory, params, options in calls:
        assert directory.parent == root
        assert params.conversions == (1 if check else 100)
        assert params.pex_cell == "adc_12b_17step"
        assert params.view == ("frida2" if family == "frida2" else "frida1")
        assert float(params.symbol_rate) == 1.6e9
        assert float(params.vin_diff.dc) == 0.05
        assert float(params.vin_cm.dc) == pytest.approx(0.7)
        assert options["check"] == check
        assert options["noise"] is True
        assert options["pex_netlist"].name == f"{directory.name}.pex.netlist"
        assert options["pex_netlist"].parent.name.startswith("20260905_")
        assert options.get("expected_disconnect", False) == (family == "frida1" and "2layer" in directory.name)
        assert sum(sim.get_cdac_weights(params.dut.cdac)) == (2303 if "radix20" in directory.name else 2047)
        if family == "frida2":
            comp = np.array([int(bit) for bit in params.seq_comp_pattern])
            logic = np.roll([int(bit) for bit in params.seq_logic_pattern], int(params.seq_logic_phase_delay_symbols))
            comp_rises = np.flatnonzero(np.diff(comp) == 1) + 1
            comp_falls = np.flatnonzero(np.diff(comp) == -1) + 1
            logic_rises = (np.flatnonzero(np.diff(logic) == 1) + 1)[1:]
            logic_falls = (np.flatnonzero(np.diff(logic) == -1) + 1)[1:]
            assert len(comp) == len(logic) == len(params.seq_init_pattern) == 256
            np.testing.assert_array_equal(comp_rises, 36 + 8 * np.arange(17))
            np.testing.assert_array_equal(comp_falls, comp_rises + 6)
            np.testing.assert_array_equal(logic_rises, comp_falls[:-1])
            np.testing.assert_array_equal(logic_falls, logic_rises + 1)
            np.testing.assert_array_equal(comp_rises[1:] - logic_falls, np.ones(16))
            assert params.seq_samp_pattern == sim.AdcTbParams().seq_samp_pattern
            assert params.seq_init_pattern == sim.AdcTbParams().seq_init_pattern


@pytest.mark.parametrize("family,count", (("frida1", 4), ("frida2", 3)))
def test_concurrent_campaign_propagates_case_failure(family, count, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        sim, "ProcessPoolExecutor", lambda **kwargs: ThreadPoolExecutor(max_workers=kwargs["max_workers"])
    )

    def fail_one(directory, *_args, **_kwargs):
        calls.append(directory.name)
        if "1layer_radix17" in directory.name:
            raise RuntimeError("simulator failed")
        return directory

    monkeypatch.setattr(sim, "_run_adc_sim", fail_one)
    with pytest.raises(RuntimeError, match="simulator failed"):
        getattr(sim, f"{family}_fixed_input_noise")(tmp_path)
    assert len(calls) == count


@pytest.mark.parametrize(
    "target,count,workers",
    (
        ("hdl21_fixed_input_noise_vs_rate", 3, 3),
        ("frida1_fixed_input_noise_vs_rate", 12, 4),
        ("frida1_supply_noise_vs_rate", 15, 4),
        ("hdl21_transfer_curve", 1, None),
        ("frida1_transfer_curve", 1, None),
    ),
)
@pytest.mark.parametrize("check", (False, True))
def test_other_campaign_recipes(target, count, workers, check, tmp_path, monkeypatch):
    calls = []

    def pool(*, max_workers, mp_context):
        assert max_workers == workers
        assert mp_context.get_start_method() == "spawn"
        return ThreadPoolExecutor(max_workers=max_workers)

    def capture(directory, params, **options):
        calls.append((directory, params, options))
        return directory

    monkeypatch.setattr(sim, "ProcessPoolExecutor", pool)
    monkeypatch.setattr(sim, "_run_adc_sim", capture)
    assert getattr(sim, target)(tmp_path, check=check) == tmp_path
    assert len(calls) == count
    assert len({directory for directory, *_ in calls}) == count
    for directory, params, options in calls:
        assert options["check"] == check
        assert params.view == ("hdl21gen" if target.startswith("hdl21") else "frida1")
        if "transfer_curve" in target:
            assert params.conversions == 151
            assert params.vin_diff == hs.LinearSweep(start=-0.75, stop=0.75, step=0.01)
            assert not options.get("noise", False)
        else:
            assert params.conversions == (1 if check else 100)
            assert float(params.vin_diff.dc) == 0.05
        if "supply_noise" in target:
            assert params.supply_series_resistance_ohm == 1.0
            assert params.supply_series_inductance_h == 1e-9
            assert params.supply_decoupling_capacitance_f == 1e-12
            assert params.supply_noise_bandwidth_hz == 25e9
            assert set(params.supply_noise_rms_v) <= {0.0, 1e-3}
            # Preserve the original supply recipe; do not enable device noise
            # as an incidental effect of consolidating the executors.
            assert not options.get("noise", False)
        elif "fixed_input_noise" in target:
            assert options["noise"] is True
    if "vs_rate" in target:
        assert {float(params.symbol_rate) for _, params, _ in calls} == {320e6, 960e6, 1.6e9}


@pytest.fixture
def captured_executor(monkeypatch):
    from types import SimpleNamespace

    from flow.analysis import io
    from flow.circuit import results
    from pdk import tsmc65
    from pdk.tsmc65 import site

    captured = {}
    monkeypatch.setattr(h.pdk, "set_default", lambda *_: None)
    monkeypatch.setattr(h.pdk, "compile", lambda *_: None)
    monkeypatch.setattr(tsmc65, "pdk_logic", object())
    monkeypatch.setattr(
        site,
        "install",
        SimpleNamespace(
            include=lambda *_: h.Literal("models"),
            include_pre_simulation=lambda: h.Literal("pre"),
        ),
    )
    monkeypatch.setattr(
        results,
        "adc_signal_names",
        lambda *args, **kwargs: {
            "time_s": "time",
            "comp": "xtop.comp_out",
            "internal": "xtop.xadc.internal",
        },
    )
    monkeypatch.setattr(
        results, "convert_spectre_adc_to_measurement", lambda *args, **kwargs: captured.update(converted=kwargs)
    )
    monkeypatch.setattr(io, "write_measurement", lambda path, value: captured.update(hdf5=path))

    def build(**kwargs):
        captured.update(kwargs)

        def run(options):
            captured["options"] = options
            if captured.get("fail"):
                raise RuntimeError("simulator/license failure")
            return {sim.AnalysisType.TRAN: SimpleNamespace(data={})}

        return SimpleNamespace(run=run)

    monkeypatch.setattr(hs, "Sim", build)
    return captured


@pytest.fixture
def pex_input(tmp_path):
    pex = tmp_path / "frida2_2layer_radix17.pex.netlist"
    ports = " ".join(port.name for port in Frida2PexAdc.port_list)
    pex.write_text(f"subckt adc_12b_17step ({ports})\ninternal\nends adc_12b_17step\n")
    (tmp_path / "signoff_summary.json").write_text(
        json.dumps(
            {
                "lvs_correct": True,
                "pex_netlist": str(pex),
                "warnings": [],
            }
        )
    )
    return pex


@pytest.mark.parametrize("view,threads", (("hdl21gen", 8), ("frida2", 8), ("frida1", 6)))
@pytest.mark.parametrize("check", (False, True))
def test_single_executor_modes(view, threads, check, captured_executor, pex_input, tmp_path):
    if view == "frida1":
        ports = " ".join(port.name for port in Frida1PexAdc.port_list)
        pex_input.write_text(f"subckt adc_12b_17step ({ports})\ninternal\nends adc_12b_17step\n")
    params = sim.AdcTbParams(
        view=view,
        pex_cell="" if view == "hdl21gen" else "adc_12b_17step",
        conversions=100,
    )
    root = tmp_path / "run"
    sim._run_adc_sim(
        root,
        params,
        pex_netlist=None if view == "hdl21gen" else pex_input,
        noise=True,
        check=check,
    )
    captured = captured_executor
    tran = next(attr for attr in captured["attrs"] if isinstance(attr, hs.Tran))
    assert tran.noise == (not check)
    assert float(tran.tstop) == pytest.approx(100e-9 if check else 16e-6)
    assert float(tran.options["strobeperiod"]) == pytest.approx(39.0625e-12)
    if not check:
        assert tran.options["noisefmax"].text == "25G"
        assert float(tran.options["noiseseed"]) == 1
    assert f"+mt={threads}" in captured["options"].simulator_args
    assert ("-ahdllint=warn" in captured["options"].simulator_args) == check
    assert captured["options"].rundir == root
    literals = "\n".join(attr.text for attr in captured["attrs"] if isinstance(attr, h.Literal))
    assert ("check_setuphold" in literals) == check
    assert ("simulator lang=spice" in literals) == (view == "hdl21gen")
    assert ("converted" in captured) == (not check)
    metadata = json.loads((root / "input.json").read_text())
    assert metadata["spectre_threads"] == threads
    assert metadata["transient_noise"] == (not check)
    if view != "hdl21gen":
        import hashlib

        assert metadata["pex_sha256"] == hashlib.sha256(pex_input.read_bytes()).hexdigest()


@pytest.mark.parametrize("check", (False, True))
def test_transfer_executor_preserves_strobes_and_waveform_limit(check, captured_executor, tmp_path):
    params = sim.AdcTbParams(
        conversions=151,
        vin_diff=hs.LinearSweep(start=-0.75, stop=0.75, step=0.01),
    )
    sim._run_adc_sim(tmp_path / "run", params, check=check, maximum_waveform_records=3)
    tran = next(attr for attr in captured_executor["attrs"] if isinstance(attr, hs.Tran))
    assert not tran.noise
    assert float(tran.options["strobeperiod"]) == pytest.approx(50e-12)
    assert float(tran.tstop) == pytest.approx(100e-9 if check else 151 * 160e-9)
    if not check:
        assert captured_executor["converted"]["maximum_waveform_records"] == 3


@pytest.mark.parametrize(
    "fault,message",
    (
        ("ports", "PEX port order differs"),
        ("nodes", "PEX waveform nodes missing"),
        ("header", "missing PEX subcircuit"),
        ("lvs", "unaccepted LVS result"),
        ("wrong_file", "PEX input differs"),
    ),
)
def test_executor_rejects_bad_pex(fault, message, captured_executor, pex_input, tmp_path):
    if fault == "ports":
        pex_input.write_text(pex_input.read_text().replace("vdd_a vin_p", "vin_p vdd_a"))
    elif fault == "nodes":
        pex_input.write_text(pex_input.read_text().replace("\ninternal\n", "\n"))
    elif fault == "header":
        pex_input.write_text("subckt other ()\nends other\n")
    else:
        summary = json.loads((tmp_path / "signoff_summary.json").read_text())
        if fault == "lvs":
            summary["lvs_correct"] = False
        else:
            summary["pex_netlist"] = "wrong.pex.netlist"
        (tmp_path / "signoff_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match=message):
        sim._run_adc_sim(
            tmp_path / "run",
            sim.AdcTbParams(view="frida2", pex_cell="adc_12b_17step"),
            pex_netlist=pex_input,
            check=True,
        )
    assert "options" not in captured_executor


def test_missing_configured_pex_fails(captured_executor, tmp_path):
    with pytest.raises(FileNotFoundError):
        sim._run_adc_sim(
            tmp_path / "run",
            sim.AdcTbParams(view="frida2", pex_cell="adc_12b_17step"),
            pex_netlist=tmp_path / "missing.pex.netlist",
            check=True,
        )
    assert "options" not in captured_executor


def test_diagnostic_simulator_error_propagates(captured_executor, tmp_path):
    captured_executor["fail"] = True
    with pytest.raises(RuntimeError, match="simulator/license failure"):
        sim._run_adc_sim(tmp_path / "run", sim.AdcTbParams(), check=True)


@pytest.mark.parametrize(
    "view,target,warning,accepted",
    (
        ("frida1", "frida1_2layer_radix17", "expected LVS mismatch: disconnected historical MOM layer", True),
        ("frida1", "frida1_2layer_radix20", "expected LVS mismatch: disconnected historical MOM layer", True),
        ("frida1", "frida1_1layer_radix17", "expected LVS mismatch: disconnected historical MOM layer", False),
        ("frida2", "frida2_2layer_radix17", "expected LVS mismatch: disconnected historical MOM layer", False),
        ("frida1", "frida1_2layer_radix17", "unrelated mismatch", False),
    ),
)
def test_historical_lvs_exception_is_narrow(view, target, warning, accepted, captured_executor, pex_input, tmp_path):
    renamed = pex_input.with_name(f"{target}.pex.netlist")
    pex_input.rename(renamed)
    if view == "frida1":
        ports = " ".join(port.name for port in Frida1PexAdc.port_list)
        renamed.write_text(f"subckt adc_12b_17step ({ports})\ninternal\nends adc_12b_17step\n")
    (tmp_path / "signoff_summary.json").write_text(
        json.dumps(
            {
                "lvs_correct": False,
                "pex_netlist": f"/worker/{renamed.name}",
                "warnings": [warning],
            }
        )
    )
    kwargs = {"pex_netlist": renamed, "expected_disconnect": True, "check": True}
    params = sim.AdcTbParams(view=view, pex_cell="adc_12b_17step")
    if accepted:
        sim._run_adc_sim(tmp_path / "run", params, **kwargs)
        assert captured_executor["options"]
    else:
        with pytest.raises(ValueError, match="unaccepted LVS"):
            sim._run_adc_sim(tmp_path / "run", params, **kwargs)


def test_adc_main_lists_only_experiment_targets_in_family_order(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["sim"])
    sim.main()
    assert capsys.readouterr().out.splitlines()[1:] == [
        "  hdl21_fixed_input_noise_vs_rate",
        "  hdl21_transfer_curve",
        "  frida1_fixed_input_noise",
        "  frida1_fixed_input_noise_vs_rate",
        "  frida1_transfer_curve",
        "  frida1_supply_noise_vs_rate",
        "  frida2_fixed_input_noise",
    ]
    functions = [
        name
        for name, value in vars(sim).items()
        if inspect.isfunction(value) and value.__module__ == sim.__name__ and name.startswith("_")
    ]
    assert functions == ["_run_adc_sim"]


def test_main_runs_full_experiment(tmp_path, monkeypatch):
    calls = []

    def frida2_fixed_input_noise(directory, **options):
        calls.append((directory, options))

    monkeypatch.setattr(sim, "frida2_fixed_input_noise", frida2_fixed_input_noise)
    monkeypatch.setattr(sim, "__file__", str(tmp_path / "flow/adc/sim.py"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "sim",
            "frida2_fixed_input_noise",
        ],
    )
    sim.main()
    directory, options = calls[0]
    assert directory.parent == tmp_path / "build/sim/adc/frida2_fixed_input_noise"
    assert directory.is_dir()
    assert options == {}
