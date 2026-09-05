"""Software-only checks for the native HDL21 ADC simulation interface."""

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from threading import Barrier

import hdl21 as h
import hdl21.sim as hs
import pytest

from flow.adc.subckt import (
    Frida1PexAdc,
    Frida2PexAdc,
    Frida65a1LayerRadix20PexAdc,
    Frida65a2LayerRadix17PexAdc,
    Frida65a2LayerRadix20PexAdc,
    Frida65aPexAdc,
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


@pytest.mark.parametrize("view", ("frida65a", "hdl21gen"))
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
        Frida65aPexAdc,
        Frida65a1LayerRadix20PexAdc,
        Frida65a2LayerRadix17PexAdc,
        Frida65a2LayerRadix20PexAdc,
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
    tb = sim.AdcTb(sim.AdcTbParams(view="frida65a", pex_cell=pex_cell, conversions=1))

    assert tb.xadc.of.module.name == pex_cell


def test_pex_cell_rejects_unknown_and_generated_views() -> None:
    with pytest.raises(ValueError, match="unsupported FRIDA65A PEX cell"):
        sim.AdcTb(sim.AdcTbParams(view="frida65a", pex_cell="adc_unknown", conversions=1))
    with pytest.raises(ValueError, match="applies only to extracted views"):
        sim.AdcTb(sim.AdcTbParams(view="hdl21gen", pex_cell="adc_1layer_radix17", conversions=1))


def test_c0_rename_preserves_the_physical_initialization_voltages() -> None:
    for bus in ("dac_astate_p", "dac_bstate_p", "dac_astate_n", "dac_bstate_n"):
        pattern = tuple(int(stage in (0, 3, 8, 14)) for stage in range(16))
        old = sim.AdcTb(sim.AdcTbParams(view="frida65a", pex_cell="adc_12b_17step", **{bus: pattern}))
        new = sim.AdcTb(sim.AdcTbParams(view="frida2", pex_cell="adc_12b_17step", **{bus: pattern}))
        assert old.xadc.of.module is Frida1PexAdc
        assert new.xadc.of.module is Frida2PexAdc
        for stage, state in enumerate(pattern):
            assert float(getattr(old, f"v{bus}_{15 - stage}").of.params.dc) == pytest.approx(1.2 * state)
            assert float(getattr(new, f"v{bus}_{stage}").of.params.dc) == pytest.approx(1.2 * state)


def test_supply_noise_testbench_repeats_independent_rail_networks() -> None:
    params = sim.AdcTbParams(
        view="frida65a",
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


def test_extracted_flavors_share_one_campaign_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    flavor_names = (
        "adc_1layer_radix17",
        "adc_1layer_radix20",
        "adc_2layer_radix17",
        "adc_2layer_radix20",
    )
    calls = []

    def record_flavor_campaign(run_dir: Path) -> Path:
        calls.append(run_dir)
        return run_dir

    for flavor_name in flavor_names:
        monkeypatch.setattr(
            sim, f"_run_frida_1_{flavor_name.removeprefix('adc_')}_noise_vs_rate", record_flavor_campaign
        )

    assert sim.frida_1_noise_vs_rate(tmp_path) == tmp_path
    assert calls == [tmp_path / flavor_name for flavor_name in flavor_names]
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(flavor_names)


def test_frida2_noise_recipe_uses_hundred_conversions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    pex_netlist = tmp_path / "adc.pex.netlist"

    def record_campaign(
        run_dir: Path,
        cases: tuple[tuple[str, sim.AdcTbParams], ...],
        netlist: Path,
        *,
        netlist_only: bool = False,
    ) -> Path:
        captured.update(cases=cases, pex_netlist=netlist)
        return run_dir

    monkeypatch.setattr(sim, "_run_extracted_adc_noise", record_campaign)

    assert sim._run_frida2_noise_10msps(tmp_path, pex_netlist) == tmp_path
    case_name, params = captured["cases"][0]
    assert case_name == "10msps_cm700mv_dc50mv"
    assert params.pex_cell == "adc_12b_17step"
    assert params.view == "frida2"
    assert params.conversions == 100
    assert float(params.symbol_rate) == pytest.approx(1.6e9)
    assert captured["pex_netlist"] == pex_netlist


def test_find_latest_signed_off_pex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = tmp_path / "repository"
    module = repository / "flow/adc/sim.py"
    run = repository / "build/layout/adc/frida2_2layer_radix17/20260903_170000"
    run.mkdir(parents=True)
    pex = run / "frida2_2layer_radix17.pex.netlist"
    pex.write_text("pex\n", encoding="utf-8")
    (run / "signoff_summary.json").write_text(
        json.dumps({"lvs_correct": True, "pex_netlist": str(pex)}), encoding="utf-8"
    )
    monkeypatch.setattr(sim, "__file__", str(module))

    assert sim._find_latest_signed_off_pex("frida2_2layer_radix17") == pex


def test_extracted_noise_runner_enables_transient_noise() -> None:
    assert "noise=True" in inspect.getsource(sim._run_extracted_adc_noise)


@pytest.mark.parametrize("family,count", (("frida1", 4), ("frida2", 3)))
@pytest.mark.parametrize("netlist_only", (False, True))
def test_hundred_conversion_campaigns(family, count, netlist_only, tmp_path, monkeypatch):
    calls = []
    started = Barrier(count)
    pex = tmp_path / "input.pex.netlist"
    pex.write_text("test input")
    monkeypatch.setattr(sim, "_find_latest_signed_off_pex", lambda target: pex)

    def pool(*, max_workers, mp_context):
        assert max_workers == count
        assert mp_context.get_start_method() == "spawn"
        # Threads keep the mock visible; the barrier requires every flavor to
        # enter before any can finish. A serialized runner fails this test.
        return ThreadPoolExecutor(max_workers=max_workers)

    def capture(run_dir, cases, netlist, *, netlist_only=False):
        calls.append((run_dir, cases[0], netlist, netlist_only))
        started.wait(timeout=10)
        return run_dir

    monkeypatch.setattr(sim, "ProcessPoolExecutor", pool)
    monkeypatch.setattr(sim, "_run_extracted_adc_noise", capture)
    getattr(sim, f"{family}_10msps")(tmp_path / "campaign", netlist_only=netlist_only)
    assert len(calls) == count
    assert len({name for _, (name, _), _, _ in calls}) == count
    assert sum(json.loads((directory / "input.json").read_text())["spectre_threads"] for directory, *_ in calls) == 24
    for directory, (name, params), netlist, preflight in calls:
        assert directory.name == name
        assert directory.parent == tmp_path / "campaign"
        assert json.loads((directory / "input.json").read_text())["spectre_threads"] == (8 if family == "frida2" else 6)
        assert params.conversions == 100
        assert params.pex_cell == "adc_12b_17step"
        assert params.view == ("frida2" if family == "frida2" else "frida65a")
        assert float(params.symbol_rate) == 1.6e9
        assert float(params.vin_diff.dc) == 0.05
        assert float(params.vin_cm.dc) == pytest.approx(0.7)
        assert netlist == pex and preflight == netlist_only
        assert sum(sim.get_cdac_weights(params.dut.cdac)) == (2303 if "radix20" in name else 2047)


@pytest.mark.parametrize("family,count", (("frida1", 4), ("frida2", 3)))
def test_concurrent_campaign_propagates_case_failure(family, count, tmp_path, monkeypatch):
    pex = tmp_path / "input.pex.netlist"
    pex.write_text("test input")
    calls = []
    monkeypatch.setattr(sim, "_find_latest_signed_off_pex", lambda target: pex)
    monkeypatch.setattr(
        sim, "ProcessPoolExecutor", lambda **kwargs: ThreadPoolExecutor(max_workers=kwargs["max_workers"])
    )

    def fail_one(run_dir, *_args, **_kwargs):
        calls.append(run_dir.name)
        if "1layer_radix17" in run_dir.name:
            raise RuntimeError("simulator failed")
        return run_dir

    monkeypatch.setattr(sim, "_run_extracted_adc_noise", fail_one)
    with pytest.raises(RuntimeError, match="simulator failed"):
        getattr(sim, f"{family}_10msps")(tmp_path / "campaign")
    assert len(calls) == count


@pytest.mark.parametrize("target,accepted", (("frida1_2layer_radix17", True), ("frida2_2layer_radix17", False)))
def test_only_historical_two_layer_campaign_accepts_known_disconnect(target, accepted, tmp_path, monkeypatch):
    monkeypatch.setattr(sim, "__file__", str(tmp_path / "flow/adc/sim.py"))
    run = tmp_path / "build/layout/adc" / target / "20260905_171235"
    run.mkdir(parents=True)
    (run / "adc.pex.netlist").write_text("pex")
    (run / "signoff_summary.json").write_text(
        json.dumps(
            {
                "lvs_correct": False,
                "pex_netlist": "/different/worker/adc.pex.netlist",
                "warnings": ["expected LVS mismatch: disconnected historical MOM layer"],
            }
        )
    )
    if accepted:
        assert sim._find_latest_signed_off_pex(target) == run / "adc.pex.netlist"
    else:
        with pytest.raises(FileNotFoundError):
            sim._find_latest_signed_off_pex(target)


def test_adc_main_owns_the_eleven_named_targets() -> None:
    source = inspect.getsource(sim.main)
    for name in (
        "frida_1_noise_vs_rate_check",
        "frida_1_noise_vs_rate",
        "frida_1_supply_noise_vs_rate",
        "frida_1_transfer_curve_check",
        "frida_1_transfer_curve",
        "frida2_2layer_radix17_10msps",
        "frida2_3layer_radix17_10msps",
        "hdl21gen_noise_vs_rate_check",
        "hdl21gen_noise_vs_rate",
        "hdl21gen_transfer_curve_check",
        "hdl21gen_transfer_curve",
    ):
        assert name in source
    assert "build/sim" not in source
