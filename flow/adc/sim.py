"""ADC testbench and named Spectre simulation targets."""

import argparse
import hashlib
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import cast

import hdl21 as h
import hdl21.sim as hs
from hdl21.prefix import G, m, p
from vlsirtools.spice import ResultFormat, SimOptions, SupportedSimulators
from vlsirtools.spice.sim_data import AnalysisType, SimResult, TranResult

from flow.adc.subckt import (
    Adc,
    AdcParams,
    Frida1PexAdc,
    Frida2PexAdc,
    Frida65a1LayerRadix20PexAdc,
    Frida65a2LayerRadix17PexAdc,
    Frida65a2LayerRadix20PexAdc,
    Frida65aPexAdc,
)
from flow.cdac import CdacParams, RedunStrat, get_cdac_weights


@h.paramclass
class AdcTbParams:
    """Parameters which determine one generated ADC testbench."""

    view = h.Param(dtype=str, desc="ADC implementation: frida65a, frida2, or hdl21gen", default="hdl21gen")
    pex_cell = h.Param(
        dtype=str,
        desc="Calibre-extracted cell; empty selects adc_1layer_radix17 for frida65a",
        default="",
    )
    dut = h.Param(
        dtype=AdcParams,
        desc="ADC DUT parameters",
        default=AdcParams(
            cdac=CdacParams(n_dac=11, n_extra=5, redun_strat=RedunStrat.SUBRDX2_OVLY),
        ),
    )
    symbol_rate = h.Param(dtype=h.Scalar, desc="DDR symbol rate", default=1.6 * G)
    conversions = h.Param(dtype=int, desc="Number of conversions", default=10_000)
    en_init = h.Param(dtype=int, desc="Enable initialization clock", default=1)
    en_samp_p = h.Param(dtype=int, desc="Enable positive sampling clock", default=1)
    en_samp_n = h.Param(dtype=int, desc="Enable negative sampling clock", default=1)
    en_comp = h.Param(dtype=int, desc="Enable comparator clock", default=1)
    en_update = h.Param(dtype=int, desc="Enable DAC-update clock", default=1)
    dac_mode = h.Param(dtype=int, desc="Select comparator-driven DAC mode", default=1)
    dac_diffcaps = h.Param(dtype=int, desc="Enable differential capacitor controls", default=1)
    dac_astate_p = h.Param(dtype=tuple[int, ...], desc="C0-first positive DAC A state", default=(0, 1) * 8)
    dac_bstate_p = h.Param(dtype=tuple[int, ...], desc="C0-first positive DAC B state", default=(0,) * 16)
    dac_astate_n = h.Param(dtype=tuple[int, ...], desc="C0-first negative DAC A state", default=(0, 1) * 8)
    dac_bstate_n = h.Param(dtype=tuple[int, ...], desc="C0-first negative DAC B state", default=(0,) * 16)
    vdd_a = h.Param(dtype=h.Vdc.Params, desc="Analog supply", default=h.Vdc.Params(dc=1200 * m))
    vdd_d = h.Param(dtype=h.Vdc.Params, desc="Digital supply", default=h.Vdc.Params(dc=1200 * m))
    vdd_dac = h.Param(dtype=h.Vdc.Params, desc="DAC supply", default=h.Vdc.Params(dc=1200 * m))
    supply_series_resistance_ohm = h.Param(
        dtype=float,
        desc="Lumped bond-wire and chip-level PDN series resistance per supply rail",
        default=0.0,
    )
    supply_series_inductance_h = h.Param(
        dtype=float,
        desc="Lumped bond-wire and chip-level PDN series inductance per supply rail",
        default=0.0,
    )
    supply_decoupling_capacitance_f = h.Param(
        dtype=float,
        desc="Local chip-level decoupling capacitance per supply rail",
        default=0.0,
    )
    supply_noise_rms_v = h.Param(
        dtype=tuple[float, ...],
        desc="Added RMS noise on VDD_A, VDD_D, and VDD_DAC",
        default=(0.0, 0.0, 0.0),
    )
    supply_noise_bandwidth_hz = h.Param(
        dtype=float,
        desc="Flat bandwidth over which each added supply-noise RMS value is defined",
        default=25.0e9,
    )
    vin_cm = h.Param(dtype=h.Vdc.Params, desc="Input common mode", default=h.Vdc.Params(dc=700 * m))
    vin_diff = h.Param(
        dtype=h.Vdc.Params | h.Vsin.Params | h.Vpwl.Params | hs.LinearSweep,
        desc="Differential input stimulus",
        default=h.Vdc.Params(dc=0.0),
    )
    seq_init_pattern = h.Param(
        dtype=str, desc="Initialization sequence", default="00000000" + "11111111" + "00000000" * 30
    )
    seq_init_phase_delay_symbols = h.Param(dtype=h.Scalar, desc="INIT phase delay", default=0.0)
    seq_samp_pattern = h.Param(
        dtype=str, desc="Sampling sequence", default="00000000" * 2 + "11111111" * 2 + "00000000" * 28
    )
    seq_samp_phase_delay_symbols = h.Param(dtype=h.Scalar, desc="SAMP phase delay", default=0.0)
    seq_comp_pattern = h.Param(
        dtype=str, desc="Comparator sequence", default="00000000" * 4 + "00001111" * 17 + "00000000" * 11
    )
    seq_comp_phase_delay_symbols = h.Param(dtype=h.Scalar, desc="COMP phase delay", default=0.0)
    seq_logic_pattern = h.Param(
        dtype=str,
        desc="SAR-logic sequence",
        default="00000000" + "00001111" + "00000000" * 3 + "11110000" * 16 + "00000000" * 11,
    )
    seq_logic_phase_delay_symbols = h.Param(dtype=h.Scalar, desc="LOGIC phase delay", default=0.0)


@h.generator
def AdcTb(params: AdcTbParams) -> h.Module:
    """Generate a complete ADC testbench for the selected DUT view."""

    if params.view not in {"frida65a", "frida2", "hdl21gen"}:
        raise ValueError(f"unsupported ADC view {params.view!r}")
    pex_adcs = {
        "": Frida65aPexAdc,
        "adc_1layer_radix17": Frida65aPexAdc,
        "adc_1layer_radix20": Frida65a1LayerRadix20PexAdc,
        "adc_2layer_radix17": Frida65a2LayerRadix17PexAdc,
        "adc_2layer_radix20": Frida65a2LayerRadix20PexAdc,
        "adc_12b_17step": Frida1PexAdc,
    }
    if params.view == "frida2":
        pex_adcs = {"adc_12b_17step": Frida2PexAdc}
        if params.pex_cell not in pex_adcs:
            raise ValueError("frida2 requires pex_cell='adc_12b_17step'")
    if params.view == "frida65a" and params.pex_cell not in pex_adcs:
        raise ValueError(f"unsupported FRIDA65A PEX cell {params.pex_cell!r}")
    if params.view == "hdl21gen" and params.pex_cell:
        raise ValueError("pex_cell applies only to extracted views")
    if not math.isfinite(float(params.symbol_rate)) or float(params.symbol_rate) <= 0.0:
        raise ValueError("ADC symbol rate must be finite and positive")
    if params.conversions <= 0:
        raise ValueError("ADC conversions must be positive")
    get_cdac_weights(params.dut.cdac)
    for name in ("vdd_a", "vdd_d", "vdd_dac", "vin_cm"):
        source = getattr(params, name)
        if source.dc is None or not math.isfinite(float(source.dc)):
            raise ValueError(f"{name}.dc must be finite")
    supply_parasitics = (
        params.supply_series_resistance_ohm,
        params.supply_series_inductance_h,
        params.supply_decoupling_capacitance_f,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in supply_parasitics):
        raise ValueError("supply RLC values must be finite and nonnegative")
    if any(supply_parasitics) and not all(value > 0.0 for value in supply_parasitics):
        raise ValueError("the supply RLC model requires positive resistance, inductance, and capacitance")
    if len(params.supply_noise_rms_v) != 3 or any(
        not math.isfinite(value) or value < 0.0 for value in params.supply_noise_rms_v
    ):
        raise ValueError("supply_noise_rms_v must contain three finite nonnegative values")
    if not math.isfinite(params.supply_noise_bandwidth_hz) or params.supply_noise_bandwidth_hz <= 0.0:
        raise ValueError("supply noise bandwidth must be finite and positive")
    for name in ("en_init", "en_samp_p", "en_samp_n", "en_comp", "en_update", "dac_mode", "dac_diffcaps"):
        if getattr(params, name) not in (0, 1):
            raise ValueError(f"{name} must be zero or one")
    for name in ("dac_astate_p", "dac_bstate_p", "dac_astate_n", "dac_bstate_n"):
        value = getattr(params, name)
        if len(value) != 16 or any(bit not in (0, 1) for bit in value):
            raise ValueError(f"{name} must contain exactly sixteen binary values")
    patterns = (
        params.seq_init_pattern,
        params.seq_samp_pattern,
        params.seq_comp_pattern,
        params.seq_logic_pattern,
    )
    if any(not pattern or set(pattern) - {"0", "1"} for pattern in patterns):
        raise ValueError("ADC sequencer patterns must be non-empty binary strings")
    if len({len(pattern) for pattern in patterns}) != 1 or len(patterns[0]) % 8:
        raise ValueError("ADC sequencer patterns must have equal whole-word lengths")

    tb = h.Module(name=f"AdcTb_{params.view}")
    tb.vss = h.Port(desc="Simulator ground")
    tb.vdd_a, tb.vdd_d, tb.vdd_dac, tb.vin_cm, tb.vin_diff = h.Signals(5)
    tb.vin = h.Diff()
    tb.seq_init, tb.seq_samp, tb.seq_comp, tb.seq_logic, tb.comp_out = h.Signals(5)
    tb.dac_astate_p, tb.dac_bstate_p, tb.dac_astate_n, tb.dac_bstate_n = (h.Signal(width=16) for _ in range(4))
    tb.dac_state_p, tb.dac_state_n = h.Signal(width=16), h.Signal(width=16)
    symbol_period_s = 1.0 / float(params.symbol_rate)
    pattern_period_s = len(patterns[0]) * symbol_period_s
    transition_s = min(float(100 * p), symbol_period_s / 20.0)
    supply_rails = (
        ("vdd_a", params.vdd_a, params.supply_noise_rms_v[0]),
        ("vdd_d", params.vdd_d, params.supply_noise_rms_v[1]),
        ("vdd_dac", params.vdd_dac, params.supply_noise_rms_v[2]),
    )
    rlc_enabled = all(value > 0.0 for value in supply_parasitics)
    for rail_name, source_params, noise_rms_v in supply_rails:
        rail = getattr(tb, rail_name)
        source_node = h.Signal(name=f"{rail_name}_source") if rlc_enabled else rail
        if rlc_enabled:
            series_node = h.Signal(name=f"{rail_name}_series")
            setattr(tb, f"{rail_name}_source", source_node)
            setattr(tb, f"{rail_name}_series", series_node)
            setattr(
                tb,
                f"r{rail_name}",
                h.Resistor(r=params.supply_series_resistance_ohm)(p=source_node, n=series_node),
            )
            setattr(
                tb,
                f"l{rail_name}",
                h.Inductor(l=params.supply_series_inductance_h)(p=series_node, n=rail),
            )
            setattr(
                tb,
                f"c{rail_name}",
                h.Capacitor(c=params.supply_decoupling_capacitance_f)(p=rail, n=tb.vss),
            )
        if noise_rms_v > 0.0:
            noise_density_v2_per_hz = noise_rms_v**2 / params.supply_noise_bandwidth_hz
            tb.literals.append(
                h.Literal(
                    f"v{rail_name} ({source_node.name} vss) vsource dc={float(source_params.dc):.12g} "
                    f"noisevec=[0 {noise_density_v2_per_hz:.12g} "
                    f"{params.supply_noise_bandwidth_hz:.12g} {noise_density_v2_per_hz:.12g}]"
                )
            )
        else:
            setattr(tb, f"v{rail_name}", h.Vdc(source_params)(p=source_node, n=tb.vss))
    tb.vvin_cm = h.Vdc(params.vin_cm)(p=tb.vin_cm, n=tb.vss)
    if isinstance(params.vin_diff, h.Vdc.Params):
        tb.vvin_diff = h.Vdc(params.vin_diff)(p=tb.vin_diff, n=tb.vss)
    elif isinstance(params.vin_diff, h.Vsin.Params):
        tb.vvin_diff = h.Vsin(
            voff=params.vin_diff.voff,
            vamp=params.vin_diff.vamp,
            freq=params.vin_diff.freq,
            td=0.0 if params.vin_diff.td is None else params.vin_diff.td,
            phase=0.0 if params.vin_diff.phase is None else params.vin_diff.phase,
        )(p=tb.vin_diff, n=tb.vss)
    elif isinstance(params.vin_diff, hs.LinearSweep):
        wave = h.Pwl.staircase(
            start=params.vin_diff.start,
            stop=params.vin_diff.stop,
            step=params.vin_diff.step,
            dwell=pattern_period_s,
            transition=100 * p,
            transition_at="end",
        )
        level_count = len(wave.points) // 2
        if level_count != params.conversions:
            raise ValueError(
                f"ADC linear input sweep contains {level_count} values, but conversions={params.conversions}"
            )
        tb.vvin_diff = h.Vpwl(wave=wave)(p=tb.vin_diff, n=tb.vss)
    elif isinstance(params.vin_diff, h.Vpwl.Params):
        tb.vvin_diff = h.Vpwl(params.vin_diff)(p=tb.vin_diff, n=tb.vss)
    else:
        raise TypeError(f"unsupported ADC differential source {type(params.vin_diff).__name__}")
    tb.evin_p = h.Vcvs(gain=0.5)(p=tb.vin.p, n=tb.vin_cm, cp=tb.vin_diff, cn=tb.vss)
    tb.evin_n = h.Vcvs(gain=-0.5)(p=tb.vin.n, n=tb.vin_cm, cp=tb.vin_diff, cn=tb.vss)

    for name, pattern, phase_symbols in (
        ("init", params.seq_init_pattern, float(params.seq_init_phase_delay_symbols)),
        ("samp", params.seq_samp_pattern, float(params.seq_samp_phase_delay_symbols)),
        ("comp", params.seq_comp_pattern, float(params.seq_comp_phase_delay_symbols)),
        ("logic", params.seq_logic_pattern, float(params.seq_logic_phase_delay_symbols)),
    ):
        whole_symbols = math.floor(phase_symbols)
        rotation = whole_symbols % len(pattern)
        if rotation:
            pattern = pattern[-rotation:] + pattern[:-rotation]
        setattr(
            tb,
            f"vseq_{name}",
            h.Vbit(
                data=pattern,
                period=symbol_period_s,
                val0=0.0,
                val1=float(params.vdd_d.dc),
                delay=(phase_symbols - whole_symbols) * symbol_period_s,
                rise=transition_s,
                fall=transition_s,
                rptstart=1,
                rpttimes=-1,
            )(p=getattr(tb, f"seq_{name}"), n=tb.vss),
        )
    for name in ("en_init", "en_samp_p", "en_samp_n", "en_comp", "en_update", "dac_mode", "dac_diffcaps"):
        signal = h.Signal(name=name)
        setattr(tb, name, signal)
        setattr(tb, f"v{name}", h.Vdc(dc=float(params.vdd_d.dc) * getattr(params, name))(p=signal, n=tb.vss))
    for bus_name in ("dac_astate_p", "dac_bstate_p", "dac_astate_n", "dac_bstate_n"):
        for stage, state in enumerate(getattr(params, bus_name)):
            # Only the immutable FRIDA-1 namespace needs the old 15-first map.
            bus_index = 15 - stage if params.view == "frida65a" else stage
            setattr(
                tb,
                f"v{bus_name}_{bus_index}",
                h.Vdc(dc=float(params.vdd_d.dc) * state)(p=getattr(tb, bus_name)[bus_index], n=tb.vss),
            )

    connections = {
        "vin_p": tb.vin.p,
        "vin_n": tb.vin.n,
        "seq_init": tb.seq_init,
        "seq_samp": tb.seq_samp,
        "seq_comp": tb.seq_comp,
        "seq_update": tb.seq_logic,
        "en_init": tb.en_init,
        "en_samp_p": tb.en_samp_p,
        "en_samp_n": tb.en_samp_n,
        "en_comp": tb.en_comp,
        "en_update": tb.en_update,
        "dac_mode": tb.dac_mode,
        "dac_diffcaps": tb.dac_diffcaps,
        "dac_astate_p": tb.dac_astate_p,
        "dac_bstate_p": tb.dac_bstate_p,
        "dac_astate_n": tb.dac_astate_n,
        "dac_bstate_n": tb.dac_bstate_n,
        "comp_out": tb.comp_out,
        "vdd_a": tb.vdd_a,
        "vss_a": tb.vss,
        "vdd_d": tb.vdd_d,
        "vss_d": tb.vss,
        "vdd_dac": tb.vdd_dac,
        "vss_dac": tb.vss,
    }
    if params.view == "hdl21gen":
        tb.xadc = Adc(params.dut)(
            **connections,
            dac_state_p=tb.dac_state_p,
            dac_state_n=tb.dac_state_n,
        )
    else:
        pex_connections = {
            name: value
            for name, value in connections.items()
            if name not in {"dac_astate_p", "dac_bstate_p", "dac_astate_n", "dac_bstate_n"}
        }
        for bus_name in ("dac_astate_p", "dac_bstate_p", "dac_astate_n", "dac_bstate_n"):
            for bit in range(16):
                pex_connections[f"{bus_name}_{bit}"] = getattr(tb, bus_name)[bit]
        tb.xadc = pex_adcs[params.pex_cell]()(**pex_connections)
    return tb


def frida_1_noise_vs_rate_check(run_dir: Path) -> Path:
    """Run the extracted ADC briefly at three rates with circuit checks."""

    from flow.circuit.results import adc_signal_names
    from pdk import tsmc65
    from pdk.tsmc65 import site

    parameters = (
        AdcTbParams(
            view="frida65a",
            symbol_rate=320e6,
            conversions=1,
            vin_diff=h.Vdc.Params(dc=0.05),
            seq_logic_phase_delay_symbols=2.0,
        ),
        AdcTbParams(
            view="frida65a",
            symbol_rate=960e6,
            conversions=1,
            vin_diff=h.Vdc.Params(dc=0.05),
            seq_logic_phase_delay_symbols=2.0,
        ),
        AdcTbParams(
            view="frida65a",
            symbol_rate=1.6e9,
            conversions=1,
            vin_diff=h.Vdc.Params(dc=0.05),
            seq_logic_phase_delay_symbols=2.0,
        ),
    )
    h.pdk.set_default(tsmc65.pdk_logic)
    simulations = []
    for params in parameters:
        tb = AdcTb(params)
        h.pdk.compile(tb)
        signal_names = adc_signal_names(params.view, pex_cell=params.pex_cell)
        save_targets = [
            re.sub(r"([/<>-])", r"\\\1", raw_name)
            for canonical_name, raw_name in signal_names.items()
            if canonical_name != "time_s"
        ]
        simulations.append(
            hs.Sim(
                tb=tb,
                attrs=[
                    site.install.include(h.pdk.Corner.TYP),
                    site.install.include_pre_simulation(),
                    hs.Include(path=site.ADC_PEX_NETLIST),
                    hs.Options(name="temp", value=25.0),
                    hs.Options(name="save", value="selected"),
                    hs.Save(save_targets),
                    h.Literal(
                        "check_caps static_capacitor type=distr\n"
                        "check_erc static_erc floatbulk=all floatgate=no_top_moscap dangle=no_top "
                        "gate2power=on gate2ground=on\n"
                        "check_highz static_highz node=[*] fanout=gate_has_driver_no_moscap\n"
                        "check_dcpath static_dcpath net=[xtop.vdd_a xtop.vdd_d xtop.vdd_dac 0]\n"
                        "check_rcdelay static_rcdelay node=[*] maxnrise=20 maxnfall=20\n"
                        "check_stack static_stack count=3\n"
                        "check_topology static_topology node=[*] pin2gnd=on\n"
                        "check_nodecap dyn_nodecap node=[xtop.vin_p xtop.vin_n xtop.comp_out] time=[50n 99n]\n"
                        "check_setuphold dyn_setuphold node=[xtop.comp_out] ref_node=xtop.seq_logic "
                        "setup_time=50p hold_time=50p\n"
                        "check_power dyn_subcktpwr inst=[xtop.xadc] depth=1 port=[*] power=on"
                    ),
                    hs.Tran(
                        tstop=100e-9,
                        name="tran",
                        options={
                            "strobeperiod": min(1.0 / float(params.symbol_rate) / 16.0, 50e-12),
                            "strobeoutput": "strobeonly",
                        },
                    ),
                ],
            )
        )
    hs.run(
        simulations,
        SimOptions(
            simulator=SupportedSimulators.SPECTRE,
            fmt=ResultFormat.NONE,
            rundir=run_dir,
            simulator_args=(
                "+preset=mx",
                "+mt=4",
                "+lqtimeout",
                "3600",
                "+escchars",
                "+log",
                "spectre.log",
                "-ahdllint=warn",
                "-ahdllint_log",
                "ahdllint.log",
            ),
        ),
    )
    return run_dir


def frida_1_transfer_curve_check(run_dir: Path) -> Path:
    """Run the extracted-ADC transfer testbench briefly with circuit checks."""

    from flow.circuit.results import adc_signal_names
    from pdk import tsmc65
    from pdk.tsmc65 import site

    params = AdcTbParams(
        view="frida65a",
        symbol_rate=1.6e9,
        conversions=151,
        vin_diff=hs.LinearSweep(start=-0.75, stop=0.75, step=0.01),
        seq_logic_phase_delay_symbols=2.0,
    )
    h.pdk.set_default(tsmc65.pdk_logic)
    tb = AdcTb(params)
    h.pdk.compile(tb)
    save_targets = [
        re.sub(r"([/<>-])", r"\\\1", raw_name)
        for canonical_name, raw_name in adc_signal_names(params.view).items()
        if canonical_name != "time_s"
    ]
    simulation = hs.Sim(
        tb=tb,
        attrs=[
            site.install.include(h.pdk.Corner.TYP),
            site.install.include_pre_simulation(),
            hs.Include(path=site.ADC_PEX_NETLIST),
            hs.Options(name="temp", value=25.0),
            hs.Options(name="save", value="selected"),
            hs.Save(save_targets),
            h.Literal(
                "check_caps static_capacitor type=distr\n"
                "check_erc static_erc floatbulk=all floatgate=no_top_moscap dangle=no_top "
                "gate2power=on gate2ground=on\n"
                "check_highz static_highz node=[*] fanout=gate_has_driver_no_moscap\n"
                "check_dcpath static_dcpath net=[xtop.vdd_a xtop.vdd_d xtop.vdd_dac 0]\n"
                "check_rcdelay static_rcdelay node=[*] maxnrise=20 maxnfall=20\n"
                "check_stack static_stack count=3\n"
                "check_topology static_topology node=[*] pin2gnd=on\n"
                "check_nodecap dyn_nodecap node=[xtop.vin_p xtop.vin_n xtop.comp_out] time=[50n 99n]\n"
                "check_setuphold dyn_setuphold node=[xtop.comp_out] ref_node=xtop.seq_logic "
                "setup_time=50p hold_time=50p\n"
                "check_power dyn_subcktpwr inst=[xtop.xadc] depth=1 port=[*] power=on"
            ),
            hs.Tran(tstop=100e-9, name="tran", options={"strobeperiod": 50e-12, "strobeoutput": "strobeonly"}),
        ],
    )
    simulation.run(
        SimOptions(
            simulator=SupportedSimulators.SPECTRE,
            fmt=ResultFormat.NONE,
            rundir=run_dir,
            simulator_args=(
                "+preset=mx",
                "+mt=4",
                "+lqtimeout",
                "3600",
                "+escchars",
                "+log",
                "spectre.log",
                "-ahdllint=warn",
                "-ahdllint_log",
                "ahdllint.log",
            ),
        )
    )
    return run_dir


def hdl21gen_noise_vs_rate_check(run_dir: Path) -> Path:
    """Run the generated ADC briefly at three rates with circuit checks."""

    from flow.circuit.results import adc_signal_names
    from pdk import tsmc65
    from pdk.tsmc65 import site

    parameters = (
        AdcTbParams(
            view="hdl21gen",
            symbol_rate=320e6,
            conversions=1,
            vin_diff=h.Vdc.Params(dc=0.05),
            seq_logic_phase_delay_symbols=2.0,
        ),
        AdcTbParams(
            view="hdl21gen",
            symbol_rate=960e6,
            conversions=1,
            vin_diff=h.Vdc.Params(dc=0.05),
            seq_logic_phase_delay_symbols=2.0,
        ),
        AdcTbParams(
            view="hdl21gen",
            symbol_rate=1.6e9,
            conversions=1,
            vin_diff=h.Vdc.Params(dc=0.05),
            seq_logic_phase_delay_symbols=2.0,
        ),
    )
    h.pdk.set_default(tsmc65.pdk_logic)
    standard_cells = (
        *site.STANDARD_CELL_SPICE_NETLISTS,
        Path(__file__).resolve().parents[2] / "design/spice/adc_digital.sp",
    )
    simulations = []
    for params in parameters:
        tb = AdcTb(params)
        h.pdk.compile(tb)
        save_targets = [
            re.sub(r"([/<>-])", r"\\\1", raw_name)
            for canonical_name, raw_name in adc_signal_names(params.view).items()
            if canonical_name != "time_s"
        ]
        simulations.append(
            hs.Sim(
                tb=tb,
                attrs=[
                    site.install.include(h.pdk.Corner.TYP),
                    site.install.include_pre_simulation(),
                    h.Literal(
                        "\n".join(
                            (
                                "simulator lang=spice",
                                *(f'.include "{path}"' for path in standard_cells),
                                "simulator lang=spectre",
                            )
                        )
                    ),
                    hs.Options(name="temp", value=25.0),
                    hs.Options(name="save", value="selected"),
                    hs.Save(save_targets),
                    h.Literal(
                        "check_caps static_capacitor type=distr\n"
                        "check_erc static_erc floatbulk=all floatgate=no_top_moscap dangle=no_top "
                        "gate2power=on gate2ground=on\n"
                        "check_highz static_highz node=[*] fanout=gate_has_driver_no_moscap\n"
                        "check_dcpath static_dcpath net=[xtop.vdd_a xtop.vdd_d xtop.vdd_dac 0]\n"
                        "check_rcdelay static_rcdelay node=[*] maxnrise=20 maxnfall=20\n"
                        "check_stack static_stack count=3\n"
                        "check_topology static_topology node=[*] pin2gnd=on\n"
                        "check_nodecap dyn_nodecap node=[xtop.vin_p xtop.vin_n xtop.comp_out] time=[50n 99n]\n"
                        "check_setuphold dyn_setuphold node=[xtop.comp_out] ref_node=xtop.seq_logic "
                        "setup_time=50p hold_time=50p\n"
                        "check_power dyn_subcktpwr inst=[xtop.xadc] depth=1 port=[*] power=on"
                    ),
                    hs.Tran(
                        tstop=100e-9,
                        name="tran",
                        options={
                            "strobeperiod": min(1.0 / float(params.symbol_rate) / 16.0, 50e-12),
                            "strobeoutput": "strobeonly",
                        },
                    ),
                ],
            )
        )
    hs.run(
        simulations,
        SimOptions(
            simulator=SupportedSimulators.SPECTRE,
            fmt=ResultFormat.NONE,
            rundir=run_dir,
            simulator_args=(
                "+preset=mx",
                "+mt=4",
                "+lqtimeout",
                "3600",
                "+escchars",
                "+log",
                "spectre.log",
                "-ahdllint=warn",
                "-ahdllint_log",
                "ahdllint.log",
            ),
        ),
    )
    return run_dir


def hdl21gen_transfer_curve_check(run_dir: Path) -> Path:
    """Run the generated-ADC transfer testbench briefly with circuit checks."""

    from flow.circuit.results import adc_signal_names
    from pdk import tsmc65
    from pdk.tsmc65 import site

    params = AdcTbParams(
        view="hdl21gen",
        symbol_rate=1.6e9,
        conversions=151,
        vin_diff=hs.LinearSweep(start=-0.75, stop=0.75, step=0.01),
        seq_logic_phase_delay_symbols=2.0,
    )
    h.pdk.set_default(tsmc65.pdk_logic)
    standard_cells = (
        *site.STANDARD_CELL_SPICE_NETLISTS,
        Path(__file__).resolve().parents[2] / "design/spice/adc_digital.sp",
    )
    tb = AdcTb(params)
    h.pdk.compile(tb)
    save_targets = [
        re.sub(r"([/<>-])", r"\\\1", raw_name)
        for canonical_name, raw_name in adc_signal_names(params.view).items()
        if canonical_name != "time_s"
    ]
    simulation = hs.Sim(
        tb=tb,
        attrs=[
            site.install.include(h.pdk.Corner.TYP),
            site.install.include_pre_simulation(),
            h.Literal(
                "\n".join(
                    (
                        "simulator lang=spice",
                        *(f'.include "{path}"' for path in standard_cells),
                        "simulator lang=spectre",
                    )
                )
            ),
            hs.Options(name="temp", value=25.0),
            hs.Options(name="save", value="selected"),
            hs.Save(save_targets),
            h.Literal(
                "check_caps static_capacitor type=distr\n"
                "check_erc static_erc floatbulk=all floatgate=no_top_moscap dangle=no_top "
                "gate2power=on gate2ground=on\n"
                "check_highz static_highz node=[*] fanout=gate_has_driver_no_moscap\n"
                "check_dcpath static_dcpath net=[xtop.vdd_a xtop.vdd_d xtop.vdd_dac 0]\n"
                "check_rcdelay static_rcdelay node=[*] maxnrise=20 maxnfall=20\n"
                "check_stack static_stack count=3\n"
                "check_topology static_topology node=[*] pin2gnd=on\n"
                "check_nodecap dyn_nodecap node=[xtop.vin_p xtop.vin_n xtop.comp_out] time=[50n 99n]\n"
                "check_setuphold dyn_setuphold node=[xtop.comp_out] ref_node=xtop.seq_logic "
                "setup_time=50p hold_time=50p\n"
                "check_power dyn_subcktpwr inst=[xtop.xadc] depth=1 port=[*] power=on"
            ),
            hs.Tran(tstop=100e-9, name="tran", options={"strobeperiod": 50e-12, "strobeoutput": "strobeonly"}),
        ],
    )
    simulation.run(
        SimOptions(
            simulator=SupportedSimulators.SPECTRE,
            fmt=ResultFormat.NONE,
            rundir=run_dir,
            simulator_args=(
                "+preset=mx",
                "+mt=4",
                "+lqtimeout",
                "3600",
                "+escchars",
                "+log",
                "spectre.log",
                "-ahdllint=warn",
                "-ahdllint_log",
                "ahdllint.log",
            ),
        )
    )
    return run_dir


def _run_extracted_adc_noise(
    run_dir: Path,
    cases: tuple[tuple[str, AdcTbParams], ...],
    pex_netlist: Path,
    *,
    netlist_only: bool = False,
) -> Path:
    """Run one configured extracted-ADC noise campaign."""

    if not pex_netlist.is_file():
        raise FileNotFoundError(pex_netlist)
    # Fixed family budgets: four FRIDA-1 jobs or three FRIDA-2 jobs per worker.
    spectre_threads = 6 if cases[0][1].view == "frida65a" else 8
    from flow.analysis.io import write_measurement
    from flow.circuit.results import adc_signal_names, convert_spectre_adc_to_measurement
    from pdk import tsmc65
    from pdk.tsmc65 import site

    h.pdk.set_default(tsmc65.pdk_logic)
    simulations = []
    for case_name, params in cases:
        tb = AdcTb(params)
        h.pdk.compile(tb)
        signal_names = adc_signal_names(params.view, pex_cell=params.pex_cell)
        # Calibre's interface is positional; validate the actual extracted header.
        text = pex_netlist.read_text().replace("\\\n", " ")
        cell = tb.xadc.of.module
        header = re.search(rf"^subckt {re.escape(cell.name)}\s*\(([^)]*)\)", text, re.MULTILINE | re.IGNORECASE)
        if header is None:
            raise ValueError(f"missing PEX subcircuit {cell.name}")
        ports = [re.sub(r"<(\d+)>", r"_\1", port.replace("\\", "").lower()) for port in header[1].split()]
        if ports != [port.name for port in cell.port_list]:
            raise ValueError(f"PEX port order differs from {cell.name}")
        nodes = set(text.replace("\\", "").split())
        missing = [name for name in signal_names.values() if name.startswith("xtop.xadc.") and name[10:] not in nodes]
        if missing:
            raise ValueError(f"PEX waveform nodes missing: {missing}")
        save_targets = [
            re.sub(r"([/<>-])", r"\\\1", raw_name)
            for canonical_name, raw_name in signal_names.items()
            if canonical_name != "time_s"
        ]
        tstop_s = params.conversions * len(params.seq_init_pattern) / float(params.symbol_rate)
        simulations.append(
            hs.Sim(
                name=case_name,
                tb=tb,
                attrs=[
                    site.install.include(h.pdk.Corner.TYP),
                    site.install.include_pre_simulation(),
                    hs.Include(path=pex_netlist),
                    hs.Options(name="temp", value=25.0),
                    hs.Options(name="save", value="selected"),
                    hs.Save(save_targets),
                    hs.Tran(
                        tstop=tstop_s,
                        name="tran",
                        noise=True,
                        options={
                            "strobeperiod": min(1.0 / float(params.symbol_rate) / 16.0, 50e-12),
                            "strobeoutput": "strobeonly",
                            "noisefmin": 1.0 / tstop_s,
                            "noisefmax": "25G",
                            "noiseseed": 1,
                        },
                    ),
                ],
            )
        )
    if netlist_only:
        from vlsirtools.spice.spectre import SpectreSim

        for (case_name, _), simulation in zip(cases, simulations, strict=True):
            destination = run_dir if len(cases) == 1 else run_dir / case_name
            backend = SpectreSim(hs.to_proto(simulation), SimOptions(rundir=destination))
            backend.setup()
            backend.write_netlist()
        return run_dir
    # HDL21 types cover scalar and sequence inputs, all result formats, and all analyses;
    # this call requests a SIM_DATA list containing one transient per simulation.
    results = cast(
        list[SimResult],
        hs.run(
            simulations,
            SimOptions(
                simulator=SupportedSimulators.SPECTRE,
                fmt=ResultFormat.SIM_DATA,
                rundir=run_dir,
                simulator_args=(
                    "+preset=mx",
                    f"+mt={spectre_threads}",
                    "+lqtimeout",
                    "3600",
                    "+escchars",
                    "+log",
                    "spectre.log",
                ),
            ),
        ),
    )
    for index, ((case_name, params), result) in enumerate(zip(cases, results, strict=True)):
        transient = cast(TranResult, result[AnalysisType.TRAN])
        if len(cases) == 1:
            case_dir = run_dir
        else:
            case_dir = run_dir / case_name
            (run_dir / str(index)).rename(case_dir)
        measurement = convert_spectre_adc_to_measurement(
            transient.data,
            params=params,
            raw_path=case_dir / "netlist.raw",
            signal_names=adc_signal_names(params.view, pex_cell=params.pex_cell),
        )
        write_measurement(case_dir / "result.h5", measurement)
    return run_dir


def _run_frida_1_1layer_radix17_noise_vs_rate(run_dir: Path) -> Path:
    """Run the one-layer radix-17 extracted ADC at 2, 6, and 10 Msps."""

    from pdk.tsmc65 import site

    dut = AdcParams(
        adc_bits=12,
        cdac=CdacParams(
            n_dac=11,
            n_extra=5,
            redun_strat=RedunStrat.SUBRDX2_OVLY,
            weights=(768, 512, 320, 192, 96, 64, 32, 24, 12, 10, 5, 4, 4, 2, 1, 1),
        ),
    )
    cases = (
        (
            "2msps_cm700mv_dc50mv",
            AdcTbParams(
                view="frida65a",
                pex_cell="adc_1layer_radix17",
                dut=dut,
                symbol_rate=320e6,
                conversions=100,
                vin_diff=h.Vdc.Params(dc=0.05),
                seq_logic_phase_delay_symbols=2.0,
            ),
        ),
        (
            "6msps_cm700mv_dc50mv",
            AdcTbParams(
                view="frida65a",
                pex_cell="adc_1layer_radix17",
                dut=dut,
                symbol_rate=960e6,
                conversions=100,
                vin_diff=h.Vdc.Params(dc=0.05),
                seq_logic_phase_delay_symbols=2.0,
            ),
        ),
        (
            "10msps_cm700mv_dc50mv",
            AdcTbParams(
                view="frida65a",
                pex_cell="adc_1layer_radix17",
                dut=dut,
                symbol_rate=1.6e9,
                conversions=100,
                vin_diff=h.Vdc.Params(dc=0.05),
                seq_logic_phase_delay_symbols=2.0,
            ),
        ),
    )
    return _run_extracted_adc_noise(run_dir, cases, site.ADC_PEX_NETLIST)


def _run_frida_1_1layer_radix20_noise_vs_rate(run_dir: Path) -> Path:
    """Run the one-layer radix-20 extracted ADC at 2, 6, and 10 Msps."""

    from pdk.tsmc65 import site

    dut = AdcParams(
        adc_bits=12,
        cdac=CdacParams(
            n_dac=11,
            n_extra=5,
            redun_strat=RedunStrat.SUBRDX2_OVLY,
            weights=(768, 512, 320, 192, 128, 64, 64, 64, 64, 64, 32, 16, 8, 4, 2, 1),
        ),
    )
    cases = (
        (
            "2msps_cm700mv_dc50mv",
            AdcTbParams(
                view="frida65a",
                pex_cell="adc_1layer_radix20",
                dut=dut,
                symbol_rate=320e6,
                conversions=100,
                vin_diff=h.Vdc.Params(dc=0.05),
                seq_logic_phase_delay_symbols=2.0,
            ),
        ),
        (
            "6msps_cm700mv_dc50mv",
            AdcTbParams(
                view="frida65a",
                pex_cell="adc_1layer_radix20",
                dut=dut,
                symbol_rate=960e6,
                conversions=100,
                vin_diff=h.Vdc.Params(dc=0.05),
                seq_logic_phase_delay_symbols=2.0,
            ),
        ),
        (
            "10msps_cm700mv_dc50mv",
            AdcTbParams(
                view="frida65a",
                pex_cell="adc_1layer_radix20",
                dut=dut,
                symbol_rate=1.6e9,
                conversions=100,
                vin_diff=h.Vdc.Params(dc=0.05),
                seq_logic_phase_delay_symbols=2.0,
            ),
        ),
    )
    pex_netlist = site.ADC_PEX_NETLIST.parent / "adc_1layer_radix20/adc_1layer_radix20.pex.netlist"
    return _run_extracted_adc_noise(run_dir, cases, pex_netlist)


def _run_frida_1_2layer_radix17_noise_vs_rate(run_dir: Path) -> Path:
    """Run the two-layer radix-17 extracted ADC at 2, 6, and 10 Msps."""

    from pdk.tsmc65 import site

    dut = AdcParams(
        adc_bits=12,
        cdac=CdacParams(
            n_dac=11,
            n_extra=5,
            redun_strat=RedunStrat.SUBRDX2_OVLY,
            weights=(768, 512, 320, 192, 96, 64, 32, 24, 12, 10, 5, 4, 4, 2, 1, 1),
        ),
    )
    cases = (
        (
            "2msps_cm700mv_dc50mv",
            AdcTbParams(
                view="frida65a",
                pex_cell="adc_2layer_radix17",
                dut=dut,
                symbol_rate=320e6,
                conversions=100,
                vin_diff=h.Vdc.Params(dc=0.05),
                seq_logic_phase_delay_symbols=2.0,
            ),
        ),
        (
            "6msps_cm700mv_dc50mv",
            AdcTbParams(
                view="frida65a",
                pex_cell="adc_2layer_radix17",
                dut=dut,
                symbol_rate=960e6,
                conversions=100,
                vin_diff=h.Vdc.Params(dc=0.05),
                seq_logic_phase_delay_symbols=2.0,
            ),
        ),
        (
            "10msps_cm700mv_dc50mv",
            AdcTbParams(
                view="frida65a",
                pex_cell="adc_2layer_radix17",
                dut=dut,
                symbol_rate=1.6e9,
                conversions=100,
                vin_diff=h.Vdc.Params(dc=0.05),
                seq_logic_phase_delay_symbols=2.0,
            ),
        ),
    )
    pex_netlist = site.ADC_PEX_NETLIST.parent / "adc_2layer_radix17/adc_2layer_radix17.pex.netlist"
    return _run_extracted_adc_noise(run_dir, cases, pex_netlist)


def _run_frida_1_2layer_radix20_noise_vs_rate(run_dir: Path) -> Path:
    """Run the two-layer radix-20 extracted ADC at 2, 6, and 10 Msps."""

    from pdk.tsmc65 import site

    dut = AdcParams(
        adc_bits=12,
        cdac=CdacParams(
            n_dac=11,
            n_extra=5,
            redun_strat=RedunStrat.SUBRDX2_OVLY,
            weights=(768, 512, 320, 192, 128, 64, 64, 64, 64, 64, 32, 16, 8, 4, 2, 1),
        ),
    )
    cases = (
        (
            "2msps_cm700mv_dc50mv",
            AdcTbParams(
                view="frida65a",
                pex_cell="adc_2layer_radix20",
                dut=dut,
                symbol_rate=320e6,
                conversions=100,
                vin_diff=h.Vdc.Params(dc=0.05),
                seq_logic_phase_delay_symbols=2.0,
            ),
        ),
        (
            "6msps_cm700mv_dc50mv",
            AdcTbParams(
                view="frida65a",
                pex_cell="adc_2layer_radix20",
                dut=dut,
                symbol_rate=960e6,
                conversions=100,
                vin_diff=h.Vdc.Params(dc=0.05),
                seq_logic_phase_delay_symbols=2.0,
            ),
        ),
        (
            "10msps_cm700mv_dc50mv",
            AdcTbParams(
                view="frida65a",
                pex_cell="adc_2layer_radix20",
                dut=dut,
                symbol_rate=1.6e9,
                conversions=100,
                vin_diff=h.Vdc.Params(dc=0.05),
                seq_logic_phase_delay_symbols=2.0,
            ),
        ),
    )
    pex_netlist = site.ADC_PEX_NETLIST.parent / "adc_2layer_radix20/adc_2layer_radix20.pex.netlist"
    return _run_extracted_adc_noise(run_dir, cases, pex_netlist)


def frida_1_noise_vs_rate(run_dir: Path) -> Path:
    """Run all four extracted ADC flavors at 2, 6, and 10 Msps."""

    flavor_campaigns = (
        ("adc_1layer_radix17", _run_frida_1_1layer_radix17_noise_vs_rate),
        ("adc_1layer_radix20", _run_frida_1_1layer_radix20_noise_vs_rate),
        ("adc_2layer_radix17", _run_frida_1_2layer_radix17_noise_vs_rate),
        ("adc_2layer_radix20", _run_frida_1_2layer_radix20_noise_vs_rate),
    )
    for flavor_name, run_flavor_campaign in flavor_campaigns:
        flavor_run_dir = run_dir / flavor_name
        flavor_run_dir.mkdir()
        run_flavor_campaign(flavor_run_dir)
    return run_dir


def _run_frida2_noise_10msps(run_dir: Path, pex_netlist: Path, *, netlist_only: bool = False) -> Path:
    cases = (
        (
            "10msps_cm700mv_dc50mv",
            AdcTbParams(
                view="frida2",
                pex_cell="adc_12b_17step",
                dut=AdcParams(adc_bits=12, cdac=CdacParams()),
                symbol_rate=1.6 * G,
                conversions=100,
                vin_diff=h.Vdc.Params(dc=0.05),
                seq_logic_phase_delay_symbols=2.0,
            ),
        ),
    )
    return _run_extracted_adc_noise(run_dir, cases, pex_netlist, netlist_only=netlist_only)


def _find_latest_signed_off_pex(target: str) -> Path:
    root = Path(__file__).resolve().parents[2] / "build" / "layout" / "adc" / target
    for candidate in sorted(root.glob("*/signoff_summary.json"), reverse=True):
        summary = json.loads(candidate.read_text(encoding="utf-8"))
        # Resolve bundled artifacts relative to their summary on remote workers.
        pex_netlist = candidate.parent / Path(summary["pex_netlist"]).name
        expected_disconnect = target in ("frida1_2layer_radix17", "frida1_2layer_radix20")
        accepted = summary["lvs_correct"] or (
            expected_disconnect
            and summary.get("warnings") == ["expected LVS mismatch: disconnected historical MOM layer"]
        )
        if accepted and pex_netlist.is_file() and pex_netlist.stat().st_size:
            return pex_netlist
    raise FileNotFoundError(f"no accepted PEX run exists beneath {root}")


def frida2_2layer_radix17_10msps(run_dir: Path, *, netlist_only: bool = False) -> Path:
    """Run 100 connected two-layer FRIDA-2 conversions at 10 MS/s."""

    return _run_frida2_noise_10msps(
        run_dir, _find_latest_signed_off_pex("frida2_2layer_radix17"), netlist_only=netlist_only
    )


def frida2_3layer_radix17_10msps(run_dir: Path, *, netlist_only: bool = False) -> Path:
    """Run 100 connected three-layer FRIDA-2 conversions at 10 MS/s."""

    return _run_frida2_noise_10msps(
        run_dir, _find_latest_signed_off_pex("frida2_3layer_radix17"), netlist_only=netlist_only
    )


def frida1_10msps(run_dir: Path, *, netlist_only: bool = False) -> Path:
    """Run four FRIDA-1 flavors concurrently: 100 conversions, six threads each."""

    jobs = []
    for target, weights in (
        ("frida1_1layer_radix17", (768, 512, 320, 192, 96, 64, 32, 24, 12, 10, 5, 4, 4, 2, 1, 1)),
        ("frida1_1layer_radix20", (768, 512, 320, 192, 128, 64, 64, 64, 64, 64, 32, 16, 8, 4, 2, 1)),
        ("frida1_2layer_radix17", (768, 512, 320, 192, 96, 64, 32, 24, 12, 10, 5, 4, 4, 2, 1, 1)),
        ("frida1_2layer_radix20", (768, 512, 320, 192, 128, 64, 64, 64, 64, 64, 32, 16, 8, 4, 2, 1)),
    ):
        params = AdcTbParams(
            view="frida65a",
            pex_cell="adc_12b_17step",
            dut=AdcParams(adc_bits=12, cdac=CdacParams(weights=weights)),
            symbol_rate=1.6 * G,
            conversions=100,
            vin_diff=h.Vdc.Params(dc=0.05),
            seq_logic_phase_delay_symbols=2.0,
        )
        pex = _find_latest_signed_off_pex(target)
        case_dir = run_dir / target
        case_dir.mkdir(parents=True)
        (case_dir / "input.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "pex_netlist": str(pex),
                    "pex_sha256": hashlib.sha256(pex.read_bytes()).hexdigest(),
                    "expected_historical_disconnect": "2layer" in target,
                    "spectre_threads": 6,
                },
                indent=2,
            )
            + "\n"
        )
        jobs.append((case_dir, ((target, params),), pex))
    # Four six-thread simulations use 24 of a Juno/Jupiter worker's 28 cores.
    # Spawn isolates HDL21/PDK caches and writes each HDF5 as its case finishes.
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(_run_extracted_adc_noise, *job, netlist_only=netlist_only) for job in jobs]
        for future in futures:
            future.result()
    return run_dir


def frida2_10msps(run_dir: Path, *, netlist_only: bool = False) -> Path:
    """Run three FRIDA-2 stacks concurrently: 100 conversions, eight threads each."""

    jobs = []
    params = AdcTbParams(
        view="frida2",
        pex_cell="adc_12b_17step",
        dut=AdcParams(adc_bits=12, cdac=CdacParams()),
        symbol_rate=1.6 * G,
        conversions=100,
        vin_diff=h.Vdc.Params(dc=0.05),
        seq_logic_phase_delay_symbols=2.0,
    )
    for target in ("frida2_1layer_radix17", "frida2_2layer_radix17", "frida2_3layer_radix17"):
        pex = _find_latest_signed_off_pex(target)
        case_dir = run_dir / target
        case_dir.mkdir(parents=True)
        (case_dir / "input.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "pex_netlist": str(pex),
                    "pex_sha256": hashlib.sha256(pex.read_bytes()).hexdigest(),
                    "expected_historical_disconnect": False,
                    "spectre_threads": 8,
                },
                indent=2,
            )
            + "\n"
        )
        jobs.append((case_dir, ((target, params),), pex))
    # Three eight-thread simulations use the same 24-core worker budget.
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(_run_extracted_adc_noise, *job, netlist_only=netlist_only) for job in jobs]
        for future in futures:
            future.result()
    return run_dir


def frida_1_supply_noise_vs_rate(run_dir: Path) -> Path:
    """Run the 15 extracted-ADC rate and supply-noise combinations."""

    from flow.analysis.io import write_measurement
    from flow.circuit.results import adc_signal_names, convert_spectre_adc_to_measurement
    from pdk import tsmc65
    from pdk.tsmc65 import site

    rates = ((2, 320e6), (6, 960e6), (10, 1.6e9))
    noise_rms_v = 1e-3
    noise_cases = (
        ("none", (0.0, 0.0, 0.0)),
        ("vdda", (noise_rms_v, 0.0, 0.0)),
        ("vddd", (0.0, noise_rms_v, 0.0)),
        ("vddac", (0.0, 0.0, noise_rms_v)),
        ("all", (noise_rms_v, noise_rms_v, noise_rms_v)),
    )
    cases = tuple(
        (
            f"{rate_msps}msps_{noise_name}",
            AdcTbParams(
                view="frida65a",
                symbol_rate=symbol_rate,
                conversions=100,
                vin_diff=h.Vdc.Params(dc=0.05),
                seq_logic_phase_delay_symbols=2.0,
                supply_series_resistance_ohm=1.0,
                supply_series_inductance_h=1e-9,
                supply_decoupling_capacitance_f=1e-12,
                supply_noise_rms_v=rail_noise_rms_v,
                supply_noise_bandwidth_hz=25e9,
            ),
        )
        for rate_msps, symbol_rate in rates
        for noise_name, rail_noise_rms_v in noise_cases
    )
    h.pdk.set_default(tsmc65.pdk_logic)
    simulations = []
    for case_name, params in cases:
        tb = AdcTb(params)
        h.pdk.compile(tb)
        save_targets = [
            re.sub(r"([/<>-])", r"\\\1", raw_name)
            for canonical_name, raw_name in adc_signal_names(params.view).items()
            if canonical_name != "time_s"
        ]
        save_targets.extend(("xtop.vdd_a", "xtop.vdd_d", "xtop.vdd_dac"))
        tstop_s = params.conversions * len(params.seq_init_pattern) / float(params.symbol_rate)
        simulations.append(
            hs.Sim(
                name=case_name,
                tb=tb,
                attrs=[
                    site.install.include(h.pdk.Corner.TYP),
                    site.install.include_pre_simulation(),
                    hs.Include(path=site.ADC_PEX_NETLIST),
                    hs.Options(name="temp", value=25.0),
                    hs.Options(name="save", value="selected"),
                    hs.Save(save_targets),
                    hs.Tran(
                        tstop=tstop_s,
                        name="tran",
                        options={
                            "strobeperiod": min(1.0 / float(params.symbol_rate) / 16.0, 50e-12),
                            "strobeoutput": "strobeonly",
                            "noisefmin": 1.0 / tstop_s,
                            "noisefmax": "25G",
                            "noiseseed": 1,
                        },
                    ),
                ],
            )
        )
    results = cast(
        list[SimResult],
        hs.run(
            simulations,
            SimOptions(
                simulator=SupportedSimulators.SPECTRE,
                fmt=ResultFormat.SIM_DATA,
                rundir=run_dir,
                simulator_args=("+preset=mx", "+mt=4", "+lqtimeout", "3600", "+escchars", "+log", "spectre.log"),
            ),
        ),
    )
    for index, ((case_name, params), result) in enumerate(zip(cases, results, strict=True)):
        transient = cast(TranResult, result[AnalysisType.TRAN])
        case_dir = run_dir / case_name
        (run_dir / str(index)).rename(case_dir)
        measurement = convert_spectre_adc_to_measurement(
            transient.data,
            params=params,
            raw_path=case_dir / "netlist.raw",
            signal_names=adc_signal_names(params.view),
            maximum_waveform_records=3,
        )
        write_measurement(case_dir / "result.h5", measurement)
    return run_dir


def frida_1_transfer_curve(run_dir: Path) -> Path:
    """Run the extracted ADC from -750 mV to +750 mV in 10 mV steps."""

    from flow.analysis.io import write_measurement
    from flow.circuit.results import adc_signal_names, convert_spectre_adc_to_measurement
    from pdk import tsmc65
    from pdk.tsmc65 import site

    params = AdcTbParams(
        view="frida65a",
        symbol_rate=1.6e9,
        conversions=151,
        vin_diff=hs.LinearSweep(start=-0.75, stop=0.75, step=0.01),
        seq_logic_phase_delay_symbols=2.0,
    )
    h.pdk.set_default(tsmc65.pdk_logic)
    tb = AdcTb(params)
    h.pdk.compile(tb)
    save_targets = [
        re.sub(r"([/<>-])", r"\\\1", raw_name)
        for canonical_name, raw_name in adc_signal_names(params.view).items()
        if canonical_name != "time_s"
    ]
    tstop_s = params.conversions * len(params.seq_init_pattern) / float(params.symbol_rate)
    simulation = hs.Sim(
        tb=tb,
        attrs=[
            site.install.include(h.pdk.Corner.TYP),
            site.install.include_pre_simulation(),
            hs.Include(path=site.ADC_PEX_NETLIST),
            hs.Options(name="temp", value=25.0),
            hs.Options(name="save", value="selected"),
            hs.Save(save_targets),
            hs.Tran(tstop=tstop_s, name="tran", options={"strobeperiod": 50e-12, "strobeoutput": "strobeonly"}),
        ],
    )
    # HDL21 types cover every result format and analysis; this call requests
    # SIM_DATA containing one transient.
    result = cast(
        SimResult,
        simulation.run(
            SimOptions(
                simulator=SupportedSimulators.SPECTRE,
                fmt=ResultFormat.SIM_DATA,
                rundir=run_dir,
                simulator_args=("+preset=mx", "+mt=4", "+lqtimeout", "3600", "+escchars", "+log", "spectre.log"),
            )
        ),
    )
    transient = cast(TranResult, result[AnalysisType.TRAN])
    measurement = convert_spectre_adc_to_measurement(
        transient.data,
        params=params,
        raw_path=run_dir / "netlist.raw",
        signal_names=adc_signal_names(params.view),
        maximum_waveform_records=3,
    )
    write_measurement(run_dir / "result.h5", measurement)
    return run_dir


def hdl21gen_noise_vs_rate(run_dir: Path) -> Path:
    """Run 100 generated-ADC conversions at 2, 6, and 10 Msps."""

    from flow.analysis.io import write_measurement
    from flow.circuit.results import adc_signal_names, convert_spectre_adc_to_measurement
    from pdk import tsmc65
    from pdk.tsmc65 import site

    parameters = (
        AdcTbParams(
            view="hdl21gen",
            symbol_rate=320e6,
            conversions=100,
            vin_diff=h.Vdc.Params(dc=0.05),
            seq_logic_phase_delay_symbols=2.0,
        ),
        AdcTbParams(
            view="hdl21gen",
            symbol_rate=960e6,
            conversions=100,
            vin_diff=h.Vdc.Params(dc=0.05),
            seq_logic_phase_delay_symbols=2.0,
        ),
        AdcTbParams(
            view="hdl21gen",
            symbol_rate=1.6e9,
            conversions=100,
            vin_diff=h.Vdc.Params(dc=0.05),
            seq_logic_phase_delay_symbols=2.0,
        ),
    )
    h.pdk.set_default(tsmc65.pdk_logic)
    standard_cells = (
        *site.STANDARD_CELL_SPICE_NETLISTS,
        Path(__file__).resolve().parents[2] / "design/spice/adc_digital.sp",
    )
    simulations = []
    for params in parameters:
        tb = AdcTb(params)
        h.pdk.compile(tb)
        save_targets = [
            re.sub(r"([/<>-])", r"\\\1", raw_name)
            for canonical_name, raw_name in adc_signal_names(params.view).items()
            if canonical_name != "time_s"
        ]
        tstop_s = params.conversions * len(params.seq_init_pattern) / float(params.symbol_rate)
        simulations.append(
            hs.Sim(
                tb=tb,
                attrs=[
                    site.install.include(h.pdk.Corner.TYP),
                    site.install.include_pre_simulation(),
                    h.Literal(
                        "\n".join(
                            (
                                "simulator lang=spice",
                                *(f'.include "{path}"' for path in standard_cells),
                                "simulator lang=spectre",
                            )
                        )
                    ),
                    hs.Options(name="temp", value=25.0),
                    hs.Options(name="save", value="selected"),
                    hs.Save(save_targets),
                    hs.Tran(
                        tstop=tstop_s,
                        name="tran",
                        noise=True,
                        options={
                            "strobeperiod": min(1.0 / float(params.symbol_rate) / 16.0, 50e-12),
                            "strobeoutput": "strobeonly",
                            "noisefmin": 1.0 / tstop_s,
                            "noisefmax": "25G",
                            "noiseseed": 1,
                        },
                    ),
                ],
            )
        )
    # HDL21 types cover scalar and sequence inputs, all result formats, and all analyses;
    # this call requests a SIM_DATA list containing one transient per simulation.
    results = cast(
        list[SimResult],
        hs.run(
            simulations,
            SimOptions(
                simulator=SupportedSimulators.SPECTRE,
                fmt=ResultFormat.SIM_DATA,
                rundir=run_dir,
                simulator_args=("+preset=mx", "+mt=4", "+lqtimeout", "3600", "+escchars", "+log", "spectre.log"),
            ),
        ),
    )
    for index, (params, result) in enumerate(zip(parameters, results, strict=True)):
        transient = cast(TranResult, result[AnalysisType.TRAN])
        case_dir = run_dir / str(index)
        measurement = convert_spectre_adc_to_measurement(
            transient.data,
            params=params,
            raw_path=case_dir / "netlist.raw",
            signal_names=adc_signal_names(params.view),
            maximum_waveform_records=3,
        )
        write_measurement(case_dir / "result.h5", measurement)
    return run_dir


def hdl21gen_transfer_curve(run_dir: Path) -> Path:
    """Run the generated ADC from -750 mV to +750 mV in 10 mV steps."""

    from flow.analysis.io import write_measurement
    from flow.circuit.results import adc_signal_names, convert_spectre_adc_to_measurement
    from pdk import tsmc65
    from pdk.tsmc65 import site

    params = AdcTbParams(
        view="hdl21gen",
        symbol_rate=1.6e9,
        conversions=151,
        vin_diff=hs.LinearSweep(start=-0.75, stop=0.75, step=0.01),
        seq_logic_phase_delay_symbols=2.0,
    )
    h.pdk.set_default(tsmc65.pdk_logic)
    standard_cells = (
        *site.STANDARD_CELL_SPICE_NETLISTS,
        Path(__file__).resolve().parents[2] / "design/spice/adc_digital.sp",
    )
    tb = AdcTb(params)
    h.pdk.compile(tb)
    save_targets = [
        re.sub(r"([/<>-])", r"\\\1", raw_name)
        for canonical_name, raw_name in adc_signal_names(params.view).items()
        if canonical_name != "time_s"
    ]
    tstop_s = params.conversions * len(params.seq_init_pattern) / float(params.symbol_rate)
    simulation = hs.Sim(
        tb=tb,
        attrs=[
            site.install.include(h.pdk.Corner.TYP),
            site.install.include_pre_simulation(),
            h.Literal(
                "\n".join(
                    (
                        "simulator lang=spice",
                        *(f'.include "{path}"' for path in standard_cells),
                        "simulator lang=spectre",
                    )
                )
            ),
            hs.Options(name="temp", value=25.0),
            hs.Options(name="save", value="selected"),
            hs.Save(save_targets),
            hs.Tran(tstop=tstop_s, name="tran", options={"strobeperiod": 50e-12, "strobeoutput": "strobeonly"}),
        ],
    )
    # HDL21 types cover every result format and analysis; this call requests
    # SIM_DATA containing one transient.
    result = cast(
        SimResult,
        simulation.run(
            SimOptions(
                simulator=SupportedSimulators.SPECTRE,
                fmt=ResultFormat.SIM_DATA,
                rundir=run_dir,
                simulator_args=("+preset=mx", "+mt=4", "+lqtimeout", "3600", "+escchars", "+log", "spectre.log"),
            )
        ),
    )
    transient = cast(TranResult, result[AnalysisType.TRAN])
    measurement = convert_spectre_adc_to_measurement(
        transient.data,
        params=params,
        raw_path=run_dir / "netlist.raw",
        signal_names=adc_signal_names(params.view),
        maximum_waveform_records=3,
    )
    write_measurement(run_dir / "result.h5", measurement)
    return run_dir


def main() -> None:
    """Create one output directory and run one named ADC target."""

    targets = {
        target.__name__: target
        for target in (
            frida_1_noise_vs_rate_check,
            frida_1_noise_vs_rate,
            frida_1_supply_noise_vs_rate,
            frida_1_transfer_curve_check,
            frida_1_transfer_curve,
            frida2_2layer_radix17_10msps,
            frida2_3layer_radix17_10msps,
            frida1_10msps,
            frida2_10msps,
            hdl21gen_noise_vs_rate_check,
            hdl21gen_noise_vs_rate,
            hdl21gen_transfer_curve_check,
            hdl21gen_transfer_curve,
        )
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", choices=sorted(targets))
    parser.add_argument(
        "--netlist-only", action="store_true", help="Preflight a 100-conversion campaign without Spectre"
    )
    args = parser.parse_args()
    if args.target is None:
        print("Available ADC simulation targets:")
        for name in sorted(targets):
            print(f"  {name}")
        return
    run_dir = (
        Path(__file__).resolve().parents[2]
        / "build"
        / "sim"
        / "adc"
        / args.target
        / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    if args.netlist_only:
        if args.target not in (
            "frida1_10msps",
            "frida2_10msps",
            "frida2_2layer_radix17_10msps",
            "frida2_3layer_radix17_10msps",
        ):
            parser.error("--netlist-only applies to the 100-conversion extracted campaigns")
        if args.target == "frida1_10msps":
            frida1_10msps(run_dir, netlist_only=True)
        elif args.target == "frida2_10msps":
            frida2_10msps(run_dir, netlist_only=True)
        elif args.target == "frida2_2layer_radix17_10msps":
            frida2_2layer_radix17_10msps(run_dir, netlist_only=True)
        else:
            frida2_3layer_radix17_10msps(run_dir, netlist_only=True)
    else:
        targets[args.target](run_dir)


if __name__ == "__main__":
    main()
