"""Comparator testbench and named TSMC65 Spectre simulation targets."""

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import cast

import hdl21 as h
import hdl21.sim as hs
from hdl21.primitives import C, MosType, R, Vdc, Vpulse, Vpwl
from vlsirtools.spice import ResultFormat, SimOptions, SupportedSimulators
from vlsirtools.spice.sim_data import AnalysisType, SimResult, TranResult

from flow.analysis.io import write_measurement
from flow.circuit.results import comp_signal_names, convert_spectre_comp_to_measurement

from .subckt import Bias, Comp, CompParams, Stages, State, is_valid_comp_params


@h.paramclass
class CompTbParams:
    """Parameters which determine one generated comparator testbench."""

    vdd = h.Param(dtype=h.Scalar, desc="Supply voltage", default=1.2)
    comp = h.Param(dtype=CompParams, desc="Comparator parameters", default=CompParams())
    vin_cm_values_v = h.Param(dtype=tuple[h.Scalar, ...], desc="Input common-mode values", default=(0.8,))
    vin_diff_values_v = h.Param(
        dtype=tuple[h.Scalar, ...],
        desc="Differential-input values",
        default=tuple(step * 100e-6 for step in range(-30, 31)),
    )
    conversions = h.Param(dtype=int, desc="Decisions per input point", default=100)
    reset_time_s = h.Param(dtype=h.Scalar, desc="Reset interval", default=10e-9)
    evaluation_time_s = h.Param(dtype=h.Scalar, desc="Evaluation interval", default=30e-9)
    transition_time_s = h.Param(dtype=h.Scalar, desc="Clock and input transition", default=100e-12)
    source_resistance_ohm = h.Param(dtype=h.Scalar, desc="Input source resistance", default=1_000.0)
    source_capacitance_f = h.Param(dtype=h.Scalar, desc="Input source capacitance", default=100e-15)
    output_load_f = h.Param(dtype=h.Scalar, desc="Output load per side", default=10e-15)


@h.generator
def CompTb(params: CompTbParams) -> h.Module:
    """Generate a complete swept comparator testbench."""

    if not math.isfinite(float(params.vdd)) or float(params.vdd) <= 0.0:
        raise ValueError("comparator vdd must be finite and positive")
    common_modes = tuple(float(value) for value in params.vin_cm_values_v)
    differential_values = tuple(float(value) for value in params.vin_diff_values_v)
    if (
        not common_modes
        or not all(math.isfinite(value) for value in common_modes)
        or len(set(common_modes)) != len(common_modes)
    ):
        raise ValueError("comparator common modes must be finite and unique")
    if (
        not differential_values
        or not all(math.isfinite(value) for value in differential_values)
        or len(set(differential_values)) != len(differential_values)
    ):
        raise ValueError("comparator differential inputs must be finite and unique")
    if params.conversions <= 0:
        raise ValueError("comparator conversions must be positive")
    for name in (
        "reset_time_s",
        "evaluation_time_s",
        "transition_time_s",
        "source_resistance_ohm",
        "source_capacitance_f",
        "output_load_f",
    ):
        if not math.isfinite(float(getattr(params, name))) or float(getattr(params, name)) <= 0.0:
            raise ValueError(f"comparator {name} must be finite and positive")
    if float(params.transition_time_s) >= min(float(params.reset_time_s), float(params.evaluation_time_s)):
        raise ValueError("comparator transition must be shorter than reset and evaluation")
    cycle_s = float(params.reset_time_s) + float(params.evaluation_time_s)

    @h.module
    class CompTb:
        vss = h.Port(desc="Simulator ground")
        vdd, vin_cm, vin_diff, in_p, in_n, clk, clk_b, out_p, out_n = h.Signals(9)
        vin = h.Diff()

    CompTb.vvdd = Vdc(dc=params.vdd)(p=CompTb.vdd, n=CompTb.vss)
    CompTb.rsrc_p = R(r=params.source_resistance_ohm)(p=CompTb.vin.p, n=CompTb.in_p)
    CompTb.rsrc_n = R(r=params.source_resistance_ohm)(p=CompTb.vin.n, n=CompTb.in_n)
    CompTb.csrc_p = C(c=params.source_capacitance_f)(p=CompTb.in_p, n=CompTb.vss)
    CompTb.csrc_n = C(c=params.source_capacitance_f)(p=CompTb.in_n, n=CompTb.vss)
    CompTb.vclk = Vpulse(
        v1=0.0,
        v2=params.vdd,
        period=cycle_s,
        width=params.evaluation_time_s,
        rise=params.transition_time_s,
        fall=params.transition_time_s,
        delay=params.reset_time_s,
    )(p=CompTb.clk, n=CompTb.vss)
    CompTb.vclk_b = Vpulse(
        v1=params.vdd,
        v2=0.0,
        period=cycle_s,
        width=params.evaluation_time_s,
        rise=params.transition_time_s,
        fall=params.transition_time_s,
        delay=params.reset_time_s,
    )(p=CompTb.clk_b, n=CompTb.vss)
    CompTb.cload_p = C(c=params.output_load_f)(p=CompTb.out_p, n=CompTb.vss)
    CompTb.cload_n = C(c=params.output_load_f)(p=CompTb.out_n, n=CompTb.vss)
    CompTb.dut = Comp(params.comp)(
        inp=CompTb.in_p,
        inn=CompTb.in_n,
        outp=CompTb.out_p,
        outn=CompTb.out_n,
        clk=CompTb.clk,
        clkb=CompTb.clk_b,
        vdd=CompTb.vdd,
        vss=CompTb.vss,
    )
    common_mode_schedule = []
    differential_schedule = []
    for common_mode in params.vin_cm_values_v:
        for differential in params.vin_diff_values_v:
            common_mode_schedule.append(common_mode)
            differential_schedule.append(differential)
    dwell_s = params.conversions * cycle_s
    CompTb.vvin_cm = Vpwl(
        wave=h.Pwl.steps(
            values=common_mode_schedule, dwell=dwell_s, transition=params.transition_time_s, transition_at="start"
        )
    )(p=CompTb.vin_cm, n=CompTb.vss)
    CompTb.vvin_diff = Vpwl(
        wave=h.Pwl.steps(
            values=differential_schedule, dwell=dwell_s, transition=params.transition_time_s, transition_at="start"
        )
    )(p=CompTb.vin_diff, n=CompTb.vss)
    CompTb.evin_p = h.Vcvs(gain=0.5)(p=CompTb.vin.p, n=CompTb.vin_cm, cp=CompTb.vin_diff, cn=CompTb.vss)
    CompTb.evin_n = h.Vcvs(gain=-0.5)(p=CompTb.vin.n, n=CompTb.vin_cm, cp=CompTb.vin_diff, cn=CompTb.vss)
    return CompTb


def _run_comp_sim(
    run_dir: Path,
    params: CompTbParams,
    *,
    candidate_id: str,
    candidate_label: str,
    topology_index: int,
    size_profile: str,
    check: bool = False,
) -> Path:
    """Execute one configured comparator case, not an entire campaign.

    Targets define experiments: topology, sizing, stimuli, and case concurrency.
    This executor owns compilation, the shared Spectre recipe, short diagnostic
    checks, and measurement output. Each campaign worker compiles in an isolated
    process and uses one Spectre thread; no compiled HDL21 graph crosses workers.
    """
    from pdk import tsmc65
    from pdk.tsmc65 import site

    run_dir.mkdir(parents=True, exist_ok=True)
    if not is_valid_comp_params(params.comp):
        raise ValueError(f"invalid comparator case {candidate_id}")
    try:
        h.pdk.set_default(tsmc65.pdk_logic)
        tb = CompTb(params)
        h.pdk.compile(tb)
        tstop_s = (
            len(params.vin_cm_values_v)
            * len(params.vin_diff_values_v)
            * params.conversions
            * (float(params.reset_time_s) + float(params.evaluation_time_s))
        )
        attrs = [
            site.install.include(h.pdk.Corner.TYP),
            site.install.include_pre_simulation(),
            hs.Options(name="temp", value=25.0),
            hs.Options(name="save", value="selected"),
            hs.Save([raw for canonical, raw in comp_signal_names().items() if canonical != "time_s"]),
        ]
        if check:
            attrs.append(
                h.Literal(
                    "check_caps static_capacitor type=distr\n"
                    "check_erc static_erc floatbulk=all floatgate=no_top_moscap dangle=no_top "
                    "gate2power=on gate2ground=on\n"
                    "check_highz static_highz node=[*] fanout=gate_has_driver_no_moscap\n"
                    "check_dcpath static_dcpath net=[xtop.vdd 0]\n"
                    "check_stack static_stack count=3\n"
                    "check_topology static_topology node=[*] pin2gnd=on\n"
                    "check_nodecap dyn_nodecap node=[xtop.in_p xtop.in_n xtop.out_p xtop.out_n] time=[10n 39n]\n"
                    "check_setuphold dyn_setuphold node=[xtop.out_p xtop.out_n] ref_node=xtop.clk "
                    "setup_time=50p hold_time=50p"
                )
            )
        tran_options = {"strobeperiod": 500e-12, "strobeoutput": "strobeonly"}
        if not check:
            tran_options.update(noisefmin=1.0 / tstop_s, noisefmax="25G", noiseseed=1)
        attrs.append(hs.Tran(tstop=tstop_s, name="tran", noise=not check, options=tran_options))
        simulation = hs.Sim(tb=tb, attrs=attrs)
        started = time.perf_counter()
        result = simulation.run(
            SimOptions(
                simulator=SupportedSimulators.SPECTRE,
                fmt=ResultFormat.NONE if check else ResultFormat.SIM_DATA,
                rundir=run_dir,
                simulator_args=(
                    "+preset=mx",
                    "+mt=1",
                    "+lqtimeout",
                    "3600",
                    "+escchars",
                    "+log",
                    "spectre.log",
                    *(("-ahdllint=warn", "-ahdllint_log", "ahdllint.log") if check else ()),
                ),
            )
        )
        runtime_s = time.perf_counter() - started
        if not check:
            transient = cast(TranResult, cast(SimResult, result)[AnalysisType.TRAN])
            measurement = convert_spectre_comp_to_measurement(
                transient.data,
                params=params,
                raw_path=run_dir / "netlist.raw",
                signal_names=comp_signal_names(),
                candidate_id=candidate_id,
                candidate_label=candidate_label,
                topology_index=topology_index,
                size_profile=size_profile,
                compiled_tb=tb,
                spectre_runtime_s=runtime_s,
            )
            write_measurement(run_dir / "result.h5", measurement)
    finally:
        CompTb.Cache.reset()
        Comp.Cache.reset()
    return run_dir


def hdl21_comp_perf_vs_size(run_dir: Path, *, check: bool = False) -> Path:
    """Run 296 generated sizes plus FRIDA-1; checks cover six representative cases."""

    run_dir.mkdir(parents=True, exist_ok=True)
    if check:
        cases = (
            (
                "fabricated_single_nmos_switched",
                CompParams(
                    diffpair_w=37,
                    tail_w=5,
                    rst_w=8,
                    latch_on_w=25,
                    latch_init_w=33,
                    srlatch_n_w=4,
                    srlatch_p_w=8,
                    diffpair_l=5,
                    tail_l=13,
                    rst_l=1,
                    latch_on_l=6,
                    latch_init_l=17,
                    preamp_diff_xtors=MosType.NMOS,
                    preamp_bias=Bias.SWITCHED,
                    comp_stages=Stages.SINGLE,
                    latch_inner_init_xtors=State.CLOCK,
                ),
            ),
            (
                "half_single_pmos_dynamic",
                CompParams(
                    diffpair_w=19,
                    tail_w=3,
                    rst_w=4,
                    latch_on_w=13,
                    latch_init_w=17,
                    srlatch_n_w=2,
                    srlatch_p_w=4,
                    diffpair_l=5,
                    tail_l=13,
                    rst_l=1,
                    latch_on_l=6,
                    latch_init_l=17,
                    preamp_diff_xtors=MosType.PMOS,
                    preamp_bias=Bias.DYNAMIC,
                    comp_stages=Stages.SINGLE,
                    latch_inner_init_xtors=State.SIGNAL,
                ),
            ),
            (
                "double_double_nmos_switched",
                CompParams(
                    diffpair_w=74,
                    tail_w=10,
                    rst_w=16,
                    latch_on_w=50,
                    latch_init_w=66,
                    srlatch_n_w=8,
                    srlatch_p_w=16,
                    diffpair_l=5,
                    tail_l=13,
                    rst_l=1,
                    latch_on_l=6,
                    latch_init_l=17,
                    preamp_diff_xtors=MosType.NMOS,
                    preamp_bias=Bias.SWITCHED,
                    comp_stages=Stages.DOUBLE,
                    latch_inner_on_xtors=State.SIGNAL,
                    latch_outer_on_xtors=State.OMIT,
                    latch_inner_init_xtors=State.CLOCK,
                    latch_outer_init_xtors=State.OMIT,
                ),
            ),
            (
                "fabricated_double_pmos_dynamic",
                CompParams(
                    diffpair_w=37,
                    tail_w=5,
                    rst_w=8,
                    latch_on_w=25,
                    latch_init_w=33,
                    srlatch_n_w=4,
                    srlatch_p_w=8,
                    diffpair_l=5,
                    tail_l=13,
                    rst_l=1,
                    latch_on_l=6,
                    latch_init_l=17,
                    preamp_diff_xtors=MosType.PMOS,
                    preamp_bias=Bias.DYNAMIC,
                    comp_stages=Stages.DOUBLE,
                    latch_inner_on_xtors=State.CLOCK,
                    latch_outer_on_xtors=State.SIGNAL,
                    latch_inner_init_xtors=State.SIGNAL,
                    latch_outer_init_xtors=State.OMIT,
                ),
            ),
            (
                "half_double_nmos_dynamic",
                CompParams(
                    diffpair_w=19,
                    tail_w=3,
                    rst_w=4,
                    latch_on_w=13,
                    latch_init_w=17,
                    srlatch_n_w=2,
                    srlatch_p_w=4,
                    diffpair_l=5,
                    tail_l=13,
                    rst_l=1,
                    latch_on_l=6,
                    latch_init_l=17,
                    preamp_diff_xtors=MosType.NMOS,
                    preamp_bias=Bias.DYNAMIC,
                    comp_stages=Stages.DOUBLE,
                    latch_inner_on_xtors=State.SIGNAL,
                    latch_outer_on_xtors=State.CLOCK,
                    latch_inner_init_xtors=State.CLOCK,
                    latch_outer_init_xtors=State.CLOCK,
                ),
            ),
            (
                "double_double_pmos_switched",
                CompParams(
                    diffpair_w=74,
                    tail_w=10,
                    rst_w=16,
                    latch_on_w=50,
                    latch_init_w=66,
                    srlatch_n_w=8,
                    srlatch_p_w=16,
                    diffpair_l=5,
                    tail_l=13,
                    rst_l=1,
                    latch_on_l=6,
                    latch_init_l=17,
                    preamp_diff_xtors=MosType.PMOS,
                    preamp_bias=Bias.SWITCHED,
                    comp_stages=Stages.DOUBLE,
                    latch_inner_on_xtors=State.CLOCK,
                    latch_outer_on_xtors=State.CLOCK,
                    latch_inner_init_xtors=State.SIGNAL,
                    latch_outer_init_xtors=State.SIGNAL,
                ),
            ),
        )
        cases = [(name, name, 0, "diagnostic", comp) for name, comp in cases]
    else:
        topologies = []
        for diff_type in (MosType.NMOS, MosType.PMOS):
            for bias in Bias:
                for stages in Stages:
                    for inner_on in State:
                        for outer_on in State:
                            for inner_init in (State.CLOCK, State.SIGNAL):
                                for outer_init in (State.OMIT, State.CLOCK, State.SIGNAL):
                                    topology = {
                                        "comp_stages": stages,
                                        "preamp_diff_xtors": diff_type,
                                        "preamp_bias": bias,
                                        "latch_inner_on_xtors": inner_on,
                                        "latch_outer_on_xtors": outer_on,
                                        "latch_inner_init_xtors": inner_init,
                                        "latch_outer_init_xtors": outer_init,
                                    }
                                    probe = CompParams(
                                        diffpair_w=37,
                                        tail_w=5,
                                        rst_w=8,
                                        latch_on_w=25,
                                        latch_init_w=33,
                                        srlatch_n_w=4,
                                        srlatch_p_w=8,
                                        diffpair_l=5,
                                        tail_l=13,
                                        rst_l=1,
                                        latch_on_l=6,
                                        latch_init_l=17,
                                        **topology,
                                    )
                                    if is_valid_comp_params(probe):
                                        topologies.append(topology)
        if len(topologies) != 148:
            raise RuntimeError(f"expected 148 valid comparator topologies, got {len(topologies)}")

        cases = []
        for topology_index, topology in enumerate(topologies):
            for size_profile, widths in (
                (
                    "half",
                    {
                        "diffpair_w": 19,
                        "tail_w": 3,
                        "rst_w": 4,
                        "latch_on_w": 13,
                        "latch_init_w": 17,
                        "srlatch_n_w": 2,
                        "srlatch_p_w": 4,
                    },
                ),
                (
                    "double",
                    {
                        "diffpair_w": 74,
                        "tail_w": 10,
                        "rst_w": 16,
                        "latch_on_w": 50,
                        "latch_init_w": 66,
                        "srlatch_n_w": 8,
                        "srlatch_p_w": 16,
                    },
                ),
            ):
                comp = CompParams(
                    **topology,
                    **widths,
                    diffpair_l=5,
                    tail_l=13,
                    rst_l=1,
                    latch_on_l=6,
                    latch_init_l=17,
                )
                digest = hashlib.sha256(repr(comp).encode()).hexdigest()[:8]
                candidate_id = f"c{topology_index:03d}_{size_profile}_{digest}"
                topology_label = "-".join(
                    (
                        comp.preamp_diff_xtors.name.lower(),
                        comp.preamp_bias.name.lower(),
                        comp.comp_stages.name.lower(),
                        f"inner-on-{comp.latch_inner_on_xtors.name.lower()}",
                        f"outer-on-{comp.latch_outer_on_xtors.name.lower()}",
                        f"inner-init-{comp.latch_inner_init_xtors.name.lower()}",
                        f"outer-init-{comp.latch_outer_init_xtors.name.lower()}",
                    )
                )
                cases.append(
                    (
                        candidate_id,
                        f"{topology_label}, {size_profile}",
                        topology_index,
                        size_profile,
                        comp,
                    )
                )
        baseline = CompParams(
            diffpair_w=37,
            tail_w=5,
            rst_w=8,
            latch_on_w=25,
            latch_init_w=33,
            srlatch_n_w=4,
            srlatch_p_w=8,
            diffpair_l=5,
            tail_l=13,
            rst_l=1,
            latch_on_l=6,
            latch_init_l=17,
        )
        baseline_topology = {
            "comp_stages": baseline.comp_stages,
            "preamp_diff_xtors": baseline.preamp_diff_xtors,
            "preamp_bias": baseline.preamp_bias,
            "latch_inner_on_xtors": baseline.latch_inner_on_xtors,
            "latch_outer_on_xtors": baseline.latch_outer_on_xtors,
            "latch_inner_init_xtors": baseline.latch_inner_init_xtors,
            "latch_outer_init_xtors": baseline.latch_outer_init_xtors,
        }
        cases.append(
            (
                "frida1_fabricated_baseline",
                "FRIDA-1 fabricated comparator dimensions",
                topologies.index(baseline_topology),
                "fabricated",
                baseline,
            )
        )
        if len(cases) != 297 or len({case[0] for case in cases}) != 297:
            raise RuntimeError("comparator campaign must contain 297 unique cases")

    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=24, mp_context=get_context("spawn")) as executor:
        futures = {
            executor.submit(
                _run_comp_sim,
                run_dir / candidate_id,
                CompTbParams(
                    comp=comp,
                    **({"vin_cm_values_v": (0.8,), "vin_diff_values_v": (0.0,), "conversions": 1} if check else {}),
                ),
                candidate_id=candidate_id,
                candidate_label=label,
                topology_index=topology_index,
                size_profile=size_profile,
                check=check,
            ): candidate_id
            for candidate_id, label, topology_index, size_profile, comp in cases
        }
        for future in as_completed(futures):
            candidate_id = futures[future]
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - report every independent case failure
                failures[candidate_id] = repr(error)
    if failures:
        failure_path = run_dir / "failures.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError(f"{len(failures)} comparator cases failed; see {failure_path}")
    return run_dir


def frida1_fixed_input_noise(run_dir: Path, *, check: bool = False) -> Path:
    """Run the FRIDA-1 comparator S-curve, or one noise-free diagnostic decision."""

    params = CompTbParams(
        comp=CompParams(
            diffpair_w=37,
            tail_w=5,
            rst_w=8,
            latch_on_w=25,
            latch_init_w=33,
            srlatch_n_w=4,
            srlatch_p_w=8,
            diffpair_l=5,
            tail_l=13,
            rst_l=1,
            latch_on_l=6,
            latch_init_l=17,
        ),
        **({"vin_cm_values_v": (0.8,), "vin_diff_values_v": (0.0,), "conversions": 1} if check else {}),
    )
    return _run_comp_sim(
        run_dir,
        params,
        candidate_id="frida1_fabricated_baseline",
        candidate_label="FRIDA-1 fabricated comparator dimensions",
        topology_index=37,
        size_profile="fabricated",
        check=check,
    )


def main() -> None:
    """Create one output directory and run one named comparator target."""

    targets = {
        target.__name__: target
        for target in (
            hdl21_comp_perf_vs_size,
            frida1_fixed_input_noise,
        )
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", choices=list(targets))
    args = parser.parse_args()
    if args.target is None:
        print("Available comparator simulation targets:")
        for name in list(targets):
            print(f"  {name}")
        return
    run_dir = (
        Path(__file__).resolve().parents[2]
        / "build"
        / "sim"
        / "comp"
        / args.target
        / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    targets[args.target](run_dir)


if __name__ == "__main__":
    main()
