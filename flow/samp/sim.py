"""Sampler testbench and named TSMC65 Spectre simulation targets."""

import argparse
from datetime import datetime
from pathlib import Path

import hdl21 as h
import hdl21.sim as hs
from hdl21.prefix import f, p
from hdl21.primitives import C, Vdc, Vpulse
from vlsirtools.spice import ResultFormat, SimOptions, SupportedSimulators

from .subckt import Samp, SampParams


@h.paramclass
class SampTbParams:
    """Parameters which determine the generated sampler testbench."""

    vdd = h.Param(dtype=h.Scalar, desc="Supply voltage", default=1.2)
    samp = h.Param(dtype=SampParams, desc="Sampler parameters", default=SampParams())
    cload = h.Param(dtype=h.Prefixed, desc="Output load capacitance", default=1 * f)
    input_voltage = h.Param(dtype=h.Scalar, desc="DC sampler input voltage", default=0.6)
    clock_period_s = h.Param(dtype=h.Scalar, desc="Clock period", default=100e-9)
    clock_high_time_s = h.Param(dtype=h.Scalar, desc="Clock high time", default=50e-9)
    clock_transition_time_s = h.Param(dtype=h.Scalar, desc="Clock rise and fall time", default=100e-12)
    clock_delay_s = h.Param(dtype=h.Scalar, desc="Clock delay", default=0.0)


@h.generator
def SampTb(params: SampTbParams) -> h.Module:
    """Generate a complementary-clock sampler testbench."""

    if not 0.0 < float(params.clock_high_time_s) < float(params.clock_period_s):
        raise ValueError("sampler clock high time must lie inside one period")
    if float(params.clock_transition_time_s) <= 0.0 or float(params.clock_delay_s) < 0.0:
        raise ValueError("sampler clock transition must be positive and delay non-negative")

    @h.module
    class SampTb:
        vss = h.Port(desc="Simulator ground")
        vdd, clk, clk_b, din, dout = h.Signals(5)

    SampTb.vvdd = Vdc(dc=params.vdd)(p=SampTb.vdd, n=SampTb.vss)
    SampTb.vclk = Vpulse(
        v1=0.0,
        v2=params.vdd,
        period=params.clock_period_s,
        width=params.clock_high_time_s,
        rise=params.clock_transition_time_s,
        fall=params.clock_transition_time_s,
        delay=params.clock_delay_s,
    )(p=SampTb.clk, n=SampTb.vss)
    SampTb.vclk_b = Vpulse(
        v1=params.vdd,
        v2=0.0,
        period=params.clock_period_s,
        width=params.clock_high_time_s,
        rise=params.clock_transition_time_s,
        fall=params.clock_transition_time_s,
        delay=params.clock_delay_s,
    )(p=SampTb.clk_b, n=SampTb.vss)
    SampTb.vdin = Vdc(dc=params.input_voltage)(p=SampTb.din, n=SampTb.vss)
    SampTb.cload = C(c=params.cload)(p=SampTb.dout, n=SampTb.vss)
    SampTb.dut = Samp(params.samp)(
        din=SampTb.din,
        dout=SampTb.dout,
        clk=SampTb.clk,
        clk_b=SampTb.clk_b,
        vdd=SampTb.vdd,
        vss=SampTb.vss,
    )
    return SampTb


def frida1_transient(run_dir: Path, *, check: bool = False) -> Path:
    """Run five periods of the fabricated-size sampler transient."""

    from pdk import tsmc65
    from pdk.tsmc65 import site

    run_dir.mkdir(parents=True, exist_ok=True)
    params = SampTbParams()
    h.pdk.set_default(tsmc65.pdk_logic)
    tb = SampTb(params)
    h.pdk.compile(tb)
    simulation = hs.Sim(
        tb=tb,
        attrs=[
            site.install.include(h.pdk.Corner.TYP),
            site.install.include_pre_simulation(),
            hs.Options(name="temp", value=25.0),
            hs.Options(name="save", value="selected"),
            hs.Save([tb.din, tb.dout, tb.clk, tb.clk_b, "xtop.vvdd:p"]),
            *(
                [
                    h.Literal(
                        "check_caps static_capacitor type=distr\n"
                        "check_erc static_erc floatbulk=all floatgate=no_top_moscap dangle=no_top "
                        "gate2power=on gate2ground=on\n"
                        "check_highz static_highz node=[*] fanout=gate_has_driver_no_moscap\n"
                        "check_dcpath static_dcpath net=[xtop.vdd 0]\n"
                        "check_topology static_topology node=[*] pin2gnd=on\n"
                        "check_nodecap dyn_nodecap node=[xtop.din xtop.dout] time=[25n 75n]"
                    ),
                ]
                if check
                else []
            ),
            hs.Tran(
                tstop=params.clock_period_s if check else 5 * params.clock_period_s,
                name="tran",
                options={"strobeperiod": 100 * p, "strobeoutput": "strobeonly"},
            ),
        ],
    )
    simulation.run(
        SimOptions(
            simulator=SupportedSimulators.SPECTRE,
            fmt=ResultFormat.NONE if check else ResultFormat.SIM_DATA,
            rundir=run_dir,
            simulator_args=(
                "+preset=mx",
                "+mt=4",
                "+lqtimeout",
                "3600",
                "+escchars",
                "+log",
                "spectre.log",
                *(("-ahdllint=warn", "-ahdllint_log", "ahdllint.log") if check else ()),
            ),
        )
    )
    return run_dir


def main() -> None:
    """Create one output directory and run one named sampler target."""

    targets = {target.__name__: target for target in (frida1_transient,)}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", choices=sorted(targets))
    args = parser.parse_args()
    if args.target is None:
        print("Available sampler simulation targets:")
        for name in sorted(targets):
            print(f"  {name}")
        return
    run_dir = (
        Path(__file__).resolve().parents[2]
        / "build"
        / "sim"
        / "samp"
        / args.target
        / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    targets[args.target](run_dir)


if __name__ == "__main__":
    main()
