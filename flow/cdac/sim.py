"""CDAC testbench and named TSMC65 Spectre simulation targets."""

import argparse
from datetime import datetime
from pathlib import Path

import hdl21 as h
import hdl21.sim as hs
from hdl21.prefix import f, p
from hdl21.primitives import C, Vdc, Vpwl
from vlsirtools.spice import ResultFormat, SimOptions, SupportedSimulators

from .subckt import Cdac, CdacParams


@h.paramclass
class CdacTbParams:
    """Parameters which determine the generated CDAC testbench."""

    vdd = h.Param(dtype=h.Scalar, desc="Supply voltage", default=1.2)
    cdac = h.Param(dtype=CdacParams, desc="CDAC parameters", default=CdacParams())
    code_dwell_s = h.Param(dtype=h.Scalar, desc="Time held at each input code", default=200e-9)
    transition_time_s = h.Param(dtype=h.Scalar, desc="Input-code transition time", default=100e-12)


@h.generator
def CdacTb(params: CdacTbParams) -> h.Module:
    """Generate a complete code-ramp CDAC testbench."""

    if float(params.code_dwell_s) <= 0.0 or float(params.transition_time_s) <= 0.0:
        raise ValueError("CDAC dwell and transition times must be positive")
    n_stages = params.cdac.n_dac + params.cdac.n_extra

    @h.module
    class CdacTb:
        vss = h.Port(desc="Simulator ground")
        vdd, top = h.Signals(2)
        dac_bits = h.Signal(width=n_stages)

    CdacTb.vvdd = Vdc(dc=params.vdd)(p=CdacTb.vdd, n=CdacTb.vss)
    CdacTb.cload = C(c=100 * f)(p=CdacTb.top, n=CdacTb.vss)
    CdacTb.dut = Cdac(params.cdac)(top=CdacTb.top, dac=CdacTb.dac_bits, vdd=CdacTb.vdd, vss=CdacTb.vss)
    stage_values: list[list[h.Scalar]] = [[] for _ in range(n_stages)]
    for code in range(2**params.cdac.n_dac):
        # Treat the integer only as a packed display stimulus: its highest
        # position is C0 and its lowest position is C15. With redundant stages,
        # this short baseline ramp therefore exercises the low-weight tail.
        for stage in range(n_stages):
            packed_bit = n_stages - stage - 1
            stage_values[stage].append(params.vdd if (code >> packed_bit) & 1 else 0.0)
    for stage, values in enumerate(stage_values):
        setattr(
            CdacTb,
            f"vdac_{stage}",
            Vpwl(
                wave=h.Pwl.steps(
                    values=values,
                    dwell=params.code_dwell_s + params.transition_time_s,
                    transition=params.transition_time_s,
                    transition_at="end",
                )
            )(p=CdacTb.dac_bits[stage], n=CdacTb.vss),
        )
    return CdacTb


def frida1_transfer_curve(run_dir: Path, *, check: bool = False) -> Path:
    """Run the complete fabricated-size CDAC code ramp."""

    from pdk import tsmc65
    from pdk.tsmc65 import site

    run_dir.mkdir(parents=True, exist_ok=True)
    params = CdacTbParams()
    h.pdk.set_default(tsmc65.pdk_logic)
    tb = CdacTb(params)
    h.pdk.compile(tb)
    n_codes = 2**params.cdac.n_dac
    simulation = hs.Sim(
        tb=tb,
        attrs=[
            site.install.include(h.pdk.Corner.TYP),
            site.install.include_pre_simulation(),
            hs.Options(name="temp", value=25.0),
            hs.Options(name="save", value="selected"),
            hs.Save([tb.top, tb.dac_bits, "xtop.vvdd:p"]),
            *(
                [
                    h.Literal(
                        "check_caps static_capacitor type=distr\n"
                        "check_erc static_erc floatbulk=all floatgate=no_top_moscap dangle=no_top "
                        "gate2power=on gate2ground=on\n"
                        "check_highz static_highz node=[*] fanout=gate_has_driver_no_moscap\n"
                        "check_dcpath static_dcpath net=[xtop.vdd 0]\n"
                        "check_topology static_topology node=[*] pin2gnd=on\n"
                        "check_nodecap dyn_nodecap node=[xtop.top] time=[100n 200n]"
                    ),
                ]
                if check
                else []
            ),
            hs.Tran(
                tstop=(
                    params.code_dwell_s + params.transition_time_s
                    if check
                    else n_codes * params.code_dwell_s + (n_codes - 1) * params.transition_time_s
                ),
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
    """Create one output directory and run one named CDAC target."""

    targets = {target.__name__: target for target in (frida1_transfer_curve,)}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", choices=sorted(targets))
    args = parser.parse_args()
    if args.target is None:
        print("Available CDAC simulation targets:")
        for name in sorted(targets):
            print(f"  {name}")
        return
    run_dir = (
        Path(__file__).resolve().parents[2]
        / "build"
        / "sim"
        / "cdac"
        / args.target
        / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    targets[args.target](run_dir)


if __name__ == "__main__":
    main()
