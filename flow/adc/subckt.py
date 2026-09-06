"""
ADC generator for FRIDA.

Creates a complete SAR ADC by composing:
- Digital control block (ExternalModule from synthesized netlist)
- 2x CDAC with integrated drivers
- 2x Sampling switches
- 1x Comparator

Architecture:
    Adc (HDL21 generator)
    ├── Xadc_digital (ExternalModule - static 16-stage)
    ├── Xcdac_p (HDL21 Cdac)
    ├── Xcdac_n (HDL21 Cdac)
    ├── Xsamp_p (HDL21 Samp)
    ├── Xsamp_n (HDL21 Samp)
    └── Xcomp (HDL21 Comp)
"""

import hdl21 as h
from hdl21.primitives import MosType

from ..cdac import Cdac, CdacParams, get_cdac_weights
from ..comp import Comp, CompParams
from ..samp import Samp, SampParams


@h.paramclass
class AdcParams:
    """ADC parameters."""

    # Top-level ADC parameter
    adc_bits = h.Param(dtype=int, desc="Nominal normalized ADC resolution", default=12)
    # Component parameters
    cdac = h.Param(
        dtype=CdacParams,
        desc="CDAC parameters",
        default=CdacParams(n_dac=11, n_extra=5),
    )
    samp = h.Param(dtype=SampParams, desc="Sampler parameters", default=SampParams())
    comp = h.Param(dtype=CompParams, desc="Comparator parameters", default=CompParams())


# ==== Digital Block ExternalModule ====

# The checked-in adc_digital netlist is the fabricated FRIDA-1 block. Its
# physical bus counts from bit 15 down to bit 0. The Adc generator below is the
# sole adapter between that immutable implementation and the project-wide
# C0-first conversion-stage convention.
#
# Port list from spice/adc_digital.cdl (simplified - main signals only):
# - Sequencer inputs: seq_init, seq_samp, seq_comp, seq_update
# - Enable inputs: en_init, en_samp_p, en_samp_n, en_comp, en_update
# - Control: dac_mode, dac_diffcaps
# - DAC state inputs: dac_astate_p[15:0], dac_bstate_p[15:0],
#                     dac_astate_n[15:0], dac_bstate_n[15:0]
# - Comparator feedback: comp_out_p, comp_out_n
# - Clock outputs: clk_samp_p, clk_samp_p_b, clk_samp_n, clk_samp_n_b, clk_comp
# - DAC state outputs: dac_state_p_main[15:0], dac_state_n_main[15:0]
# - Supplies: vdd_d, vss_d

Frida1AdcDigital = h.ExternalModule(
    name="adc_digital",
    port_list=[
        # Sequencer clock inputs
        h.Input(name="seq_init"),
        h.Input(name="seq_samp"),
        h.Input(name="seq_comp"),
        h.Input(name="seq_update"),
        # Enable inputs
        h.Input(name="en_init"),
        h.Input(name="en_samp_p"),
        h.Input(name="en_samp_n"),
        h.Input(name="en_comp"),
        h.Input(name="en_update"),
        # Control inputs
        h.Input(name="dac_mode"),
        h.Input(name="dac_diffcaps"),
        # DAC initial state inputs (16-bit buses)
        h.Input(name="dac_astate_p", width=16),
        h.Input(name="dac_bstate_p", width=16),
        h.Input(name="dac_astate_n", width=16),
        h.Input(name="dac_bstate_n", width=16),
        # Comparator feedback
        h.Input(name="comp_out_p"),
        h.Input(name="comp_out_n"),
        # Clock outputs
        h.Output(name="clk_samp_p"),
        h.Output(name="clk_samp_p_b"),
        h.Output(name="clk_samp_n"),
        h.Output(name="clk_samp_n_b"),
        h.Output(name="clk_comp"),
        # DAC state outputs (main capacitors only - no diffcaps)
        h.Output(name="dac_state_p_main", width=16),
        h.Output(name="dac_state_p_diff", width=16),  # Unused but in netlist
        h.Output(name="dac_state_n_main", width=16),
        h.Output(name="dac_state_n_diff", width=16),  # Unused but in netlist
        # DAC driver polarity outputs
        h.Output(name="dac_invert_p_main"),
        h.Output(name="dac_invert_p_diff"),
        h.Output(name="dac_invert_n_main"),
        h.Output(name="dac_invert_n_diff"),
        # Single-ended comparator readout
        h.Output(name="comp_out"),
        # Supplies
        h.Inout(name="vdd_d"),
        h.Inout(name="vss_d"),
    ],
    desc="Fabricated FRIDA-1 digital block with legacy reversed stage bus",
)


# Exact positional signature shared by the four Calibre-extracted FRIDA-1 ADC flavors.
_FRIDA1_PEX_ADC_PORT_NAMES = (
    "vdd_a",
    "vin_p",
    "vss_a",
    "dac_mode",
    "dac_diffcaps",
    "seq_init",
    "en_init",
    "seq_samp",
    "en_samp_p",
    "en_samp_n",
    "seq_comp",
    "en_comp",
    "seq_update",
    "en_update",
    "comp_out",
    "vin_n",
    "vdd_d",
    "vss_d",
    "vdd_dac",
    "vss_dac",
    "dac_bstate_p_4",
    "dac_bstate_p_11",
    "dac_astate_p_0",
    "dac_astate_p_1",
    "dac_astate_p_2",
    "dac_astate_p_3",
    "dac_astate_p_4",
    "dac_astate_p_5",
    "dac_astate_p_6",
    "dac_astate_p_8",
    "dac_astate_p_9",
    "dac_astate_p_10",
    "dac_astate_p_11",
    "dac_astate_p_12",
    "dac_astate_p_13",
    "dac_astate_p_14",
    "dac_astate_p_15",
    "dac_bstate_p_0",
    "dac_bstate_p_2",
    "dac_bstate_p_3",
    "dac_bstate_p_5",
    "dac_bstate_p_6",
    "dac_bstate_p_7",
    "dac_bstate_p_8",
    "dac_bstate_p_10",
    "dac_bstate_p_12",
    "dac_bstate_p_13",
    "dac_bstate_p_14",
    "dac_bstate_p_15",
    "dac_astate_p_7",
    "dac_bstate_p_1",
    "dac_bstate_p_9",
    "dac_astate_n_0",
    "dac_astate_n_1",
    "dac_astate_n_2",
    "dac_astate_n_3",
    "dac_astate_n_4",
    "dac_astate_n_5",
    "dac_astate_n_6",
    "dac_astate_n_7",
    "dac_astate_n_8",
    "dac_astate_n_9",
    "dac_astate_n_10",
    "dac_astate_n_11",
    "dac_astate_n_12",
    "dac_astate_n_13",
    "dac_astate_n_14",
    "dac_astate_n_15",
    "dac_bstate_n_0",
    "dac_bstate_n_1",
    "dac_bstate_n_2",
    "dac_bstate_n_3",
    "dac_bstate_n_4",
    "dac_bstate_n_5",
    "dac_bstate_n_6",
    "dac_bstate_n_7",
    "dac_bstate_n_8",
    "dac_bstate_n_9",
    "dac_bstate_n_10",
    "dac_bstate_n_11",
    "dac_bstate_n_13",
    "dac_bstate_n_14",
    "dac_bstate_n_15",
    "dac_bstate_n_12",
)

Frida1_1LayerRadix17PexAdc = h.ExternalModule(
    name="adc_1layer_radix17",
    port_list=[h.Inout(name=name) for name in _FRIDA1_PEX_ADC_PORT_NAMES],
    desc="Calibre xRC extracted FRIDA-1 ADC in exact positional pin order",
)
Frida1_1LayerRadix20PexAdc = h.ExternalModule(
    name="adc_1layer_radix20",
    port_list=[h.Inout(name=name) for name in _FRIDA1_PEX_ADC_PORT_NAMES],
    desc="Calibre xRC extracted FRIDA-1 one-layer radix-20 ADC",
)
Frida1_2LayerRadix17PexAdc = h.ExternalModule(
    name="adc_2layer_radix17",
    port_list=[h.Inout(name=name) for name in _FRIDA1_PEX_ADC_PORT_NAMES],
    desc="Calibre xRC extracted FRIDA-1 two-layer radix-17 ADC",
)
Frida1_2LayerRadix20PexAdc = h.ExternalModule(
    name="adc_2layer_radix20",
    port_list=[h.Inout(name=name) for name in _FRIDA1_PEX_ADC_PORT_NAMES],
    desc="Calibre xRC extracted FRIDA-1 two-layer radix-20 ADC",
)
Frida1PexAdc = h.ExternalModule(
    name="adc_12b_17step",
    port_list=[h.Inout(name=name) for name in _FRIDA1_PEX_ADC_PORT_NAMES],
    desc="Calibre xRC extracted FRIDA-1 ADC with standardized top-cell name",
)
Frida2PexAdc = h.ExternalModule(
    name="adc_12b_17step",
    port_list=[
        h.Inout(
            name=f"{name.rsplit('_', 1)[0]}_{15 - int(name.rsplit('_', 1)[1])}"
            if name.startswith("dac_") and "state_" in name
            else name
        )
        for name in _FRIDA1_PEX_ADC_PORT_NAMES
    ],
    desc="Calibre xRC extracted C0-first FRIDA-2 ADC; exact renamed positional signature",
)


@h.generator
def Adc(p: AdcParams) -> h.Module:
    """
    SAR ADC generator.

    Composes digital control with analog blocks (CDAC, sampler, comparator).

    The fabricated digital block fixes this implementation at 16 capacitor
    stages and 17 comparator decisions.
    """
    n_stages = len(get_cdac_weights(p.cdac))
    if n_stages != 16:
        raise ValueError(
            f"ADC currently requires 16 CDAC stages (got {n_stages}). CDAC params must give 16 physical stages."
        )

    @h.module
    class Adc:
        """SAR ADC module."""

        # External IO
        vin_p = h.Input(desc="Positive analog input")
        vin_n = h.Input(desc="Negative analog input")

        # Sequencer clock inputs (directly connected to digital block)
        seq_init = h.Input(desc="Initialize sequence")
        seq_samp = h.Input(desc="Sample sequence")
        seq_comp = h.Input(desc="Compare sequence")
        seq_update = h.Input(desc="Update sequence")

        # Enable inputs
        en_init = h.Input(desc="Enable initialization")
        en_samp_p = h.Input(desc="Enable positive sampler")
        en_samp_n = h.Input(desc="Enable negative sampler")
        en_comp = h.Input(desc="Enable comparator")
        en_update = h.Input(desc="Enable update")

        # Control inputs
        dac_mode = h.Input(desc="DAC mode control")
        dac_diffcaps = h.Input(desc="Differential caps control")

        # DAC initial states use C0-first conversion-stage indices.
        dac_astate_p = h.Input(width=16, desc="C0-first DAC A-state positive")
        dac_bstate_p = h.Input(width=16, desc="C0-first DAC B-state positive")
        dac_astate_n = h.Input(width=16, desc="C0-first DAC A-state negative")
        dac_bstate_n = h.Input(width=16, desc="C0-first DAC B-state negative")

        # Digital outputs (for readout)
        dac_state_p = h.Output(width=16, desc="C0-first DAC state positive")
        dac_state_n = h.Output(width=16, desc="C0-first DAC state negative")
        comp_out = h.Output(desc="Single-ended comparator readout")

        # Supplies
        vdd_a = h.Inout(desc="Analog supply")
        vss_a = h.Inout(desc="Analog ground")
        vdd_d = h.Inout(desc="Digital supply")
        vss_d = h.Inout(desc="Digital ground")
        vdd_dac = h.Inout(desc="DAC-driver supply")
        vss_dac = h.Inout(desc="DAC-driver ground")

        # Internal signals
        clk_samp_p = h.Signal(desc="Sample clock positive")
        clk_samp_p_b = h.Signal(desc="Sample clock positive bar")
        clk_samp_n = h.Signal(desc="Sample clock negative")
        clk_samp_n_b = h.Signal(desc="Sample clock negative bar")
        clk_comp = h.Signal(desc="Comparator clock")
        clk_comp_b = h.Signal(desc="Comparator clock complement")
        comp_out_p = h.Signal(desc="Comparator output positive")
        comp_out_n = h.Signal(desc="Comparator output negative")
        cdac_top_p = h.Signal(desc="CDAC top plate positive")
        cdac_top_n = h.Signal(desc="CDAC top plate negative")
        dac_state_p_diff = h.Signal(width=16)  # Unused
        dac_state_n_diff = h.Signal(width=16)  # Unused
        dac_invert_p_main = h.Signal()
        dac_invert_p_diff = h.Signal()
        dac_invert_n_main = h.Signal()
        dac_invert_n_diff = h.Signal()

    # Adapt the immutable FRIDA-1 digital netlist at this one boundary. HDL21
    # flattens the first concatenation part onto physical bit 15, so listing
    # logical stages C0 through C15 reverses the legacy physical bus exactly.
    Adc.xdigital = Frida1AdcDigital()(
        seq_init=Adc.seq_init,
        seq_samp=Adc.seq_samp,
        seq_comp=Adc.seq_comp,
        seq_update=Adc.seq_update,
        en_init=Adc.en_init,
        en_samp_p=Adc.en_samp_p,
        en_samp_n=Adc.en_samp_n,
        en_comp=Adc.en_comp,
        en_update=Adc.en_update,
        dac_mode=Adc.dac_mode,
        dac_diffcaps=Adc.dac_diffcaps,
        dac_astate_p=h.Concat(*(Adc.dac_astate_p[stage] for stage in range(16))),
        dac_bstate_p=h.Concat(*(Adc.dac_bstate_p[stage] for stage in range(16))),
        dac_astate_n=h.Concat(*(Adc.dac_astate_n[stage] for stage in range(16))),
        dac_bstate_n=h.Concat(*(Adc.dac_bstate_n[stage] for stage in range(16))),
        comp_out_p=Adc.comp_out_p,
        comp_out_n=Adc.comp_out_n,
        clk_samp_p=Adc.clk_samp_p,
        clk_samp_p_b=Adc.clk_samp_p_b,
        clk_samp_n=Adc.clk_samp_n,
        clk_samp_n_b=Adc.clk_samp_n_b,
        clk_comp=Adc.clk_comp,
        dac_state_p_main=h.Concat(*(Adc.dac_state_p[stage] for stage in range(16))),
        dac_state_p_diff=Adc.dac_state_p_diff,
        dac_state_n_main=h.Concat(*(Adc.dac_state_n[stage] for stage in range(16))),
        dac_state_n_diff=Adc.dac_state_n_diff,
        dac_invert_p_main=Adc.dac_invert_p_main,
        dac_invert_p_diff=Adc.dac_invert_p_diff,
        dac_invert_n_main=Adc.dac_invert_n_main,
        dac_invert_n_diff=Adc.dac_invert_n_diff,
        comp_out=Adc.comp_out,
        vdd_d=Adc.vdd_d,
        vss_d=Adc.vss_d,
    )

    # The generated comparator consumes both clock polarities, whereas the
    # synthesized digital block exposes only ``clk_comp``. Generate its true
    # complement locally instead of borrowing an unrelated sampling clock.
    Adc.MP_clk_comp_b = h.Mos(tp=MosType.PMOS, vth=p.comp.rst_vth, w=10, l=1)(
        d=Adc.clk_comp_b,
        g=Adc.clk_comp,
        s=Adc.vdd_a,
        b=Adc.vdd_a,
    )
    Adc.MN_clk_comp_b = h.Mos(tp=MosType.NMOS, vth=p.comp.rst_vth, w=10, l=1)(
        d=Adc.clk_comp_b,
        g=Adc.clk_comp,
        s=Adc.vss_a,
        b=Adc.vss_a,
    )

    # Instantiate positive CDAC
    Adc.xcdac_p = Cdac(p.cdac)(
        top=Adc.cdac_top_p,
        dac=Adc.dac_state_p,
        vdd=Adc.vdd_dac,
        vss=Adc.vss_dac,
    )

    # Instantiate negative CDAC
    Adc.xcdac_n = Cdac(p.cdac)(
        top=Adc.cdac_top_n,
        dac=Adc.dac_state_n,
        vdd=Adc.vdd_dac,
        vss=Adc.vss_dac,
    )

    # Instantiate positive sampler
    Adc.xsamp_p = Samp(p.samp)(
        din=Adc.vin_p,
        dout=Adc.cdac_top_p,
        clk=Adc.clk_samp_p,
        clk_b=Adc.clk_samp_p_b,
        vdd=Adc.vdd_a,
        vss=Adc.vss_a,
    )

    # Instantiate negative sampler
    Adc.xsamp_n = Samp(p.samp)(
        din=Adc.vin_n,
        dout=Adc.cdac_top_n,
        clk=Adc.clk_samp_n,
        clk_b=Adc.clk_samp_n_b,
        vdd=Adc.vdd_a,
        vss=Adc.vss_a,
    )

    # Instantiate comparator
    Adc.xcomp = Comp(p.comp)(
        inp=Adc.cdac_top_p,
        inn=Adc.cdac_top_n,
        outp=Adc.comp_out_p,
        outn=Adc.comp_out_n,
        clk=Adc.clk_comp,
        clkb=Adc.clk_comp_b,
        vdd=Adc.vdd_a,
        vss=Adc.vss_a,
    )

    return Adc
