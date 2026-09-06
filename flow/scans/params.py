"""Physical ADC scan parameters and configuration builders."""

import math
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import hdl21 as h
from yaml import safe_load

from flow.adc import AdcParams
from flow.adc.sim import AdcTbParams as _AdcTbParams
from flow.cdac import CdacParams, RedunStrat


@h.paramclass
class AdcScanParams:
    """Physical conditions wrapped around one ADC testbench configuration."""

    tb = h.Param(
        dtype=_AdcTbParams,
        desc="ADC stimulus and digital configuration",
        default=_AdcTbParams(view="frida1"),
    )
    temperature_c = h.Param(dtype=h.Scalar, desc="Test temperature in degrees Celsius", default=25.0)
    board_id = h.Param(dtype=str | None, desc="Physical board identifier", default=None)
    observed_adc = h.Param(dtype=int | None, desc="ADC routed to comparator output", default=None)
    active_adc_mask = h.Param(
        dtype=tuple[int, ...] | None, desc="ADC enable mask from ADC15 through ADC0", default=None
    )
    campaign = h.Param(dtype=str, desc="Acquisition campaign", default="adc")
    sampling_mode = h.Param(dtype=str, desc="Comparator input mode", default="track")
    sweep_stage = h.Param(dtype=str, desc="Fixed, coarse, or fine sweep stage", default="fixed")
    sweep_min_v = h.Param(dtype=h.Scalar | None, desc="Adaptive sweep lower bound", default=None)
    sweep_max_v = h.Param(dtype=h.Scalar | None, desc="Adaptive sweep upper bound", default=None)
    sweep_step_v = h.Param(dtype=h.Scalar | None, desc="Adaptive sweep step", default=None)
    requested_dac_rail_percent = h.Param(
        dtype=h.Scalar | None, desc="Requested P-side DAC rail percentage", default=None
    )
    cdac_side = h.Param(dtype=str | None, desc="Selected CDAC side", default=None)
    cdac_element = h.Param(dtype=int | None, desc="Selected C0-first CDAC stage", default=None)
    cdac_direction = h.Param(dtype=str | None, desc="Selected CDAC transition direction", default=None)
    settling_time_s = h.Param(dtype=h.Scalar, desc="Post-update settling delay", default=0.0)
    vdd_io = h.Param(dtype=h.Vdc.Params, desc="Physical I/O supply", default=h.Vdc.Params(dc=1.2))


def validate_params(params: AdcScanParams) -> None:
    """Validate physical scan fields and their nested testbench configuration."""

    if not isinstance(params, AdcScanParams):
        raise TypeError("physical scans require AdcScanParams")
    tb = params.tb
    if tb.view != "frida1":
        raise ValueError("physical scans require tb.view='frida1'")
    if not math.isfinite(float(params.temperature_c)):
        raise ValueError("temperature_c must be finite")
    if not math.isfinite(float(params.settling_time_s)) or float(params.settling_time_s) < 0.0:
        raise ValueError("settling_time_s must be finite and non-negative")
    if params.vdd_io.dc is None or not math.isfinite(float(params.vdd_io.dc)):
        raise ValueError("vdd_io.dc must be finite")
    if not math.isfinite(float(tb.symbol_rate)) or float(tb.symbol_rate) <= 0.0:
        raise ValueError("tb.symbol_rate must be finite and positive")
    if tb.conversions <= 0:
        raise ValueError("tb.conversions must be positive")
    for name in ("vdd_a", "vdd_d", "vdd_dac", "vin_cm"):
        source = getattr(tb, name)
        if source.dc is None or not math.isfinite(float(source.dc)):
            raise ValueError(f"tb.{name}.dc must be finite")
    if isinstance(tb.vin_diff, h.Vdc.Params):
        source_values = (tb.vin_diff.dc,)
    elif isinstance(tb.vin_diff, h.Vsin.Params):
        source_values = (tb.vin_diff.voff, tb.vin_diff.vamp, tb.vin_diff.freq)
        if tb.vin_diff.vamp is None or float(tb.vin_diff.vamp) < 0.0:
            raise ValueError("tb.vin_diff sine amplitude must be non-negative")
        if tb.vin_diff.freq is None or float(tb.vin_diff.freq) <= 0.0:
            raise ValueError("tb.vin_diff sine frequency must be positive")
    elif isinstance(tb.vin_diff, h.Vpwl.Params):
        wave = tb.vin_diff.wave
        if (isinstance(wave, str) and not wave.strip()) or (not isinstance(wave, str) and not wave.points):
            raise ValueError("tb.vin_diff PWL wave must not be empty")
        source_values = ()
    else:
        raise TypeError(f"unsupported tb.vin_diff source {type(tb.vin_diff).__name__}")
    if any(value is None or not math.isfinite(float(value)) for value in source_values):
        raise ValueError("tb.vin_diff source values must be finite")
    for name in ("en_init", "en_samp_p", "en_samp_n", "en_comp", "en_update", "dac_mode", "dac_diffcaps"):
        if getattr(tb, name) not in (0, 1):
            raise ValueError(f"tb.{name} must be zero or one")
    for name in ("dac_astate_p", "dac_bstate_p", "dac_astate_n", "dac_bstate_n"):
        values = getattr(tb, name)
        if len(values) != 16 or any(value not in (0, 1) for value in values):
            raise ValueError(f"tb.{name} must contain exactly sixteen binary values")
    patterns = (tb.seq_init_pattern, tb.seq_samp_pattern, tb.seq_comp_pattern, tb.seq_logic_pattern)
    if any(not pattern or set(pattern) - {"0", "1"} for pattern in patterns):
        raise ValueError("tb sequencer patterns must be non-empty binary strings")
    if len({len(pattern) for pattern in patterns}) != 1 or len(patterns[0]) % 8:
        raise ValueError("tb sequencer patterns must have equal whole-word lengths")

    selection = (params.board_id, params.observed_adc, params.active_adc_mask)
    if any(value is not None for value in selection):
        if not all(value is not None for value in selection):
            raise ValueError("board_id, observed_adc, and active_adc_mask must be set together")
        assert params.board_id is not None and params.observed_adc is not None and params.active_adc_mask is not None
        if not params.board_id.strip() or not 0 <= params.observed_adc < 16:
            raise ValueError("board_id must be non-empty and observed_adc must be in 0..15")
        if len(params.active_adc_mask) != 16 or any(bit not in (0, 1) for bit in params.active_adc_mask):
            raise ValueError("active_adc_mask must contain exactly sixteen binary values")
        if params.active_adc_mask[15 - params.observed_adc] != 1:
            raise ValueError("active_adc_mask must include observed_adc")

    campaigns = {"adc", "adc_ramp", "adc_transfer", "comp_common_mode", "comp_sampling_noise", "cdac_ab"}
    if params.campaign not in campaigns:
        raise ValueError(f"campaign must be one of {sorted(campaigns)}")
    if params.sampling_mode not in {"track", "hold"}:
        raise ValueError("sampling_mode must be 'track' or 'hold'")
    if params.sweep_stage not in {"fixed", "coarse", "fine"}:
        raise ValueError("sweep_stage must be fixed, coarse, or fine")
    sweep = (params.sweep_min_v, params.sweep_max_v, params.sweep_step_v)
    if any(value is not None for value in sweep):
        if not all(value is not None for value in sweep):
            raise ValueError("sweep_min_v, sweep_max_v, and sweep_step_v must be set together")
        lower, upper, step = (float(value) for value in sweep if value is not None)
        if not all(math.isfinite(value) for value in (lower, upper, step)) or lower > upper or step <= 0.0:
            raise ValueError("adaptive sweep bounds must be finite, ordered, and use a positive step")
        if isinstance(tb.vin_diff, h.Vdc.Params) and not lower <= float(tb.vin_diff.dc) <= upper:
            raise ValueError("tb.vin_diff.dc must lie inside adaptive sweep bounds")

    if params.requested_dac_rail_percent is not None:
        rail = float(params.requested_dac_rail_percent)
        if not math.isfinite(rail) or not 0.0 <= rail <= 100.0:
            raise ValueError("requested_dac_rail_percent must be in 0..100")
    if params.campaign == "comp_sampling_noise":
        if params.requested_dac_rail_percent is None:
            raise ValueError("comp_sampling_noise requires requested_dac_rail_percent")
    elif params.requested_dac_rail_percent is not None:
        raise ValueError("requested_dac_rail_percent is only valid for comp_sampling_noise")

    selectors = (params.cdac_side, params.cdac_element, params.cdac_direction)
    if params.campaign == "cdac_ab":
        if any(value is None for value in selectors):
            raise ValueError("cdac_ab requires side, element, and direction")
        if params.cdac_side not in {"p", "n"} or params.cdac_element is None or not 0 <= params.cdac_element < 16:
            raise ValueError("invalid CDAC side or element")
        if params.cdac_direction not in {"1to0", "0to1"} or params.sampling_mode != "hold":
            raise ValueError("invalid CDAC direction or sampling mode")
    elif any(value is not None for value in selectors):
        raise ValueError("CDAC selectors are only valid for cdac_ab")


@cache
def load_board_map() -> dict[str, Any]:
    """Load the physical-board inventory and calibration map."""

    return safe_load((Path(__file__).resolve().parent / "map_board.yaml").read_text())


def convert_sample_rate_to_baud(params: _AdcTbParams, sample_rate_hz: float) -> float:
    """Convert an active ADC conversion rate into its serializer rate."""

    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive")
    patterns = (params.seq_init_pattern, params.seq_samp_pattern, params.seq_comp_pattern, params.seq_logic_pattern)
    active = [index for index in range(len(patterns[0])) if any(pattern[index] == "1" for pattern in patterns)]
    if not active:
        raise ValueError("sequencer patterns contain no active symbols")
    return sample_rate_hz * (active[-1] - active[0] + 1)


def build_adc_variants(
    *,
    board_id: str,
    adc_indices: Sequence[int],
    active_conversion_rates_hz: Sequence[float],
    logic_offsets_symbols: Sequence[float],
    conversions: int,
    vin_cm_v: float,
    vin_diff: h.Vdc.Params | h.Vsin.Params | h.Vpwl.Params,
    campaign: str = "adc",
) -> list[AdcScanParams]:
    """Build complete physical ADC configurations for one campaign."""

    adc_indices = tuple(adc_indices)
    if not adc_indices or len(set(adc_indices)) != len(adc_indices):
        raise ValueError("adc_indices must contain unique selections")
    if not active_conversion_rates_hz or not logic_offsets_symbols:
        raise ValueError("conversion rates and logic offsets must not be empty")
    board_map = load_board_map()
    board = board_map["boards"][board_id]
    variants = []
    for adc_index in adc_indices:
        flavor = board["adc_channels"][adc_index]
        dut = AdcParams(
            adc_bits=12,
            cdac=CdacParams(
                n_dac=11,
                n_extra=5,
                redun_strat=RedunStrat.SUBRDX2_OVLY,
                weights=tuple(board_map["adc_flavors"][flavor]["cdac_weights"]),
            ),
        )
        active_adc_mask = tuple(int(index == adc_index) for index in reversed(range(16)))
        template = _AdcTbParams(view="frida1", dut=dut)
        for logic_offset in logic_offsets_symbols:
            for conversion_rate in active_conversion_rates_hz:
                params = AdcScanParams(
                    tb=_AdcTbParams(
                        view="frida1",
                        dut=dut,
                        symbol_rate=convert_sample_rate_to_baud(template, float(conversion_rate)),
                        conversions=conversions,
                        vin_cm=h.Vdc.Params(dc=vin_cm_v),
                        vin_diff=vin_diff,
                        seq_logic_phase_delay_symbols=float(template.seq_comp_phase_delay_symbols)
                        + float(logic_offset),
                    ),
                    board_id=board_id,
                    observed_adc=adc_index,
                    active_adc_mask=active_adc_mask,
                    campaign=campaign,
                )
                validate_params(params)
                variants.append(params)
    return variants
