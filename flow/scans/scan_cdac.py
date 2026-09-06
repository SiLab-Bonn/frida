"""Acquire physical A-to-B CDAC transition S-curves as typed HDF5 points.

The public :func:`scan` function acquires a supplied list of complete
``AdcScanParams`` objects. Named hardware campaigns live in
``flow.scans.runner``.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from functools import cache
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import hdl21 as h
import numpy as np
from bitarray import bitarray
from pyvisa.errors import VisaIOError

from flow.adc import AdcParams
from flow.adc.sim import AdcTbParams
from flow.analysis.io import read_measurement, write_measurement
from flow.analysis.types import CdacExtDaq, CdacExtWave, MeasCdacExt, MeasInfo
from flow.cdac import CdacParams, RedunStrat, get_cdac_weights
from flow.scans.fastrx import (
    calculate_single_sample_fastrx_capture_alignment,
    convert_fastrx_words_to_comp,
)
from flow.scans.params import AdcScanParams, load_board_map, validate_params
from flow.scans.plldrp import calculate_pll_frequency, select_pll_configuration, set_pll_divider
from flow.scans.scan_adc import (
    convert_params_to_spi_fmt,
    convert_vdiff_input_to_awg_supply,
)
from flow.scans.scope import wait_for_scope_armed, wait_for_scope_capture
from flow.scans.seqgen import convert_params_to_seqgen_fmt


def build_alternating_sweep_values(
    center_v: float,
    minimum_v: float,
    maximum_v: float,
    step_v: float = 1.0e-3,
) -> tuple[float, ...]:
    """Build center, lower, upper CDAC points until both safety limits are reached."""

    if not all(math.isfinite(value) for value in (center_v, minimum_v, maximum_v, step_v)):
        raise ValueError("adaptive sweep values must be finite")
    if minimum_v > center_v or center_v > maximum_v:
        raise ValueError("adaptive sweep center must lie inside its bounds")
    if step_v <= 0.0:
        raise ValueError("adaptive sweep step must be positive")
    values = [center_v]
    lower_steps = math.floor((center_v - minimum_v) / step_v + 1.0e-9)
    upper_steps = math.floor((maximum_v - center_v) / step_v + 1.0e-9)
    for step_index in range(1, max(lower_steps, upper_steps) + 1):
        if step_index <= lower_steps:
            values.append(max(minimum_v, center_v - step_index * step_v))
        if step_index <= upper_steps:
            values.append(min(maximum_v, center_v + step_index * step_v))
    return tuple(values)


def find_probability_bracket(
    observations: Mapping[float, tuple[int, int]],
) -> tuple[float, float] | None:
    """Return the closest measured <=10% and >=90% voltage pair."""

    low_probability_limit = 0.10
    high_probability_limit = 0.90
    low_probability = [
        voltage for voltage, (ones, total) in observations.items() if ones / total <= low_probability_limit
    ]
    high_probability = [
        voltage for voltage, (ones, total) in observations.items() if ones / total >= high_probability_limit
    ]
    if not low_probability or not high_probability:
        return None
    first, second = min(
        ((low, high) for low in low_probability for high in high_probability),
        key=lambda pair: (abs(pair[1] - pair[0]), min(pair), max(pair)),
    )
    return min(first, second), max(first, second)


def build_fine_sweep_variants(
    template: AdcScanParams,
    observations: Mapping[float, tuple[int, int]],
) -> list[AdcScanParams]:
    """Compose the 100 µV/1,000-trial CDAC interval enclosed by a coarse bracket."""

    step_v = 100.0e-6
    conversions = 1_000
    bracket = find_probability_bracket(observations)
    if bracket is None:
        raise ValueError("fine CDAC sweep requires a measured 10%..90% bracket")
    minimum_v, maximum_v = bracket
    safety_minimum_v = minimum_v if template.sweep_min_v is None else float(template.sweep_min_v)
    safety_maximum_v = maximum_v if template.sweep_max_v is None else float(template.sweep_max_v)
    point_count = round((maximum_v - minimum_v) / step_v) + 1
    return [
        replace(
            template,
            sweep_stage="fine",
            sweep_min_v=safety_minimum_v,
            sweep_max_v=safety_maximum_v,
            sweep_step_v=step_v,
            tb=replace(
                template.tb,
                conversions=conversions,
                vin_diff=h.Vdc.Params(dc=minimum_v + index * step_v),
            ),
        )
        for index in range(point_count)
    ]


def build_next_coarse_sweep_variant(
    template: AdcScanParams,
    observations: Mapping[float, tuple[int, int]],
) -> AdcScanParams | None:
    """Compose the next monotonic coarse point needed to bracket one CDAC curve."""

    low_probability_limit = 0.10
    high_probability_limit = 0.90
    if template.sweep_min_v is None or template.sweep_max_v is None or template.sweep_step_v is None:
        raise ValueError("adaptive coarse sweep requires min, max, and step controls in AdcScanParams")
    sweep_values = build_alternating_sweep_values(
        float(template.tb.vin_diff.dc),
        float(template.sweep_min_v),
        float(template.sweep_max_v),
        float(template.sweep_step_v),
    )
    probabilities = {voltage: ones / total for voltage, (ones, total) in observations.items()}
    has_low = any(probability <= low_probability_limit for probability in probabilities.values())
    has_high = any(probability >= high_probability_limit for probability in probabilities.values())
    if observations and has_high and not has_low:
        minimum_measured_v = min(observations)
        sweep_values = tuple(voltage for voltage in sweep_values if voltage < minimum_measured_v)
    elif observations and has_low and not has_high:
        maximum_measured_v = max(observations)
        sweep_values = tuple(voltage for voltage in sweep_values if voltage > maximum_measured_v)

    for voltage in sweep_values:
        if voltage not in observations:
            return replace(template, tb=replace(template.tb, vin_diff=h.Vdc.Params(dc=voltage)))
    return None


def build_next_fine_sweep_variant(
    template: AdcScanParams,
    fine_observations: Mapping[float, tuple[int, int]],
    coarse_observations: Mapping[float, tuple[int, int]],
) -> AdcScanParams | None:
    """Extend a completed fine CDAC interval until its own bracket exists."""

    step_v = 100.0e-6
    conversions = 1_000
    low_probability_limit = 0.10
    high_probability_limit = 0.90
    if not fine_observations:
        raise ValueError("fine extension requires measured fine observations")
    if find_probability_bracket(fine_observations) is not None:
        return None
    coarse_by_voltage = {round(voltage, 12): result for voltage, result in coarse_observations.items()}
    for voltage, (fine_ones, fine_total) in fine_observations.items():
        coarse_result = coarse_by_voltage.get(round(voltage, 12))
        if coarse_result is None:
            continue
        coarse_ones, coarse_total = coarse_result
        fine_probability = fine_ones / fine_total
        coarse_probability = coarse_ones / coarse_total
        if (fine_probability <= low_probability_limit and coarse_probability >= high_probability_limit) or (
            fine_probability >= high_probability_limit and coarse_probability <= low_probability_limit
        ):
            return None
    if template.sweep_min_v is None or template.sweep_max_v is None:
        raise ValueError("fine extension requires the original adaptive safety bounds")
    if len(coarse_observations) < 2:
        raise ValueError("fine extension requires at least two coarse observations")

    coarse_voltages = np.asarray(sorted(coarse_observations), dtype=np.float64)
    coarse_probabilities = np.asarray(
        [coarse_observations[voltage][0] / coarse_observations[voltage][1] for voltage in coarse_voltages],
        dtype=np.float64,
    )
    trend = float(
        np.dot(
            coarse_voltages - np.mean(coarse_voltages),
            coarse_probabilities - np.mean(coarse_probabilities),
        )
    )
    increasing = trend >= 0.0
    has_low = any(ones / total <= low_probability_limit for ones, total in fine_observations.values())
    has_high = any(ones / total >= high_probability_limit for ones, total in fine_observations.values())
    measured_minimum_v = min(fine_observations)
    measured_maximum_v = max(fine_observations)
    candidates = []
    if not has_high:
        candidates.append(measured_maximum_v + step_v if increasing else measured_minimum_v - step_v)
    if not has_low:
        candidates.append(measured_minimum_v - step_v if increasing else measured_maximum_v + step_v)
    safety_minimum_v = float(template.sweep_min_v)
    safety_maximum_v = float(template.sweep_max_v)
    for candidate_v in candidates:
        candidate_v = round(candidate_v / step_v) * step_v
        if safety_minimum_v - 1.0e-12 <= candidate_v <= safety_maximum_v + 1.0e-12:
            return replace(
                template,
                sweep_stage="fine",
                sweep_step_v=step_v,
                tb=replace(template.tb, conversions=conversions, vin_diff=h.Vdc.Params(dc=candidate_v)),
            )
    return None


@cache
def _convert_dac_rail_percent_to_codes_cached(
    rail_percent: float,
    weights: tuple[int, ...],
) -> tuple[str, ...]:
    """Return every closest weighted C0-to-C15 state deterministically."""

    if not math.isfinite(rail_percent) or not 0.0 <= rail_percent <= 100.0:
        raise ValueError("rail_percent must be finite and in 0..100")
    if len(weights) != 16 or any(
        isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0 for weight in weights
    ):
        raise ValueError("weights must contain exactly 16 positive integers")
    target = rail_percent * sum(weights) / 100.0
    best_error = math.inf
    best_codes: list[str] = []
    for value in range(1 << 16):
        code = f"{value:016b}"
        realized = sum(weight for bit, weight in zip(code, weights, strict=True) if bit == "1")
        error = abs(realized - target)
        if error < best_error:
            best_error = error
            best_codes = [code]
        elif error == best_error:
            best_codes.append(code)
    return tuple(best_codes)


def _convert_dac_rail_percent_to_codes(
    rail_percent: float,
    weights: Sequence[int],
) -> tuple[str, ...]:
    """Return cached closest weighted C0-to-C15 states."""

    return _convert_dac_rail_percent_to_codes_cached(rail_percent, tuple(weights))


def _predict_cdac_step_v(params: AdcScanParams) -> float:
    """Predict the selected element's unsigned physical top-plate step for planning.

    The fabricated active-high XOR makes ``dac_diffcaps=0`` switch the main
    and difference banks together (main+diff), while ``dac_diffcaps=1`` is
    normal complementary operation (main-diff).
    """

    if params.campaign != "cdac_ab" or params.cdac_element is None:
        raise ValueError("CDAC step prediction requires a selected cdac_ab element")
    weights = get_cdac_weights(params.tb.dut.cdac)
    total_weights = [65 * math.ceil(weight / 64) for weight in weights]
    switched_weight = weights[params.cdac_element] if params.tb.dac_diffcaps else total_weights[params.cdac_element]
    topplate_parasitic_weight = 0.0
    if params.board_id is not None and params.observed_adc is not None:
        board_map = load_board_map()
        board = board_map["boards"][params.board_id]
        flavor = board["adc_channels"][params.observed_adc]
        topplate_parasitic_weight = float(board_map["adc_flavors"][flavor].get("cdac_topplate_parasitic_weight", 0.0))
    return float(params.tb.vdd_dac.dc) * switched_weight / (sum(total_weights) + topplate_parasitic_weight)


def _calculate_cdac_plate_voltages(params: AdcScanParams) -> tuple[float, float, float, float]:
    """Return P/N top-plate voltages before and after the selected switch."""

    if not isinstance(params.tb.vin_diff, h.Vdc.Params):
        raise TypeError("CDAC plate planning requires a DC differential input")
    if params.cdac_side is None or params.cdac_direction is None:
        raise ValueError("CDAC plate planning requires side and direction")
    vin_cm_v = float(params.tb.vin_cm.dc)
    vin_diff_v = float(params.tb.vin_diff.dc)
    before_p_v = vin_cm_v + vin_diff_v / 2.0
    before_n_v = vin_cm_v - vin_diff_v / 2.0
    signed_step_v = _predict_cdac_step_v(params) * (1.0 if params.cdac_direction == "0to1" else -1.0)
    after_p_v = before_p_v + (signed_step_v if params.cdac_side == "p" else 0.0)
    after_n_v = before_n_v + (signed_step_v if params.cdac_side == "n" else 0.0)
    return before_p_v, before_n_v, after_p_v, after_n_v


def _calculate_cdac_input_bounds(params: AdcScanParams) -> tuple[float, float]:
    """Calculate the complete safe external-Vdiff interval for one transition."""

    if params.cdac_side is None or params.cdac_direction is None:
        raise ValueError("CDAC input bounds require side and direction")
    if params.board_id is None:
        raise ValueError("physical CDAC input bounds require board_id")
    vin_cm_v = float(params.tb.vin_cm.dc)
    lower_plate_v = 0.4
    upper_plate_v = float(params.tb.vdd_a.dc)
    step_sign = 1.0 if params.cdac_direction == "0to1" else -1.0
    signed_step_v = step_sign * _predict_cdac_step_v(params)
    minimum_v = max(
        2.0 * (lower_plate_v - vin_cm_v),
        2.0 * (vin_cm_v - upper_plate_v),
    )
    maximum_v = min(
        2.0 * (upper_plate_v - vin_cm_v),
        2.0 * (vin_cm_v - lower_plate_v),
    )
    if params.cdac_side == "p":
        minimum_v = max(minimum_v, 2.0 * (lower_plate_v - vin_cm_v - signed_step_v))
        maximum_v = min(maximum_v, 2.0 * (upper_plate_v - vin_cm_v - signed_step_v))
    else:
        minimum_v = max(minimum_v, 2.0 * (vin_cm_v + signed_step_v - upper_plate_v))
        maximum_v = min(maximum_v, 2.0 * (vin_cm_v + signed_step_v - lower_plate_v))
    calibration = load_board_map()["boards"][params.board_id]["input_calibration"]
    maximum_calibrated_v = float(calibration["maximum_abs_vdiff_v"])
    minimum_v = max(minimum_v, -maximum_calibrated_v)
    maximum_v = min(maximum_v, maximum_calibrated_v)
    if minimum_v > maximum_v:
        raise ValueError("CDAC transition has no safe calibrated external-input interval")
    return minimum_v, maximum_v


def _cdac_curve_key(params: AdcScanParams) -> tuple[Any, ...]:
    """Identify one A-to-B transition curve while excluding its input point."""

    return (
        params.board_id,
        params.observed_adc,
        params.cdac_side,
        params.cdac_element,
        params.cdac_direction,
        params.tb.dac_mode,
        params.tb.dac_diffcaps,
        float(params.settling_time_s),
        float(params.tb.vin_cm.dc),
        params.tb.dac_astate_p,
        params.tb.dac_bstate_p,
        params.tb.dac_astate_n,
        params.tb.dac_bstate_n,
    )


def _cdac_point_stem(params: AdcScanParams) -> str:
    """Return the stable filename portion used for resuming one CDAC point."""

    assert params.board_id is not None
    assert params.observed_adc is not None
    assert params.cdac_side is not None
    assert params.cdac_element is not None
    assert params.cdac_direction is not None
    assert isinstance(params.tb.vin_diff, h.Vdc.Params)
    return (
        (
            f"{params.board_id}_adc{params.observed_adc:02d}_cdac_ab_"
            f"{params.cdac_side}_c{params.cdac_element}_{params.cdac_direction}_"
            f"mode{params.tb.dac_mode}_diff{params.tb.dac_diffcaps}_"
            f"settle{float(params.settling_time_s) * 1e9:06.2f}ns_{params.sweep_stage}_"
            f"vcm{float(params.tb.vin_cm.dc) * 1e3:07.2f}mv_"
            f"vdiff{float(params.tb.vin_diff.dc) * 1e3:+09.3f}mv"
        )
        .replace("+", "p")
        .replace("-", "m")
    )


def _validate_cdac_resume_curves(
    existing_curves: Mapping[tuple[Any, ...], Sequence[MeasCdacExt]],
) -> None:
    """Allow reuse only after one uninterrupted acquisition marked the curve complete."""

    for measurements in existing_curves.values():
        session_ids = {measurement.info.readbacks.get("acquisition_session_id") for measurement in measurements}
        completed = [
            measurement for measurement in measurements if measurement.info.readbacks.get("curve_complete") is True
        ]
        latest_timestamp = max(measurement.info.timestamp_utc for measurement in measurements)
        if (
            None in session_ids
            or len(session_ids) != 1
            or len(completed) != 1
            or completed[0].info.timestamp_utc != latest_timestamp
        ):
            params = measurements[0].param
            assert params.observed_adc is not None
            assert params.cdac_side is not None
            assert params.cdac_element is not None
            assert params.cdac_direction is not None
            raise ValueError(
                "CDAC run directory contains an interrupted or mixed-session curve for "
                f"ADC{params.observed_adc:02d} {params.cdac_side} C{params.cdac_element} "
                f"{params.cdac_direction} diff{params.tb.dac_diffcaps}; start a new run directory"
            )


def _build_cdac_params(
    *,
    adc_index: int,
    side: str,
    element: int,
    direction: str,
    dac_diffcaps: int,
    vin_diff_v: float,
    conversions: int,
    sweep_stage: str,
    settling_time_s: float = 20.0e-9,
    sweep_min_v: float | None = None,
    sweep_max_v: float | None = None,
    sweep_step_v: float | None = None,
) -> AdcScanParams:
    """Compose one complete physical A-to-B CDAC point."""

    board_id = "00"
    board_map = load_board_map()
    flavor = board_map["boards"][board_id]["adc_channels"][adc_index]
    cap_weights = tuple(board_map["adc_flavors"][flavor]["cdac_weights"])
    dut = AdcParams(
        adc_bits=12,
        cdac=CdacParams(
            n_dac=11,
            n_extra=5,
            redun_strat=RedunStrat.SUBRDX2_OVLY,
            weights=cap_weights,
        ),
    )
    if direction == "1to0":
        a_p = [1] * 16
        a_n = [1] * 16
        b_p = a_p.copy()
        b_n = a_n.copy()
        (b_p if side == "p" else b_n)[element] = 0
    elif direction == "0to1":
        a_p = [0] * 16
        a_n = [0] * 16
        b_p = a_p.copy()
        b_n = a_n.copy()
        (b_p if side == "p" else b_n)[element] = 1
    else:
        raise ValueError("direction must be '1to0' or '0to1'")

    symbol_rate_bps = 1.6e9
    params = AdcScanParams(
        tb=AdcTbParams(
            view="frida1",
            dut=dut,
            symbol_rate=symbol_rate_bps,
            conversions=conversions,
            dac_mode=0,
            dac_diffcaps=dac_diffcaps,
            dac_astate_p=tuple(a_p),
            dac_bstate_p=tuple(b_p),
            dac_astate_n=tuple(a_n),
            dac_bstate_n=tuple(b_n),
            vin_cm=h.Vdc.Params(dc=0.8),
            vin_diff=h.Vdc.Params(dc=vin_diff_v),
        ),
        board_id=board_id,
        observed_adc=adc_index,
        active_adc_mask=tuple(int(index == adc_index) for index in reversed(range(16))),
        campaign="cdac_ab",
        sampling_mode="hold",
        sweep_stage=sweep_stage,
        sweep_min_v=sweep_min_v,
        sweep_max_v=sweep_max_v,
        sweep_step_v=sweep_step_v,
        cdac_side=side,
        cdac_element=element,
        cdac_direction=direction,
        settling_time_s=settling_time_s,
    )
    sequence_words = len(params.tb.seq_init_pattern) // 8
    word_period_s = 8.0 / symbol_rate_bps
    update_word = 21
    comp_word = update_word + 1 + math.ceil(settling_time_s / word_period_s)
    if comp_word >= sequence_words - 2:
        raise ValueError("settling_time_s leaves no room for comparator capture")
    comp_words = ["00000000"] * sequence_words
    init_pattern = "00000000" + "11111111" + "00000000" * 30
    # Keep the physical top plates connected long enough to erase charge from
    # the preceding voltage point before opening both sampling switches.
    samp_pattern = "00000000" * 3 + "11111111" * 17 + "00000000" * 12
    logic_pattern = "00000000" + "00001111" + "00000000" * 19 + "00001111" + "00000000" * 10
    comp_words[comp_word] = "00001111"
    params = replace(
        params,
        tb=replace(
            params.tb,
            seq_init_pattern=init_pattern,
            seq_samp_pattern=samp_pattern,
            seq_comp_pattern="".join(comp_words),
            seq_logic_pattern=logic_pattern,
        ),
    )
    validate_params(params)
    plate_voltages = _calculate_cdac_plate_voltages(params)
    if any(voltage < 0.4 or voltage > float(params.tb.vdd_a.dc) for voltage in plate_voltages):
        raise ValueError(f"CDAC point drives a predicted top plate outside 0.4..VDD_A: {plate_voltages}")
    return params


def _expected_transition_v(
    adc_index: int,
    side: str,
    element: int,
    direction: str,
    dac_diffcaps: int,
    comparator_offset_v: float,
) -> float:
    """Calculate the external Vdiff which cancels the planned CDAC step.

    A manually accepted C0 measurement may scale the PEX step solely to
    center later adaptive sweeps.  The measured P50 remains the only input to
    capacitance extraction.
    """

    probe = _build_cdac_params(
        adc_index=adc_index,
        side=side,
        element=element,
        direction=direction,
        dac_diffcaps=dac_diffcaps,
        vin_diff_v=0.0,
        conversions=1,
        sweep_stage="fixed",
    )
    board = load_board_map()["boards"]["00"]
    center_calibrations = board.get("cdac_sweep_center_calibration", {})
    center_calibration = center_calibrations.get(adc_index, center_calibrations.get(str(adc_index)))
    center_scale = 1.0
    if center_calibration is not None:
        scales = center_calibration["scale_by_diffcaps"]
        if len(scales) != 2:
            raise ValueError("CDAC sweep-center calibration requires two diffcaps scales")
        center_scale = float(scales[dac_diffcaps])
        if not math.isfinite(center_scale) or center_scale <= 0.0:
            raise ValueError("CDAC sweep-center calibration scales must be finite and positive")
    step_sign = 1.0 if direction == "0to1" else -1.0
    side_sign = 1.0 if side == "p" else -1.0
    return comparator_offset_v - side_sign * step_sign * center_scale * _predict_cdac_step_v(probe)


def build_capacitor_variants(
    *,
    adc_indices: Sequence[int],
    coarse_step_v: float,
    coarse_trials: int,
    selected_curves: Collection[tuple[int, str, int, str, int]] | None = None,
) -> list[AdcScanParams]:
    """Build adaptive seeds for all or selected ADC00–ADC03 A-to-B curves."""

    selected_curves = None if selected_curves is None else set(selected_curves)
    board = load_board_map()["boards"]["00"]
    calibrations = board.get("comparator_calibration", {})
    variants = []
    for adc_index in adc_indices:
        calibration = calibrations.get(adc_index, calibrations.get(str(adc_index)))
        if calibration is None:
            raise ValueError(f"ADC{adc_index:02d} has no accepted comparator_calibration")
        comparator_offset_v = float(calibration["offset_v"])
        for side in ("p", "n"):
            for element in range(16):
                for direction in ("1to0", "0to1"):
                    for dac_diffcaps in (0, 1):
                        curve = (adc_index, side, element, direction, dac_diffcaps)
                        if selected_curves is not None and curve not in selected_curves:
                            continue
                        center_v = _expected_transition_v(
                            adc_index,
                            side,
                            element,
                            direction,
                            dac_diffcaps,
                            comparator_offset_v,
                        )
                        seed = _build_cdac_params(
                            adc_index=adc_index,
                            side=side,
                            element=element,
                            direction=direction,
                            dac_diffcaps=dac_diffcaps,
                            vin_diff_v=center_v,
                            conversions=coarse_trials,
                            sweep_stage="coarse",
                        )
                        minimum_v, maximum_v = _calculate_cdac_input_bounds(seed)
                        seed = replace(
                            seed,
                            sweep_min_v=minimum_v,
                            sweep_max_v=maximum_v,
                            sweep_step_v=coarse_step_v,
                        )
                        validate_params(seed)
                        variants.append(seed)
    if selected_curves is not None:
        observed_curves = {
            (
                params.observed_adc,
                params.cdac_side,
                params.cdac_element,
                params.cdac_direction,
                params.tb.dac_diffcaps,
            )
            for params in variants
        }
        if observed_curves != selected_curves:
            raise ValueError("selected CDAC curves are not in the capacitor campaign")
    return variants


def scan(
    variants: Sequence[AdcScanParams],
    *,
    run_dir: Path,
    capture_scope_per_curve: bool = True,
) -> Path:
    """Acquire or resume complete A-to-B variants in one run directory."""

    topplate_precondition_s = 0.01
    smu_settle_s = 0.5
    vin_cm_settle_s = 0.5
    si570_settle_s = 0.02
    input_settle_s = 0.1
    fine_batch_trials = 100
    fine_batch_interval_s = 0.5
    drift_checkpoint_low_probability = 0.15
    drift_checkpoint_high_probability = 0.85
    capture_timeout_s = 5.0
    scope_timeout_s = 5.0
    scope_tracks = {"vin_diff_v": 1, "seq_comp_v": 2, "seq_logic_v": 3, "comp_out_v": 4}

    queue = list(variants)
    if not isinstance(capture_scope_per_curve, bool):
        raise TypeError("capture_scope_per_curve must be boolean")
    if not queue:
        raise ValueError("CDAC scan requires at least one parameter variant")
    for params in queue:
        validate_params(params)
        if params.campaign != "cdac_ab":
            raise ValueError(f"scan_cdac cannot run campaign {params.campaign!r}")
        if params.board_id is None or params.observed_adc is None or params.active_adc_mask is None:
            raise ValueError("every physical CDAC point must select its board and ADC")
        if not isinstance(params.tb.vin_diff, h.Vdc.Params):
            raise TypeError("physical CDAC S-curves require fixed DC inputs")
        _calculate_cdac_plate_voltages(params)

    first = queue[0]
    static_signature = (
        first.board_id,
        float(first.tb.symbol_rate),
        float(first.tb.vdd_a.dc),
        float(first.tb.vdd_d.dc),
        float(first.tb.vdd_dac.dc),
        float(first.vdd_io.dc),
    )
    if any(
        (
            params.board_id,
            float(params.tb.symbol_rate),
            float(params.tb.vdd_a.dc),
            float(params.tb.vdd_d.dc),
            float(params.tb.vdd_dac.dc),
            float(params.vdd_io.dc),
        )
        != static_signature
        for params in queue[1:]
    ):
        raise ValueError("one CDAC run requires one board, clock rate, and supply configuration")
    assert first.board_id is not None
    board_map = load_board_map()
    board = board_map["boards"][first.board_id]
    supply_limits = board["supply_limits"]
    minimum_supply_v = float(supply_limits["minimum_voltage_v"])
    maximum_supply_v = float(supply_limits["maximum_voltage_v"])
    signal_headroom_v = float(supply_limits["signal_headroom_v"])
    fixed_vdd_io_v = float(board["fixed_vdd_io_v"])
    if not math.isclose(float(first.vdd_io.dc), fixed_vdd_io_v, abs_tol=1.0e-12):
        raise ValueError(
            f"VDD_IO is fixed at {fixed_vdd_io_v:g} V on {first.board_id}; "
            f"CDAC run requests {float(first.vdd_io.dc):g} V"
        )
    for rail, field in (("VDD_A", "vdd_a"), ("VDD_D", "vdd_d"), ("VDD_DAC", "vdd_dac")):
        requested_v = float(getattr(first.tb, field).dc)
        if not minimum_supply_v <= requested_v <= maximum_supply_v:
            raise ValueError(
                f"{rail} request {requested_v:g} V is outside {minimum_supply_v:g}..{maximum_supply_v:g} V"
            )
    calibration: Mapping[str, Any] = board["input_calibration"]
    for params in queue:
        assert isinstance(params.tb.vin_diff, h.Vdc.Params)
        vin_cm_v = float(params.tb.vin_cm.dc)
        minimum_input_v = -signal_headroom_v
        maximum_input_v = float(params.tb.vdd_a.dc) + signal_headroom_v
        if not float(calibration["minimum_vin_cm_v"]) <= vin_cm_v <= float(calibration["maximum_vin_cm_v"]):
            raise ValueError(f"Vin_cm={vin_cm_v:g} V is outside the calibrated input range")
        planned_vdiff_values = [float(params.tb.vin_diff.dc)]
        if params.sweep_min_v is not None and params.sweep_max_v is not None:
            planned_vdiff_values.extend((float(params.sweep_min_v), float(params.sweep_max_v)))
        for vin_diff_v in planned_vdiff_values:
            if abs(vin_diff_v) > float(calibration["maximum_abs_vdiff_v"]):
                raise ValueError("requested Vdiff exceeds the calibrated range")
            _awg_voltage_v, vin_cm_supply_v = convert_vdiff_input_to_awg_supply(
                vin_diff_v,
                vin_cm_v,
                calibration,
            )
            if not 0.0 <= vin_cm_supply_v <= maximum_supply_v:
                raise ValueError("calibrated Vin_cm supply request is unsafe")
            vin_p_v = vin_cm_v + vin_diff_v / 2.0
            vin_n_v = vin_cm_v - vin_diff_v / 2.0
            if not (
                minimum_input_v - 1.0e-12 <= vin_p_v <= maximum_input_v + 1.0e-12
                and minimum_input_v - 1.0e-12 <= vin_n_v <= maximum_input_v + 1.0e-12
            ):
                raise ValueError(
                    f"CDAC inputs {(vin_p_v, vin_n_v)} V are outside {minimum_input_v:g}..{maximum_input_v:g} V"
                )
            planned = replace(params, tb=replace(params.tb, vin_diff=h.Vdc.Params(dc=vin_diff_v)))
            plate_voltages = _calculate_cdac_plate_voltages(planned)
            if any(
                voltage < 0.4 - 1.0e-12 or voltage > float(params.tb.vdd_a.dc) + 1.0e-12 for voltage in plate_voltages
            ):
                raise ValueError(f"predicted CDAC plate voltage is unsafe: {plate_voltages}")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    existing_paths: dict[str, Path] = {}
    coarse_observations: dict[tuple[Any, ...], dict[float, tuple[int, int]]] = {}
    fine_observations: dict[tuple[Any, ...], dict[float, tuple[int, int]]] = {}
    existing_curves: dict[tuple[Any, ...], list[MeasCdacExt]] = {}
    scope_captured_curves: set[tuple[Any, ...]] = set()
    drift_checkpoint_curves: set[tuple[Any, ...]] = set()
    for path in sorted(run_dir.glob("*.h5")):
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasCdacExt):
            raise TypeError(f"CDAC run directory contains {type(measurement).__name__}: {path}")
        stem = _cdac_point_stem(measurement.param)
        if stem in existing_paths:
            raise ValueError(f"CDAC run directory contains duplicate point {stem}")
        existing_paths[stem] = path
        if measurement.param.sweep_stage in {"coarse", "fine"}:
            key = _cdac_curve_key(measurement.param)
            if measurement.wave is not None:
                scope_captured_curves.add(key)
            if float(measurement.info.readbacks.get("capture_batch_interval_s", 0.0)) > 0.0:
                drift_checkpoint_curves.add(key)
            existing_curves.setdefault(key, []).append(measurement)
            voltage = float(measurement.param.tb.vin_diff.dc)
            observations_by_stage = (
                coarse_observations if measurement.param.sweep_stage == "coarse" else fine_observations
            )
            observations_by_stage.setdefault(key, {})[voltage] = (
                int(np.sum(measurement.daq.decision)),
                len(measurement.daq.decision),
            )

    _validate_cdac_resume_curves(existing_curves)

    coarse_templates = {_cdac_curve_key(params): params for params in queue if params.sweep_stage == "coarse"}
    fine_scheduled = {_cdac_curve_key(params) for params in queue if params.sweep_stage == "fine"}
    requested_stems = [_cdac_point_stem(params) for params in queue]
    if len(requested_stems) != len(set(requested_stems)):
        raise ValueError("CDAC campaign contains duplicate parameter points")

    acquisition_session_id = datetime.now().astimezone().isoformat(timespec="microseconds")

    from gpib_ctypes import make_default_gpib

    make_default_gpib()
    from basil.dut import Dut

    map_dir = Path(__file__).resolve().parent
    daq_dut = Dut(str(map_dir / "map_fpga.yaml"))
    awg_dut = Dut(str(map_dir / "map_awg.yaml"))
    vin_cm_dut = Dut(str(map_dir / "map_supply.yaml"))
    smu_dut = Dut(str(map_dir / "map_smu.yaml"))
    scope_dut = Dut(str(map_dir / "map_scope.yaml")) if capture_scope_per_curve else None
    initialized_duts = []
    daq = awg = vin_cm_supply = scope = None
    smus: list[tuple[Any, str, str]] = []
    instrument_identities: dict[str, str] = {}
    active_vin_cm_supply_v: float | None = None
    active_vin_cm_measured_v: float | None = None
    active_vin_cm_measured_a: float | None = None
    awg_enabled = False
    awg_configured = False
    awg_verified_curves: set[tuple[Any, ...]] = set()
    programmed_spi_bytes: bytes | None = None
    spi_bytes_by_curve: dict[tuple[Any, ...], bytes] = {}

    try:
        duts = (daq_dut, awg_dut, vin_cm_dut, smu_dut, *((scope_dut,) if scope_dut is not None else ()))
        for dut in duts:
            dut.init()
            initialized_duts.append(dut)
        daq = daq_dut
        awg = awg_dut["awg"]
        vin_cm_supply = vin_cm_dut["vocm_supply"]
        scope = None if scope_dut is None else scope_dut["scope"]
        smus = [
            (smu_dut["smu1"], "VDD_A", "vdd_a"),
            (smu_dut["smu2"], "VDD_D", "vdd_d"),
            (smu_dut["smu3"], "VDD_DAC", "vdd_dac"),
        ]
        instrument_identities = {
            "awg": str(awg.get_name()).strip(),
            "vin_cm_supply": str(vin_cm_supply.get_name()).strip(),
            **{field: str(smu.get_name()).strip() for smu, _rail, field in smus},
        }
        if scope is not None:
            instrument_identities["scope"] = str(scope.get_name()).strip()

        awg.set_enable(0)
        vin_cm_supply.set_enable(0)
        vin_cm_supply.set_voltage(0.0)
        vin_cm_supply.set_voltage_range("P25V")
        vin_cm_supply.set_current_limit(float(supply_limits["vin_cm_current_limit_a"]))
        loaded_tolerance_v = float(supply_limits["loaded_voltage_tolerance_v"])
        supply_readbacks: dict[str, dict[str, float]] = {}
        for smu, rail, field in smus:
            requested_v = float(getattr(first, field).dc)
            smu.off()
            smu.set_voltage(0.0)
            smu.source_volt()
            smu.four_wire_off()
            smu.set_voltage_range(float(supply_limits["smu_voltage_range_v"]))
            smu.set_current_limit(float(supply_limits["smu_current_compliance_a"]))
            smu.current_sense_autorange_on()
            smu.set_current_nplc(10.0)
            smu.autozero_on()
            smu.set_voltage(requested_v)
        for smu, _rail, _field in smus:
            smu.on()
        sleep(smu_settle_s)
        for smu, rail, field in smus:
            for attempt in range(3):
                try:
                    measured_v = float(smu.get_voltage())
                    measured_a = float(smu.get_current())
                    break
                except (UnicodeDecodeError, ValueError, VisaIOError) as error:
                    if attempt == 2:
                        raise RuntimeError(f"{rail} readback failed after three attempts") from error
                    smu._intf._resource.clear()
                    sleep(0.1)
            requested_v = float(getattr(first, field).dc)
            if measured_v > maximum_supply_v + 5e-3 or measured_v < requested_v - loaded_tolerance_v:
                raise RuntimeError(f"{rail} loaded readback {measured_v:g} V is unsafe")
            supply_readbacks[field] = {
                "requested_voltage_v": requested_v,
                "measured_voltage_v": measured_v,
                "measured_current_a": measured_a,
            }

        if scope is not None:
            scope.set_acquire_state("STOP")
            scope.set_acquire_mode("SAMPLE")
            scope.set_acquire_stop_after("SEQUENCE")
            scope.set_horizontal_record_length(10_000)
            scope._intf.write("HORizontal:POSition 20")
            for signal_name, channel in scope_tracks.items():
                scope._intf.write(f"DISplay:GLObal:CH{channel}:STATE ON")
                scope.set_coupling("DC", channel=channel)
                scope.set_vertical_scale(0.1 if signal_name == "vin_diff_v" else 0.2, channel=channel)
                scope.set_vertical_position(0.0, channel=channel)
                scope.set_vertical_offset(0.0, channel=channel)
                scope.set_bandwidth(200.0e6 if signal_name == "vin_diff_v" else 2.0e9, channel=channel)
            scope.set_trigger_type("EDGE")
            scope.set_trigger_source(channel=2)
            scope.set_trigger_edge_slope("RISE")
            # The differential sequencer probe is centered around zero and swings
            # to roughly +/-0.6 V; trigger at its zero crossing.
            scope.set_trigger_level(0.0, channel=2)
            scope.set_trigger_mode("NORMAL")

        daq["gpio0"]["RST_B"] = 0
        daq["gpio0"]["AMP_EN"] = 1
        daq["gpio0"]["RX_LOOPBACK"] = 0
        daq["gpio0"]["SPI_LOOPBACK"] = 0
        daq["gpio0"]["DBG_FIFO"] = 0
        daq["gpio0"]["RX_TIEHIGH"] = 0
        daq["gpio0"]["SEQ_START"] = 0
        daq["gpio0"]["RX_EN_MUX"] = 1
        daq["gpio0"].write()
        daq["gpio0"]["RST_B"] = 1
        daq["gpio0"].write()

        symbol_rate_bps = float(first.tb.symbol_rate)
        si570_frequency_hz, pll_divider_n = select_pll_configuration(symbol_rate_bps)
        sequencer_frequency_hz, serializer_frequency_hz = calculate_pll_frequency(
            pll_divider_n,
            input_frequency_hz=si570_frequency_hz,
        )
        daq["si570"].frequency_change(si570_frequency_hz / 1e6)
        sleep(si570_settle_s)
        set_pll_divider(daq["gpio2"], pll_divider_n)
        data_size = int(daq["fastrx0"].get_size())
        expected_data_size = len(get_cdac_weights(first.tb.dut.cdac)) + 1
        if data_size != expected_data_size:
            raise RuntimeError(f"FastRX DATA_SIZE={data_size}, expected {expected_data_size} from the configured CDAC")

        next_file_index = len(existing_paths)
        for variant_index, original_params in enumerate(queue):
            point_started = monotonic()
            curve_key = _cdac_curve_key(original_params)
            observations = coarse_observations.get(curve_key, {})
            if original_params.sweep_stage == "coarse" and find_probability_bracket(observations) is not None:
                if curve_key not in fine_scheduled:
                    queue[variant_index + 1 : variant_index + 1] = build_fine_sweep_variants(
                        coarse_templates[curve_key],
                        observations,
                    )
                    fine_scheduled.add(curve_key)
                continue
            point_stem = _cdac_point_stem(original_params)
            if point_stem in existing_paths:
                if original_params.sweep_stage == "coarse":
                    next_variant = build_next_coarse_sweep_variant(
                        coarse_templates[curve_key],
                        observations,
                    )
                    if next_variant is not None:
                        queue.insert(variant_index + 1, next_variant)
                elif original_params.sweep_stage == "fine":
                    next_is_same_fine = (
                        variant_index + 1 < len(queue)
                        and queue[variant_index + 1].sweep_stage == "fine"
                        and _cdac_curve_key(queue[variant_index + 1]) == curve_key
                    )
                    if not next_is_same_fine and curve_key in coarse_templates:
                        next_variant = build_next_fine_sweep_variant(
                            coarse_templates[curve_key],
                            fine_observations.get(curve_key, {}),
                            observations,
                        )
                        if next_variant is not None:
                            queue.insert(variant_index + 1, next_variant)
                continue
            scan_params = original_params
            params = scan_params.tb
            assert scan_params.observed_adc is not None
            assert isinstance(params.vin_diff, h.Vdc.Params)
            vin_diff_v = float(params.vin_diff.dc)
            vin_cm_v = float(params.vin_cm.dc)
            plate_voltages = _calculate_cdac_plate_voltages(scan_params)
            if any(voltage < 0.4 or voltage > float(params.vdd_a.dc) for voltage in plate_voltages):
                raise ValueError(f"predicted CDAC plate voltage is unsafe: {plate_voltages}")
            if abs(vin_diff_v) > float(calibration["maximum_abs_vdiff_v"]):
                raise ValueError("requested Vdiff exceeds the calibrated range")
            if not float(calibration["minimum_vin_cm_v"]) <= vin_cm_v <= float(calibration["maximum_vin_cm_v"]):
                raise ValueError(f"Vin_cm={vin_cm_v:g} V is outside the calibrated input range")

            awg_voltage_v, vin_cm_supply_v = convert_vdiff_input_to_awg_supply(vin_diff_v, vin_cm_v, calibration)
            if not 0.0 <= vin_cm_supply_v <= maximum_supply_v:
                raise ValueError("calibrated Vin_cm supply request is unsafe")
            vin_cm_changed = active_vin_cm_supply_v is None or not math.isclose(
                vin_cm_supply_v,
                active_vin_cm_supply_v,
                abs_tol=1.0e-12,
            )
            if vin_cm_changed:
                if awg_enabled:
                    awg.set_enable(0)
                    awg_enabled = False
                vin_cm_supply.set_enable(0)
                vin_cm_supply.set_voltage(vin_cm_supply_v)
                vin_cm_supply.set_enable(1)
                active_vin_cm_supply_v = vin_cm_supply_v
                sleep(vin_cm_settle_s)
                active_vin_cm_measured_v = float(vin_cm_supply.get_voltage())
                active_vin_cm_measured_a = float(vin_cm_supply.get_current())
            assert active_vin_cm_measured_v is not None
            assert active_vin_cm_measured_a is not None
            if not awg_configured:
                awg.set_DC(f"DEF,DEF,{awg_voltage_v}")
                awg.set_output_load("INFinity")
                awg_configured = True
            else:
                awg.set_voltage_offset(awg_voltage_v)
            if not awg_enabled:
                awg.set_enable(1)
                awg_enabled = True
            sleep(input_settle_s)
            awg_readback_checked = curve_key not in awg_verified_curves
            awg_readback_v: float | None = None
            if awg_readback_checked:
                awg_readback_v = float(str(awg.get_voltage_offset()).strip().split(",")[0])
                if not math.isclose(awg_readback_v, awg_voltage_v, abs_tol=0.5e-3):
                    raise RuntimeError(f"AWG offset readback {awg_readback_v:g} V does not match {awg_voltage_v:g} V")
                awg_verified_curves.add(curve_key)

            capture_alignment = calculate_single_sample_fastrx_capture_alignment(
                params,
                **board["capture_timing_model"],
            )
            phase_advance = capture_alignment.control_phase_advance_symbols
            if phase_advance:
                params = replace(
                    params,
                    seq_init_phase_delay_symbols=float(params.seq_init_phase_delay_symbols) - phase_advance,
                    seq_samp_phase_delay_symbols=float(params.seq_samp_phase_delay_symbols) - phase_advance,
                    seq_comp_phase_delay_symbols=float(params.seq_comp_phase_delay_symbols) - phase_advance,
                    seq_logic_phase_delay_symbols=float(params.seq_logic_phase_delay_symbols) - phase_advance,
                )
                scan_params = replace(scan_params, tb=params)
                validate_params(scan_params)
            rx_sen_start_word = capture_alignment.rx_sen_start_word
            comp_idelay_taps = capture_alignment.comp_idelay_taps
            daq["gpio1"].read()
            if not daq["gpio1"]["COMP_IDELAY_RDY"].tovalue():
                raise RuntimeError("comparator IDELAYCTRL is not ready")
            daq["gpio1"]["COMP_IDELAY_TAPS"] = comp_idelay_taps
            daq["gpio1"]["COMP_IDELAY_LOAD"] = 1
            daq["gpio1"].write()
            daq["gpio1"]["COMP_IDELAY_LOAD"] = 0
            daq["gpio1"].write()

            spi_bytes = spi_bytes_by_curve.get(curve_key)
            if spi_bytes is None:
                spi_bytes = convert_params_to_spi_fmt(scan_params)
                spi_bytes_by_curve[curve_key] = spi_bytes
            spi_readback_checked = spi_bytes != programmed_spi_bytes
            spi_mismatches = 0
            if spi_readback_checked:
                for _write_index in range(2):
                    daq["spi0"].set_data(list(spi_bytes))
                    daq["spi0"].set_size(180)
                    daq["spi0"].start()
                    daq["spi0"].wait_for_ready()
                raw_spi = bytes(daq["spi0"].get_data(size=23))
                readback_bits = bitarray()
                readback_bits.frombytes(raw_spi)
                expected_bits = bitarray()
                expected_bits.frombytes(spi_bytes)
                spi_mismatches = (expected_bits[:180][1:] ^ readback_bits[:180][1:]).count(1)
                if spi_mismatches:
                    raise RuntimeError(f"SPI configuration readback has {spi_mismatches} mismatches")
                programmed_spi_bytes = spi_bytes

            sequence_words = len(params.seq_init_pattern) // 8
            setup_init_words = ["00000000"] * sequence_words
            setup_samp_words = ["00000000", *(["11111111"] * (sequence_words - 1))]
            setup_comp_words = ["00000000"] * sequence_words
            setup_logic_words = ["00000000"] * sequence_words
            setup_init_words[1] = "11111111"
            setup_logic_words[1] = "00001111"
            setup_params = replace(
                params,
                seq_init_pattern="".join(setup_init_words),
                seq_samp_pattern="".join(setup_samp_words),
                seq_comp_pattern="".join(setup_comp_words),
                seq_logic_pattern="".join(setup_logic_words),
                seq_init_phase_delay_symbols=0.0,
                seq_samp_phase_delay_symbols=0.0,
                seq_comp_phase_delay_symbols=0.0,
                seq_logic_phase_delay_symbols=0.0,
            )
            setup_rx_sen_pattern = "0" * sequence_words
            daq["seq0"].reset()
            daq["fastrx0"].reset()
            daq["fastrx0"].set_en(False)
            daq["seq0"].set_data(convert_params_to_seqgen_fmt(setup_params, setup_rx_sen_pattern))
            daq["seq0"].set_size(sequence_words)
            daq["seq0"].set_clk_divide(1)
            daq["seq0"].set_repeat(1)
            daq["seq0"].set_en_ext_start(False)
            daq["seq0"].start()
            deadline = monotonic() + capture_timeout_s
            while not daq["seq0"].is_done():
                if monotonic() >= deadline:
                    raise TimeoutError("one-shot CDAC A-state setup did not finish")
                sleep(0.001)
            # seq_gen holds its final word after DONE, keeping SAMP high while
            # the new external voltage settles onto both selected top plates.
            sleep(topplate_precondition_s)

            daq["seq0"].reset()
            daq["fastrx0"].reset()
            sleep(0.001)
            rx_sen_pattern = "0" * rx_sen_start_word + "1" + "0" * (sequence_words - rx_sen_start_word - 1)
            daq["seq0"].set_data(convert_params_to_seqgen_fmt(params, rx_sen_pattern))
            daq["seq0"].set_size(sequence_words)
            daq["seq0"].set_clk_divide(1)
            daq["seq0"].set_en_ext_start(False)
            daq["fastrx0"].set_en(True)
            daq["fifo0"]["RESET"]
            daq["fifo0"].get_data()

            conversion_period_s = len(params.seq_init_pattern) / symbol_rate_bps
            capture_scope = scope is not None and curve_key not in scope_captured_curves
            acquisition_count_before: int | None = None
            if capture_scope:
                scope.set_horizontal_scale(conversion_period_s / 8.0)
                scope.set_acquire_state("RUN")
                acquisition_count_before = wait_for_scope_armed(scope, timeout_s=scope_timeout_s)
            capture_started = monotonic()
            if scan_params.sweep_stage == "fine":
                complete_batches, remainder = divmod(params.conversions, fine_batch_trials)
                trial_batches = (
                    *((fine_batch_trials,) * complete_batches),
                    *((remainder,) if remainder else ()),
                )
            else:
                trial_batches = (params.conversions,)
            time_distribute_batches = False
            raw_batches = []
            for batch_index, batch_trials in enumerate(trial_batches):
                daq["seq0"].set_repeat(batch_trials)
                batch_timeout_s = max(capture_timeout_s, 2.0 * batch_trials * conversion_period_s + 2.0)
                deadline = monotonic() + batch_timeout_s
                daq["seq0"].start()
                while not daq["seq0"].is_done():
                    if monotonic() >= deadline:
                        raise TimeoutError(f"sequencer did not finish {batch_trials} CDAC trials")
                    sleep(0.001)
                while int(daq["fifo0"]["FIFO_SIZE"]) < 4 * batch_trials:
                    if monotonic() >= deadline:
                        raise TimeoutError("FastRX did not deliver every CDAC word")
                    sleep(0.001)
                raw_batch = daq["fifo0"].get_data()
                if len(raw_batch) != batch_trials:
                    raise RuntimeError(f"expected {batch_trials} FastRX words, received {len(raw_batch)}")
                raw_batches.extend(raw_batch)
                if scan_params.sweep_stage == "fine" and batch_index == 0:
                    first_decisions, _first_frames = convert_fastrx_words_to_comp(raw_batch, data_size=data_size)
                    first_batch_probability = float(np.mean(first_decisions))
                    time_distribute_batches = (
                        curve_key not in drift_checkpoint_curves
                        and drift_checkpoint_low_probability
                        <= first_batch_probability
                        <= drift_checkpoint_high_probability
                    )
                    if time_distribute_batches:
                        drift_checkpoint_curves.add(curve_key)
                if time_distribute_batches and batch_index + 1 < len(trial_batches):
                    sleep(fine_batch_interval_s)
            raw_data = raw_batches
            if len(raw_data) != params.conversions:
                raise RuntimeError(f"expected {params.conversions} FastRX words, received {len(raw_data)}")
            capture_elapsed_s = monotonic() - capture_started
            scope_waveforms = None
            scope_elapsed_s = 0.0
            if capture_scope:
                assert scope is not None
                assert acquisition_count_before is not None
                scope_started = monotonic()
                wait_for_scope_capture(scope, acquisition_count_before, timeout_s=scope_timeout_s)
                scope_waveforms = scope.get_waveforms(
                    {channel: name.removesuffix("_v") for name, channel in scope_tracks.items()}
                )
                missing_channels = sorted(set(scope_tracks.values()).difference(scope_waveforms))
                if missing_channels:
                    raise RuntimeError(f"scope did not return channels {missing_channels}")
                scope_elapsed_s = monotonic() - scope_started
                scope_captured_curves.add(curve_key)
            fastrx_lost_count = int(daq["fastrx0"].get_lost_count())
            if fastrx_lost_count:
                raise RuntimeError(f"FastRX lost {fastrx_lost_count} CDAC words")

            fastrx_words = np.asarray(raw_data, dtype=np.uint32)
            decisions, frames = convert_fastrx_words_to_comp(fastrx_words, data_size=data_size)
            curve_complete = False
            curve_error: str | None = None
            if original_params.sweep_stage == "coarse":
                observations = coarse_observations.setdefault(curve_key, {})
                observations[vin_diff_v] = (int(np.sum(decisions)), len(decisions))
                if find_probability_bracket(observations) is not None and curve_key not in fine_scheduled:
                    queue[variant_index + 1 : variant_index + 1] = build_fine_sweep_variants(
                        coarse_templates[curve_key],
                        observations,
                    )
                    fine_scheduled.add(curve_key)
                elif find_probability_bracket(observations) is None:
                    next_variant = build_next_coarse_sweep_variant(
                        coarse_templates[curve_key],
                        observations,
                    )
                    if next_variant is not None:
                        queue.insert(variant_index + 1, next_variant)
                    else:
                        curve_error = "coarse CDAC sweep exhausted its safe bounds without a 10%..90% bracket"
            elif original_params.sweep_stage == "fine":
                observations_fine = fine_observations.setdefault(curve_key, {})
                observations_fine[vin_diff_v] = (int(np.sum(decisions)), len(decisions))
                next_is_same_fine = (
                    variant_index + 1 < len(queue)
                    and queue[variant_index + 1].sweep_stage == "fine"
                    and _cdac_curve_key(queue[variant_index + 1]) == curve_key
                )
                if not next_is_same_fine:
                    if curve_key not in coarse_templates:
                        curve_error = "fine CDAC sweep has no coarse template for completion validation"
                    else:
                        next_variant = build_next_fine_sweep_variant(
                            coarse_templates[curve_key],
                            observations_fine,
                            coarse_observations[curve_key],
                        )
                        if next_variant is not None:
                            queue.insert(variant_index + 1, next_variant)
                        elif find_probability_bracket(observations_fine) is not None:
                            curve_complete = True
                        else:
                            curve_error = "fine CDAC sweep exhausted its safe bounds without a 10%..90% bracket"
            trial_index = np.arange(params.conversions, dtype=np.int64)
            wave = None
            if scope_waveforms is not None:
                reference = scope_waveforms[scope_tracks["vin_diff_v"]]
                scope_time_s = reference.x_scale.offset + np.arange(len(reference.data)) * reference.x_scale.slope
                wave_signals = {
                    name: np.asarray(scope_waveforms[channel].data, dtype=np.float64)[None, :]
                    for name, channel in scope_tracks.items()
                }
                wave = CdacExtWave(
                    trial_index=np.asarray([0], dtype=np.int64),
                    time_s=scope_time_s,
                    **wave_signals,
                )
            readbacks: dict[str, str | int | float | bool] = {
                "si570_frequency_hz": si570_frequency_hz,
                "pll_divider_n": pll_divider_n,
                "sequencer_frequency_hz": sequencer_frequency_hz,
                "serializer_frequency_hz": serializer_frequency_hz,
                "rx_sen_start_word": rx_sen_start_word,
                "comp_idelay_taps": comp_idelay_taps,
                "capture_control_phase_advance_symbols": phase_advance,
                "capture_setup_margin_s": capture_alignment.setup_margin_s,
                "capture_hold_margin_s": capture_alignment.hold_margin_s,
                "capture_batch_count": len(trial_batches),
                "capture_batch_trials": max(trial_batches),
                "capture_batch_interval_s": fine_batch_interval_s if time_distribute_batches else 0.0,
                "drift_checkpoint": time_distribute_batches,
                "capture_elapsed_s": capture_elapsed_s,
                "input_settle_s": input_settle_s,
                "scope_waveform_captured": capture_scope,
                "scope_elapsed_s": scope_elapsed_s,
                "topplate_precondition_s": topplate_precondition_s,
                "sample_aperture_s": params.seq_samp_pattern.count("1") / symbol_rate_bps,
                "acquisition_session_id": acquisition_session_id,
                "curve_complete": curve_complete,
                "vin_cm_supply_set_v": vin_cm_supply_v,
                "vin_cm_supply_measured_v": active_vin_cm_measured_v,
                "vin_cm_supply_measured_a": active_vin_cm_measured_a,
                "awg_voltage_set_v": awg_voltage_v,
                "awg_readback_checked": awg_readback_checked,
                "predicted_cdac_step_v": _predict_cdac_step_v(params),
                "cdac_topplate_parasitic_weight": float(
                    board_map["adc_flavors"][board["adc_channels"][scan_params.observed_adc]].get(
                        "cdac_topplate_parasitic_weight",
                        0.0,
                    )
                ),
                "predicted_before_p_v": plate_voltages[0],
                "predicted_before_n_v": plate_voltages[1],
                "predicted_after_p_v": plate_voltages[2],
                "predicted_after_n_v": plate_voltages[3],
                "spi_readback_checked": spi_readback_checked,
                "spi_mismatches": spi_mismatches,
                "fastrx_lost_count": fastrx_lost_count,
            }
            if awg_readback_v is not None:
                readbacks["awg_voltage_readback_v"] = awg_readback_v
            for field, values in supply_readbacks.items():
                for quantity, value in values.items():
                    readbacks[f"{field}_{quantity}"] = value

            h5_path = run_dir / f"{next_file_index:05d}_{point_stem}.h5"
            before_p = np.tile(np.asarray(params.dac_astate_p, dtype=np.uint8), (params.conversions, 1))
            before_n = np.tile(np.asarray(params.dac_astate_n, dtype=np.uint8), (params.conversions, 1))
            after_p = np.tile(np.asarray(params.dac_bstate_p, dtype=np.uint8), (params.conversions, 1))
            after_n = np.tile(np.asarray(params.dac_bstate_n, dtype=np.uint8), (params.conversions, 1))
            measurement = MeasCdacExt(
                info=MeasInfo(
                    schema_version=2,
                    measurement_type="MeasCdacExt",
                    backend="physical",
                    timestamp_utc=datetime.now().astimezone(),
                    instruments=instrument_identities,
                    readbacks=readbacks,
                ),
                param=scan_params,
                daq=CdacExtDaq(
                    trial_index=trial_index,
                    dac_state_p=after_p,
                    dac_state_n=after_n,
                    vin_diff_v=np.full(params.conversions, vin_diff_v),
                    decision=decisions,
                    dac_state_before_p=before_p,
                    dac_state_before_n=before_n,
                    vin_cm_v=np.full(params.conversions, vin_cm_v),
                    fastrx_word=fastrx_words,
                    fastrx_frame=frames,
                ),
                wave=wave,
            )
            write_measurement(h5_path, measurement)
            existing_paths[point_stem] = h5_path
            next_file_index += 1
            print(
                f"[{variant_index + 1}/{len(queue)}] ADC{scan_params.observed_adc:02d} "
                f"{scan_params.cdac_side.upper()} C{scan_params.cdac_element} "
                f"{scan_params.cdac_direction} "
                f"P(decision=1)={float(np.mean(decisions)):.4f} "
                f"elapsed={monotonic() - point_started:.3f}s: {h5_path}"
            )
            if curve_error is not None:
                raise RuntimeError(curve_error)
        return run_dir
    finally:
        if daq is not None:
            try:
                daq["gpio0"]["RST_B"] = 0
                daq["gpio0"]["AMP_EN"] = 0
                daq["gpio0"]["RX_LOOPBACK"] = 0
                daq["gpio0"]["SPI_LOOPBACK"] = 0
                daq["gpio0"]["DBG_FIFO"] = 0
                daq["gpio0"]["RX_TIEHIGH"] = 0
                daq["gpio0"]["SEQ_START"] = 0
                daq["gpio0"]["RX_EN_MUX"] = 0
                daq["gpio0"].write()
                daq["si570"].frequency_change(200.0)
                sleep(si570_settle_s)
                set_pll_divider(daq["gpio2"], 2)
            except Exception as error:  # noqa: BLE001 - best-effort safety shutdown
                print(f"Warning: could not restore the FPGA: {error}")
        if awg is not None:
            try:
                awg.set_DC("DEF,DEF,0")
                awg.set_enable(0)
            except Exception as error:  # noqa: BLE001 - best-effort safety shutdown
                print(f"Warning: could not disable the AWG: {error}")
        if vin_cm_supply is not None:
            try:
                vin_cm_supply.set_enable(0)
                vin_cm_supply.set_voltage(0.0)
            except Exception as error:  # noqa: BLE001 - best-effort safety shutdown
                print(f"Warning: could not disable Vin_cm: {error}")
        for smu, _rail, _field in smus:
            try:
                smu.off()
                smu.set_voltage(0.0)
            except Exception as error:  # noqa: BLE001 - best-effort safety shutdown
                print(f"Warning: could not disable an SMU: {error}")
        for dut in reversed(initialized_duts):
            dut.close()
