"""Software-only tests for A-to-B CDAC scan construction."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import hdl21 as h
import numpy as np
import pytest

from flow.adc.sim import AdcTbParams
from flow.analysis.types import CdacExtDaq, CdacExtWave, InfoValue, MeasCdacExt, MeasInfo
from flow.scans.fastrx import calculate_single_sample_fastrx_capture_alignment
from flow.scans.params import AdcScanParams, load_board_map, validate_params
from flow.scans.scan_cdac import (
    _build_cdac_params,
    _calculate_cdac_input_bounds,
    _calculate_cdac_plate_voltages,
    _cdac_point_stem,
    _convert_dac_rail_percent_to_codes,
    _expected_transition_v,
    _predict_cdac_step_v,
    _validate_cdac_resume_curves,
    build_alternating_sweep_values,
    build_capacitor_variants,
    build_fine_sweep_variants,
    build_next_coarse_sweep_variant,
    build_next_fine_sweep_variant,
)
from flow.scans.scan_cdac import (
    scan as run_cdac_scan,
)

RADIX17 = (768, 512, 320, 192, 96, 64, 32, 24, 12, 10, 5, 4, 4, 2, 1, 1)
RADIX20 = (768, 512, 320, 192, 128, 64, 64, 64, 64, 64, 32, 16, 8, 4, 2, 1)


def build_cdac_test_variants() -> list[AdcScanParams]:
    """Build one calibrated C0 transition point per characterized ADC."""

    calibrations = load_board_map()["boards"]["00"]["comparator_calibration"]
    return [
        _build_cdac_params(
            adc_index=adc_index,
            side="p",
            element=0,
            direction="1to0",
            dac_diffcaps=0,
            vin_diff_v=_expected_transition_v(
                adc_index,
                "p",
                0,
                "1to0",
                0,
                float(calibrations[adc_index]["offset_v"]),
            ),
            conversions=128,
            sweep_stage="fixed",
        )
        for adc_index in range(4)
    ]


def test_adaptive_sweep_alternates_and_fine_builder_uses_measured_bracket() -> None:
    assert build_alternating_sweep_values(0.0, -2e-3, 3e-3) == (
        0.0,
        -1e-3,
        1e-3,
        -2e-3,
        2e-3,
        3e-3,
    )
    template = _build_cdac_params(
        adc_index=0,
        side="p",
        element=0,
        direction="1to0",
        dac_diffcaps=0,
        vin_diff_v=0.0,
        conversions=128,
        sweep_stage="coarse",
    )
    seed = replace(
        template,
        sweep_min_v=-2e-3,
        sweep_max_v=3e-3,
        sweep_step_v=1e-3,
    )
    fine = build_fine_sweep_variants(
        seed,
        {
            -2e-3: (128, 128),
            -1e-3: (80, 128),
            0.0: (10, 128),
            1e-3: (0, 128),
        },
    )

    assert len(fine) == 21
    assert all(params.tb.conversions == 1_000 and params.sweep_stage == "fine" for params in fine)
    assert float(fine[0].tb.vin_diff.dc) == pytest.approx(-2e-3)
    assert float(fine[-1].tb.vin_diff.dc) == pytest.approx(0.0)

    resumed = build_next_coarse_sweep_variant(seed, {0.0: (50, 128), -1e-3: (0, 128)})
    assert resumed is not None
    assert float(resumed.tb.vin_diff.dc) == pytest.approx(1e-3)
    search_lower = build_next_coarse_sweep_variant(seed, {0.0: (128, 128), -1e-3: (128, 128)})
    assert search_lower is not None
    assert float(search_lower.tb.vin_diff.dc) == pytest.approx(-2e-3)
    search_upper = build_next_coarse_sweep_variant(seed, {0.0: (0, 128), 1e-3: (0, 128)})
    assert search_upper is not None
    assert float(search_upper.tb.vin_diff.dc) == pytest.approx(2e-3)
    bounded_fine = build_fine_sweep_variants(seed, {-2e-3: (128, 128), 0.0: (10, 128)})
    assert float(bounded_fine[0].sweep_min_v) == pytest.approx(-2e-3)
    assert float(bounded_fine[0].sweep_max_v) == pytest.approx(3e-3)
    extension = build_next_fine_sweep_variant(
        seed,
        {-1e-3: (800, 1_000), 0.0: (200, 1_000)},
        {-2e-3: (128, 128), 0.0: (10, 128)},
    )
    assert extension is not None
    assert extension.sweep_stage == "fine"
    assert extension.tb.conversions == 1_000
    assert float(extension.tb.vin_diff.dc) == pytest.approx(-1.1e-3)
    assert (
        build_next_fine_sweep_variant(
            seed,
            {-1e-3: (950, 1_000), 0.0: (50, 1_000)},
            {-2e-3: (128, 128), 0.0: (10, 128)},
        )
        is None
    )
    assert (
        build_next_fine_sweep_variant(
            seed,
            {-2e-3 + 1e-15: (0, 1_000), -1.9e-3: (0, 1_000)},
            {-2e-3: (128, 128), 0.0: (10, 128)},
        )
        is None
    )


def _resume_measurement(
    params: AdcScanParams,
    timestamp: datetime,
    *,
    session_id: str | None,
    curve_complete: bool,
) -> MeasCdacExt:
    tb = params.tb
    trials = tb.conversions
    after_p = np.tile(tb.dac_bstate_p, (trials, 1))
    after_n = np.tile(tb.dac_bstate_n, (trials, 1))
    readbacks: dict[str, InfoValue] = {"curve_complete": curve_complete}
    if session_id is not None:
        readbacks["acquisition_session_id"] = session_id
    return MeasCdacExt(
        info=MeasInfo(
            schema_version=1,
            measurement_type="MeasCdacExt",
            backend="physical",
            timestamp_utc=timestamp,
            readbacks=readbacks,
        ),
        param=params,
        daq=CdacExtDaq(
            trial_index=np.arange(trials),
            dac_state_p=after_p,
            dac_state_n=after_n,
            vin_diff_v=np.full(trials, float(tb.vin_diff.dc)),
            decision=np.zeros(trials, dtype=np.uint8),
        ),
        wave=CdacExtWave(
            trial_index=np.asarray([0], dtype=np.int64),
            time_s=np.asarray([0.0, 1e-9]),
            vin_diff_v=np.zeros((1, 2)),
            seq_comp_v=np.zeros((1, 2)),
            comp_out_v=np.zeros((1, 2)),
        ),
    )


def test_cdac_resume_requires_one_completed_uninterrupted_curve() -> None:
    coarse = _build_cdac_params(
        adc_index=0,
        side="p",
        element=0,
        direction="1to0",
        dac_diffcaps=0,
        vin_diff_v=0.20,
        conversions=8,
        sweep_stage="coarse",
    )
    fine = replace(
        coarse,
        tb=replace(coarse.tb, vin_diff=h.Vdc.Params(dc=0.199), conversions=1_000),
        sweep_stage="fine",
    )
    started = datetime(2026, 8, 4, 8, 0, tzinfo=UTC)
    coarse_measurement = _resume_measurement(
        coarse,
        started,
        session_id="session-a",
        curve_complete=False,
    )
    completed_measurement = _resume_measurement(
        fine,
        started + timedelta(minutes=1),
        session_id="session-a",
        curve_complete=True,
    )

    _validate_cdac_resume_curves({("curve",): [coarse_measurement, completed_measurement]})

    with pytest.raises(ValueError, match="interrupted or mixed-session"):
        _validate_cdac_resume_curves({("curve",): [coarse_measurement]})
    with pytest.raises(ValueError, match="interrupted or mixed-session"):
        _validate_cdac_resume_curves(
            {
                ("curve",): [
                    coarse_measurement,
                    replace(
                        completed_measurement,
                        info=replace(
                            completed_measurement.info,
                            readbacks={
                                "acquisition_session_id": "session-b",
                                "curve_complete": True,
                            },
                        ),
                    ),
                ]
            }
        )
    with pytest.raises(ValueError, match="interrupted or mixed-session"):
        _validate_cdac_resume_curves(
            {
                ("curve",): [
                    replace(coarse_measurement, info=replace(coarse_measurement.info, readbacks={})),
                    completed_measurement,
                ]
            }
        )


@pytest.mark.parametrize("weights", (RADIX17, RADIX20))
def test_dac_rail_codes_return_all_deterministic_closest_states(weights: tuple[int, ...]) -> None:
    codes = _convert_dac_rail_percent_to_codes(50.0, weights)

    assert codes == tuple(sorted(codes))
    realized = [sum(weight for bit, weight in zip(code, weights, strict=True) if bit == "1") for code in codes]
    target = sum(weights) / 2.0
    assert len({abs(value - target) for value in realized}) == 1
    assert abs(realized[0] - target) == min(
        abs(sum(weight for bit, weight in zip(f"{value:016b}", weights, strict=True) if bit == "1") - target)
        for value in range(1 << 16)
    )
    assert _convert_dac_rail_percent_to_codes(0.0, weights) == ("0" * 16,)
    assert _convert_dac_rail_percent_to_codes(100.0, weights) == ("1" * 16,)


def test_dac_rail_codes_preserve_exact_ties() -> None:
    codes = _convert_dac_rail_percent_to_codes(3.125, (2,) * 16)

    assert len(codes) == 17
    assert codes[0] == "0" * 16
    assert codes[-1] == "1" + "0" * 15


def test_single_sample_alignment_covers_late_cdac_sequence() -> None:
    timing = load_board_map()["boards"]["00"]["capture_timing_model"]
    template = build_cdac_test_variants()[0]
    for rate_mbd in range(80, 1601):
        params = replace(template, tb=replace(template.tb, symbol_rate=rate_mbd * 1.0e6))
        alignment = calculate_single_sample_fastrx_capture_alignment(params.tb, **timing)
        assert 0 <= alignment.rx_sen_start_word < 31
        assert 0 <= alignment.comp_idelay_taps < 32
        assert alignment.setup_margin_s >= timing["minimum_capture_margin_s"]
        assert alignment.hold_margin_s >= timing["minimum_capture_margin_s"]


@pytest.mark.parametrize("side", ("p", "n"))
@pytest.mark.parametrize("direction", ("1to0", "0to1"))
@pytest.mark.parametrize("dac_diffcaps", (0, 1))
def test_cdac_params_encode_only_selected_a_to_b_transition(
    side: str,
    direction: str,
    dac_diffcaps: int,
) -> None:
    params = _build_cdac_params(
        adc_index=0,
        side=side,
        element=7,
        direction=direction,
        dac_diffcaps=dac_diffcaps,
        vin_diff_v=0.0,
        conversions=8,
        sweep_stage="fixed",
    )

    tb = params.tb
    before_p = np.asarray(tb.dac_astate_p)
    before_n = np.asarray(tb.dac_astate_n)
    after_p = np.asarray(tb.dac_bstate_p)
    after_n = np.asarray(tb.dac_bstate_n)
    changed_p = np.flatnonzero(before_p != after_p).tolist()
    changed_n = np.flatnonzero(before_n != after_n).tolist()
    assert changed_p == ([7] if side == "p" else [])
    assert changed_n == ([7] if side == "n" else [])
    expected_before = 1 if direction == "1to0" else 0
    expected_after = 1 - expected_before
    assert (before_p if side == "p" else before_n)[7] == expected_before
    assert (after_p if side == "p" else after_n)[7] == expected_after
    assert tb.dac_diffcaps == dac_diffcaps
    assert tb.seq_logic_pattern.count("1") == 8
    assert tb.seq_init_pattern.count("1") == 8
    assert tb.seq_samp_pattern.count("1") == 136
    assert tb.seq_comp_pattern.count("1") == 4
    samp_words = [tb.seq_samp_pattern[index : index + 8] for index in range(0, 256, 8)]
    logic_words = [tb.seq_logic_pattern[index : index + 8] for index in range(0, 256, 8)]
    comp_words = [tb.seq_comp_pattern[index : index + 8] for index in range(0, 256, 8)]
    assert samp_words[3:20] == ["11111111"] * 17
    assert all(word == "00000000" for index, word in enumerate(samp_words) if not 3 <= index < 20)
    update_word = 21
    assert [index for index, word in enumerate(logic_words) if "1" in word] == [1, update_word]
    assert [index for index, word in enumerate(comp_words) if "1" in word] == [update_word + 5]
    minimum_v, maximum_v = _calculate_cdac_input_bounds(params)
    for boundary_v in (minimum_v, maximum_v):
        boundary = replace(params, tb=replace(tb, vin_diff=h.Vdc.Params(dc=boundary_v)))
        assert all(
            0.4 - 1e-12 <= voltage <= float(tb.vdd_a.dc) + 1e-12 for voltage in _calculate_cdac_plate_voltages(boundary)
        )
    outside = (
        replace(params, tb=replace(tb, vin_diff=h.Vdc.Params(dc=minimum_v - 1e-6))),
        replace(params, tb=replace(tb, vin_diff=h.Vdc.Params(dc=maximum_v + 1e-6))),
    )
    assert all(
        any(voltage < 0.4 or voltage > float(tb.vdd_a.dc) for voltage in _calculate_cdac_plate_voltages(candidate))
        for candidate in outside
    )


def test_cdac_expected_transition_applies_comparator_offset_with_physical_sign() -> None:
    comparator_offset_v = 7e-3
    falling = _expected_transition_v(0, "p", 15, "1to0", 0, comparator_offset_v)
    rising = _expected_transition_v(0, "p", 15, "0to1", 0, comparator_offset_v)

    assert falling > comparator_offset_v
    assert rising < comparator_offset_v
    assert (falling + rising) / 2.0 == pytest.approx(comparator_offset_v)


@pytest.mark.parametrize("adc_index", (0, 2))
@pytest.mark.parametrize("dac_diffcaps", (0, 1))
def test_cdac_expected_transition_uses_accepted_c0_scale(
    adc_index: int,
    dac_diffcaps: int,
) -> None:
    comparator_offset_v = 7e-3
    probe = _build_cdac_params(
        adc_index=adc_index,
        side="p",
        element=0,
        direction="1to0",
        dac_diffcaps=dac_diffcaps,
        vin_diff_v=0.0,
        conversions=8,
        sweep_stage="fixed",
    )
    scale = load_board_map()["boards"]["00"]["cdac_sweep_center_calibration"][adc_index]["scale_by_diffcaps"][
        dac_diffcaps
    ]
    accepted_scaled_weights = {
        0: (641.9997915881323, 621.1102678208147),
        2: (828.1190540265467, 799.9696164433147),
    }
    switched_weight = 780.0 if dac_diffcaps == 0 else 768.0

    falling = _expected_transition_v(
        adc_index,
        "p",
        0,
        "1to0",
        dac_diffcaps,
        comparator_offset_v,
    )

    assert scale * switched_weight == pytest.approx(accepted_scaled_weights[adc_index][dac_diffcaps])
    assert falling - comparator_offset_v == pytest.approx(scale * _predict_cdac_step_v(probe))


@pytest.mark.parametrize("adc_index", range(4))
@pytest.mark.parametrize("dac_diffcaps", (0, 1))
def test_cdac_step_prediction_includes_flavor_topplate_parasitics(
    adc_index: int,
    dac_diffcaps: int,
) -> None:
    params = _build_cdac_params(
        adc_index=adc_index,
        side="p",
        # C15 makes the fabricated mode distinction unambiguous: its physical
        # main+diff weight is 65, while normal main-diff operation is weight 1.
        element=15,
        direction="1to0",
        dac_diffcaps=dac_diffcaps,
        vin_diff_v=0.0,
        conversions=8,
        sweep_stage="fixed",
    )
    board_map = load_board_map()
    flavor = board_map["boards"]["00"]["adc_channels"][adc_index]
    flavor_config = board_map["adc_flavors"][flavor]
    weights = flavor_config["cdac_weights"]
    total_weights = [65 * np.ceil(weight / 64) for weight in weights]
    switched_weight = weights[15] if dac_diffcaps else total_weights[15]
    expected = (
        float(params.tb.vdd_dac.dc)
        * switched_weight
        / (sum(total_weights) + flavor_config["cdac_topplate_parasitic_weight"])
    )

    assert _predict_cdac_step_v(params) == pytest.approx(expected)


def test_cdac_test_plate_predictions_are_safe_and_adc00_through_adc03() -> None:
    variants = build_cdac_test_variants()
    calibrations = load_board_map()["boards"]["00"]["comparator_calibration"]

    assert [params.observed_adc for params in variants] == [0, 1, 2, 3]
    for params in variants:
        calibration = calibrations[params.observed_adc]
        expected_center_v = _expected_transition_v(
            params.observed_adc,
            "p",
            0,
            "1to0",
            0,
            float(calibration["offset_v"]),
        )
        assert float(params.tb.vin_diff.dc) == pytest.approx(expected_center_v)
        assert "_c0_" in _cdac_point_stem(params)
        assert all(0.4 <= voltage <= float(params.tb.vdd_a.dc) for voltage in _calculate_cdac_plate_voltages(params))


def test_default_cdac_campaign_cardinality_selection_and_point_uniqueness() -> None:
    """Cover every explicit ADC00--ADC03 CDAC campaign axis."""

    capacitor_variants = build_capacitor_variants(
        adc_indices=(0, 1, 2, 3),
        coarse_step_v=1.0e-3,
        coarse_trials=128,
    )
    assert len(capacitor_variants) == 512
    assert {
        (params.observed_adc, params.cdac_side, params.cdac_element, params.cdac_direction, params.tb.dac_diffcaps)
        for params in capacitor_variants
    } == {
        (adc_index, side, element, direction, diffcaps)
        for adc_index in range(4)
        for side in ("p", "n")
        for element in range(16)
        for direction in ("1to0", "0to1")
        for diffcaps in (0, 1)
    }
    stems = [_cdac_point_stem(params) for params in capacitor_variants]
    assert len(stems) == len(set(stems))


def test_cdac_builder_rejects_unsafe_top_plate_before_hardware() -> None:
    with pytest.raises(ValueError, match="outside 0.4..VDD_A"):
        _build_cdac_params(
            adc_index=0,
            side="p",
            element=0,
            direction="1to0",
            dac_diffcaps=0,
            vin_diff_v=0.81,
            conversions=8,
            sweep_stage="fixed",
        )


@pytest.mark.parametrize("supply_v", (1.1, 1.3))
def test_cdac_preflight_accepts_programmable_supply_boundaries(
    tmp_path,
    supply_v: float,
) -> None:
    boundary = replace(
        build_cdac_test_variants()[0],
        tb=replace(
            build_cdac_test_variants()[0].tb,
            vdd_a=h.Vdc.Params(dc=supply_v),
            vdd_d=h.Vdc.Params(dc=supply_v),
            vdd_dac=h.Vdc.Params(dc=supply_v),
        ),
    )
    uncalibrated = replace(boundary, tb=replace(boundary.tb, vin_diff=h.Vdc.Params(dc=1.01)))
    run_dir = tmp_path / "not-created"

    with pytest.raises(ValueError, match="calibrated range"):
        run_cdac_scan((boundary, uncalibrated), run_dir=run_dir, capture_scope_per_curve=False)
    assert not run_dir.exists()


def test_cdac_preflight_rejects_supply_and_fixed_io_before_hardware(tmp_path) -> None:
    params = build_cdac_test_variants()[0]
    for invalid, message in (
        (replace(params, tb=replace(params.tb, vdd_a=h.Vdc.Params(dc=1.099))), "VDD_A request"),
        (replace(params, tb=replace(params.tb, vdd_dac=h.Vdc.Params(dc=1.301))), "VDD_DAC request"),
        (replace(params, vdd_io=h.Vdc.Params(dc=1.3)), "VDD_IO is fixed"),
    ):
        run_dir = tmp_path / message.replace(" ", "_")
        with pytest.raises(ValueError, match=message):
            run_cdac_scan((invalid,), run_dir=run_dir, capture_scope_per_curve=False)
        assert not run_dir.exists()


def test_cdac_scan_axis_validation_rejects_incomplete_params() -> None:
    params = AdcScanParams(tb=AdcTbParams(view="frida1"), campaign="cdac_ab", sampling_mode="hold")
    with pytest.raises(ValueError, match="requires side, element, and direction"):
        validate_params(params)
