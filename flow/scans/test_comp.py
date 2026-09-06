"""Software-only tests for comparator scan construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import hdl21 as h
import numpy as np
import pytest

from flow.adc.sim import AdcTbParams
from flow.cdac import get_cdac_weights
from flow.scans.fastrx import (
    calculate_single_sample_fastrx_capture_alignment,
    convert_fastrx_words_to_comp,
)
from flow.scans.params import AdcScanParams, load_board_map, validate_params
from flow.scans.scan_cdac import _convert_dac_rail_percent_to_codes
from flow.scans.scan_comp import (
    _build_comp_params,
    _comp_point_stem,
    build_common_mode_variants,
    build_sampling_noise_variants,
)
from flow.scans.scan_comp import (
    scan as run_comp_scan,
)
from flow.scans.seqgen import convert_params_to_seqgen_fmt


def build_comp_test_variants() -> list[AdcScanParams]:
    """Build one balanced track point for each characterized ADC."""

    return [
        _build_comp_params(
            adc_index=adc_index,
            campaign="comp_common_mode",
            sampling_mode="track",
            sweep_stage="fixed",
            vin_cm_v=0.8,
            vin_diff_v=0.0,
            conversions=128,
        )
        for adc_index in range(4)
    ]


def test_single_sample_seqgen_has_exactly_one_rx_sen_word() -> None:
    params = build_comp_test_variants()[0]
    memory = convert_params_to_seqgen_fmt(params.tb, "0" * 10 + "1" + "0" * 21)
    control = np.asarray(memory, dtype=np.uint8).reshape(-1, 8)[:, 4]

    assert len(memory) == 32 * 8
    assert np.flatnonzero(control).tolist() == [10]
    without_capture = convert_params_to_seqgen_fmt(params.tb, "0" * 32)
    assert not np.any(np.asarray(without_capture, dtype=np.uint8).reshape(-1, 8)[:, 4])


def test_single_sample_alignment_covers_comparator_sequence() -> None:
    timing = load_board_map()["boards"]["00"]["capture_timing_model"]
    template = build_comp_test_variants()[0]
    for rate_mbd in range(80, 1601):
        params = replace(template, tb=replace(template.tb, symbol_rate=rate_mbd * 1.0e6))
        alignment = calculate_single_sample_fastrx_capture_alignment(params.tb, **timing)
        assert 0 <= alignment.rx_sen_start_word < 31
        assert 0 <= alignment.comp_idelay_taps < 32
        assert alignment.setup_margin_s >= timing["minimum_capture_margin_s"]
        assert alignment.hold_margin_s >= timing["minimum_capture_margin_s"]


def test_partial_fastrx_decoder_validates_headers_frames_and_unused_bits() -> None:
    frames = np.arange(2051, dtype=np.uint32) % (1 << 11)
    decisions = np.arange(2051, dtype=np.uint32) & 1
    words = (np.uint32(1 << 28) | (frames << 17) | decisions).astype(np.uint32)

    decoded, decoded_frames = convert_fastrx_words_to_comp(words, data_size=17)
    np.testing.assert_array_equal(decoded, decisions)
    np.testing.assert_array_equal(decoded_frames, frames)

    invalid_identifier = words[:2].copy()
    invalid_identifier[1] &= np.uint32(0x0FFFFFFF)
    with pytest.raises(RuntimeError, match="identifier"):
        convert_fastrx_words_to_comp(invalid_identifier, data_size=17)
    invalid_frame = words[:2].copy()
    invalid_frame[1] ^= np.uint32(1 << 17)
    with pytest.raises(RuntimeError, match="frame"):
        convert_fastrx_words_to_comp(invalid_frame, data_size=17)
    invalid_payload = words[:2].copy()
    invalid_payload[1] |= np.uint32(1 << 1)
    with pytest.raises(RuntimeError, match="unused"):
        convert_fastrx_words_to_comp(invalid_payload, data_size=17)


def test_comparator_balanced_track_params_encode_expected_point() -> None:
    variants = build_comp_test_variants()

    assert [params.observed_adc for params in variants] == [0, 1, 2, 3]
    for params in variants:
        tb = params.tb
        assert params.campaign == "comp_common_mode"
        assert params.sampling_mode == "track"
        assert set(tb.seq_samp_pattern) == {"1"}
        assert "1" not in tb.seq_init_pattern
        assert "1" not in tb.seq_logic_pattern
        weights = tuple(get_cdac_weights(tb.dut.cdac))
        expected = _convert_dac_rail_percent_to_codes(50.0, weights)[0]
        assert tb.dac_astate_p == tuple(int(bit) for bit in expected)
        assert tb.dac_astate_p == tb.dac_bstate_p == tb.dac_astate_n == tb.dac_bstate_n


def test_comparator_setup_latches_sampling_enable_before_continuous_track() -> None:
    measured = build_comp_test_variants()[0]
    sequence_words = len(measured.tb.seq_init_pattern) // 8
    init_words = ["00000000"] * sequence_words
    samp_words = ["00000000", *(["11111111"] * (sequence_words - 1))]
    comp_words = ["00000000"] * sequence_words
    logic_words = ["00000000"] * sequence_words
    init_words[1] = "11111111"
    logic_words[1] = "00001111"
    setup = replace(
        measured,
        tb=replace(
            measured.tb,
            seq_init_pattern="".join(init_words),
            seq_samp_pattern="".join(samp_words),
            seq_comp_pattern="".join(comp_words),
            seq_logic_pattern="".join(logic_words),
            seq_init_phase_delay_symbols=0.0,
            seq_samp_phase_delay_symbols=0.0,
            seq_comp_phase_delay_symbols=0.0,
            seq_logic_phase_delay_symbols=0.0,
        ),
    )

    assert set(measured.tb.seq_samp_pattern) == {"1"}
    assert setup.tb.seq_samp_pattern[:8] == "00000000"
    assert set(setup.tb.seq_samp_pattern[8:]) == {"1"}
    assert setup.tb.seq_init_pattern[8:16] == "11111111"
    assert setup.tb.seq_logic_pattern[8:16] == "00001111"
    assert "1" not in setup.tb.seq_comp_pattern


def test_default_comparator_campaign_cardinality_selection_and_point_uniqueness() -> None:
    """Cover every explicit ADC00--ADC03 comparator campaign axis."""

    common_mode_variants = build_common_mode_variants(
        adc_indices=(0, 1, 2, 3),
        common_mode_values_v=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
        minimum_v=0.0,
        maximum_v=25.0e-3,
        step_v=100.0e-6,
        conversions=1_000,
    )
    sampling_variants = build_sampling_noise_variants(
        adc_indices=(0, 1, 2, 3),
        coupling_percentages=(0.0, 25.0, 50.0, 75.0, 100.0),
        vin_cm_v=0.7,
        minimum_v=0.0,
        maximum_v=25.0e-3,
        step_v=100.0e-6,
        conversions=1_000,
    )
    comparator_campaigns = {
        "common_mode": (common_mode_variants, 6_024),
        "sampling_noise": (sampling_variants, 10_040),
    }
    for variants, expected_count in comparator_campaigns.values():
        assert len(variants) == expected_count
        assert {params.observed_adc for params in variants} == {0, 1, 2, 3}
        stems = [_comp_point_stem(params) for params in variants]
        assert len(stems) == len(set(stems))

    common_modes = {round(float(params.tb.vin_cm.dc), 12) for params in comparator_campaigns["common_mode"][0]}
    assert common_modes == {round(index * 0.1, 12) for index in range(7, 13)}
    assert {
        (float(params.sweep_min_v), float(params.sweep_max_v)) for params in comparator_campaigns["common_mode"][0]
    } == {(0.0, 25.0e-3)}
    assert {
        (params.sweep_stage, params.tb.conversions, float(params.sweep_step_v))
        for params in comparator_campaigns["common_mode"][0]
    } == {("fine", 1_000, 100.0e-6)}

    assert {params.sampling_mode for params in sampling_variants} == {"track", "hold"}
    assert {float(params.tb.vin_cm.dc) for params in sampling_variants} == {0.7}
    assert {float(params.requested_dac_rail_percent) for params in sampling_variants} == {
        0.0,
        25.0,
        50.0,
        75.0,
        100.0,
    }
    assert {(float(params.sweep_min_v), float(params.sweep_max_v)) for params in sampling_variants} == {(0.0, 25.0e-3)}
    assert {
        (params.sweep_stage, params.tb.conversions, float(params.sweep_step_v)) for params in sampling_variants
    } == {("fine", 1_000, 100.0e-6)}
    assert all(params.tb.dac_astate_p == params.tb.dac_bstate_p for params in sampling_variants)
    assert all(params.tb.dac_astate_n == params.tb.dac_bstate_n for params in sampling_variants)
    for params in sampling_variants:
        if float(params.requested_dac_rail_percent) == 50.0:
            assert params.tb.dac_astate_n == params.tb.dac_astate_p
        else:
            assert params.tb.dac_astate_n == tuple(1 - bit for bit in params.tb.dac_astate_p)

    curve_point_counts = Counter(
        (
            params.observed_adc,
            float(params.requested_dac_rail_percent),
            params.sampling_mode,
        )
        for params in sampling_variants
    )
    assert set(curve_point_counts.values()) == {251}

    repair_variants = build_sampling_noise_variants(
        adc_indices=(0, 1, 2, 3),
        coupling_percentages=(0.0, 25.0, 50.0, 75.0, 100.0),
        vin_cm_v=0.7,
        minimum_v=0.0,
        maximum_v=25.0e-3,
        step_v=100.0e-6,
        conversions=1_000,
        selected_curves={(1, 100.0, "track"), (2, 75.0, "track")},
    )
    assert len(repair_variants) == 502
    assert {
        (
            params.observed_adc,
            float(params.requested_dac_rail_percent),
            params.sampling_mode,
        )
        for params in repair_variants
    } == {(1, 100.0, "track"), (2, 75.0, "track")}


@pytest.mark.parametrize(
    ("params", "message"),
    (
        (
            AdcScanParams(tb=AdcTbParams(view="frida1"), campaign="comp_sampling_noise"),
            "requested_dac_rail_percent",
        ),
        (AdcScanParams(tb=AdcTbParams(view="frida1"), settling_time_s=-1.0), "non-negative"),
    ),
)
def test_comparator_scan_axis_validation_rejects_incomplete_params(
    params: AdcScanParams,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_params(params)


def test_hold_comp_builder_places_sampling_before_comparison() -> None:
    params = _build_comp_params(
        adc_index=0,
        campaign="comp_sampling_noise",
        sampling_mode="hold",
        sweep_stage="coarse",
        vin_cm_v=0.7,
        vin_diff_v=0.0,
        conversions=128,
        requested_dac_rail_percent=50.0,
        settling_time_s=20e-9,
    )
    samp_words = [params.tb.seq_samp_pattern[index : index + 8] for index in range(0, 256, 8)]
    comp_words = [params.tb.seq_comp_pattern[index : index + 8] for index in range(0, 256, 8)]

    assert samp_words[1:4] == ["11111111"] * 3
    assert all(word == "00000000" for word in samp_words[4:])
    comp_word = next(index for index, word in enumerate(comp_words) if "1" in word)
    assert comp_word == 8
    assert (comp_word - 4) * 8 / float(params.tb.symbol_rate) == pytest.approx(params.settling_time_s)


@pytest.mark.parametrize(
    ("vin_cm_v", "vin_diff_v"),
    ((0.4, 0.9), (0.8, 0.9)),
)
def test_comparator_preflight_accepts_signal_headroom_boundary_before_rejecting_beyond_it(
    tmp_path,
    vin_cm_v: float,
    vin_diff_v: float,
) -> None:
    boundary = _build_comp_params(
        adc_index=0,
        campaign="comp_common_mode",
        sampling_mode="track",
        sweep_stage="fixed",
        vin_cm_v=vin_cm_v,
        vin_diff_v=vin_diff_v,
        conversions=8,
    )
    beyond = replace(boundary, tb=replace(boundary.tb, vin_diff=h.Vdc.Params(dc=vin_diff_v + 2.0e-6)))
    run_dir = tmp_path / "not-created"

    with pytest.raises(ValueError, match="comparator inputs"):
        run_comp_scan((boundary, beyond), run_dir=run_dir, capture_scope_per_curve=False)
    assert not run_dir.exists()


@pytest.mark.parametrize("supply_v", (1.1, 1.3))
def test_comparator_preflight_accepts_programmable_supply_boundaries(
    tmp_path,
    supply_v: float,
) -> None:
    source = _build_comp_params(
        adc_index=0,
        campaign="comp_common_mode",
        sampling_mode="track",
        sweep_stage="fixed",
        vin_cm_v=0.8,
        vin_diff_v=0.0,
        conversions=8,
    )
    boundary = replace(
        source,
        tb=replace(
            source.tb,
            vdd_a=h.Vdc.Params(dc=supply_v),
            vdd_d=h.Vdc.Params(dc=supply_v),
            vdd_dac=h.Vdc.Params(dc=supply_v),
        ),
    )
    uncalibrated = replace(boundary, tb=replace(boundary.tb, vin_diff=h.Vdc.Params(dc=1.01)))
    run_dir = tmp_path / "not-created"

    with pytest.raises(ValueError, match="calibrated range"):
        run_comp_scan((boundary, uncalibrated), run_dir=run_dir, capture_scope_per_curve=False)
    assert not run_dir.exists()


def test_comparator_preflight_accepts_official_1p2_v_common_mode_endpoint(tmp_path) -> None:
    official = _build_comp_params(
        adc_index=0,
        campaign="comp_common_mode",
        sampling_mode="track",
        sweep_stage="fine",
        vin_cm_v=1.2,
        vin_diff_v=25.0e-3,
        conversions=1_000,
        sweep_min_v=0.0,
        sweep_max_v=25.0e-3,
        sweep_step_v=100.0e-6,
    )
    uncalibrated = replace(
        official,
        tb=replace(official.tb, vin_diff=h.Vdc.Params(dc=1.01)),
        sweep_min_v=None,
        sweep_max_v=None,
        sweep_step_v=None,
    )
    run_dir = tmp_path / "not-created"

    with pytest.raises(ValueError, match="calibrated range"):
        run_comp_scan((official, uncalibrated), run_dir=run_dir, capture_scope_per_curve=False)
    assert not run_dir.exists()


def test_comparator_preflight_rejects_supply_and_fixed_io_before_hardware(tmp_path) -> None:
    params = _build_comp_params(
        adc_index=0,
        campaign="comp_common_mode",
        sampling_mode="track",
        sweep_stage="fixed",
        vin_cm_v=0.8,
        vin_diff_v=0.0,
        conversions=8,
    )
    for invalid, message in (
        (replace(params, tb=replace(params.tb, vdd_a=h.Vdc.Params(dc=1.099))), "VDD_A request"),
        (replace(params, tb=replace(params.tb, vdd_d=h.Vdc.Params(dc=1.301))), "VDD_D request"),
        (replace(params, vdd_io=h.Vdc.Params(dc=1.3)), "VDD_IO is fixed"),
    ):
        run_dir = tmp_path / message.replace(" ", "_")
        with pytest.raises(ValueError, match=message):
            run_comp_scan((invalid,), run_dir=run_dir, capture_scope_per_curve=False)
        assert not run_dir.exists()
