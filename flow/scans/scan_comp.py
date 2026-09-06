"""Acquire physical comparator S-curves as fixed-input typed HDF5 points.

The public :func:`scan` function acquires a supplied list of complete
``AdcScanParams`` objects. Named hardware campaigns live in
``flow.scans.runner``.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
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
from flow.analysis.types import CompDaq, CompExtWave, MeasCompExt, MeasInfo
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


def _comp_curve_key(params: AdcScanParams) -> tuple[Any, ...]:
    """Identify one S-curve while excluding its input and acquisition stage."""

    return (
        params.board_id,
        params.observed_adc,
        params.campaign,
        params.sampling_mode,
        float(params.tb.vin_cm.dc),
        None if params.requested_dac_rail_percent is None else float(params.requested_dac_rail_percent),
        params.tb.dac_mode,
        params.tb.dac_diffcaps,
        float(params.settling_time_s),
        params.tb.dac_astate_p,
        params.tb.dac_bstate_p,
        params.tb.dac_astate_n,
        params.tb.dac_bstate_n,
    )


def _comp_point_stem(params: AdcScanParams) -> str:
    """Return the stable filename portion used for resuming one point."""

    assert params.board_id is not None
    assert params.observed_adc is not None
    assert isinstance(params.tb.vin_diff, h.Vdc.Params)
    rail_label = ""
    if params.requested_dac_rail_percent is not None:
        rail_label = f"_pcpl{float(params.requested_dac_rail_percent):06.2f}pct"
    return (
        (
            f"{params.board_id}_adc{params.observed_adc:02d}_{params.campaign}_"
            f"{params.sampling_mode}_mode{params.tb.dac_mode}_diff{params.tb.dac_diffcaps}_"
            f"settle{float(params.settling_time_s) * 1e9:06.2f}ns_{params.sweep_stage}{rail_label}_"
            f"vcm{float(params.tb.vin_cm.dc) * 1e3:07.2f}mv_"
            f"vdiff{float(params.tb.vin_diff.dc) * 1e3:+08.3f}mv"
        )
        .replace("+", "p")
        .replace("-", "m")
    )


def _build_comp_params(
    *,
    adc_index: int,
    campaign: str,
    sampling_mode: str,
    sweep_stage: str,
    vin_cm_v: float,
    vin_diff_v: float,
    conversions: int,
    dac_state_p: tuple[int, ...] | None = None,
    dac_state_n: tuple[int, ...] | None = None,
    requested_dac_rail_percent: float | None = None,
    settling_time_s: float = 0.0,
    sweep_min_v: float | None = None,
    sweep_max_v: float | None = None,
    sweep_step_v: float | None = None,
) -> AdcScanParams:
    """Compose one complete physical comparator point without hardware I/O."""

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
    if dac_state_p is None:
        from flow.scans.scan_cdac import _convert_dac_rail_percent_to_codes

        dac_state_p = tuple(int(bit) for bit in _convert_dac_rail_percent_to_codes(50.0, cap_weights)[0])
    if dac_state_n is None:
        dac_state_n = dac_state_p

    symbol_rate_bps = 1.6e9
    params = AdcScanParams(
        tb=AdcTbParams(
            view="frida1",
            dut=dut,
            symbol_rate=symbol_rate_bps,
            conversions=conversions,
            dac_mode=0,
            dac_astate_p=dac_state_p,
            dac_bstate_p=dac_state_p,
            dac_astate_n=dac_state_n,
            dac_bstate_n=dac_state_n,
            vin_cm=h.Vdc.Params(dc=vin_cm_v),
            vin_diff=h.Vdc.Params(dc=vin_diff_v),
        ),
        board_id=board_id,
        observed_adc=adc_index,
        active_adc_mask=tuple(int(index == adc_index) for index in reversed(range(16))),
        campaign=campaign,
        sampling_mode=sampling_mode,
        sweep_stage=sweep_stage,
        sweep_min_v=sweep_min_v,
        sweep_max_v=sweep_max_v,
        sweep_step_v=sweep_step_v,
        requested_dac_rail_percent=requested_dac_rail_percent,
        settling_time_s=settling_time_s,
    )
    sequence_words = len(params.tb.seq_init_pattern) // 8
    word_period_s = 8.0 / symbol_rate_bps
    settle_words = math.ceil(settling_time_s / word_period_s)
    init_words = ["00000000"] * sequence_words
    logic_words = ["00000000"] * sequence_words
    comp_words = ["00000000"] * sequence_words
    if sampling_mode == "track":
        samp_words = ["11111111"] * sequence_words
        comp_word = 8 + settle_words
    elif sampling_mode == "hold":
        samp_words = ["00000000"] * sequence_words
        samp_words[1:4] = ["11111111"] * 3
        sampling_reopen_word = 4
        comp_word = sampling_reopen_word + settle_words
    else:
        raise ValueError("sampling_mode must be 'track' or 'hold'")
    if comp_word >= sequence_words - 2:
        raise ValueError("settling_time_s leaves no room for comparator capture")
    comp_words[comp_word] = "00001111"

    params = replace(
        params,
        tb=replace(
            params.tb,
            seq_init_pattern="".join(init_words),
            seq_samp_pattern="".join(samp_words),
            seq_comp_pattern="".join(comp_words),
            seq_logic_pattern="".join(logic_words),
        ),
    )
    validate_params(params)
    return params


def _fixed_grid_values(minimum_v: float, maximum_v: float, step_v: float) -> tuple[float, ...]:
    """Return one finite inclusive hardware-scan grid."""

    if not all(math.isfinite(value) for value in (minimum_v, maximum_v, step_v)):
        raise ValueError("comparator scan grid values must be finite")
    if minimum_v > maximum_v:
        raise ValueError("comparator scan minimum must not exceed its maximum")
    if step_v <= 0.0:
        raise ValueError("comparator scan step must be positive")
    steps = (maximum_v - minimum_v) / step_v
    if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("comparator scan endpoints must align to its step")
    return tuple(minimum_v + index * step_v for index in range(round(steps) + 1))


def build_common_mode_variants(
    *,
    adc_indices: Sequence[int],
    common_mode_values_v: Sequence[float],
    minimum_v: float,
    maximum_v: float,
    step_v: float,
    conversions: int,
) -> list[AdcScanParams]:
    """Build fixed-grid comparator common-mode S-curves."""

    vin_diff_values_v = _fixed_grid_values(minimum_v, maximum_v, step_v)
    variants = []
    # Keep the slow GPIB common-mode supply fixed while all four ADCs are
    # measured at one Vin_cm value.
    for vin_cm_v in common_mode_values_v:
        for adc_index in adc_indices:
            for vin_diff_v in vin_diff_values_v:
                variants.append(
                    _build_comp_params(
                        adc_index=adc_index,
                        campaign="comp_common_mode",
                        sampling_mode="track",
                        sweep_stage="fine",
                        vin_cm_v=vin_cm_v,
                        vin_diff_v=vin_diff_v,
                        conversions=conversions,
                        sweep_min_v=minimum_v,
                        sweep_max_v=maximum_v,
                        sweep_step_v=step_v,
                    )
                )
    return variants


def build_sampling_noise_variants(
    *,
    adc_indices: Sequence[int],
    coupling_percentages: Sequence[float],
    vin_cm_v: float,
    minimum_v: float,
    maximum_v: float,
    step_v: float,
    conversions: int,
    selected_curves: Collection[tuple[int, float, str]] | None = None,
) -> list[AdcScanParams]:
    """Build all or selected complementary-CDAC track/hold S-curves."""

    from flow.scans.scan_cdac import _convert_dac_rail_percent_to_codes

    board_map = load_board_map()
    board = board_map["boards"]["00"]
    vin_diff_values_v = _fixed_grid_values(minimum_v, maximum_v, step_v)
    variants = []
    for adc_index in adc_indices:
        flavor = board["adc_channels"][adc_index]
        weights = tuple(board_map["adc_flavors"][flavor]["cdac_weights"])
        for coupling_percent_p in coupling_percentages:
            state_p = tuple(int(bit) for bit in _convert_dac_rail_percent_to_codes(coupling_percent_p, weights)[0])
            if coupling_percent_p == 50.0:
                state_n = state_p
            else:
                state_n = tuple(1 - bit for bit in state_p)
            for sampling_mode in ("track", "hold"):
                for vin_diff_v in vin_diff_values_v:
                    variants.append(
                        _build_comp_params(
                            adc_index=adc_index,
                            campaign="comp_sampling_noise",
                            sampling_mode=sampling_mode,
                            sweep_stage="fine",
                            vin_cm_v=vin_cm_v,
                            vin_diff_v=vin_diff_v,
                            conversions=conversions,
                            dac_state_p=state_p,
                            dac_state_n=state_n,
                            requested_dac_rail_percent=coupling_percent_p,
                            settling_time_s=20.0e-9 if sampling_mode == "hold" else 0.0,
                            sweep_min_v=minimum_v,
                            sweep_max_v=maximum_v,
                            sweep_step_v=step_v,
                        )
                    )
    if selected_curves is None:
        return variants
    selected_curves = set(selected_curves)
    filtered = [
        params
        for params in variants
        if (
            params.observed_adc,
            float(params.requested_dac_rail_percent),
            params.sampling_mode,
        )
        in selected_curves
    ]
    observed_curves = {
        (
            params.observed_adc,
            float(params.requested_dac_rail_percent),
            params.sampling_mode,
        )
        for params in filtered
    }
    if observed_curves != selected_curves:
        raise ValueError("selected comparator sampling-noise curves are not in the campaign")
    return filtered


def scan(
    variants: Sequence[AdcScanParams],
    *,
    run_dir: Path,
    capture_scope_per_curve: bool = True,
) -> Path:
    """Acquire or resume complete comparator parameters in one run directory."""

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
    scope_tracks = {"vin_diff_v": 1, "seq_comp_v": 2, "comp_out_v": 4}

    queue = list(variants)
    if not isinstance(capture_scope_per_curve, bool):
        raise TypeError("capture_scope_per_curve must be boolean")
    if not queue:
        raise ValueError("comparator scan requires at least one parameter variant")
    for params in queue:
        validate_params(params)
        if not params.campaign.startswith("comp_"):
            raise ValueError(f"scan_comp cannot run campaign {params.campaign!r}")
        if params.board_id is None or params.observed_adc is None or params.active_adc_mask is None:
            raise ValueError("every physical comparator point must select its board and ADC")
        if not isinstance(params.tb.vin_diff, h.Vdc.Params):
            raise TypeError("physical comparator S-curves require fixed DC inputs")
        if params.sweep_stage == "coarse":
            raise ValueError("adaptive coarse sweeps belong to scan_cdac; comparator scans require explicit points")

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
        raise ValueError("one comparator run requires one board, clock rate, and supply configuration")
    assert first.board_id is not None
    board = load_board_map()["boards"][first.board_id]
    supply_limits = board["supply_limits"]
    minimum_supply_v = float(supply_limits["minimum_voltage_v"])
    maximum_supply_v = float(supply_limits["maximum_voltage_v"])
    signal_headroom_v = float(supply_limits["signal_headroom_v"])
    fixed_vdd_io_v = float(board["fixed_vdd_io_v"])
    if not math.isclose(float(first.vdd_io.dc), fixed_vdd_io_v, abs_tol=1.0e-12):
        raise ValueError(
            f"VDD_IO is fixed at {fixed_vdd_io_v:g} V on {first.board_id}; "
            f"comparator run requests {float(first.vdd_io.dc):g} V"
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
                    f"comparator inputs {(vin_p_v, vin_n_v)} V are outside {minimum_input_v:g}..{maximum_input_v:g} V"
                )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    existing_paths: dict[str, Path] = {}
    scope_captured_curves: set[tuple[Any, ...]] = set()
    drift_checkpoint_curves: set[tuple[Any, ...]] = set()
    for path in sorted(run_dir.glob("*.h5")):
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasCompExt):
            raise TypeError(f"comparator run directory contains {type(measurement).__name__}: {path}")
        stem = _comp_point_stem(measurement.param)
        if stem in existing_paths:
            raise ValueError(f"comparator run directory contains duplicate point {stem}")
        existing_paths[stem] = path
        key = _comp_curve_key(measurement.param)
        if measurement.wave is not None:
            scope_captured_curves.add(key)
        if float(measurement.info.readbacks.get("capture_batch_interval_s", 0.0)) > 0.0:
            drift_checkpoint_curves.add(key)

    requested_stems = [_comp_point_stem(params) for params in queue]
    if len(requested_stems) != len(set(requested_stems)):
        raise ValueError("comparator campaign contains duplicate parameter points")

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
                scope.set_vertical_scale(0.01 if signal_name == "vin_diff_v" else 0.2, channel=channel)
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
            curve_key = _comp_curve_key(original_params)
            point_stem = _comp_point_stem(original_params)
            if point_stem in existing_paths:
                continue
            scan_params = original_params
            params = scan_params.tb
            assert scan_params.observed_adc is not None
            assert isinstance(params.vin_diff, h.Vdc.Params)
            vin_diff_v = float(params.vin_diff.dc)
            vin_cm_v = float(params.vin_cm.dc)

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
                    raise TimeoutError("one-shot comparator A-state setup did not finish")
                sleep(0.001)

            rx_sen_pattern = "0" * rx_sen_start_word + "1" + "0" * (sequence_words - rx_sen_start_word - 1)
            sequencer_memory = convert_params_to_seqgen_fmt(params, rx_sen_pattern)
            daq["seq0"].reset()
            daq["fastrx0"].reset()
            sleep(0.001)
            daq["seq0"].set_data(sequencer_memory)
            daq["seq0"].set_size(len(params.seq_init_pattern) // 8)
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
            if params.sweep_stage == "fine":
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
                        raise TimeoutError(f"sequencer did not finish {batch_trials} comparator trials")
                    sleep(0.001)
                expected_fifo_bytes = 4 * batch_trials
                while int(daq["fifo0"]["FIFO_SIZE"]) < expected_fifo_bytes:
                    if monotonic() >= deadline:
                        raise TimeoutError("FastRX did not deliver every comparator word")
                    sleep(0.001)
                raw_batch = daq["fifo0"].get_data()
                if len(raw_batch) != batch_trials:
                    raise RuntimeError(f"expected {batch_trials} FastRX words, received {len(raw_batch)}")
                raw_batches.extend(raw_batch)
                if params.sweep_stage == "fine" and batch_index == 0:
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
                raise RuntimeError(f"FastRX lost {fastrx_lost_count} comparator words")

            fastrx_words = np.asarray(raw_data, dtype=np.uint32)
            decisions, frames = convert_fastrx_words_to_comp(fastrx_words, data_size=data_size)
            trial_index = np.arange(params.conversions, dtype=np.int64)
            wave = None
            if scope_waveforms is not None:
                reference = scope_waveforms[scope_tracks["vin_diff_v"]]
                scope_time_s = reference.x_scale.offset + np.arange(len(reference.data)) * reference.x_scale.slope
                wave_signals = {
                    name: np.asarray(scope_waveforms[channel].data, dtype=np.float64)[None, :]
                    for name, channel in scope_tracks.items()
                }
                wave = CompExtWave(
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
                "vin_cm_supply_set_v": vin_cm_supply_v,
                "vin_cm_supply_measured_v": active_vin_cm_measured_v,
                "vin_cm_supply_measured_a": active_vin_cm_measured_a,
                "awg_voltage_set_v": awg_voltage_v,
                "awg_readback_checked": awg_readback_checked,
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
            measurement = MeasCompExt(
                info=MeasInfo(
                    schema_version=2,
                    measurement_type="MeasCompExt",
                    backend="physical",
                    timestamp_utc=datetime.now().astimezone(),
                    instruments=instrument_identities,
                    readbacks=readbacks,
                ),
                param=scan_params,
                daq=CompDaq(
                    trial_index=trial_index,
                    vin_diff_v=np.full(params.conversions, vin_diff_v),
                    vin_cm_v=np.full(params.conversions, vin_cm_v),
                    decision=decisions,
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
                f"P(decision=1)={float(np.mean(decisions)):.4f} "
                f"elapsed={monotonic() - point_started:.3f}s: {h5_path}"
            )
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
