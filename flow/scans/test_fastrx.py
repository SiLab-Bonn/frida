"""Compare physical comparator decisions against the FastRX 17-bit capture.

ADC01 converts a fixed +50 mV differential input at 0.8 V common mode. The
default run tests 80--1600 MBd in 40 MBd steps and LOGIC phase offsets -3..+3.
For each point the analytical capture equation selects RX_SEN and IDELAY,
FastRX records repeated conversions, and the scope records:

    CH1 ADC differential input
    CH2 COMP
    CH3 LOGIC
    CH4 COMP_OUT

The first 17 scope decisions are compared bit-for-bit with the first FastRX
word. Every scope waveform, ADC conversion record, and noise histogram is
saved under a new ``build/test_fastrx/<timestamp>`` directory.

Run the quick physical smoke point:

    uv run pytest -q -s -m "hw and not slow" flow/scans/test_fastrx.py -k physical

Run the complete 273-point physical campaign:

    uv run pytest -q -s -m "hw and slow" flow/scans/test_fastrx.py -k physical
"""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import hdl21 as h
import numpy as np
import pytest
from basil.HL.tektronix_oscilloscope import response_value
from bitarray import bitarray
from yaml import safe_load

from flow.adc import AdcParams
from flow.adc.sim import AdcTbParams
from flow.analysis.adc import (
    analyze_adc_code_distribution,
    analyze_adc_noise_sweep,
    analyze_scope_wave_to_bits,
)
from flow.analysis.io import (
    scope_records_to_adc_wave,
    write_measurement,
)
from flow.analysis.plots import (
    plot_adc_code_distribution,
    plot_adc_fastrx_scope_comparison,
    plot_adc_noise_sweep,
)
from flow.analysis.types import AdcDaq, MeasAdcExt, MeasInfo
from flow.cdac import CdacParams, RedunStrat, get_cdac_weights
from flow.scans.fastrx import calculate_fastrx_capture_alignment, convert_fastrx_words_to_adc
from flow.scans.params import AdcScanParams, load_board_map, validate_params
from flow.scans.plldrp import calculate_pll_frequency, select_pll_configuration, set_pll_divider
from flow.scans.scan_adc import (
    convert_dac_caps_to_adc_weights,
    convert_params_to_spi_fmt,
    convert_vdiff_input_to_awg_supply,
)
from flow.scans.scope import (
    FRIDA_SCOPE_CHANNELS,
    wait_for_scope_armed,
    wait_for_scope_capture,
)
from flow.scans.seqgen import convert_params_to_seqgen_fmt

MAP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "build" / "test_fastrx"

BOARD_ID = "00"
ADC_INDEX = 1
VIN_DIFF_V = 0.050
VIN_CM_V = 0.800
VDD_V = 1.200
SMU_VOLTAGE_RANGE_V = 2.0
SMU_CURRENT_COMPLIANCE_A = 500.0e-6
VIN_CM_CURRENT_LIMIT_A = 10.0e-3

DEFAULT_SYMBOL_RATES_BPS = tuple(rate_mbd * 1.0e6 for rate_mbd in range(80, 1601, 40))
DEFAULT_LOGIC_OFFSETS = (-3, -2, -1, 0, 1, 2, 3)
DEFAULT_CONVERSIONS = 1_000
SMOKE_SYMBOL_RATE_BPS = 800.0e6
SMOKE_LOGIC_OFFSET = 0
SMOKE_CONVERSIONS = 100

SCOPE_TRACKS = {channel: track for track, channel in FRIDA_SCOPE_CHANNELS.items()}
SCOPE_TRIGGER_CHANNEL = FRIDA_SCOPE_CHANNELS["seq_logic"]
SCOPE_RECORD_LENGTH = 10_000
SCOPE_BANDWIDTH_HZ = 2.0e9
SCOPE_VERTICAL_SCALE_V = 0.2
SCOPE_CAPTURE_TIMEOUT_S = 5.0
SCOPE_DOWNLOAD_SETTLE_S = 0.1
SETUP_SETTLE_S = 0.2
SMU_SETTLE_S = 0.5
SI570_SETTLE_S = 0.02
FASTRX_TIMEOUT_S = 5.0
FASTRX_TRAILING_DRAIN_S = 0.01


@pytest.mark.hw
@pytest.mark.parametrize(
    ("symbol_rates_bps", "logic_offsets", "conversions"),
    (
        pytest.param(
            (SMOKE_SYMBOL_RATE_BPS,),
            (SMOKE_LOGIC_OFFSET,),
            SMOKE_CONVERSIONS,
            id="smoke",
        ),
        pytest.param(
            DEFAULT_SYMBOL_RATES_BPS,
            DEFAULT_LOGIC_OFFSETS,
            DEFAULT_CONVERSIONS,
            marks=pytest.mark.slow,
            id="full",
        ),
    ),
)
def test_physical_fastrx_matches_scope(
    symbol_rates_bps: tuple[float, ...],
    logic_offsets: tuple[int, ...],
    conversions: int,
    linux_gpib_interface: None,
) -> None:
    """Hardware: compare physical comparator decisions with FastRX capture."""

    board_map = load_board_map()
    board = board_map["boards"][BOARD_ID]
    timing_model = board["capture_timing_model"]
    flavor_name = board["adc_channels"][ADC_INDEX]
    cap_weights = tuple(board_map["adc_flavors"][flavor_name]["cdac_weights"])
    dut_params = AdcParams(
        adc_bits=12,
        cdac=CdacParams(
            n_dac=11,
            n_extra=5,
            redun_strat=RedunStrat.SUBRDX2_OVLY,
            weights=cap_weights,
        ),
    )
    active_adc_mask = tuple(int(index == ADC_INDEX) for index in reversed(range(16)))
    base_params = AdcScanParams(
        tb=AdcTbParams(
            view="frida1",
            dut=dut_params,
            conversions=conversions,
            vin_cm=h.Vdc.Params(dc=VIN_CM_V),
            vin_diff=h.Vdc.Params(dc=VIN_DIFF_V),
        ),
        board_id=BOARD_ID,
        observed_adc=ADC_INDEX,
        active_adc_mask=active_adc_mask,
    )

    maximum_supply_v = float(board["supply_limits"]["maximum_voltage_v"])
    if not 0.0 < VDD_V <= maximum_supply_v:
        raise ValueError(f"VDD={VDD_V:g} V exceeds the board limit {maximum_supply_v:g} V")
    if abs(VIN_DIFF_V) > float(board["input_calibration"]["maximum_abs_vdiff_v"]):
        raise ValueError("requested differential input exceeds the calibrated range")
    if VIN_CM_V + abs(VIN_DIFF_V) / 2.0 > maximum_supply_v:
        raise ValueError("requested ADC positive input exceeds the 1.2 V rail")
    if VIN_CM_V - abs(VIN_DIFF_V) / 2.0 < 0.0:
        raise ValueError("requested ADC negative input is below ground")

    awg_voltage_v, vin_cm_supply_v = convert_vdiff_input_to_awg_supply(
        VIN_DIFF_V,
        VIN_CM_V,
        board["input_calibration"],
    )
    if not 0.0 <= vin_cm_supply_v <= maximum_supply_v:
        raise ValueError("calibrated Vin_cm supply request is outside 0..1.2 V")

    run_timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving {len(symbol_rates_bps) * len(logic_offsets)} capture points to {run_dir}")

    from gpib_ctypes import make_default_gpib

    make_default_gpib()
    from basil.dut import Dut

    daq_dut = Dut(str(MAP_DIR / "map_fpga.yaml"))
    awg_dut = Dut(str(MAP_DIR / "map_awg.yaml"))
    vin_cm_dut = Dut(str(MAP_DIR / "map_supply.yaml"))
    smu_dut = Dut(str(MAP_DIR / "map_smu.yaml"))
    scope_dut = Dut(str(MAP_DIR / "map_scope.yaml"))
    initialized_duts = []
    daq = awg = vin_cm_supply = scope = None
    smus: list[tuple[Any, str]] = []
    scope_state: dict[str, Any] | None = None
    summary_rows: list[dict[str, Any]] = []
    measurements: list[MeasAdcExt] = []

    try:
        for instrument_dut in (
            daq_dut,
            awg_dut,
            vin_cm_dut,
            smu_dut,
            scope_dut,
        ):
            instrument_dut.init()
            initialized_duts.append(instrument_dut)

        daq = daq_dut
        awg = awg_dut["awg"]
        vin_cm_supply = vin_cm_dut["vocm_supply"]
        scope = scope_dut["scope"]
        smus = [
            (smu_dut["smu1"], "VDD_A"),
            (smu_dut["smu2"], "VDD_D"),
            (smu_dut["smu3"], "VDD_DAC"),
        ]

        # Make every stimulus safe before changing ranges or setpoints.
        awg.set_DC("DEF,DEF,0")
        awg.set_enable(0)
        vin_cm_supply.set_enable(0)
        vin_cm_supply.set_voltage(0.0)
        for smu, _rail in smus:
            smu.off()
            smu.set_voltage(0.0)

        # Configure and enable all three 1.2 V chip domains back-to-back.
        for smu, rail in smus:
            smu.source_volt()
            smu.four_wire_off()
            smu.set_voltage_range(SMU_VOLTAGE_RANGE_V)
            smu.set_current_limit(SMU_CURRENT_COMPLIANCE_A)
            smu.current_sense_autorange_on()
            smu.set_current_nplc(10.0)
            smu.autozero_on()
            smu.set_voltage(VDD_V)
            if float(smu.get_source_voltage()) > VDD_V:
                raise RuntimeError(f"{rail}: unsafe voltage setpoint readback")
            if float(smu.get_current_limit()) > SMU_CURRENT_COMPLIANCE_A:
                raise RuntimeError(f"{rail}: unsafe current-compliance readback")
        for smu, _rail in smus:
            smu.on()
        sleep(SMU_SETTLE_S)
        for smu, rail in smus:
            measured_voltage_v = float(smu.get_voltage())
            measured_current_a = float(smu.get_current())
            print(f"{rail}: {measured_voltage_v:.6f} V, {measured_current_a * 1e6:.3f} uA")
            if not 1.15 <= measured_voltage_v <= VDD_V + 5.0e-3:
                raise RuntimeError(f"{rail}: unsafe or compliance-limited voltage {measured_voltage_v:g} V")

        # Apply the fixed, calibrated differential input and common mode.
        vin_cm_supply.set_voltage_range("P25V")
        vin_cm_supply.set_current_limit(VIN_CM_CURRENT_LIMIT_A)
        vin_cm_supply.set_voltage(vin_cm_supply_v)
        vin_cm_supply.set_enable(1)
        awg.set_output_load("INFinity")
        awg.set_DC(f"DEF,DEF,{awg_voltage_v}")
        awg.set_enable(1)
        sleep(SETUP_SETTLE_S)
        print(
            f"Stimulus: target Vdiff={VIN_DIFF_V:g} V, Vin_cm={VIN_CM_V:g} V; "
            f"AWG={awg_voltage_v:.6f} V, Vin_cm supply={vin_cm_supply_v:.6f} V"
        )

        # Select the physical ADC comparator and sequencer-controlled RX_SEN.
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

        # Program and verify ADC01 once; only FPGA timing changes in the sweep.
        spi_bytes = convert_params_to_spi_fmt(base_params)
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

        # Preserve scope acquisition state. Per-channel settings are
        # intentionally left in the agreed 2 GHz, 200 mV/div, zero-offset form.
        scope_state = {
            "acquire_state": response_value(scope.get_acquire_state()),
            "acquire_stop_after": response_value(scope.get_acquire_stop_after()),
            "acquire_mode": response_value(scope.get_acquire_mode()),
            "horizontal_scale": response_value(scope.get_horizontal_scale()),
            "horizontal_record_length": response_value(scope.get_horizontal_record_length()),
            "horizontal_position": response_value(scope._intf.query("HORizontal:POSition?")),
            "trigger_mode": response_value(scope.get_trigger_mode()),
            "trigger_type": response_value(scope.get_trigger_type()),
            "trigger_source": response_value(scope.get_triggr_source()),
            "trigger_slope": response_value(scope.get_trigger_edge_slope()),
            "trigger_level": response_value(scope.get_trigger_level(channel=SCOPE_TRIGGER_CHANNEL)),
            "display": {
                channel: response_value(scope._intf.query(f"DISplay:GLObal:CH{channel}:STATE?"))
                for channel in SCOPE_TRACKS
            },
        }
        scope.set_acquire_state("STOP")
        scope.set_acquire_mode("SAMPLE")
        scope.set_acquire_stop_after("SEQUENCE")
        scope.set_horizontal_record_length(SCOPE_RECORD_LENGTH)
        for channel in SCOPE_TRACKS:
            scope._intf.write(f"DISplay:GLObal:CH{channel}:STATE ON")
            scope.set_coupling("DC", channel=channel)
            scope.set_vertical_scale(SCOPE_VERTICAL_SCALE_V, channel=channel)
            scope.set_vertical_position(0.0, channel=channel)
            scope.set_vertical_offset(0.0, channel=channel)
            scope.set_bandwidth(SCOPE_BANDWIDTH_HZ, channel=channel)
        scope.set_trigger_type("EDGE")
        scope.set_trigger_source(channel=SCOPE_TRIGGER_CHANNEL)
        scope.set_trigger_edge_slope("RISE")
        scope.set_trigger_level(0.0, channel=SCOPE_TRIGGER_CHANNEL)
        scope.set_trigger_mode("NORMAL")

        cap_weights = get_cdac_weights(base_params.tb.dut.cdac)
        code_weights = convert_dac_caps_to_adc_weights(cap_weights)
        data_size = int(daq["fastrx0"].get_size())
        if data_size != len(code_weights):
            raise RuntimeError(f"FastRX DATA_SIZE={data_size}, expected {len(code_weights)}")
        sequence_words = len(base_params.tb.seq_init_pattern) // 8

        for point_index, (symbol_rate_bps, logic_offset) in enumerate(
            ((symbol_rate_bps, logic_offset) for logic_offset in logic_offsets for symbol_rate_bps in symbol_rates_bps),
            start=1,
        ):
            params = dataclasses.replace(
                base_params,
                tb=dataclasses.replace(
                    base_params.tb,
                    symbol_rate=symbol_rate_bps,
                    seq_logic_phase_delay_symbols=logic_offset,
                ),
            )
            validate_params(params)
            alignment = calculate_fastrx_capture_alignment(
                params.tb,
                **timing_model,
            )
            phase_advance = alignment.control_phase_advance_symbols
            if phase_advance:
                params = dataclasses.replace(
                    params,
                    tb=dataclasses.replace(
                        params.tb,
                        seq_init_phase_delay_symbols=float(params.tb.seq_init_phase_delay_symbols) - phase_advance,
                        seq_samp_phase_delay_symbols=float(params.tb.seq_samp_phase_delay_symbols) - phase_advance,
                        seq_comp_phase_delay_symbols=float(params.tb.seq_comp_phase_delay_symbols) - phase_advance,
                        seq_logic_phase_delay_symbols=float(params.tb.seq_logic_phase_delay_symbols) - phase_advance,
                    ),
                )
                validate_params(params)
            rx_sen_start_word = alignment.rx_sen_start_word
            comp_idelay_taps = alignment.comp_idelay_taps
            rx_sen_stop_word = rx_sen_start_word + len(code_weights)
            rx_sen_pattern = (
                "0" * rx_sen_start_word + "1" * len(code_weights) + "0" * (sequence_words - rx_sen_stop_word)
            )
            sequencer_memory = convert_params_to_seqgen_fmt(
                params.tb,
                rx_sen_pattern,
            )

            si570_frequency_hz, pll_divider_n = select_pll_configuration(symbol_rate_bps)
            _sequencer_frequency_hz, serializer_frequency_hz = calculate_pll_frequency(
                pll_divider_n,
                input_frequency_hz=si570_frequency_hz,
            )
            actual_symbol_rate_bps = 2.0 * serializer_frequency_hz
            if not math.isclose(actual_symbol_rate_bps, symbol_rate_bps, rel_tol=1e-12):
                raise RuntimeError("PLL selection did not reproduce the requested symbol rate")

            print(
                f"\n[{point_index}/{len(symbol_rates_bps) * len(logic_offsets)}] "
                f"{symbol_rate_bps / 1e6:g} MBd, LOGIC {logic_offset:+d}: "
                f"phase advance={phase_advance}, RX_SEN={rx_sen_start_word}, tap={comp_idelay_taps}, "
                f"predicted setup={alignment.setup_margin_s * 1e9:.3f} ns, "
                f"hold={alignment.hold_margin_s * 1e9:.3f} ns"
            )

            daq["si570"].frequency_change(si570_frequency_hz / 1e6)
            sleep(SI570_SETTLE_S)
            set_pll_divider(daq["gpio2"], pll_divider_n)

            # Program comparator IDELAY through the GPIO1 transaction exercised
            # independently by test_gpio.py.
            daq["gpio1"].read()
            if not daq["gpio1"]["COMP_IDELAY_RDY"].tovalue():
                raise RuntimeError("comparator IDELAYCTRL is not ready")
            daq["gpio1"]["COMP_IDELAY_TAPS"] = comp_idelay_taps
            daq["gpio1"]["COMP_IDELAY_LOAD"] = 1
            daq["gpio1"].write()
            daq["gpio1"]["COMP_IDELAY_LOAD"] = 0
            daq["gpio1"].write()

            # Program the 64-bit sequencer memory with the same public Basil
            # calls exercised by test_seqgen.py.
            daq["seq0"].reset()
            daq["fastrx0"].reset()
            sleep(0.001)
            daq["seq0"].set_data(sequencer_memory)
            daq["seq0"].set_size(sequence_words)
            daq["seq0"].set_clk_divide(1)
            daq["seq0"].set_repeat(conversions)
            daq["seq0"].set_en_ext_start(False)
            daq["fastrx0"].set_en(True)
            daq["fifo0"]["RESET"]
            daq["fifo0"].get_data()

            sequence_period_s = len(params.tb.seq_init_pattern) / symbol_rate_bps
            scope.set_acquire_state("STOP")
            # Request about two complete patterns across the ten horizontal
            # divisions. This leaves margin for the scope's discrete timebase
            # rounding at rates such as 240 MBd.
            scope.set_horizontal_scale(sequence_period_s / 5.0)
            actual_horizontal_scale_s = float(response_value(scope.get_horizontal_scale()))
            horizontal_divisions = float(response_value(scope.get_horizontal_divisions()))
            record_span_s = actual_horizontal_scale_s * horizontal_divisions
            post_trigger_s = 1.02 * sequence_period_s
            if post_trigger_s >= record_span_s:
                raise RuntimeError(
                    f"scope span {record_span_s:g} s does not cover one sequence ({sequence_period_s:g} s)"
                )
            horizontal_position_percent = 100.0 * (1.0 - post_trigger_s / record_span_s)
            scope._intf.write(f"HORizontal:POSition {horizontal_position_percent:g}")
            scope._intf.write("ACQuire:NUMACq:RESET")
            scope.set_acquire_state("RUN")
            acquisition_count_before = wait_for_scope_armed(
                scope,
                timeout_s=SCOPE_CAPTURE_TIMEOUT_S,
            )

            capture_deadline = monotonic() + max(
                FASTRX_TIMEOUT_S,
                2.0 * conversions * sequence_period_s + 2.0,
            )
            daq["seq0"].start()
            wait_for_scope_capture(
                scope,
                acquisition_count_before,
                timeout_s=SCOPE_CAPTURE_TIMEOUT_S,
            )
            while not daq["seq0"].is_done():
                if monotonic() >= capture_deadline:
                    raise TimeoutError("sequencer did not finish the requested conversions")
                sleep(0.001)
            expected_fifo_bytes = 4 * conversions
            while int(daq["fifo0"]["FIFO_SIZE"]) < expected_fifo_bytes:
                if monotonic() >= capture_deadline:
                    raise TimeoutError("FastRX did not deliver every requested conversion")
                sleep(0.001)
            sleep(FASTRX_TRAILING_DRAIN_S)
            raw_data = list(daq["fifo0"].get_data())
            if len(raw_data) != conversions:
                raise RuntimeError(f"received {len(raw_data)} FastRX words, expected {conversions}")
            if int(daq["fastrx0"].get_lost_count()):
                raise RuntimeError("FastRX reported lost words")

            sleep(SCOPE_DOWNLOAD_SETTLE_S)
            waveforms = scope.get_waveforms(SCOPE_TRACKS)
            missing_channels = sorted(set(SCOPE_TRACKS).difference(waveforms))
            if missing_channels:
                raise RuntimeError(f"scope did not return channels {missing_channels}")

            stem = (
                f"adc01_{symbol_rate_bps / 1e6:04.0f}mbd_"
                f"logic{logic_offset:+d}_rx{rx_sen_start_word:02d}_"
                f"tap{comp_idelay_taps:02d}"
            )
            fastrx_words = np.asarray(raw_data, dtype=np.uint32)
            bout_values, dout_raw_values, dout_values = convert_fastrx_words_to_adc(
                fastrx_words,
                data_size,
                code_weights,
                params.tb.dut.adc_bits,
            )
            mean_code = float(np.mean(dout_values))
            sigma_code = float(np.std(dout_values))
            measurement = MeasAdcExt(
                info=MeasInfo(
                    schema_version=1,
                    measurement_type="MeasAdcExt",
                    backend="physical",
                    timestamp_utc=datetime.now().astimezone(),
                    instruments={
                        "awg": str(awg.get_name()).strip(),
                        "vin_cm_supply": str(vin_cm_supply.get_name()).strip(),
                        "scope": str(scope.get_name()).strip(),
                    },
                    readbacks={
                        "rx_sen_start_word": rx_sen_start_word,
                        "comp_idelay_taps": comp_idelay_taps,
                        "control_phase_advance_symbols": phase_advance,
                        "scope_comp_out_inverted": False,
                        "predicted_setup_margin_s": alignment.setup_margin_s,
                        "predicted_hold_margin_s": alignment.hold_margin_s,
                    },
                ),
                param=params,
                daq=AdcDaq(
                    conversion_index=np.arange(conversions),
                    bout=bout_values,
                    dout_raw=dout_raw_values,
                    dout=dout_values,
                    vin_diff_v=np.full(conversions, VIN_DIFF_V),
                    fastrx_word=fastrx_words,
                ),
                wave=scope_records_to_adc_wave(
                    [waveforms],
                    [0],
                    {
                        "vin_diff_v": FRIDA_SCOPE_CHANNELS["adc_vdiff"],
                        "seq_comp_v": FRIDA_SCOPE_CHANNELS["seq_comp"],
                        "seq_logic_v": FRIDA_SCOPE_CHANNELS["seq_logic"],
                        "comp_out_v": FRIDA_SCOPE_CHANNELS["comp_out"],
                    },
                ),
            )
            scope_analysis = analyze_scope_wave_to_bits(measurement)
            measurement = dataclasses.replace(
                measurement,
                info=dataclasses.replace(
                    measurement.info,
                    readbacks=(
                        measurement.info.readbacks | {"scope_fastrx_bit_mismatches": scope_analysis.mismatch_count}
                    ),
                ),
            )
            print(
                f"scope={scope_analysis.scope_bit_string}, FastRX={scope_analysis.fastrx_bit_string}, "
                f"mismatches={scope_analysis.mismatch_count}/17; "
                f"mean={mean_code:.3f} LSB, sigma={sigma_code:.3f} LSB"
            )
            h5_path = run_dir / f"{stem}.h5"
            write_measurement(h5_path, measurement)
            measurements.append(measurement)
            plot_adc_fastrx_scope_comparison(
                measurement,
                scope_analysis,
                output_path=run_dir / f"{stem}_scope",
            )
            plot_adc_code_distribution(
                [measurement],
                analyze_adc_code_distribution([measurement]),
                output_path=run_dir / f"{stem}_hist",
            )

            summary_rows.append(
                {
                    "symbol_rate_bps": symbol_rate_bps,
                    "active_conversion_rate_hz": symbol_rate_bps / 160.0,
                    "active_conversion_rate_msps": symbol_rate_bps / 160.0e6,
                    "logic_offset_symbols": logic_offset,
                    "comparator_time_percent": 50.0 + 12.5 * logic_offset,
                    "rx_sen_start_word": rx_sen_start_word,
                    "comp_idelay_taps": comp_idelay_taps,
                    "control_phase_advance_symbols": phase_advance,
                    "predicted_earliest_data_arrival_s": alignment.earliest_data_arrival_s,
                    "predicted_latest_data_arrival_s": alignment.latest_data_arrival_s,
                    "predicted_capture_edge_s": alignment.capture_edge_s,
                    "predicted_setup_margin_s": alignment.setup_margin_s,
                    "predicted_hold_margin_s": alignment.hold_margin_s,
                    "scope_bits": scope_analysis.scope_bit_string,
                    "fastrx_bits": scope_analysis.fastrx_bit_string,
                    "bit_mismatches": scope_analysis.mismatch_count,
                    "mean_dout_lsb": mean_code,
                    "sigma_dout_lsb": sigma_code,
                    "minimum_dout_lsb": int(np.min(dout_values)),
                    "maximum_dout_lsb": int(np.max(dout_values)),
                    "measurement_h5": str(h5_path),
                }
            )

        plot_adc_noise_sweep(
            measurements,
            analyze_adc_noise_sweep(measurements),
            output_path=run_dir / "decision_variation_vs_conversion_rate",
        )

        mismatched_points = [row for row in summary_rows if row["bit_mismatches"]]
        if mismatched_points:
            raise AssertionError(
                f"{len(mismatched_points)}/{len(summary_rows)} points did not "
                f"match all 17 scope and FastRX bits; see {run_dir}"
            )
        print(f"PASS: all {len(summary_rows)} points matched all 17 scope/FastRX bits; measurements: {run_dir}")
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
                sleep(SI570_SETTLE_S)
                set_pll_divider(daq["gpio2"], 2)
            except Exception as error:  # noqa: BLE001 - best-effort safety shutdown
                print(f"WARNING: could not restore FPGA defaults: {error}")
        if awg is not None:
            try:
                awg.set_DC("DEF,DEF,0")
                awg.set_enable(0)
            except Exception as error:  # noqa: BLE001 - best-effort safety shutdown
                print(f"WARNING: could not disable and zero the AWG: {error}")
        if vin_cm_supply is not None:
            try:
                vin_cm_supply.set_enable(0)
                vin_cm_supply.set_voltage(0.0)
            except Exception as error:  # noqa: BLE001 - best-effort safety shutdown
                print(f"WARNING: could not disable and zero Vin_cm: {error}")
        for smu, rail in smus:
            try:
                smu.off()
                smu.set_voltage(0.0)
            except Exception as error:  # noqa: BLE001 - best-effort safety shutdown
                print(f"WARNING: could not disable and zero {rail}: {error}")
        if scope is not None and scope_state is not None:
            try:
                scope.set_acquire_state("STOP")
                scope.set_acquire_mode(scope_state["acquire_mode"])
                scope.set_horizontal_scale(scope_state["horizontal_scale"])
                scope.set_horizontal_record_length(scope_state["horizontal_record_length"])
                scope._intf.write(f"HORizontal:POSition {scope_state['horizontal_position']}")
                scope.set_trigger_mode(scope_state["trigger_mode"])
                scope.set_trigger_type(scope_state["trigger_type"])
                scope._intf.write(f"TRIGger:A:EDGe:SOUrce {scope_state['trigger_source']}")
                scope.set_trigger_edge_slope(scope_state["trigger_slope"])
                scope.set_trigger_level(
                    scope_state["trigger_level"],
                    channel=SCOPE_TRIGGER_CHANNEL,
                )
                for channel in SCOPE_TRACKS:
                    scope.set_vertical_scale(SCOPE_VERTICAL_SCALE_V, channel=channel)
                    scope.set_vertical_position(0.0, channel=channel)
                    scope.set_vertical_offset(0.0, channel=channel)
                    scope.set_bandwidth(SCOPE_BANDWIDTH_HZ, channel=channel)
                    scope._intf.write(f"DISplay:GLObal:CH{channel}:STATE {scope_state['display'][channel]}")
                scope.set_acquire_stop_after(scope_state["acquire_stop_after"])
                scope.set_acquire_state(scope_state["acquire_state"])
            except Exception as error:  # noqa: BLE001 - best-effort state restoration
                print(f"WARNING: could not fully restore scope state: {error}")
        for instrument_dut in reversed(initialized_duts):
            instrument_dut.close()


INTERNAL_EXPECTED_BITS = "10110100101100101"
INTERNAL_TEST_REPEATS = 64
INTERNAL_CAPTURE_TIMEOUT_S = 2.0


@pytest.mark.hw
def test_fastrx_captures_exact_internal_17_bit_pattern() -> None:
    """Hardware: capture every test bit in order with contiguous frame IDs."""

    params = AdcTbParams()
    timing_model = load_board_map()["boards"]["00"]["capture_timing_model"]
    capture_start_words = sorted(
        {
            calculate_fastrx_capture_alignment(
                dataclasses.replace(params, symbol_rate=rate_mbd * 1.0e6),
                **timing_model,
            ).rx_sen_start_word
            for rate_mbd in range(80, 1601, 80)
        }
    )
    sequence_words = len(params.seq_init_pattern) // 8
    first_capture_stop_word = capture_start_words[0] + len(INTERNAL_EXPECTED_BITS)
    first_rx_sen_pattern = (
        "0" * capture_start_words[0]
        + "1" * len(INTERNAL_EXPECTED_BITS)
        + "0" * (sequence_words - first_capture_stop_word)
    )
    memory = convert_params_to_seqgen_fmt(params, first_rx_sen_pattern)

    # Initialize only the hardware blocks participating in this loopback.
    config = safe_load((MAP_DIR / "map_fpga.yaml").read_text())
    driver_names = {"fifo0", "seq0", "gpio0", "fastrx0"}
    register_names = {"seq0", "gpio0"}
    config["hw_drivers"] = [driver for driver in config["hw_drivers"] if driver["name"] in driver_names]
    config["registers"] = [register for register in config["registers"] if register["name"] in register_names]

    from basil.dut import Dut

    daq = Dut(config)
    seq = gpio = None
    original_seq = original_gpio = None
    try:
        daq.init()
        seq = daq["seq0"]
        gpio = daq["gpio0"]
        gpio_fields = (
            "RST_B",
            "AMP_EN",
            "RX_LOOPBACK",
            "SPI_LOOPBACK",
            "DBG_FIFO",
            "RX_TIEHIGH",
            "SEQ_START",
            "RX_EN_MUX",
        )
        gpio.read()
        original_gpio = {field: gpio[field].tovalue() for field in gpio_fields}
        original_seq = {
            "memory": bytes(seq.get_data(size=len(memory))),
            "size": int(seq.get_size()),
            "clk_divide": int(seq.get_clk_divide()),
            "repeat": int(seq.get_repeat()),
            "en_ext_start": int(seq.get_en_ext_start()),
        }

        gpio["RX_LOOPBACK"] = 1
        gpio["DBG_FIFO"] = 0
        gpio["RX_TIEHIGH"] = 0
        gpio["SEQ_START"] = 0
        gpio["RX_EN_MUX"] = 1
        gpio.write()

        for capture_start_word in capture_start_words:
            capture_stop_word = capture_start_word + len(INTERNAL_EXPECTED_BITS)
            rx_sen_pattern = (
                "0" * capture_start_word
                + "1" * len(INTERNAL_EXPECTED_BITS)
                + "0" * (sequence_words - capture_stop_word)
            )
            memory = convert_params_to_seqgen_fmt(params, rx_sen_pattern)
            # Byte lane four carries RX_SEN in bit zero and RX_TEST in bit one.
            for bit_index, bit in enumerate(INTERNAL_EXPECTED_BITS):
                if bit == "1":
                    memory[(capture_start_word + bit_index) * 8 + 4] |= 1 << 1

            seq.reset()
            daq["fastrx0"].reset()
            sleep(0.001)
            seq.set_data(memory)
            seq.set_size(sequence_words)
            seq.set_clk_divide(1)
            seq.set_repeat(INTERNAL_TEST_REPEATS)
            seq.set_en_ext_start(False)
            daq["fastrx0"].set_en(True)
            assert int(daq["fastrx0"].get_size()) == len(INTERNAL_EXPECTED_BITS)

            daq["fifo0"]["RESET"]
            daq["fifo0"].get_data()
            seq.start()
            deadline = monotonic() + INTERNAL_CAPTURE_TIMEOUT_S
            while not seq.is_done():
                if monotonic() >= deadline:
                    raise TimeoutError("sequencer did not finish the FastRX loopback")
                sleep(0.001)

            expected_fifo_bytes = 4 * INTERNAL_TEST_REPEATS
            while int(daq["fifo0"]["FIFO_SIZE"]) < expected_fifo_bytes:
                if monotonic() >= deadline:
                    raise TimeoutError("FastRX did not deliver every loopback frame")
                sleep(0.001)
            sleep(0.01)

            words = list(daq["fifo0"].get_data())
            assert len(words) == INTERNAL_TEST_REPEATS
            expected_data = int(INTERNAL_EXPECTED_BITS, 2)
            for frame_index, word in enumerate(words):
                identifier, frame, data = daq["fastrx0"].parse_word(int(word))
                assert identifier == 1
                assert frame == frame_index
                assert data == expected_data, (
                    f"start word {capture_start_word}, frame {frame_index}: "
                    f"captured {data:017b}, expected {INTERNAL_EXPECTED_BITS}"
                )
            assert int(daq["fastrx0"].get_lost_count()) == 0
    finally:
        try:
            if seq is not None and original_seq is not None:
                seq.reset()
                seq.set_data(original_seq["memory"])
                seq.set_size(original_seq["size"])
                seq.set_clk_divide(original_seq["clk_divide"])
                seq.set_repeat(original_seq["repeat"])
                seq.set_en_ext_start(original_seq["en_ext_start"])
            if gpio is not None and original_gpio is not None:
                for field, value in original_gpio.items():
                    gpio[field] = value
                gpio.write()
            if "daq" in locals():
                daq["fastrx0"].reset()
                daq["fifo0"]["RESET"]
                daq["fifo0"].get_data()
        finally:
            daq.close()
