"""Software-only tests for pure scan helpers; no hardware I/O is performed."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import hdl21 as h
import numpy as np
import pytest
from bitarray import bitarray

from flow.adc import AdcParams
from flow.adc.sim import AdcTbParams
from flow.analysis.plots import plot_waveforms
from flow.analysis.waveform import analyze_scope_waveforms
from flow.cdac import CdacParams, RedunStrat
from flow.scans import fastrx, scan_adc, scan_adc_noctl, seqgen
from flow.scans.params import AdcScanParams, build_adc_variants, load_board_map
from flow.scans.scope import FRIDA_SCOPE_CHANNELS, write_scope_csv
from flow.scans.test_diffamp import OUTPUT_DIR as DIFFAMP_OUTPUT_DIR
from flow.scans.test_diffamp import SCOPE_TRACKS as DIFFAMP_SCOPE_TRACKS
from flow.scans.test_diffamp import calculate_refitted_input_calibration
from flow.scans.test_fastrx import OUTPUT_DIR as FASTRX_OUTPUT_DIR
from flow.scans.test_fastrx import SCOPE_TRACKS as FASTRX_SCOPE_TRACKS
from flow.scans.test_noise import OUTPUT_DIR as NOISE_OUTPUT_DIR
from flow.scans.test_noise import SCOPE_TRACKS as NOISE_SCOPE_TRACKS
from flow.scans.test_serdes import OUTPUT_DIR as SERDES_OUTPUT_DIR
from flow.scans.test_serdes import SCOPE_TRACKS as SERDES_SCOPE_TRACKS


def serializer_params(**overrides) -> AdcTbParams:
    """Return a compact four-word parameter set for sequencer packing."""
    config = {
        "dut": AdcParams(
            cdac=CdacParams(n_dac=1, n_extra=0, redun_strat=RedunStrat.RDX2, weights=(1,)),
        ),
        "seq_init_pattern": "10000001" + "01010101" + "00000000" * 2,
        "seq_samp_pattern": "11110000" + "00000000" * 3,
        "seq_comp_pattern": "00001111" + "11111111" + "00000000" * 2,
        "seq_logic_pattern": "10101010" + "01010101" + "00000000" * 2,
    }
    config.update(overrides)
    return AdcTbParams(**config)


def test_frida_scope_channels_match_fixed_bench_cabling() -> None:
    assert FRIDA_SCOPE_CHANNELS == {
        "adc_vdiff": 1,
        "seq_comp": 2,
        "seq_logic": 3,
        "comp_out": 4,
    }
    assert DIFFAMP_SCOPE_TRACKS == {1: "vdiff_ch1"}
    assert NOISE_SCOPE_TRACKS == {1: "vdiff_ch1"}
    assert FASTRX_SCOPE_TRACKS == {
        1: "adc_vdiff",
        2: "seq_comp",
        3: "seq_logic",
        4: "comp_out",
    }
    assert SERDES_SCOPE_TRACKS == {
        2: "seq_comp",
        3: "seq_logic",
    }


def test_hardware_output_directories_match_test_modules() -> None:
    output_directories = {
        DIFFAMP_OUTPUT_DIR,
        NOISE_OUTPUT_DIR,
        FASTRX_OUTPUT_DIR,
        SERDES_OUTPUT_DIR,
    }
    assert {path.name for path in output_directories} == {
        "test_diffamp",
        "test_noise",
        "test_fastrx",
        "test_serdes",
    }
    assert {path.parent.name for path in output_directories} == {"build"}


def spi_params(**overrides) -> AdcScanParams:
    """Return physical parameters with independently recognizable SPI fields."""
    scan_config = {
        "board_id": "00",
        "observed_adc": 3,
        "active_adc_mask": tuple(int(index in (3, 9)) for index in reversed(range(16))),
    }
    tb_config = {
        "view": "frida1",
        "dac_astate_p": tuple(int(bit) for bit in "1010101010101010"),
        "dac_bstate_p": tuple(int(bit) for bit in "0101010101010101"),
        "dac_astate_n": tuple(int(bit) for bit in "1111000011110000"),
        "dac_bstate_n": tuple(int(bit) for bit in "0000111100001111"),
        "en_init": 1,
        "en_samp_p": 0,
        "en_samp_n": 1,
        "en_comp": 1,
        "en_update": 0,
        "dac_mode": 0,
        "dac_diffcaps": 1,
    }
    for name, value in overrides.items():
        if name in scan_config:
            scan_config[name] = value
        else:
            tb_config[name] = value
    return AdcScanParams(tb=AdcTbParams(**tb_config), **scan_config)


def unpack_spi_payload(payload: bytes) -> bitarray:
    """Undo wire-order reversal so tests can inspect logical SPI register positions."""
    transmitted = bitarray()
    transmitted.frombytes(payload)
    assert transmitted[180:].to01() == "0000"
    return transmitted[:180][::-1]


def test_write_scope_csv_persists_raw_aligned_acquisition(tmp_path) -> None:
    """Raw acquisition writer preserves time, voltage, and instrument codes."""

    x_scale = SimpleNamespace(offset=-1.0e-9, slope=0.5e-9, unit="s")
    waveforms = {
        1: SimpleNamespace(
            x_scale=x_scale,
            data=np.asarray([0.1, 0.2, 0.3]),
            raw_data=np.asarray([10, 20, 30]),
        ),
        3: SimpleNamespace(
            x_scale=x_scale,
            data=np.asarray([1.0, 0.0, 1.0]),
            raw_data=np.asarray([100, 0, 100]),
        ),
    }
    path = tmp_path / "scope" / "capture.csv"

    assert write_scope_csv(path, waveforms, {1: "input", 3: "logic"}) == path
    assert path.read_text().splitlines() == [
        "time_s,input_v,logic_v,input_raw,logic_raw",
        "-1e-09,0.1,1.0,10,100",
        "-5e-10,0.2,0.0,20,0",
        "0.0,0.3,1.0,30,100",
    ]
    assert plot_waveforms(
        analyze_scope_waveforms(waveforms, {1: "input", 3: "logic"}),
        output_path=tmp_path / "scope" / "capture",
    ) == (tmp_path / "scope" / "capture.pdf",)


def test_convert_params_to_seqgen_fmt_packs_serializer_lanes() -> None:
    """Verify critical 64-bit lane ordering, control placement, and zero padding."""
    memory = seqgen.convert_params_to_seqgen_fmt(serializer_params(), "0110")

    assert list(memory) == [
        0x81,
        0x0F,
        0xF0,
        0x55,
        0x00,
        0x00,
        0x00,
        0x00,
        0xAA,
        0x00,
        0xFF,
        0xAA,
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
    ]


def test_convert_params_to_seqgen_fmt_rejects_invalid_capture_and_phase() -> None:
    with pytest.raises(TypeError, match="binary string"):
        seqgen.convert_params_to_seqgen_fmt(serializer_params(), [0, 1, 1, 0])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="must contain 4"):
        seqgen.convert_params_to_seqgen_fmt(serializer_params(), "010")
    with pytest.raises(ValueError, match="only zero and one"):
        seqgen.convert_params_to_seqgen_fmt(serializer_params(), "01x0")
    with pytest.raises(ValueError, match="leave a low word"):
        seqgen.convert_params_to_seqgen_fmt(serializer_params(), "0011")

    with pytest.raises(ValueError, match="whole number"):
        seqgen.convert_params_to_seqgen_fmt(
            serializer_params(seq_comp_phase_delay_symbols=0.5),
            "0110",
        )


def test_convert_vdiff_input_to_awg_supply_applies_empirical_calibration() -> None:
    """Check the software-only amplitude, center, and supply calibration."""
    positive_awg, supply_v = scan_adc.convert_vdiff_input_to_awg_supply(
        1.0,
        0.600,
    )
    negative_awg, negative_peak_supply_v = scan_adc.convert_vdiff_input_to_awg_supply(-1.0, 0.600)
    assert positive_awg == pytest.approx(2.145589331136278)
    assert negative_awg == pytest.approx(-2.1816474284978455)
    assert supply_v == pytest.approx(0.5811476)
    assert negative_peak_supply_v == pytest.approx(supply_v)
    assert positive_awg > negative_awg

    # The endpoint midpoint is the fitted AWG center, while their separation
    # supplies the requested sine amplitude. Both depend on |Vdiff| and Vin_cm.
    assert (positive_awg + negative_awg) / 2.0 == pytest.approx(-0.0180290486807837)
    assert positive_awg - negative_awg == pytest.approx(4.327236759634124)

    zero_awg, low_common_mode_supply_v = scan_adc.convert_vdiff_input_to_awg_supply(0.0, 0.400)
    assert zero_awg == pytest.approx(-0.04582207028177367)
    assert low_common_mode_supply_v == pytest.approx(0.3806216)

    # Small DC inputs use the bounded common-mode table, including linear
    # interpolation between physically measured nodes. Larger signals retain
    # the original polynomial and cannot extrapolate beyond 0.4--0.8 V.
    small_positive_awg, _ = scan_adc.convert_vdiff_input_to_awg_supply(0.02, 0.4)
    interpolated_zero_awg, interpolated_supply_v = scan_adc.convert_vdiff_input_to_awg_supply(0.0, 0.45)
    assert small_positive_awg == pytest.approx(0.0006974289182263279)
    assert interpolated_zero_awg == pytest.approx(-0.04587662749001531)
    assert interpolated_supply_v == pytest.approx(0.43078645)
    with pytest.raises(ValueError, match="large-signal input calibration range"):
        scan_adc.convert_vdiff_input_to_awg_supply(0.026, 0.3)


def test_convert_vdiff_input_to_awg_supply_is_continuous_at_calibration_boundaries() -> None:
    """The small- and large-signal calibration models must not reset a sweep."""

    for vin_cm_v in (0.4, 0.6, 0.8):
        for boundary_v in (0.025, 0.2):
            for sign in (-1.0, 1.0):
                below, _ = scan_adc.convert_vdiff_input_to_awg_supply(
                    sign * (boundary_v - 1.0e-9),
                    vin_cm_v,
                )
                above, _ = scan_adc.convert_vdiff_input_to_awg_supply(
                    sign * (boundary_v + 1.0e-9),
                    vin_cm_v,
                )
                assert above == pytest.approx(below, abs=1.0e-7)

        requested_vdiff = np.linspace(-0.25, 0.25, 1001)
        awg_setpoints = np.asarray(
            [scan_adc.convert_vdiff_input_to_awg_supply(value, vin_cm_v)[0] for value in requested_vdiff]
        )
        assert np.all(np.diff(awg_setpoints) > 0.0)


def test_input_calibration_refit_recovers_awg_and_supply_mapping() -> None:
    magnitude_coefficients = np.asarray((2.2, 0.1, -0.02, 0.03, -0.01, 0.04))
    center_coefficients = np.asarray((-0.03, 0.02, 0.01, -0.04, 0.015, 0.025))
    supply_offset = -0.012
    supply_gain = 0.997
    rows = []
    for amplitude_v, vin_cm_v in (
        (0.1, 0.1),
        (0.1, 0.3),
        (0.1, 0.6),
        (0.1, 0.9),
        (0.2, 0.4),
        (0.2, 0.8),
        (0.5, 0.4),
        (0.5, 0.8),
        (1.0, 0.6),
    ):
        delta_v = vin_cm_v - 0.6
        basis = np.asarray((1.0, amplitude_v, amplitude_v**2, delta_v, amplitude_v * delta_v, delta_v**2))
        required_gain = float(basis @ magnitude_coefficients)
        required_center = float(basis @ center_coefficients)
        programmed_center = -0.02
        programmed_half_amplitude = 2.0 * amplitude_v
        measured_half_amplitude = programmed_half_amplitude / required_gain
        measured_offset = (programmed_center - required_center) / required_gain
        supply_set_v = supply_offset + supply_gain * vin_cm_v
        rows.append(
            {
                "target_vdiff_peak_v": amplitude_v,
                "target_vin_cm_v": vin_cm_v,
                "awg_amplitude_vpp": 2.0 * programmed_half_amplitude,
                "awg_offset_v": programmed_center,
                "measured_vdiff_vpp": 2.0 * measured_half_amplitude,
                "measured_vdiff_offset_v": measured_offset,
                "vin_cm_supply_set_v": supply_set_v,
                "vin_cm_supply_read_v": vin_cm_v,
            }
        )

    result = calculate_refitted_input_calibration(rows, awg_calibration_vin_cm_v=0.6)
    np.testing.assert_allclose(result["awg_vdiff_magnitude_coefficients"], magnitude_coefficients)
    np.testing.assert_allclose(result["awg_center_coefficients"], center_coefficients)
    assert result["vin_cm_supply_offset_v"] == pytest.approx(supply_offset)
    assert result["vin_cm_supply_gain"] == pytest.approx(supply_gain)


def test_convert_dac_caps_to_adc_weights_adds_terminal_decision() -> None:
    assert scan_adc.convert_dac_caps_to_adc_weights([8, 4, 2, 1]) == [16, 8, 4, 2, 1]
    with pytest.raises(ValueError, match="positive integers"):
        scan_adc.convert_dac_caps_to_adc_weights([8, 0, 1])


def test_convert_params_to_spi_fmt_places_fields_and_reverses_for_transmission(capsys) -> None:
    """Verify the critical 180-bit field map, ADC selection, and wire-order reversal."""
    params = spi_params()
    payload = scan_adc.convert_params_to_spi_fmt(params)
    logical = unpack_spi_payload(payload)

    assert len(payload) == 23
    assert logical[48:64][::-1].to01() == "".join(str(bit) for bit in params.tb.dac_astate_p)
    assert logical[32:48][::-1].to01() == "".join(str(bit) for bit in params.tb.dac_bstate_p)
    assert logical[16:32][::-1].to01() == "".join(str(bit) for bit in params.tb.dac_astate_n)
    assert logical[0:16][::-1].to01() == "".join(str(bit) for bit in params.tb.dac_bstate_n)

    for adc in range(16):
        base = 64 + 7 * adc
        expected = "1011001" if adc in (3, 9) else "0000000"
        assert logical[base : base + 7].to01() == expected

    assert logical[176:180][::-1].to01() == "0011"
    assert "MUX_SEL spi_bits[179:176]" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"dac_astate_p": (0,) * 15}, "exactly sixteen"),
        ({"observed_adc": 16}, "observed_adc must be in 0..15"),
        ({"active_adc_mask": (0,) * 16}, "must include observed_adc"),
    ],
)
def test_convert_params_to_spi_fmt_rejects_invalid_fields(override: dict[str, object], message: str) -> None:
    """Reject malformed DAC, ADC, and mux fields before generating an SPI payload."""
    with pytest.raises(ValueError, match=message):
        scan_adc.convert_params_to_spi_fmt(spi_params(**override))


def test_convert_fastrx_words_to_adc_preserves_bit_order_and_validates_headers() -> None:
    """Vector decoding preserves bit order, weights, frames, and identifiers."""

    data_size = 4
    code_weights = [8, 4, 2, 1]
    spi_values = (0b1011, 0b0101, 0b1110)
    words = np.asarray(
        [(1 << 28) | (frame << data_size) | spi for frame, spi in enumerate(spi_values)],
        dtype=np.uint32,
    )

    bout, dout_raw, dout = fastrx.convert_fastrx_words_to_adc(
        words,
        data_size,
        code_weights,
        adc_bits=12,
    )

    assert ["".join(str(bit) for bit in row) for row in bout] == ["1011", "0101", "1110"]
    assert dout_raw.tolist() == [11, 5, 14]
    assert dout.tolist() == [
        scan_adc.convert_dout_to_normalized_dout(raw, code_weights, adc_bits=12) for raw in (11, 5, 14)
    ]

    invalid_identifier = words.copy()
    invalid_identifier[1] &= np.uint32(0x0FFFFFFF)
    with pytest.raises(RuntimeError, match="conversion 1 has identifier"):
        fastrx.convert_fastrx_words_to_adc(
            invalid_identifier,
            data_size,
            code_weights,
            adc_bits=12,
        )

    invalid_frame = words.copy()
    invalid_frame[2] ^= np.uint32(1 << data_size)
    with pytest.raises(RuntimeError, match="conversion 2 has frame"):
        fastrx.convert_fastrx_words_to_adc(
            invalid_frame,
            data_size,
            code_weights,
            adc_bits=12,
        )


def test_convert_dout_to_normalized_dout_scales_to_twelve_bits() -> None:
    """Pin down ADC normalization endpoints and Python rounding behavior."""
    assert scan_adc.convert_dout_to_normalized_dout(0, [1, 1], adc_bits=12) == 0
    assert scan_adc.convert_dout_to_normalized_dout(2, [1, 1], adc_bits=12) == 4095
    assert scan_adc.convert_dout_to_normalized_dout(1, [1, 1], adc_bits=12) == 2048


def test_adc_preflight_rejects_input_beyond_headroom_before_hardware(
    tmp_path,
) -> None:
    base = build_adc_variants(
        board_id="00",
        adc_indices=(0,),
        active_conversion_rates_hz=(1.0e6,),
        logic_offsets_symbols=(0.0,),
        conversions=1,
        vin_cm_v=0.8,
        vin_diff=h.Vdc.Params(dc=0.05),
    )[0]
    boundary = replace(
        base,
        tb=replace(base.tb, vin_cm=h.Vdc.Params(dc=0.8), vin_diff=h.Vdc.Params(dc=0.9)),
    )
    beyond = replace(boundary, tb=replace(boundary.tb, vin_diff=h.Vdc.Params(dc=0.900002)))
    scan_outdir = tmp_path / "not-created"
    with pytest.raises(ValueError, match="ADC inputs"):
        scan_adc.scan(beyond, run_dir=scan_outdir, position="first")
    assert not scan_outdir.exists()


@pytest.mark.parametrize(
    ("field", "voltage_v", "message"),
    (
        ("vdd_a", 1.099, "VDD_A request"),
        ("vdd_dac", 1.301, "VDD_DAC request"),
        ("vdd_io", 1.3, "VDD_IO is fixed"),
    ),
)
def test_adc_preflight_rejects_supply_and_fixed_io_before_hardware(
    tmp_path,
    field: str,
    voltage_v: float,
    message: str,
) -> None:
    base = build_adc_variants(
        board_id="00",
        adc_indices=(0,),
        active_conversion_rates_hz=(1.0e6,),
        logic_offsets_symbols=(0.0,),
        conversions=1,
        vin_cm_v=0.8,
        vin_diff=h.Vdc.Params(dc=0.05),
    )[0]
    invalid = (
        replace(base, vdd_io=h.Vdc.Params(dc=voltage_v))
        if field == "vdd_io"
        else replace(base, tb=replace(base.tb, **{field: h.Vdc.Params(dc=voltage_v)}))
    )
    scan_outdir = tmp_path / "not-created"

    with pytest.raises(ValueError, match=message):
        scan_adc.scan(invalid, run_dir=scan_outdir, position="first")
    assert not scan_outdir.exists()


def test_adc_noctl_abort_opens_only_the_fpga_map(monkeypatch, tmp_path) -> None:
    import basil.dut

    constructed_paths = []

    class DummyDut:
        def __init__(self, map_path: str) -> None:
            constructed_paths.append(map_path)

        def init(self) -> None:
            pass

        def close(self) -> None:
            pass

        def __getitem__(self, _name: str):
            raise KeyError("dummy FPGA has no hardware layers")

    monkeypatch.setattr(basil.dut, "Dut", DummyDut)
    params = build_adc_variants(
        board_id="00",
        adc_indices=(0,),
        active_conversion_rates_hz=(2.0e6,),
        logic_offsets_symbols=(2.0,),
        conversions=1,
        vin_cm_v=0.7,
        vin_diff=h.Vdc.Params(dc=0.05),
    )[0]

    scan_adc_noctl.scan(params, run_dir=tmp_path / "unused", position="abort")

    assert [path.rsplit("/", 1)[-1] for path in constructed_paths] == ["map_fpga.yaml"]


def test_calculate_fastrx_capture_alignment_uses_pattern_and_path_delays() -> None:
    """Software-only: keep arbitrary rates inside the bounded data aperture."""

    timing = load_board_map()["boards"]["00"]["capture_timing_model"]
    rates_mbd = (*range(80, 1601), 80.125, 681.37, 1163.25, 1599.875)
    for rate_mbd in rates_mbd:
        symbol_rate_bps = rate_mbd * 1.0e6
        alignment = fastrx.calculate_fastrx_capture_alignment(
            AdcTbParams(symbol_rate=symbol_rate_bps),
            **timing,
        )
        sequencer_period_s = 8.0 / symbol_rate_bps
        assert alignment.first_comp_transition_symbol == 36 - alignment.control_phase_advance_symbols
        assert 0 <= alignment.control_phase_advance_symbols <= timing["maximum_control_phase_advance_symbols"]
        assert 0 <= alignment.comp_idelay_taps < timing["idelay_tap_count"]
        assert alignment.latest_data_arrival_s <= alignment.capture_edge_s
        assert alignment.setup_margin_s >= timing["minimum_capture_margin_s"]
        assert alignment.hold_margin_s >= timing["minimum_capture_margin_s"]
        assert alignment.setup_margin_s + alignment.hold_margin_s == pytest.approx(
            sequencer_period_s - timing["external_comp_delay_max_s"] + timing["external_comp_delay_min_s"]
        )

    # Hardware regression: repeated 1.52 GBd sine acquisitions were clean at
    # taps 18/20 but contained rare gross errors at taps 22/24. Preserve the
    # hold-bounded setup-side guard at 1.4 and 1.6 GBd.
    alignment_1400 = fastrx.calculate_fastrx_capture_alignment(
        AdcTbParams(symbol_rate=1.4e9),
        **timing,
    )
    alignment_1600 = fastrx.calculate_fastrx_capture_alignment(
        AdcTbParams(symbol_rate=1.6e9),
        **timing,
    )
    assert (
        alignment_1400.control_phase_advance_symbols,
        alignment_1400.rx_sen_start_word,
        alignment_1400.comp_idelay_taps,
    ) == (7, 8, 20)
    assert (
        alignment_1600.control_phase_advance_symbols,
        alignment_1600.rx_sen_start_word,
        alignment_1600.comp_idelay_taps,
    ) == (0, 9, 3)


def test_parse_pwl_wave_accepts_spice_suffixes_and_rejects_time_reversal() -> None:
    points = scan_adc.parse_pwl_wave("0 -100m 500n 100m 1u -100m")
    assert [time_s for time_s, _ in points] == pytest.approx([0.0, 500e-9, 1e-6])
    assert [voltage_v for _, voltage_v in points] == pytest.approx([-0.1, 0.1, -0.1])
    typed = scan_adc.parse_pwl_wave(h.Pwl.ramp(start=-0.1, stop=0.1, duration=500e-9))
    assert [time_s for time_s, _ in typed] == pytest.approx([0.0, 500e-9])
    assert [voltage_v for _, voltage_v in typed] == pytest.approx([-0.1, 0.1])
    with pytest.raises(ValueError, match="increase strictly"):
        scan_adc.parse_pwl_wave("1u 0 0 1")
