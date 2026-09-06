"""Software-only unit tests for ADC parameters; no hardware I/O is performed."""

import dataclasses

import hdl21 as h
import pytest
from hdl21.prefix import m

from flow.adc.sim import AdcTbParams
from flow.scans.params import (
    AdcScanParams,
    build_adc_variants,
    convert_sample_rate_to_baud,
    validate_params,
)


def test_default_params_are_valid_and_immutable() -> None:
    params = AdcScanParams(tb=AdcTbParams(view="frida1"))

    validate_params(params)
    assert isinstance(params.tb.vin_diff, h.Vdc.Params)
    assert float(params.tb.vin_diff.dc) == 0.0
    assert float(params.tb.vin_cm.dc) == pytest.approx(0.7)
    assert params.board_id is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.tb.conversions = 1


def test_analog_input_preserves_hdl21_source_type() -> None:
    inputs = (
        h.Vdc.Params(dc=100 * m),
        h.Vsin.Params(voff=0.0, vamp=100 * m, freq=1e6),
        h.Vpwl.Params(wave="0 -100m 1m 100m"),
    )

    params = [AdcTbParams(vin_diff=source) for source in inputs]

    assert [type(item.vin_diff) for item in params] == [type(source) for source in inputs]


def test_build_variants_covers_adc00_seven_offset_noise_rates() -> None:
    variants = build_adc_variants(
        board_id="00",
        adc_indices=(0,),
        active_conversion_rates_hz=tuple(rate * 0.25e6 for rate in range(2, 41)),
        logic_offsets_symbols=tuple(range(-3, 4)),
        conversions=1_000,
        vin_cm_v=0.8,
        vin_diff=h.Vdc.Params(dc=0.05),
    )

    assert len(variants) == 7 * 39
    assert {item.observed_adc for item in variants} == {0}
    assert {float(item.tb.symbol_rate) for item in variants} == {rate * 40e6 for rate in range(2, 41)}
    assert all(item.board_id == "00" for item in variants)
    assert {item.active_adc_mask for item in variants} == {
        (0,) * 15 + (1,),
    }
    assert all(item.tb.dut.adc_bits == 12 for item in variants)
    assert all(item.tb.dut.cdac.weights is not None for item in variants)
    assert all(item.tb.conversions == 1_000 for item in variants)
    assert all(isinstance(item.tb.vin_diff, h.Vdc.Params) for item in variants)
    assert {float(item.tb.symbol_rate) / 160 for item in variants} == {rate * 0.25e6 for rate in range(2, 41)}
    assert {float(item.tb.vin_diff.dc) for item in variants} == {0.05}
    assert {float(item.tb.vin_cm.dc) for item in variants} == {0.8}
    assert {float(item.tb.seq_logic_phase_delay_symbols) for item in variants} == set(range(-3, 4))


def test_build_adc_variants_covers_adc00_through_adc03_ramp() -> None:
    """Describe one repeated full-scale ramp capture for each selected ADC."""

    variants = build_adc_variants(
        board_id="00",
        adc_indices=(0, 1, 2, 3),
        active_conversion_rates_hz=(1.0e6,),
        logic_offsets_symbols=(0.0,),
        conversions=4_000_000,
        vin_cm_v=0.6,
        vin_diff=h.Vpwl.Params(wave="0 -1 0.1 1"),
        campaign="adc_ramp",
    )

    assert len(variants) == 4
    assert {item.observed_adc for item in variants} == set(range(4))
    assert all(item.board_id == "00" for item in variants)
    assert all(item.campaign == "adc_ramp" for item in variants)
    assert all(item.tb.conversions == 4_000_000 for item in variants)
    assert all(float(item.tb.symbol_rate) == 160e6 for item in variants)
    assert all(float(item.tb.vin_cm.dc) == 0.6 for item in variants)
    assert all(isinstance(item.tb.vin_diff, h.Vpwl.Params) for item in variants)
    assert {item.tb.vin_diff.wave for item in variants} == {"0 -1 0.1 1"}
    assert {item.active_adc_mask for item in variants} == {
        (0,) * (15 - adc_index) + (1,) + (0,) * adc_index for adc_index in range(4)
    }


def test_convert_sample_rate_to_baud_uses_active_pattern_span() -> None:
    params = AdcTbParams()
    assert convert_sample_rate_to_baud(params, 1e6) == 160e6

    inactive = "0" * len(params.seq_init_pattern)
    with pytest.raises(ValueError, match="no active symbols"):
        convert_sample_rate_to_baud(
            AdcTbParams(
                seq_init_pattern=inactive,
                seq_samp_pattern=inactive,
                seq_comp_pattern=inactive,
                seq_logic_pattern=inactive,
            ),
            1e6,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        convert_sample_rate_to_baud(params, 0.0)


def test_validation_rejects_invalid_configuration_relationships() -> None:
    incomplete_measurement = AdcScanParams(tb=AdcTbParams(view="frida1"), board_id="00")
    invalid_bus = AdcScanParams(tb=AdcTbParams(view="frida1", dac_astate_p=(0,) * 15 + (2,)))
    unequal_patterns = AdcScanParams(tb=AdcTbParams(view="frida1", seq_logic_pattern="01"))

    with pytest.raises(ValueError, match="must be set together"):
        validate_params(incomplete_measurement)
    with pytest.raises(ValueError, match="binary values"):
        validate_params(invalid_bus)
    with pytest.raises(ValueError, match="equal whole-word"):
        validate_params(unequal_patterns)
