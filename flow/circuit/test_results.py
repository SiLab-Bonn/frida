"""Software-only tests for VLSIR-to-FRIDA result conversion."""

import dataclasses
from pathlib import Path

import hdl21 as h
import numpy as np
import pytest

from flow.adc.sim import AdcTbParams
from flow.analysis.io import read_measurement, write_measurement
from flow.analysis.types import AdcIntWave, CompIntWave, MeasAdcInt, MeasCompInt
from flow.circuit.results import convert_spectre_adc_to_measurement, convert_spectre_comp_to_measurement
from flow.comp import CompParams
from flow.comp.sim import CompTb, CompTbParams
from pdk import tsmc65


def test_adc_waveform_conversion_writes_shared_hdf5(tmp_path: Path) -> None:
    expected_bout = "10110100101100101"
    time_step_s = 0.05e-9
    conversion_offsets_s = np.asarray([0.0, 170.0e-9])
    times_s = np.arange(0.0, 340.0e-9 + time_step_s / 2, time_step_s)
    comp_edge_times_s = np.concatenate(
        [offset_s + 5.0e-9 + np.arange(17) * 8.0e-9 for offset_s in conversion_offsets_s]
    )
    logic_edge_times_s = np.concatenate(
        [offset_s + 5.0e-9 + np.arange(16) * 8.0e-9 + 3.0e-9 for offset_s in conversion_offsets_s]
    )

    # The ADC has 17 comparator decisions but only 16 DAC-update/ LOGIC edges:
    # its final decision does not update the CDAC. Put each decision transition
    # at 97% of the COMP-to-LOGIC interval. The production 98% sampling point
    # must decode the new bit, whereas an earlier 95% sample still sees the
    # preceding decision. This also checks extrapolation of the final interval.
    seq_comp_v = np.zeros_like(times_s)
    seq_logic_v = np.zeros_like(times_s)
    for edge_s in comp_edge_times_s:
        seq_comp_v[(times_s >= edge_s) & (times_s < edge_s + 0.5e-9)] = 1.2
    for edge_s in logic_edge_times_s:
        seq_logic_v[(times_s >= edge_s) & (times_s < edge_s + 0.5e-9)] = 1.2

    comp_out_v = np.zeros_like(times_s)
    repeated_bout = expected_bout * len(conversion_offsets_s)
    for decision_index, bit in enumerate(repeated_bout):
        transition_s = comp_edge_times_s[decision_index] + 0.97 * 3.0e-9
        stop_s = (
            comp_edge_times_s[decision_index + 1] + 0.97 * 3.0e-9
            if decision_index + 1 < len(repeated_bout)
            else times_s[-1] + time_step_s
        )
        comp_out_v[(times_s >= transition_s) & (times_s < stop_s)] = 1.2 * int(bit)

    raw_wave_names = tuple(
        field.name
        for field in dataclasses.fields(AdcIntWave)
        if field.name not in {"conversion_index", "time_s", "vin_diff_v", "internal_v"}
    )
    values = {name: np.zeros_like(times_s) for name in raw_wave_names}
    values.update(
        {
            "vin_p_v": np.full_like(times_s, 0.650),
            "vin_n_v": np.full_like(times_s, 0.600),
            "seq_init_v": np.where(
                ((times_s >= 1.0e-9) & (times_s < 2.0e-9)) | ((times_s >= 171.0e-9) & (times_s < 172.0e-9)),
                1.2,
                0.0,
            ),
            "seq_comp_v": seq_comp_v,
            "seq_logic_v": seq_logic_v,
            "comp_out_v": comp_out_v,
            # Spectre voltage-source current is into the source positive
            # terminal, hence negative while each rail delivers current.
            "vdd_a_i": np.full_like(times_s, -2.0e-6),
            "vdd_d_i": np.full_like(times_s, -40.0e-6),
            "vdd_dac_i": np.full_like(times_s, -20.0e-6),
        }
    )
    values = {
        "time": times_s,
        **values,
    }
    signal_names = {"time_s": "time", **{name: name for name in raw_wave_names}}
    signal_names["comp_latch_p_v"] = "latch_internal"
    values["latch_internal"] = np.full_like(times_s, 0.875)
    raw_path = tmp_path / "netlist.raw"
    raw_path.touch()

    params = AdcTbParams(conversions=2)
    early_measurement = convert_spectre_adc_to_measurement(
        values,
        params=params,
        raw_path=raw_path,
        signal_names=signal_names,
        decision_sample_fraction=0.95,
        maximum_waveform_records=1,
    )
    assert "".join(str(bit) for bit in early_measurement.daq.bout[0]) != expected_bout

    h5_path = tmp_path / "adc.h5"
    expected = convert_spectre_adc_to_measurement(
        values,
        params=params,
        raw_path=raw_path,
        signal_names=signal_names,
    )
    write_measurement(h5_path, expected)
    measurement = read_measurement(h5_path)

    assert isinstance(measurement, MeasAdcInt)
    assert measurement.info.backend == "spice"
    assert measurement.info.readbacks["raw_format"] == "spectre_nutbin"
    assert measurement.info.readbacks["raw_max_timestep_s"] == pytest.approx(time_step_s)
    assert measurement.info.readbacks["waveform_sample_interval_s"] == pytest.approx(25e-12)
    assert measurement.info.readbacks["waveform_interpolated_from_coarser_raw"] is True
    assert measurement.info.readbacks["decision_sample_fraction"] == pytest.approx(0.98)
    assert measurement.info.readbacks["supply_power_available"] is True
    assert measurement.info.readbacks["supply_current_convention"] == "positive_current_draw"
    assert measurement.info.readbacks["vdd_d_active_average_power_w"] == pytest.approx(48.0e-6)
    assert len(measurement.daq.conversion_index) == 2
    assert all("".join(str(bit) for bit in row) == expected_bout for row in measurement.daq.bout)
    np.testing.assert_array_equal(measurement.daq.bout[0, -2:], [0, 1])
    assert measurement.daq.dout_raw[0] > 0
    assert measurement.daq.dout[0] > 0
    assert measurement.daq.vin_diff_v[0] == pytest.approx(0.050)
    np.testing.assert_array_equal(measurement.wave.conversion_index, [0, 1])
    assert measurement.wave.comp_out_v.shape == (2, 6_800)
    np.testing.assert_allclose(np.diff(measurement.wave.time_s), 25e-12)
    assert measurement.wave.seq_init_v[0, 0] == pytest.approx(0.0)
    np.testing.assert_allclose(measurement.wave.vdd_a_i, 2.0e-6)
    np.testing.assert_allclose(measurement.wave.vdd_d_i, 40.0e-6)
    np.testing.assert_allclose(measurement.wave.vdd_dac_i, 20.0e-6)
    np.testing.assert_allclose(measurement.wave.internal_v["comp_latch_p_v"], 0.875)
    assert measurement.wave.internal_v["comp_latch_p_v"].shape == measurement.wave.comp_out_v.shape


def test_comp_waveform_conversion_writes_shared_hdf5(tmp_path: Path) -> None:
    params = CompTbParams(
        comp=CompParams(diffpair_w=31),
        vin_diff_values_v=(-100e-6, 0.0, 100e-6),
        conversions=2,
    )
    time_step_s = 0.5e-9
    cycle_s = 40e-9
    trial_count = 6
    times_s = np.arange(0.0, trial_count * cycle_s + time_step_s / 2.0, time_step_s)
    decisions = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    clock = np.zeros_like(times_s)
    comp_difference = np.zeros_like(times_s)
    output_difference = np.zeros_like(times_s)
    previous_sign = -1.0
    for trial, decision in enumerate(decisions):
        start_s = trial * cycle_s
        eval_start_s = start_s + 10e-9
        stop_s = start_s + cycle_s
        active = (times_s >= eval_start_s) & (times_s < stop_s)
        clock[active] = 1.2
        sign = 1.0 if decision else -1.0
        resolved = (times_s >= eval_start_s + 2e-9) & (times_s < stop_s)
        comp_difference[resolved] = sign
        output_difference[(times_s >= start_s) & (times_s < eval_start_s + 2e-9)] = previous_sign
        output_difference[resolved] = sign
        previous_sign = sign

    values = {
        "time": times_s,
        "vin_p": np.full_like(times_s, 0.8),
        "vin_n": np.full_like(times_s, 0.8),
        "clock": clock,
        "vout_p": 0.6 + output_difference / 2.0,
        "vout_n": 0.6 - output_difference / 2.0,
        "comp_p": 0.6 + comp_difference / 2.0,
        "comp_n": 0.6 - comp_difference / 2.0,
        "vdd_i": np.full_like(times_s, -10e-6),
    }
    signal_names = {
        "time_s": "time",
        "vin_p_v": "vin_p",
        "vin_n_v": "vin_n",
        "clock_v": "clock",
        "vout_p_v": "vout_p",
        "vout_n_v": "vout_n",
        "comp_p_v": "comp_p",
        "comp_n_v": "comp_n",
        "vdd_i": "vdd_i",
    }
    raw_path = tmp_path / "netlist.raw"
    raw_path.touch()
    h.pdk.set_default(tsmc65.pdk_logic)
    compiled_tb = CompTb(params)
    h.pdk.compile(compiled_tb)

    h5_path = tmp_path / "comp.h5"
    expected = convert_spectre_comp_to_measurement(
        values,
        params=params,
        raw_path=raw_path,
        signal_names=signal_names,
        candidate_id="synthetic",
        candidate_label="Synthetic comparator",
        topology_index=0,
        size_profile="fabricated",
        compiled_tb=compiled_tb,
    )
    write_measurement(h5_path, expected)
    measurement = read_measurement(h5_path)

    assert isinstance(measurement, MeasCompInt)
    assert isinstance(measurement.wave, CompIntWave)
    assert measurement.param == params
    assert measurement.info.readbacks["candidate_id"] == "synthetic"
    total_active_area_units = measurement.info.readbacks["total_active_area_units"]
    total_active_area_um2 = measurement.info.readbacks["total_active_area_um2"]
    assert isinstance(total_active_area_units, (int, float)) and total_active_area_units > 0
    assert isinstance(total_active_area_um2, (int, float)) and total_active_area_um2 > 0
    assert measurement.info.readbacks["raw_format"] == "spectre_nutbin"
    assert measurement.info.readbacks["vdd_active_average_power_w"] == pytest.approx(12e-6)
    assert measurement.info.readbacks["energy_per_decision_j"] == pytest.approx(480e-15)
    np.testing.assert_allclose(measurement.daq.vin_diff_v, [-100e-6, -100e-6, 0.0, 0.0, 100e-6, 100e-6])
    np.testing.assert_array_equal(measurement.daq.decision, decisions)
    np.testing.assert_array_equal(measurement.wave.trial_index, np.arange(6))
    assert measurement.wave.clock_v.shape == (6, 80)
    np.testing.assert_allclose(measurement.wave.vdd_i, 10e-6)


def test_frida2_rc_waveforms_use_device_terminal_nodes() -> None:
    from flow.circuit.results import adc_signal_names

    names = adc_signal_names("frida2", pex_cell="adc_12b_17step")
    legacy = adc_signal_names("frida1", pex_cell="adc_1layer_radix17")
    assert names["vdac_p_v"] == "xtop.xadc.N_VDAC_P_XXsampswitch_p/MM0_d"
    assert names["dac_state_p_c0_v"].startswith("xtop.xadc.N_DAC_STATE_P_MAIN<0>_")
    assert legacy["dac_state_p_c0_v"].startswith("xtop.xadc.N_DAC_STATE_P_MAIN<15>_")
    assert names["dac_state_p_c0_v"].split("/", 1)[1] == legacy["dac_state_p_c0_v"].split("/", 1)[1]


@pytest.mark.parametrize("view", ("frida1", "frida2"))
def test_extracted_waveforms_cover_every_dac_stage_and_comparator_nodes(view):
    from flow.circuit.results import adc_signal_names

    names = adc_signal_names(view, pex_cell="adc_12b_17step")
    assert len(names) == len(set(names.values()))
    for side in ("p", "n"):
        for kind in ("", "diff_"):
            for stage in range(16):
                for family in ("state", "botplate"):
                    node = names[f"dac_{family}_{side}_{kind}c{stage}_v"]
                    assert f"<{stage if view == 'frida2' else 15 - stage}>_" in node
    assert names["comp_latch_p_v"].endswith("XXLATCH/MMM4_d")
    assert names["comp_latch_n_v"].endswith("XXLATCH/MMM3_d")
    assert names["comp_input_p_drain_v"].endswith("XXLATCH/MMM1_d")
    assert names["comp_input_n_drain_v"].endswith("XXLATCH/MMM2_d")
