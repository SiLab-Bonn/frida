"""Adapt VLSIR Spectre waveforms into typed FRIDA measurements."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from flow.analysis.io import interpolate_wave_records
from flow.analysis.types import AdcDaq, AdcIntWave, CompDaq, CompIntWave, MeasAdcInt, MeasCompInt, MeasInfo
from flow.cdac import get_cdac_weights

if TYPE_CHECKING:
    import hdl21 as h

    from flow.adc.sim import AdcTbParams
    from flow.comp.sim import CompTbParams


def adc_signal_names(view: str, *, pex_cell: str | None = None) -> dict[str, str]:
    """Return the Spectre-to-measurement signal map for one ADC view."""

    common = {
        "time_s": "time",
        "vin_p_v": "xtop.vin_p",
        "vin_n_v": "xtop.vin_n",
        "seq_init_v": "xtop.seq_init",
        "seq_samp_v": "xtop.seq_samp",
        "seq_comp_v": "xtop.seq_comp",
        "seq_logic_v": "xtop.seq_logic",
        "comp_out_v": "xtop.comp_out",
        "vdd_a_i": "xtop.vvdd_a:p",
        "vdd_d_i": "xtop.vvdd_d:p",
        "vdd_dac_i": "xtop.vvdd_dac:p",
    }
    if view in ("frida1", "frida2"):
        # FRIDA-1 and FRIDA-2 both use distributed xACT RC extraction. Save
        # the device-terminal nodes; lumped logical net names do not exist.
        names = {
            **common,
            "vdac_p_v": "xtop.xadc.N_VDAC_P_XXsampswitch_p/MM0_d",
            "vdac_n_v": "xtop.xadc.N_VDAC_N_XXsampswitch_n/MM0_d",
            "clk_samp_p_v": "xtop.xadc.N_CLK_SAMP_P_XXsampswitch_p/MM0_g",
            "clk_samp_p_b_v": "xtop.xadc.N_CLK_SAMP_P_B_XXsampswitch_p/MM1_g",
            "clk_samp_n_v": "xtop.xadc.N_CLK_SAMP_N_XXsampswitch_n/MM0_g",
            "clk_samp_n_b_v": "xtop.xadc.N_CLK_SAMP_N_B_XXsampswitch_n/MM1_g",
            "clk_comp_v": "xtop.xadc.N_CLK_COMP_XXcomp/XXLATCH/MMM0_g",
            "comp_out_p_v": "xtop.xadc.N_COMP_OUT_P_XXcomp/XXI3/XXI46/MM_u2_1_d",
            "comp_out_n_v": "xtop.xadc.N_COMP_OUT_N_XXcomp/XXI3/XXI47/MM_u2_1_d",
            "comp_latch_p_v": "xtop.xadc.N_XXCOMP/COMP_P_XXcomp/XXLATCH/MMM4_d",
            "comp_latch_n_v": "xtop.xadc.N_XXCOMP/COMP_N_XXcomp/XXLATCH/MMM3_d",
            "comp_input_p_drain_v": "xtop.xadc.N_XXCOMP/XXLATCH/NET031_XXcomp/XXLATCH/MMM1_d",
            "comp_input_n_drain_v": "xtop.xadc.N_XXCOMP/XXLATCH/NET037_XXcomp/XXLATCH/MMM2_d",
        }
        for stage in range(16):
            # The renamed source nets are C0-first; the synthesized gate
            # identifiers still identify the same physical driver devices.
            physical = 15 - stage
            net_index = stage if view == "frida2" else physical
            gate = f"XXxor{physical}" + ("_0" if physical >= 14 else "")
            state_device = "MMM_u2_1" if physical >= 12 else "MMM_u2"
            for side in ("p", "n"):
                for kind in ("main", "diff"):
                    suffix = f"{side}_" + ("diff_" if kind == "diff" else "") + f"c{stage}_v"
                    driver = f"XXcapdriver_{side}_{kind}/{gate}"
                    names[f"dac_state_{suffix}"] = (
                        f"xtop.xadc.N_DAC_STATE_{side.upper()}_{kind.upper()}<{net_index}>_"
                        f"{driver}/{state_device}-M_u3_g"
                    )
                    names[f"dac_botplate_{suffix}"] = (
                        f"xtop.xadc.N_DAC_DRIVE_BOTPLATE_{kind.upper()}_{side.upper()}<{net_index}>_"
                        f"{driver}/MMM_u4_1-M_u3_d"
                    )
        return names
    if view == "hdl21gen":
        return {
            **common,
            "vdac_p_v": "xtop.xadc.cdac_top_p",
            "vdac_n_v": "xtop.xadc.cdac_top_n",
            "clk_samp_p_v": "xtop.xadc.clk_samp_p",
            "clk_samp_p_b_v": "xtop.xadc.clk_samp_p_b",
            "clk_samp_n_v": "xtop.xadc.clk_samp_n",
            "clk_samp_n_b_v": "xtop.xadc.clk_samp_n_b",
            "clk_comp_v": "xtop.xadc.clk_comp",
            "comp_out_p_v": "xtop.xadc.comp_out_p",
            "comp_out_n_v": "xtop.xadc.comp_out_n",
            "dac_state_p_c0_v": "xtop.xadc.dac_state_p_0",
            "dac_state_p_c7_v": "xtop.xadc.dac_state_p_7",
            "dac_state_p_c15_v": "xtop.xadc.dac_state_p_15",
            "dac_state_n_c0_v": "xtop.xadc.dac_state_n_0",
            "dac_state_n_c7_v": "xtop.xadc.dac_state_n_7",
            "dac_state_n_c15_v": "xtop.xadc.dac_state_n_15",
            "dac_botplate_p_c0_v": "xtop.xadc.xcdac_p.bot_0",
            "dac_botplate_p_c7_v": "xtop.xadc.xcdac_p.bot_7",
            "dac_botplate_p_c15_v": "xtop.xadc.xcdac_p.bot_15",
            "dac_botplate_n_c0_v": "xtop.xadc.xcdac_n.bot_0",
            "dac_botplate_n_c7_v": "xtop.xadc.xcdac_n.bot_7",
            "dac_botplate_n_c15_v": "xtop.xadc.xcdac_n.bot_15",
        }
    raise ValueError(f"unsupported ADC view {view!r}")


def comp_signal_names() -> dict[str, str]:
    """Return the Spectre-to-measurement signal map for comparator runs."""

    return {
        "time_s": "time",
        "vin_p_v": "xtop.in_p",
        "vin_n_v": "xtop.in_n",
        "clock_v": "xtop.clk",
        "vout_p_v": "xtop.out_p",
        "vout_n_v": "xtop.out_n",
        "comp_p_v": "xtop.dut.innerp",
        "comp_n_v": "xtop.dut.innern",
        "vdd_i": "xtop.vvdd:p",
    }


def convert_spectre_adc_to_measurement(
    data: Mapping[str, Sequence[float] | np.ndarray],
    *,
    params: AdcTbParams,
    raw_path: Path,
    signal_names: Mapping[str, str],
    threshold_v: float = 0.6,
    decision_sample_fraction: float = 0.98,
    waveform_sample_interval_s: float = 25e-12,
    maximum_waveform_records: int | None = None,
) -> MeasAdcInt:
    """Decode one Spectre ADC result into the typed internal contract.

    ``signal_names`` maps canonical signals to Spectre variable names. Extra
    voltage traces are retained in ``wave.internal_v`` on the same time grid.
    Spectre reports voltage-source current into the source's
    positive terminal; the stored supply currents reverse that sign so that
    positive values mean current drawn by the ADC.
    """

    raw_wave_names = tuple(
        field.name
        for field in dataclasses.fields(AdcIntWave)
        if field.name not in {"conversion_index", "time_s", "vin_diff_v", "internal_v"}
    )
    expected_names = {"time_s", *raw_wave_names}
    missing_names = sorted(expected_names.difference(signal_names))
    extra_names = set(signal_names).difference(expected_names)
    if missing_names:
        raise ValueError(f"signal_names is missing raw AdcIntWave signals: {missing_names}")
    if any(not name.isidentifier() or not name.endswith("_v") or name in {"vin_diff_v"} for name in extra_names):
        raise ValueError("additional waveform names must be voltage identifiers ending in _v")
    raw_names = tuple(signal_names.values())
    if any(not isinstance(name, str) or not name for name in raw_names):
        raise ValueError("signal_names values must be non-empty raw variable names")
    if len(set(raw_names)) != len(raw_names):
        raise ValueError("signal_names raw variable names must be unique")
    missing = sorted(set(raw_names).difference(data))
    if missing:
        raise KeyError(f"Spectre data is missing mapped signals {missing}")
    if not 0.0 < decision_sample_fraction < 1.0:
        raise ValueError("decision_sample_fraction must lie strictly between zero and one")
    if not np.isfinite(waveform_sample_interval_s) or waveform_sample_interval_s <= 0:
        raise ValueError("waveform_sample_interval_s must be finite and positive")
    if maximum_waveform_records is not None and maximum_waveform_records <= 0:
        raise ValueError("maximum_waveform_records must be positive")

    times_s = np.asarray(data[signal_names["time_s"]], dtype=np.float64)
    signals = {name: np.asarray(data[raw], dtype=np.float64) for name, raw in signal_names.items() if name != "time_s"}
    for name in ("vdd_a_i", "vdd_d_i", "vdd_dac_i"):
        signals[name] = -signals[name]
    signals["vin_diff_v"] = signals["vin_p_v"] - signals["vin_n_v"]
    if times_s.ndim != 1 or len(times_s) < 2 or np.any(np.diff(times_s) <= 0):
        raise ValueError("Spectre time must be one-dimensional and strictly increasing")
    if any(values.shape != times_s.shape for values in signals.values()):
        raise ValueError("all mapped Spectre signals must align with Spectre time")
    if not np.all(np.isfinite(times_s)) or any(not np.all(np.isfinite(values)) for values in signals.values()):
        raise ValueError("Spectre time and mapped signals must contain only finite values")

    code_weights = np.asarray(
        [2 * weight for weight in get_cdac_weights(params.dut.cdac)] + [1],
        dtype=np.int64,
    )
    if len(code_weights) != 17:
        raise ValueError(f"ADC measurement format requires 17 decisions, got {len(code_weights)}")

    edge_indices: dict[str, np.ndarray] = {}
    for name in ("seq_init_v", "seq_comp_v", "seq_logic_v"):
        high = signals[name] > threshold_v
        edge_indices[name] = np.flatnonzero(high & np.concatenate((np.asarray([True]), ~high[:-1])))
    conversion_start_indices = edge_indices["seq_init_v"]
    # A repeating INIT can rise on the final recorded point; that is the next
    # conversion's boundary, not an additional conversion to decode.
    conversion_start_indices = conversion_start_indices[times_s[conversion_start_indices] < times_s[-1]]
    if len(conversion_start_indices) == 0:
        raise ValueError("Spectre result contains no SEQ_INIT rising edge")

    comp_edges_by_conversion = []
    logic_times_by_conversion = []
    for conversion_number, start_index in enumerate(conversion_start_indices):
        stop_index = (
            conversion_start_indices[conversion_number + 1]
            if conversion_number + 1 < len(conversion_start_indices)
            else len(times_s)
        )
        comp_edges = edge_indices["seq_comp_v"]
        comp_edges = comp_edges[(comp_edges >= start_index) & (comp_edges < stop_index)]
        if len(comp_edges) != len(code_weights):
            raise ValueError(
                f"conversion {conversion_number} contains {len(comp_edges)} COMP rising edges; "
                f"expected exactly {len(code_weights)}"
            )
        logic_edges = edge_indices["seq_logic_v"]
        logic_edges = logic_edges[(logic_edges > comp_edges[0]) & (logic_edges < stop_index)]
        logic_positions = np.searchsorted(logic_edges, comp_edges, side="right")
        matched = logic_positions < len(logic_edges)
        if np.count_nonzero(matched) != len(code_weights) - 1 or not np.all(matched[:-1]) or matched[-1]:
            raise ValueError(
                f"conversion {conversion_number} must have following LOGIC edges "
                "for its first 16 COMP edges and no update after its final decision"
            )
        matched_logic_edges = logic_edges[logic_positions[:-1]]
        if len(np.unique(matched_logic_edges)) != len(code_weights) - 1:
            raise ValueError(
                f"conversion {conversion_number} does not pair each COMP edge with a unique following LOGIC edge"
            )
        comp_edges_by_conversion.append(comp_edges)
        comp_times = times_s[comp_edges]
        logic_times = times_s[matched_logic_edges]
        final_interval_s = float(np.median(logic_times - comp_times[:-1]))
        # B16 has no LOGIC update. Its inferred capture deadline must not cross
        # the next INIT / end of the record, even when its COMP pulse is shorter.
        final_deadline_s = min(
            comp_times[-1] + final_interval_s,
            times_s[stop_index] if stop_index < len(times_s) else times_s[-1],
        )
        logic_times_by_conversion.append(np.concatenate((logic_times, [final_deadline_s])))

    comp_edge_indices = np.stack(comp_edges_by_conversion)
    comp_edge_times_s = times_s[comp_edge_indices]
    logic_edge_times_s = np.stack(logic_times_by_conversion)
    sample_times_s = comp_edge_times_s + decision_sample_fraction * (logic_edge_times_s - comp_edge_times_s)
    bout = (
        np.interp(sample_times_s.ravel(), times_s, signals["comp_out_v"]).reshape(sample_times_s.shape) > threshold_v
    ).astype(np.uint8)
    dout_raw = bout @ code_weights
    dout = np.rint(dout_raw * ((1 << params.dut.adc_bits) - 1) / np.sum(code_weights)).astype(np.int64)
    conversion_starts_s = times_s[conversion_start_indices]
    vin_diff_v = np.interp(comp_edge_times_s[:, 0], times_s, signals["vin_diff_v"])
    complete_conversions = len(conversion_start_indices)
    # HDL21 paramclasses are runtime dataclasses, although their decorator's
    # typing stub does not currently expose that fact to ty.
    params = dataclasses.replace(
        params,
        conversions=complete_conversions,
    )

    # Align waveform records to the repeated sequencer-pattern boundary, not
    # to SEQ_INIT's rising edge inside that pattern. This retains the pre-INIT
    # portion of each conversion and lets the final complete pattern fit in a
    # simulation which stops after exactly ``conversions * pattern_period``.
    waveform_starts_s = conversion_starts_s - (conversion_starts_s[0] - times_s[0])
    if complete_conversions > 1:
        record_duration_s = float(np.median(np.diff(waveform_starts_s)))
        final_record_stop_s = waveform_starts_s[-1] + record_duration_s
        if final_record_stop_s > times_s[-1] and np.isclose(
            final_record_stop_s,
            times_s[-1],
            rtol=1e-9,
            atol=0.0,
        ):
            record_duration_s = float(times_s[-1] - waveform_starts_s[-1])
    else:
        record_duration_s = min(
            float(times_s[-1] - waveform_starts_s[0]),
            len(params.seq_init_pattern) / float(params.symbol_rate),
        )
    eligible_indices = np.flatnonzero(waveform_starts_s + record_duration_s <= times_s[-1])
    if maximum_waveform_records is not None and len(eligible_indices) > maximum_waveform_records:
        selected_positions = np.unique(
            np.rint(np.linspace(0, len(eligible_indices) - 1, maximum_waveform_records)).astype(np.int64)
        )
        eligible_indices = eligible_indices[selected_positions]
    waveform_conversion_indices = eligible_indices
    waveform_starts_s = waveform_starts_s[waveform_conversion_indices]
    relative_time_s, waveform_records = interpolate_wave_records(
        times_s,
        signals,
        [(float(start_s), float(start_s + record_duration_s)) for start_s in waveform_starts_s],
        waveform_sample_interval_s,
    )

    readbacks: dict[str, str | int | float | bool] = {
        "raw_file": Path(raw_path).name,
        "raw_format": "spectre_nutbin",
        "raw_points": len(times_s),
        "raw_max_timestep_s": float(np.max(np.diff(times_s))),
        "waveform_sample_interval_s": waveform_sample_interval_s,
        "waveform_interpolated_from_coarser_raw": bool(
            np.max(np.diff(times_s)) > waveform_sample_interval_s * (1.0 + 1e-9)
        ),
        "decision_sample_fraction": decision_sample_fraction,
        "supply_power_available": True,
        "supply_current_convention": "positive_current_draw",
        "signal_map_json": json.dumps(dict(signal_names), sort_keys=True),
    }
    rail_voltages = {
        "vdd_a": float(params.vdd_a.dc),
        "vdd_d": float(params.vdd_d.dc),
        "vdd_dac": float(params.vdd_dac.dc),
    }
    for rail, voltage_v in rail_voltages.items():
        current_draw_a = signals[f"{rail}_i"]
        duration_s = float(times_s[-1] - times_s[0])
        average_current_a = float(np.trapezoid(current_draw_a, times_s) / duration_s)
        readbacks[f"{rail}_active_average_current_a"] = average_current_a
        readbacks[f"{rail}_active_average_power_w"] = voltage_v * average_current_a

    return MeasAdcInt(
        info=MeasInfo(
            schema_version=1,
            measurement_type="MeasAdcInt",
            backend="spice",
            timestamp_utc=datetime.fromtimestamp(
                Path(raw_path).stat().st_mtime,
                tz=UTC,
            ),
            instruments={"simulator": "Spectre"},
            readbacks=readbacks,
        ),
        param=params,
        daq=AdcDaq(
            conversion_index=np.arange(complete_conversions, dtype=np.int64),
            bout=bout,
            dout_raw=dout_raw,
            dout=dout,
            vin_diff_v=vin_diff_v,
        ),
        wave=AdcIntWave(
            conversion_index=waveform_conversion_indices,
            time_s=relative_time_s,
            **{name: values for name, values in waveform_records.items() if name not in extra_names},
            internal_v={name: waveform_records[name] for name in sorted(extra_names)},
        ),
    )


def convert_spectre_comp_to_measurement(
    data: Mapping[str, Sequence[float] | np.ndarray],
    *,
    params: CompTbParams,
    raw_path: Path,
    signal_names: Mapping[str, str],
    candidate_id: str,
    candidate_label: str,
    topology_index: int,
    size_profile: str,
    compiled_tb: h.Module,
    waveform_sample_interval_s: float = 500e-12,
    spectre_runtime_s: float | None = None,
) -> MeasCompInt:
    """Decode one Spectre comparator campaign result into ``MeasCompInt``.

    All decisions at every input point are retained in ``daq``. Dense
    waveforms retain every trial at the three points nearest 50% probability
    plus one representative trial everywhere else.
    """

    raw_wave_names = tuple(
        field.name for field in dataclasses.fields(CompIntWave) if field.name not in {"trial_index", "time_s"}
    )
    expected_names = {"time_s", *raw_wave_names}
    missing_names = sorted(expected_names.difference(signal_names))
    unexpected_names = sorted(set(signal_names).difference(expected_names))
    if missing_names or unexpected_names:
        raise ValueError(
            "signal_names must map exactly the raw CompIntWave signals; "
            f"missing={missing_names}, unexpected={unexpected_names}"
        )
    raw_names = tuple(signal_names.values())
    if any(not isinstance(name, str) or not name for name in raw_names):
        raise ValueError("signal_names values must be non-empty raw variable names")
    if len(set(raw_names)) != len(raw_names):
        raise ValueError("signal_names raw variable names must be unique")
    missing = sorted(set(raw_names).difference(data))
    if missing:
        raise KeyError(f"Spectre data is missing mapped signals {missing}")
    if not np.isfinite(waveform_sample_interval_s) or waveform_sample_interval_s <= 0.0:
        raise ValueError("waveform_sample_interval_s must be finite and positive")
    if not candidate_id or not candidate_label or not size_profile:
        raise ValueError("comparator candidate metadata must be non-empty")
    device_geometry_signature = []
    for name, value in compiled_tb.dut.of.namespace.items():
        call = getattr(value, "of", None)
        call_params = getattr(call, "params", None)
        width = getattr(call_params, "w", None)
        length = getattr(call_params, "l", None)
        if width is None or length is None:
            continue
        device_geometry_signature.append(
            (
                name,
                round(float(width) / 120e-9),
                round(float(length) / 60e-9),
            )
        )
    if not device_geometry_signature:
        raise ValueError("compiled comparator testbench contains no sized MOS devices")
    device_geometry_signature = tuple(device_geometry_signature)
    device_width_signature = tuple((name, width) for name, width, _length in device_geometry_signature)
    total_width_units = sum(width for _name, width in device_width_signature)
    total_active_area_units = sum(width * length for _name, width, length in device_geometry_signature)
    total_active_area_um2 = total_active_area_units * 0.12 * 0.06
    if topology_index < 0 or total_width_units <= 0 or total_active_area_units <= 0:
        raise ValueError("comparator topology index, total width, and total area must be nonnegative/positive")
    if not math.isfinite(total_active_area_um2) or total_active_area_um2 <= 0.0:
        raise ValueError("comparator physical active area must be finite and positive")
    if tuple(device_width_signature) != tuple((name, width) for name, width, _length in device_geometry_signature):
        raise ValueError("comparator width and geometry signatures must describe the same devices")
    if sum(width * length for _name, width, length in device_geometry_signature) != total_active_area_units:
        raise ValueError("comparator total active area does not match its device geometry signature")

    times_s = np.asarray(data[signal_names["time_s"]], dtype=np.float64)
    signals = {name: np.asarray(data[signal_names[name]], dtype=np.float64) for name in raw_wave_names}
    # Spectre voltage-source current is positive into the source. Store positive
    # current delivered to the DUT, matching the ADC converter and analysis.
    signals["vdd_i"] = -signals["vdd_i"]
    if times_s.ndim != 1 or len(times_s) < 2 or np.any(np.diff(times_s) <= 0):
        raise ValueError("Spectre time must be one-dimensional and strictly increasing")
    if any(values.shape != times_s.shape for values in signals.values()):
        raise ValueError("all mapped Spectre signals must align with Spectre time")
    if not np.all(np.isfinite(times_s)) or any(not np.all(np.isfinite(values)) for values in signals.values()):
        raise ValueError("Spectre time and mapped signals must contain only finite values")

    expected_trial_count = len(params.vin_cm_values_v) * len(params.vin_diff_values_v) * params.conversions
    cycle_s = float(params.reset_time_s) + float(params.evaluation_time_s)
    expected_stop_s = expected_trial_count * cycle_s
    actual_duration_s = float(times_s[-1] - times_s[0])
    if not np.isclose(actual_duration_s, expected_stop_s, rtol=1e-6, atol=waveform_sample_interval_s):
        raise ValueError(
            f"Spectre result duration {actual_duration_s:.12g} s does not match scheduled {expected_stop_s:.12g} s"
        )

    nominal_vdiff = []
    nominal_vcm = []
    point_first_trial = []
    point_index = 0
    for vcm in params.vin_cm_values_v:
        for vdiff in params.vin_diff_values_v:
            first_trial = point_index * params.conversions
            point_first_trial.append(first_trial)
            nominal_vdiff.extend([float(vdiff)] * params.conversions)
            nominal_vcm.extend([float(vcm)] * params.conversions)
            point_index += 1
    if len(nominal_vdiff) != expected_trial_count:
        raise RuntimeError("comparator nominal trial schedule is internally inconsistent")

    # Sample just before the evaluation falling edge. Interpolation uses the
    # uniformly stored raw grid but Spectre's transient-noise integration keeps
    # its much finer internal timesteps.
    decision_margin_s = max(float(params.transition_time_s) * 2.0, waveform_sample_interval_s / 2.0)
    sample_times_s = times_s[0] + (np.arange(expected_trial_count) + 1) * cycle_s - decision_margin_s
    out_p = np.interp(sample_times_s, times_s, signals["vout_p_v"])
    out_n = np.interp(sample_times_s, times_s, signals["vout_n_v"])
    decisions = (out_p > out_n).astype(np.uint8)

    # Preserve all records at the three points closest to 50% probability and
    # one representative record from every other S-curve point. Selecting from
    # the measured decisions follows offset topologies whose metastable region
    # is not centered at zero input, while bounding each production H5 file.
    representative_trials = [first + params.conversions // 2 for first in point_first_trial]
    point_probability = np.asarray(
        [np.mean(decisions[first : first + params.conversions]) for first in point_first_trial],
        dtype=np.float64,
    )
    transition_points = np.argsort(np.abs(point_probability - 0.5), kind="stable")[: min(3, len(point_first_trial))]
    transition_trials = [
        trial
        for point in transition_points
        for trial in range(point_first_trial[point], point_first_trial[point] + params.conversions)
    ]
    waveform_trial_indices = np.unique(np.asarray((*representative_trials, *transition_trials), dtype=np.int64))
    record_duration_s = cycle_s - decision_margin_s
    relative_time_s, waveform_records = interpolate_wave_records(
        times_s,
        signals,
        [
            (
                float(times_s[0] + trial * cycle_s),
                float(times_s[0] + trial * cycle_s + record_duration_s),
            )
            for trial in waveform_trial_indices
        ],
        waveform_sample_interval_s,
    )

    supply_v = float(params.vdd)
    average_current_a = float(np.trapezoid(signals["vdd_i"], times_s) / actual_duration_s)
    average_power_w = supply_v * average_current_a
    signature_text = ",".join(f"{name}:{width}" for name, width in device_width_signature)
    geometry_text = ",".join(f"{name}:{width}:{length}" for name, width, length in device_geometry_signature)
    readbacks: dict[str, str | int | float | bool] = {
        "raw_file": Path(raw_path).name,
        "raw_format": "spectre_nutbin",
        "raw_points": len(times_s),
        "raw_max_timestep_s": float(np.max(np.diff(times_s))),
        "waveform_sample_interval_s": waveform_sample_interval_s,
        "waveform_interpolated_from_coarser_raw": bool(
            np.max(np.diff(times_s)) > waveform_sample_interval_s * (1.0 + 1e-9)
        ),
        "decision_sample_margin_s": decision_margin_s,
        "supply_power_available": True,
        "supply_current_convention": "positive_current_draw",
        "vdd_v": supply_v,
        "vdd_active_average_current_a": average_current_a,
        "vdd_active_average_power_w": average_power_w,
        "energy_per_decision_j": average_power_w * cycle_s,
        "candidate_id": candidate_id,
        "candidate_label": candidate_label,
        "topology_index": topology_index,
        "size_profile": size_profile,
        "total_width_units": total_width_units,
        "device_width_signature": signature_text,
        "total_active_area_units": total_active_area_units,
        "total_active_area_um2": total_active_area_um2,
        "device_geometry_signature": geometry_text,
        "transient_noise": True,
        "transient_noise_seed": 1,
        "transient_noise_max_hz": 25e9,
    }
    if spectre_runtime_s is not None:
        if not math.isfinite(spectre_runtime_s) or spectre_runtime_s <= 0.0:
            raise ValueError("Spectre runtime must be finite and positive")
        readbacks["spectre_runtime_s"] = spectre_runtime_s

    return MeasCompInt(
        info=MeasInfo(
            schema_version=1,
            measurement_type="MeasCompInt",
            backend="spice",
            timestamp_utc=datetime.fromtimestamp(Path(raw_path).stat().st_mtime, tz=UTC),
            instruments={"simulator": "Spectre"},
            readbacks=readbacks,
        ),
        param=params,
        daq=CompDaq(
            trial_index=np.arange(expected_trial_count, dtype=np.int64),
            vin_diff_v=np.asarray(nominal_vdiff, dtype=np.float64),
            vin_cm_v=np.asarray(nominal_vcm, dtype=np.float64),
            decision=decisions,
        ),
        wave=CompIntWave(
            trial_index=waveform_trial_indices,
            time_s=relative_time_s,
            **waveform_records,
        ),
    )
