"""Explicit, manually invoked measurement -> analysis -> plot pipelines.

Each function pins its accepted input files or measurement directories. Small
campaigns list every H5 file; large campaigns use a narrow glob within a named
directory. There is intentionally no automatic discovery of the newest run
directory or of analysis pipelines.

Run one named pipeline from the repository root with:

    uv run python -m flow.analysis.runner adc_noise_vs_rate

Omit the target name to run every registered pipeline. Comparator targets write
beneath ``build/analysis/comp``; the remaining targets write beneath
``build/analysis/adc``. Each analysis domain uses one timestamped directory per
invocation.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from flow.analysis.adc import (
    analyze_adc_cdac_settling,
    analyze_adc_code_distribution,
    analyze_adc_decision_paths,
    analyze_adc_dynamic,
    analyze_adc_dynamic_sweep,
    analyze_adc_noise_sweep,
    analyze_adc_power_sweep,
    analyze_adc_power_waveform,
    analyze_adc_ramp,
    analyze_adc_sampling_noise,
    analyze_adc_transfer,
    combine_adc_noise_comparison,
)
from flow.analysis.calibration1 import analyze as analyze_calibration1
from flow.analysis.calibration2 import analyze as analyze_calibration2
from flow.analysis.calibration3 import analyze as analyze_calibration3
from flow.analysis.cdac import analyze_cdac_cap_mismatch_campaign
from flow.analysis.comp import (
    analyze_comp_candidate_sweep,
    analyze_comp_offset_noise,
    classify_comp_common_mode_validity,
)
from flow.analysis.io import read_measurement
from flow.analysis.plots import (
    plot_adc_calibration_weights,
    plot_adc_cdac_settling,
    plot_adc_code_distribution,
    plot_adc_decision_path_density,
    plot_adc_decision_paths,
    plot_adc_dynamic,
    plot_adc_noise_distribution_grid,
    plot_adc_noise_distribution_sweep,
    plot_adc_noise_sweep,
    plot_adc_power_sweep,
    plot_adc_power_waveform,
    plot_adc_ramp_histogram,
    plot_adc_ramp_nonlinearity,
    plot_adc_ramp_transfer,
    plot_adc_ramp_weights,
    plot_adc_sampling_noise,
    plot_adc_transfer,
    plot_cdac_cap_mismatch,
    plot_cdac_cap_mismatch_comparison,
    plot_comp_candidate_sweep,
    plot_comp_common_mode_campaign,
    plot_comp_noise_power_tradeoff,
    plot_comp_sampling_campaign,
    plot_waveforms,
)
from flow.analysis.types import (
    MeasAdcExt,
    MeasAdcInt,
    MeasCdacExt,
    MeasCompExt,
    MeasCompInt,
)
from flow.analysis.waveform import analyze_measurement_waveforms
from flow.scans.params import load_board_map

BASE_PATH = Path(__file__).resolve().parents[2]


def adc_transfer_curve(output_dir: Path) -> tuple[Path, ...]:
    """Plot the accepted physical ADC00 static transfer campaign."""

    meas_read_dir = BASE_PATH / "build/scan_adc/20260818_135848"
    measurements = []
    for path in sorted(meas_read_dir.glob("*.h5")):
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasAdcExt):
            raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcExt")
        measurements.append(measurement)

    analysis = analyze_adc_transfer(measurements)
    return plot_adc_transfer(
        measurements,
        analysis,
        output_path=output_dir / "adc00_transfer_curve",
    )


def adc_ramp_nonlinearity(output_dir: Path) -> tuple[Path, ...]:
    """Compare uncalibrated DOUT with BOUT decoded by accepted CDAC weights."""

    ramp_meas_read_dir = BASE_PATH / "build/scan_adc/20260812_011910"
    ramp_measurement_paths = (
        ramp_meas_read_dir
        / "0000_00_adc00_160mbd_pwl10hz_m1000top1000mv_logicp0sym_vcm600mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5",
        ramp_meas_read_dir
        / "0001_00_adc01_160mbd_pwl10hz_m1000top1000mv_logicp0sym_vcm600mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5",
        ramp_meas_read_dir
        / "0002_00_adc02_160mbd_pwl10hz_m1000top1000mv_logicp0sym_vcm600mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5",
        ramp_meas_read_dir
        / "0003_00_adc03_160mbd_pwl10hz_m1000top1000mv_logicp0sym_vcm600mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5",
    )
    cdac_meas_read_dirs = tuple(
        BASE_PATH / "build/scan_cdac" / name for name in ("20260804_171234", "20260804_193030", "20260804_193631")
    )
    ramp_measurements = []
    for path in ramp_measurement_paths:
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasAdcExt):
            raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcExt")
        ramp_measurements.append(measurement)
    ramp_by_adc = {int(measurement.param.observed_adc): measurement for measurement in ramp_measurements}
    adc_indices = tuple(sorted(ramp_by_adc))
    board_id = ramp_measurements[0].param.board_id

    cdac_measurement_runs = []
    for meas_read_dir in cdac_meas_read_dirs:
        measurements = []
        for path in sorted(meas_read_dir.glob("*.h5")):
            measurement = read_measurement(path)
            if not isinstance(measurement, MeasCdacExt):
                raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasCdacExt")
            measurements.append(measurement)
        cdac_measurement_runs.append(tuple(measurements))

    comparator_calibrations = load_board_map()["boards"][board_id].get("comparator_calibration", {})
    comparator_offset_v_by_adc = {
        adc_index: float(comparator_calibrations[adc_index]["offset_v"]) for adc_index in adc_indices
    }
    cdac_groups, _cdac_analyses = analyze_cdac_cap_mismatch_campaign(
        cdac_measurement_runs,
        adc_indices=adc_indices,
        board_id=board_id,
        comparator_offset_v_by_adc=comparator_offset_v_by_adc,
    )
    cdac_group_by_adc = {group[0].param.observed_adc: group for group in cdac_groups}
    artifacts = []
    analyses = []
    for adc_index in adc_indices:
        ramp_measurement = ramp_by_adc[adc_index]
        calibration = analyze_calibration1(
            cdac_group_by_adc[adc_index],
            comparator_offset_v=comparator_offset_v_by_adc[adc_index],
        )
        analysis = analyze_adc_ramp(ramp_measurement, calibrations=(calibration,))
        analyses.append(analysis)
        artifacts.extend(
            plot_adc_ramp_transfer(
                analysis,
                output_path=output_dir / f"adc{adc_index:02d}_ramp_transfer",
            )
        )
        artifacts.extend(
            plot_adc_ramp_histogram(
                analysis,
                output_path=output_dir / f"adc{adc_index:02d}_ramp_histogram",
            )
        )
        artifacts.extend(
            plot_adc_ramp_weights(
                analysis,
                output_path=output_dir / f"adc{adc_index:02d}_ramp_weights",
            )
        )
        artifacts.extend(
            plot_adc_ramp_nonlinearity(
                analysis,
                output_path=output_dir / f"adc{adc_index:02d}_ramp_nonlinearity",
            )
        )

    csv_path = output_dir / "adc00_adc03_ramp_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "adc_index",
                "decoding",
                "sample_count",
                "retained_sample_count",
                "reset_excluded_sample_count",
                "sample_rate_hz",
                "ramp_frequency_hz",
                "maximum_abs_dnl_lsb",
                "maximum_abs_inl_lsb",
                "missing_codes",
                "maximum_transfer_reversal_dout",
            )
        )
        for analysis in analyses:
            for curve in analysis.curves:
                writer.writerow(
                    (
                        analysis.adc_index,
                        curve.decoding,
                        analysis.sample_count,
                        analysis.retained_sample_count,
                        analysis.reset_excluded_sample_count,
                        analysis.sample_rate_hz,
                        analysis.ramp_frequency_hz,
                        curve.maximum_abs_dnl,
                        curve.maximum_abs_inl,
                        curve.missing_codes,
                        curve.maximum_transfer_reversal_dout,
                    )
                )
    artifacts.append(csv_path)
    return tuple(artifacts)


def adc_calibration(output_dir: Path) -> tuple[Path, ...]:
    """Run all three ADC00 digital calibrations and compare them uniformly."""

    ramp_meas_read_path = BASE_PATH / (
        "build/scan_adc/20260812_011910/"
        "0000_00_adc00_160mbd_pwl10hz_m1000top1000mv_logicp0sym_"
        "vcm600mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5"
    )
    measurement = read_measurement(ramp_meas_read_path)
    if not isinstance(measurement, MeasAdcExt):
        raise TypeError(f"{ramp_meas_read_path} contains {type(measurement).__name__}, expected MeasAdcExt")
    adc_index = int(measurement.param.observed_adc)
    board_id = measurement.param.board_id

    cdac_meas_read_dirs = tuple(
        BASE_PATH / "build/scan_cdac" / name for name in ("20260804_171234", "20260804_193030", "20260804_193631")
    )
    cdac_measurement_runs = []
    for meas_read_dir in cdac_meas_read_dirs:
        measurements = []
        for path in sorted(meas_read_dir.glob("*.h5")):
            cdac_measurement = read_measurement(path)
            if not isinstance(cdac_measurement, MeasCdacExt):
                raise TypeError(f"{path} contains {type(cdac_measurement).__name__}, expected MeasCdacExt")
            if cdac_measurement.param.observed_adc == adc_index:
                measurements.append(cdac_measurement)
        cdac_measurement_runs.append(tuple(measurements))

    comparator_calibrations = load_board_map()["boards"][board_id].get("comparator_calibration", {})
    comparator_offset_v = float(comparator_calibrations[adc_index]["offset_v"])
    cdac_groups, _cdac_analyses = analyze_cdac_cap_mismatch_campaign(
        cdac_measurement_runs,
        adc_indices=(adc_index,),
        board_id=board_id,
        comparator_offset_v_by_adc={adc_index: comparator_offset_v},
    )
    cdac_measurements = cdac_groups[0]

    nominal_ramp = analyze_adc_ramp(measurement)
    calibrations = (
        analyze_calibration1(
            cdac_measurements,
            comparator_offset_v=comparator_offset_v,
        ),
        analyze_calibration2(measurement, nominal_ramp),
        analyze_calibration3(measurement, nominal_ramp),
    )
    ramp = analyze_adc_ramp(measurement, calibrations=calibrations)

    artifacts = list(
        plot_adc_calibration_weights(
            calibrations,
            output_path=output_dir / f"adc{adc_index:02d}_calibration_weights",
        )
    )
    artifacts.extend(
        plot_adc_ramp_transfer(
            ramp,
            output_path=output_dir / f"adc{adc_index:02d}_calibration_transfer",
        )
    )
    artifacts.extend(
        plot_adc_ramp_histogram(
            ramp,
            output_path=output_dir / f"adc{adc_index:02d}_calibration_code_density",
        )
    )
    artifacts.extend(
        plot_adc_ramp_nonlinearity(
            ramp,
            output_path=output_dir / f"adc{adc_index:02d}_calibration_inl_dnl",
        )
    )

    calibration_by_method = {calibration.method: calibration for calibration in calibrations}
    metrics_path = output_dir / f"adc{adc_index:02d}_calibration_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "decoding",
                "training_sample_count",
                "validation_sample_count",
                "weights_from_measurement",
                "output_gain",
                "output_offset_lsb",
                "maximum_abs_dnl_lsb",
                "maximum_abs_inl_lsb",
                "missing_codes",
                "maximum_transfer_reversal_lsb",
            )
        )
        for curve in ramp.curves:
            calibration = calibration_by_method.get(curve.decoding)
            writer.writerow(
                (
                    curve.decoding,
                    0 if calibration is None else calibration.training_sample_count,
                    0 if calibration is None else calibration.validation_sample_count,
                    0 if calibration is None else int(np.count_nonzero(calibration.measured_weight_mask)),
                    1.0 if calibration is None else calibration.output_gain,
                    0.0 if calibration is None else calibration.output_offset_lsb,
                    curve.maximum_abs_dnl,
                    curve.maximum_abs_inl,
                    curve.missing_codes,
                    curve.maximum_transfer_reversal_dout,
                )
            )
    artifacts.append(metrics_path)

    weights_path = output_dir / f"adc{adc_index:02d}_calibration_weights.csv"
    with weights_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "decision_index",
                "ideal_weight_lsb",
                *(
                    field
                    for calibration in calibrations
                    for field in (
                        f"{calibration.method}_weight_lsb",
                        f"{calibration.method}_from_measurement",
                    )
                ),
            )
        )
        for decision_index in range(17):
            writer.writerow(
                (
                    decision_index,
                    calibrations[0].nominal_weights[decision_index],
                    *(
                        value
                        for calibration in calibrations
                        for value in (
                            calibration.calibrated_weights[decision_index],
                            bool(calibration.measured_weight_mask[decision_index]),
                        )
                    ),
                )
            )
    artifacts.append(weights_path)
    return tuple(artifacts)


def adc00_fixed_input_noise(output_dir: Path) -> tuple[Path, ...]:
    """Analyze the controlled and externally applied ADC00 fixed-input captures."""

    physical_meas_read_dir = BASE_PATH / "build/scan_adc/20260819_113714"
    external_meas_read_dir = BASE_PATH / "build/scan_adc/20260821_173944"
    all_active_meas_read_dir = BASE_PATH / "build/scan_adc/20260822_144348"
    ideal_meas_read_dir = BASE_PATH / "build/sim/adc/20260820_005128"
    pex_meas_read_dir = BASE_PATH / "build/sim/adc/20260820_005122"
    supply_noise_meas_read_dir = BASE_PATH / "build/sim/adc/frida65a_supply_noise_vs_rate/20260821_182756"
    measured_paths = (
        (
            "adc00",
            (
                physical_meas_read_dir
                / "0000_00_adc00_320mbd_dcp50mv_logicp2sym_vcm700mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5",
                physical_meas_read_dir
                / "0001_00_adc00_960mbd_dcp50mv_logicp2sym_vcm700mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5",
                physical_meas_read_dir
                / "0002_00_adc00_1600mbd_dcp50mv_logicp2sym_vcm700mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5",
            ),
        ),
        (
            "adc00_external",
            (
                external_meas_read_dir
                / "0000_00_adc00_320mbd_dcp50mv_logicp2sym_vcm700mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5",
                external_meas_read_dir
                / "0001_00_adc00_960mbd_dcp50mv_logicp2sym_vcm700mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5",
                external_meas_read_dir
                / "0002_00_adc00_1600mbd_dcp50mv_logicp2sym_vcm700mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5",
            ),
        ),
        (
            "adc00_all_active",
            (
                all_active_meas_read_dir
                / "0000_00_adc00_320mbd_dcp50mv_logicp2sym_vcm700mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5",
                all_active_meas_read_dir
                / "0001_00_adc00_960mbd_dcp50mv_logicp2sym_vcm700mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5",
                all_active_meas_read_dir
                / "0002_00_adc00_1600mbd_dcp50mv_logicp2sym_vcm700mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5",
            ),
        ),
    )
    simulated_paths = (
        (
            "spice_hdl21gen",
            (
                ideal_meas_read_dir / "2msps_cm700mv_dc50mv/result.h5",
                ideal_meas_read_dir / "6msps_cm700mv_dc50mv/result.h5",
                ideal_meas_read_dir / "10msps_cm700mv_dc50mv/result.h5",
            ),
        ),
        (
            "spice_frida65a_pex",
            (
                pex_meas_read_dir / "2msps_cm700mv_dc50mv/result.h5",
                pex_meas_read_dir / "6msps_cm700mv_dc50mv/result.h5",
                pex_meas_read_dir / "10msps_cm700mv_dc50mv/result.h5",
            ),
        ),
    )
    measured_sets = []
    for output_prefix, paths in measured_paths:
        measurements = []
        for path in paths:
            measurement = read_measurement(path)
            if not isinstance(measurement, MeasAdcExt):
                raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcExt")
            measurements.append(measurement)
        measured_sets.append((output_prefix, measurements, analyze_adc_noise_sweep(measurements)))

    simulated_sets = []
    for output_prefix, paths in simulated_paths:
        measurements = []
        for path in paths:
            measurement = read_measurement(path)
            if not isinstance(measurement, MeasAdcInt):
                raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcInt")
            measurements.append(measurement)
        simulated_sets.append((output_prefix, measurements, analyze_adc_noise_sweep(measurements)))

    supply_measurements_by_noise: dict[tuple[float, ...], list[MeasAdcInt]] = {}
    for path in sorted(supply_noise_meas_read_dir.glob("*/result.h5")):
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasAdcInt):
            raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcInt")
        rail_noise_rms_v = tuple(float(value) for value in measurement.param.supply_noise_rms_v)
        supply_measurements_by_noise.setdefault(rail_noise_rms_v, []).append(measurement)
    supply_noise_sets = []
    for rail_noise_rms_v, measurements in supply_measurements_by_noise.items():
        noisy_rails = tuple(
            rail
            for rail, noise_rms_v in zip(("vdda", "vddd", "vddac"), rail_noise_rms_v, strict=True)
            if noise_rms_v > 0.0
        )
        if not noisy_rails:
            noise_name = "none"
        elif len(noisy_rails) == 3:
            noise_name = "all"
        else:
            noise_name = "_".join(noisy_rails)
        measurements.sort(key=lambda measurement: float(measurement.param.symbol_rate))
        supply_noise_sets.append(
            (
                f"spice_frida65a_pex_supply_{noise_name}",
                measurements,
                analyze_adc_noise_sweep(measurements),
            )
        )
    supply_noise_order = {name: index for index, name in enumerate(("none", "vdda", "vddd", "vddac", "all"))}
    supply_noise_sets.sort(
        key=lambda item: supply_noise_order.get(item[0].removeprefix("spice_frida65a_pex_supply_"), 5)
    )
    simulated_sets.extend(supply_noise_sets)

    artifacts = []
    for output_prefix, measurements, noise_analysis in measured_sets:
        artifacts.extend(
            plot_adc_noise_sweep(
                measurements,
                noise_analysis,
                output_path=output_dir / f"{output_prefix}_50mv_noise_vs_conversion_rate",
            )
        )
    for output_prefix, measurements, noise_analysis in (*measured_sets, *simulated_sets):
        artifacts.extend(
            plot_adc_noise_distribution_sweep(
                measurements,
                noise_analysis,
                output_path=output_dir / f"{output_prefix}_50mv_output_code_distributions",
            )
        )
        for measurement, active_rate_hz in zip(
            measurements,
            noise_analysis.active_conversion_rate_hz,
            strict=True,
        ):
            artifacts.extend(
                plot_adc_decision_path_density(
                    measurement,
                    analyze_adc_decision_paths(measurement, selection="all"),
                    output_path=output_dir
                    / f"{output_prefix}_50mv_{float(active_rate_hz) / 1e6:g}msps_decision_path_density",
                )
            )
    return tuple(artifacts)


def adc_noise_density_grid(output_dir: Path) -> tuple[Path, ...]:
    """Compare the final manual-supply fixed-input captures for all 16 ADCs."""

    meas_read_dirs = (
        BASE_PATH / "build/scan_adc/20260824_165039",
        BASE_PATH / "build/scan_adc/20260824_234702",
    )
    artifacts = []
    for meas_read_dir in meas_read_dirs:
        measurements_by_adc: dict[int, list[MeasAdcExt]] = {}
        for path in sorted(meas_read_dir.glob("*.h5")):
            measurement = read_measurement(path)
            if not isinstance(measurement, MeasAdcExt):
                raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcExt")
            measurements_by_adc.setdefault(int(measurement.param.observed_adc), []).append(measurement)

        measurement_groups = tuple(tuple(measurements_by_adc[adc_index]) for adc_index in sorted(measurements_by_adc))
        analyses = tuple(analyze_adc_noise_sweep(measurements) for measurements in measurement_groups)
        first_measurement = measurement_groups[0][0]
        input_mv = float(first_measurement.param.tb.vin_diff.dc) * 1e3
        common_mode_mv = float(first_measurement.param.tb.vin_cm.dc) * 1e3
        artifacts.extend(
            plot_adc_noise_distribution_grid(
                measurement_groups,
                analyses,
                output_path=output_dir / f"adc00_adc15_{input_mv:g}mv_{common_mode_mv:g}mv_output_code_density_grid",
            )
        )
    return tuple(artifacts)


def adc_pex_flavor_paths(output_dir: Path, *, inputs: Path | None = None) -> tuple[Path, ...]:
    """Plot trajectories, settling, and held sampling noise for completed PEX flavors."""

    meas_read_dir = inputs or BASE_PATH / "build/sim/adc/frida65a_noise_vs_rate/20260827_165917"
    paths = sorted(meas_read_dir.rglob("result.h5"))
    if not paths:
        raise FileNotFoundError(f"no completed HDF5 results beneath {meas_read_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    sampling_analyses = []
    sampling_labels = []
    sampling_rows = []
    sampling_summary = []
    for path in paths:
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasAdcInt):
            raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcInt")
        noise = analyze_adc_noise_sweep((measurement,))
        input_mv = float(measurement.param.vin_diff.dc) * 1e3
        active_rate_msps = float(noise.active_conversion_rate_hz[0]) / 1e6
        flavor = "_".join(path.relative_to(meas_read_dir).parts[:-1]) or measurement.param.pex_cell
        codes = output_dir / f"{flavor}_codes.txt"
        codes.write_text(
            "sample B0_to_B16 DOUT_decimal DOUT_12bit\n"
            + "".join(
                f"{index} {''.join(str(int(bit)) for bit in bits)} {int(code)} {int(code):012b}\n"
                for index, (bits, code) in enumerate(zip(measurement.daq.bout, measurement.daq.dout, strict=True))
            )
        )
        artifacts.append(codes)
        artifacts.extend(
            plot_adc_decision_path_density(
                measurement,
                analyze_adc_decision_paths(measurement, selection="all"),
                output_path=output_dir
                / f"spice_{flavor}_{input_mv:g}mv_{active_rate_msps:g}msps_decision_path_density",
            )
        )
        artifacts.extend(
            plot_adc_cdac_settling(
                measurement,
                analyze_adc_cdac_settling(measurement),
                output_path=output_dir / f"spice_{flavor}_{input_mv:g}mv_{active_rate_msps:g}msps_cdac_settling",
            )
        )
        sampling = analyze_adc_sampling_noise(measurement)
        sampling_analyses.append(sampling)
        label = path.parent.name.replace("frida", "FRIDA-").replace("layer", "L").replace("radix", "R")
        sampling_labels.append(f"{label.replace('_', ' ')}\n{input_mv:g} mV, {active_rate_msps:g} MS/s active")
        sampling_rows.extend(
            (str(path.relative_to(meas_read_dir)), int(index), start, stop, vp, vn, vp - vn, vin)
            for index, start, stop, vp, vn, vin in zip(
                sampling.conversion_index,
                sampling.window_start_s,
                sampling.window_stop_s,
                sampling.held_p_v,
                sampling.held_n_v,
                sampling.input_diff_v,
                strict=True,
            )
        )
        sampling_summary.append(
            (
                str(path.relative_to(meas_read_dir)),
                len(sampling.conversion_index),
                float(np.mean(sampling.held_diff_v)) * 1e3,
                sampling.sigma_v * 1e6,
                float(np.mean(sampling.held_diff_v - sampling.input_diff_v)) * 1e6,
                float(np.min(sampling.window_start_s)) * 1e9,
                float(np.max(sampling.window_stop_s)) * 1e9,
            )
        )
    sampling_order = sorted(range(len(sampling_labels)), key=sampling_labels.__getitem__)
    artifacts.extend(
        plot_adc_sampling_noise(
            [sampling_analyses[index] for index in sampling_order],
            labels=[sampling_labels[index] for index in sampling_order],
            output_path=output_dir / "adc_sampling_noise",
        )
    )
    for filename, header, rows in (
        (
            "adc_sampling_levels.csv",
            (
                "source_h5",
                "conversion",
                "window_start_s",
                "window_stop_s",
                "held_p_v",
                "held_n_v",
                "held_diff_v",
                "input_diff_v",
            ),
            sampling_rows,
        ),
        (
            "adc_sampling_noise.csv",
            (
                "source_h5",
                "conversions",
                "held_mean_mv",
                "held_sd_uv",
                "mean_offset_uv",
                "earliest_window_start_ns",
                "latest_window_stop_ns",
            ),
            sampling_summary,
        ),
    ):
        destination = output_dir / filename
        with destination.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        artifacts.append(destination)
    return tuple(artifacts)


def adc_pex_cdac_settling(output_dir: Path) -> tuple[Path, ...]:
    """Plot representative internal CDAC settling for each extracted ADC flavor."""

    meas_read_dir = BASE_PATH / "build/sim/adc/frida65a_noise_vs_rate/20260827_165917"
    artifacts = []
    for path in sorted(meas_read_dir.glob("*/*/result.h5")):
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasAdcInt):
            raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcInt")
        analysis = analyze_adc_cdac_settling(measurement)
        input_mv = float(measurement.param.vin_diff.dc) * 1e3
        active_rate_msps = analysis.active_conversion_rate_hz / 1e6
        artifacts.extend(
            plot_adc_cdac_settling(
                measurement,
                analysis,
                output_path=output_dir
                / f"spice_{measurement.param.pex_cell}_{input_mv:g}mv_{active_rate_msps:g}msps_cdac_settling",
            )
        )
    return tuple(artifacts)


def adc_noise_vs_rate(output_dir: Path) -> tuple[Path, ...]:
    """Compare configured physical ADC input-referred noise across rate and backends."""

    physical_meas_read_dirs = (
        BASE_PATH / "build/scan_adc/20260801_194930",
        BASE_PATH / "build/scan_adc/20260802_021624",
    )
    sine_meas_read_dir = BASE_PATH / "build/scan_adc/20260730_215145_complete"
    ideal_meas_read_dir = BASE_PATH / "build/adc/hdl21gen_noise_vs_rate/20260801_0821"
    pex_meas_read_dir = BASE_PATH / "build/adc/frida65a_noise_vs_rate/20260731_2353"
    ideal_measurement_paths = (
        ideal_meas_read_dir / "2msps_cm600mv_dc50mv/result.h5",
        ideal_meas_read_dir / "6msps_cm600mv_dc50mv/result.h5",
        ideal_meas_read_dir / "10msps_cm600mv_dc50mv/result.h5",
    )
    pex_measurement_paths = (
        pex_meas_read_dir / "2msps_cm600mv_dc50mv/result.h5",
        pex_meas_read_dir / "6msps_cm600mv_dc50mv/result.h5",
        pex_meas_read_dir / "10msps_cm600mv_dc50mv/result.h5",
    )

    physical_measurement_sets = []
    for meas_read_dir in physical_meas_read_dirs:
        measurements = []
        for path in sorted(meas_read_dir.glob("*.h5")):
            measurement = read_measurement(path)
            if not isinstance(measurement, MeasAdcExt):
                raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcExt")
            measurements.append(measurement)
        physical_measurement_sets.append(tuple(measurements))
    physical_measurement_sets.sort(key=lambda measurements: float(measurements[0].param.tb.vin_diff.dc))

    sine_measurements = []
    for path in sorted(sine_meas_read_dir.glob("*.h5")):
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasAdcExt):
            raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcExt")
        sine_measurements.append(measurement)

    ideal_measurements = []
    for path in ideal_measurement_paths:
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasAdcInt):
            raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcInt")
        ideal_measurements.append(measurement)
    pex_measurements = []
    for path in pex_measurement_paths:
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasAdcInt):
            raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcInt")
        pex_measurements.append(measurement)

    adc_indices = tuple(
        sorted(
            {
                int(measurement.param.observed_adc)
                for measurements in physical_measurement_sets
                for measurement in measurements
            }
        )
    )
    ideal_noise = analyze_adc_noise_sweep(ideal_measurements)
    pex_noise = analyze_adc_noise_sweep(pex_measurements)
    artifacts = []
    for adc_index in adc_indices:
        adc_physical_measurement_sets = [
            [measurement for measurement in measurements if measurement.param.observed_adc == adc_index]
            for measurements in physical_measurement_sets
        ]
        adc_sine_measurements = [
            measurement for measurement in sine_measurements if measurement.param.observed_adc == adc_index
        ]
        physical_noise_sweeps = [
            analyze_adc_noise_sweep(measurements) for measurements in adc_physical_measurement_sets
        ]
        sine_dynamic = analyze_adc_dynamic_sweep(adc_sine_measurements)
        comparison_measurements = [
            *adc_physical_measurement_sets[0],
            *(measurement for measurements in adc_physical_measurement_sets for measurement in measurements),
            *adc_sine_measurements,
        ]
        series_labels = [
            *("Input stimulus noise" for _ in adc_physical_measurement_sets[0]),
            *(
                f"Measured ({float(measurements[0].param.tb.vin_diff.dc) * 1e3:g} mV DC)"
                for measurements in adc_physical_measurement_sets
                for _ in measurements
            ),
            *("Measured (1 V sine)" for _ in adc_sine_measurements),
        ]
        simulated_noise_sweeps = []
        if adc_index == adc_indices[0]:
            simulated_noise_sweeps.extend((ideal_noise, pex_noise))
            comparison_measurements.extend((*ideal_measurements, *pex_measurements))
            series_labels.extend(
                (
                    *("SPICE Ideal (50 mV DC)" for _ in ideal_measurements),
                    *("SPICE PEX (50 mV DC)" for _ in pex_measurements),
                )
            )

        comparison = combine_adc_noise_comparison(
            physical_noise_sweeps,
            sine_dynamic,
            simulated_noise_sweeps,
            series_labels=series_labels,
        )
        artifacts.extend(
            plot_adc_noise_sweep(
                comparison_measurements,
                comparison,
                output_path=output_dir / f"adc{adc_index:02d}_noise_vs_conversion_rate",
            )
        )
    return tuple(artifacts)


def adc_code_distributions(output_dir: Path) -> tuple[Path, ...]:
    """Plot configured ADC fixed-input distributions and selected decision paths."""

    physical_meas_read_dirs = (
        BASE_PATH / "build/scan_adc/20260801_194930",
        BASE_PATH / "build/scan_adc/20260802_021624",
    )
    ideal_meas_read_dir = BASE_PATH / "build/adc/hdl21gen_noise_vs_rate/20260801_0821"
    ideal_measurement_paths = (
        ideal_meas_read_dir / "2msps_cm600mv_dc50mv/result.h5",
        ideal_meas_read_dir / "6msps_cm600mv_dc50mv/result.h5",
        ideal_meas_read_dir / "10msps_cm600mv_dc50mv/result.h5",
    )
    decision_path_rates_hz = (2.0e6, 10.0e6)

    physical_measurement_sets = []
    for meas_read_dir in physical_meas_read_dirs:
        measurements = []
        for path in sorted(meas_read_dir.glob("*.h5")):
            measurement = read_measurement(path)
            if not isinstance(measurement, MeasAdcExt):
                raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcExt")
            measurements.append(measurement)
        physical_measurement_sets.append(tuple(measurements))
    physical_measurement_sets.sort(key=lambda measurements: float(measurements[0].param.tb.vin_diff.dc))
    adc_indices = tuple(
        sorted(
            {
                int(measurement.param.observed_adc)
                for measurements in physical_measurement_sets
                for measurement in measurements
            }
        )
    )

    ideal_measurements = []
    for path in ideal_measurement_paths:
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasAdcInt):
            raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcInt")
        ideal_measurements.append(measurement)
    ideal_noise = analyze_adc_noise_sweep(ideal_measurements)

    artifacts = []
    for measurements in physical_measurement_sets:
        input_mv = float(measurements[0].param.tb.vin_diff.dc) * 1e3
        for adc_index in adc_indices:
            adc_measurements = [
                measurement for measurement in measurements if measurement.param.observed_adc == adc_index
            ]
            artifacts.extend(
                plot_adc_noise_distribution_sweep(
                    adc_measurements,
                    analyze_adc_noise_sweep(adc_measurements),
                    output_path=output_dir / f"adc{adc_index:02d}_{input_mv:g}mv_dc_output_code_distributions",
                )
            )

    for measurement, active_rate_hz in zip(
        ideal_measurements,
        ideal_noise.active_conversion_rate_hz,
        strict=True,
    ):
        rate_msps = float(active_rate_hz) / 1e6
        artifacts.extend(
            plot_adc_code_distribution(
                [measurement],
                analyze_adc_code_distribution([measurement]),
                output_path=output_dir / f"spice_hdl21gen_{rate_msps:g}msps_output_code_histogram",
            )
        )
        artifacts.extend(
            plot_adc_decision_paths(
                measurement,
                analyze_adc_decision_paths(measurement, selection="all"),
                output_path=output_dir / f"spice_hdl21gen_{rate_msps:g}msps_decision_paths",
            )
        )

    for measurements in physical_measurement_sets:
        input_mv = float(measurements[0].param.tb.vin_diff.dc) * 1e3
        for adc_index in adc_indices:
            adc_measurements = [
                measurement for measurement in measurements if measurement.param.observed_adc == adc_index
            ]
            noise = analyze_adc_noise_sweep(adc_measurements)
            for requested_rate_hz in decision_path_rates_hz:
                measurement_index = int(
                    np.flatnonzero(np.isclose(noise.active_conversion_rate_hz, requested_rate_hz))[0]
                )
                measurement = adc_measurements[measurement_index]
                rate_msps = float(noise.active_conversion_rate_hz[measurement_index]) / 1e6
                analysis = analyze_adc_decision_paths(measurement, selection="all")
                output_path = output_dir / (
                    f"adc{adc_index:02d}_{input_mv:g}mv_{rate_msps:g}msps_decision_path_density"
                )
                artifacts.extend(plot_adc_decision_path_density(measurement, analysis, output_path=output_path))
    return tuple(artifacts)


def adc_power_vs_rate(output_dir: Path) -> tuple[Path, ...]:
    """Plot measured and simulated power sweeps plus detailed waveforms."""

    sine_meas_read_dir = BASE_PATH / "build/scan_adc/20260730_215145_complete"
    ideal_meas_read_dir = BASE_PATH / "build/adc/hdl21gen_noise_vs_rate/20260801_0821"
    pex_meas_read_dir = BASE_PATH / "build/adc/frida65a_noise_vs_rate/20260731_2353"
    simulation_measurement_paths = {
        "ideal": (
            ideal_meas_read_dir / "2msps_cm600mv_dc50mv/result.h5",
            ideal_meas_read_dir / "6msps_cm600mv_dc50mv/result.h5",
            ideal_meas_read_dir / "10msps_cm600mv_dc50mv/result.h5",
        ),
        "pex": (
            pex_meas_read_dir / "2msps_cm600mv_dc50mv/result.h5",
            pex_meas_read_dir / "6msps_cm600mv_dc50mv/result.h5",
            pex_meas_read_dir / "10msps_cm600mv_dc50mv/result.h5",
        ),
    }

    sine_measurements = []
    for path in sorted(sine_meas_read_dir.glob("*.h5")):
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasAdcExt):
            raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcExt")
        sine_measurements.append(measurement)
    adc_indices = tuple(sorted({int(measurement.param.observed_adc) for measurement in sine_measurements}))

    simulation_measurements = {}
    for source, paths in simulation_measurement_paths.items():
        measurements = []
        for path in paths:
            measurement = read_measurement(path)
            if not isinstance(measurement, MeasAdcInt):
                raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcInt")
            measurements.append(measurement)
        simulation_measurements[source] = measurements

    simulation_power = {
        source: analyze_adc_power_sweep(measurements) for source, measurements in simulation_measurements.items()
    }

    artifacts = []
    physical_power = {}
    for adc_index in adc_indices:
        measurements = [measurement for measurement in sine_measurements if measurement.param.observed_adc == adc_index]
        analysis = analyze_adc_power_sweep(measurements)
        physical_power[adc_index] = (measurements, analysis)
        artifacts.extend(
            plot_adc_power_sweep(
                measurements,
                analysis,
                output_path=output_dir / f"adc_power_vs_conversion_rate_adc{adc_index:02d}",
            )
        )
    for source in ("ideal", "pex"):
        measurements = simulation_measurements[source]
        analysis = simulation_power[source]
        artifacts.extend(
            plot_adc_power_sweep(
                measurements,
                analysis,
                output_path=output_dir / f"spice_{source}_power_vs_conversion_rate",
            )
        )
        detail_index = int(np.argmax(analysis.active_conversion_rate_hz))
        detail_measurement = measurements[detail_index]
        detail_rate_msps = float(analysis.active_conversion_rate_hz[detail_index]) / 1e6
        artifacts.extend(
            plot_adc_power_waveform(
                analyze_adc_power_waveform(detail_measurement),
                output_path=output_dir / f"spice_{source}_{detail_rate_msps:g}msps_supply_power",
            )
        )
    for adc_index, (measurements, analysis) in physical_power.items():
        detail_index = int(np.argmin(analysis.active_conversion_rate_hz))
        detail_measurement = measurements[detail_index]
        detail_rate_mbd = float(detail_measurement.param.tb.symbol_rate) / 1e6
        artifacts.extend(
            plot_waveforms(
                analyze_measurement_waveforms(detail_measurement),
                output_path=output_dir / f"adc{adc_index:02d}_{detail_rate_mbd:g}mbd_sine_waveforms",
            )
        )
        artifacts.extend(
            plot_adc_dynamic(
                detail_measurement,
                analyze_adc_dynamic(detail_measurement),
                output_path=output_dir / f"adc{adc_index:02d}_{detail_rate_mbd:g}mbd_sine_fit_and_spectrum",
            )
        )
    return tuple(artifacts)


def adc_rate_characterization(output_dir: Path) -> tuple[Path, ...]:
    """Run the configured ADC rate-sweep noise, code-distribution, and power analyses."""

    return (
        *adc_noise_vs_rate(output_dir),
        *adc_code_distributions(output_dir),
        *adc_power_vs_rate(output_dir),
    )


def adc_noise_vs_comp_time(output_dir: Path) -> tuple[Path, ...]:
    """Plot ADC input-referred noise versus conversion rate and comparator timing."""

    meas_read_dirs = (
        BASE_PATH / "build/scan_adc/20260802_081407",
        BASE_PATH / "build/loopback_fastrx/20260729_181030",
    )
    artifacts = []
    for meas_read_dir in meas_read_dirs:
        measurements = []
        for path in sorted(meas_read_dir.glob("*.h5")):
            measurement = read_measurement(path)
            if not isinstance(measurement, MeasAdcExt):
                raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasAdcExt")
            measurements.append(measurement)

        adc_index = int(measurements[0].param.observed_adc)
        noise = analyze_adc_noise_sweep(measurements)
        artifacts.extend(
            plot_adc_noise_sweep(
                measurements,
                noise,
                output_path=output_dir / f"adc{adc_index:02d}_noise_vs_conversion_rate_and_logic_offset",
            )
        )
    return tuple(artifacts)


def comp_system_common_mode(output_dir: Path) -> tuple[Path, ...]:
    """Analyze and plot separate ADC00–ADC03 comparator common-mode campaigns."""

    meas_read_dir = BASE_PATH / "build/scan_comp/20260805_171216"
    measurements = []
    for path in sorted(meas_read_dir.glob("*.h5")):
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasCompExt):
            raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasCompExt")
        measurements.append(measurement)

    artifacts = []
    adc_indices = tuple(sorted({int(measurement.param.observed_adc) for measurement in measurements}))
    for adc_index in adc_indices:
        adc_measurements = [measurement for measurement in measurements if measurement.param.observed_adc == adc_index]
        grouped: dict[float, list[MeasCompExt]] = {}
        for measurement in adc_measurements:
            grouped.setdefault(float(measurement.param.tb.vin_cm.dc), []).append(measurement)
        groups = [grouped[value] for value in sorted(grouped)]
        analyses = [analyze_comp_offset_noise(group) for group in groups]
        analyses = list(classify_comp_common_mode_validity(groups, analyses))
        artifacts.extend(
            plot_comp_common_mode_campaign(
                groups,
                analyses,
                output_path=output_dir / f"adc{adc_index:02d}_comparator_common_mode",
            )
        )
    return tuple(artifacts)


def comp_system_sampling_noise(output_dir: Path) -> tuple[Path, ...]:
    """Analyze and plot separate ADC00–ADC03 track/hold comparator campaigns."""

    base_meas_read_dir = BASE_PATH / "build/scan_comp/20260805_183915"
    correction_meas_read_dir = BASE_PATH / "build/scan_comp/20260805_192902"
    base_measurements = []
    for path in sorted(base_meas_read_dir.glob("*.h5")):
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasCompExt):
            raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasCompExt")
        base_measurements.append(measurement)
    correction_measurements = []
    for path in sorted(correction_meas_read_dir.glob("*.h5")):
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasCompExt):
            raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasCompExt")
        correction_measurements.append(measurement)

    corrected_curve_keys = {
        (
            int(measurement.param.observed_adc),
            float(measurement.param.requested_dac_rail_percent),
            measurement.param.sampling_mode,
        )
        for measurement in correction_measurements
    }
    measurements = [
        measurement
        for measurement in base_measurements
        if (
            int(measurement.param.observed_adc),
            float(measurement.param.requested_dac_rail_percent),
            measurement.param.sampling_mode,
        )
        not in corrected_curve_keys
    ]
    measurements.extend(correction_measurements)

    artifacts = []
    adc_indices = tuple(sorted({int(measurement.param.observed_adc) for measurement in measurements}))
    for adc_index in adc_indices:
        adc_measurements = [measurement for measurement in measurements if measurement.param.observed_adc == adc_index]
        grouped: dict[tuple[float, str], list[MeasCompExt]] = {}
        for measurement in adc_measurements:
            group_key = (
                float(measurement.param.requested_dac_rail_percent),
                measurement.param.sampling_mode,
            )
            grouped.setdefault(group_key, []).append(measurement)
        groups = [grouped[key] for key in sorted(grouped)]
        analyses = [analyze_comp_offset_noise(group) for group in groups]
        artifacts.extend(
            plot_comp_sampling_campaign(
                groups,
                analyses,
                output_path=output_dir / f"adc{adc_index:02d}_comparator_sampling_noise",
            )
        )
    return tuple(artifacts)


def comp_candidate_sweep(output_dir: Path) -> tuple[Path, ...]:
    """Analyze the complete generated-comparator noise/power/timing campaign."""

    meas_read_dir = BASE_PATH / "build/comp/frida65_candidate_scurve_power/candidates"
    measurements = []
    for path in sorted(meas_read_dir.glob("*/result.h5")):
        measurement = read_measurement(path)
        if not isinstance(measurement, MeasCompInt):
            raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasCompInt")
        measurements.append(measurement)
    analysis = analyze_comp_candidate_sweep(measurements)
    artifacts = list(
        plot_comp_candidate_sweep(
            measurements,
            analysis,
            output_path=output_dir / "comp_candidate_noise_power_settling",
        )
    )
    artifacts.extend(
        plot_comp_noise_power_tradeoff(
            analysis,
            output_path=output_dir / "comp_candidate_noise_power_tradeoff",
        )
    )
    csv_path = output_dir / "comp_candidate_noise_power_settling.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "area_order",
                "candidate_id",
                "candidate_label",
                "size_profile",
                "topology_index",
                "total_width_units",
                "total_active_area_units",
                "total_active_area_um2",
                "device_count",
                "comp_stages",
                "preamp_diff_xtors",
                "preamp_bias",
                "latch_inner_on_xtors",
                "latch_outer_on_xtors",
                "latch_inner_init_xtors",
                "latch_outer_init_xtors",
                "diffpair_w",
                "diffpair_l",
                "tail_w",
                "tail_l",
                "rst_w",
                "rst_l",
                "latch_on_w",
                "latch_on_l",
                "latch_init_w",
                "latch_init_l",
                "srlatch_n_w",
                "srlatch_p_w",
                "validity",
                "offset_v",
                "noise_sigma_v",
                "average_power_w",
                "energy_per_decision_j",
                "maximum_clock_to_decision_s",
                "maximum_settling_s",
                "unresolved_fraction",
            )
        )
        measurement_by_id = {
            str(measurement.info.readbacks["candidate_id"]): measurement for measurement in measurements
        }
        for index, candidate_id in enumerate(analysis.candidate_id):
            comp = measurement_by_id[candidate_id].param.comp
            writer.writerow(
                (
                    index,
                    candidate_id,
                    analysis.candidate_label[index],
                    analysis.size_profile[index],
                    analysis.topology_index[index],
                    analysis.total_width_units[index],
                    analysis.total_active_area_units[index],
                    analysis.total_active_area_um2[index],
                    analysis.device_count[index],
                    comp.comp_stages.name,
                    comp.preamp_diff_xtors.name,
                    comp.preamp_bias.name,
                    comp.latch_inner_on_xtors.name,
                    comp.latch_outer_on_xtors.name,
                    comp.latch_inner_init_xtors.name,
                    comp.latch_outer_init_xtors.name,
                    comp.diffpair_w,
                    comp.diffpair_l,
                    comp.tail_w,
                    comp.tail_l,
                    comp.rst_w,
                    comp.rst_l,
                    comp.latch_on_w,
                    comp.latch_on_l,
                    comp.latch_init_w,
                    comp.latch_init_l,
                    comp.srlatch_n_w,
                    comp.srlatch_p_w,
                    analysis.validity[index],
                    analysis.offset_v[index],
                    analysis.noise_sigma_v[index],
                    analysis.average_power_w[index],
                    analysis.energy_per_decision_j[index],
                    analysis.maximum_clock_to_decision_s[index],
                    analysis.maximum_settling_s[index],
                    analysis.unresolved_fraction[index],
                )
            )
    artifacts.append(csv_path)
    return tuple(artifacts)


def cdac_system_cap_mismatch(output_dir: Path) -> tuple[Path, ...]:
    """Extract and plot ADC00–ADC03 capacitor mismatch from A-to-B transitions."""

    meas_read_dirs = tuple(
        BASE_PATH / "build/scan_cdac" / name for name in ("20260804_171234", "20260804_193030", "20260804_193631")
    )
    measurement_runs = []
    for meas_read_dir in meas_read_dirs:
        measurements = []
        for path in sorted(meas_read_dir.glob("*.h5")):
            measurement = read_measurement(path)
            if not isinstance(measurement, MeasCdacExt):
                raise TypeError(f"{path} contains {type(measurement).__name__}, expected MeasCdacExt")
            measurements.append(measurement)
        measurement_runs.append(tuple(measurements))

    adc_indices = tuple(
        sorted(
            {int(measurement.param.observed_adc) for measurements in measurement_runs for measurement in measurements}
        )
    )
    board_id = measurement_runs[0][0].param.board_id
    comparator_calibrations = load_board_map()["boards"][board_id].get("comparator_calibration", {})
    adc_groups, analyses = analyze_cdac_cap_mismatch_campaign(
        measurement_runs,
        adc_indices=adc_indices,
        board_id=board_id,
        comparator_offset_v_by_adc={
            adc_index: float(comparator_calibrations[adc_index]["offset_v"]) for adc_index in adc_indices
        },
    )
    artifacts = []
    for adc_measurements, analysis in zip(adc_groups, analyses, strict=True):
        artifacts.extend(
            plot_cdac_cap_mismatch(
                adc_measurements,
                analysis,
                output_path=output_dir / f"adc{analysis.adc_index:02d}_cdac_cap_mismatch",
            )
        )
    artifacts.extend(
        plot_cdac_cap_mismatch_comparison(
            adc_groups,
            analyses,
            output_path=output_dir / "adc00_adc03_cdac_cap_mismatch_comparison",
        )
    )
    return tuple(artifacts)


TARGETS: dict[str, Callable[[Path], tuple[Path, ...]]] = {
    target.__name__: target
    for target in (
        adc_transfer_curve,
        adc_ramp_nonlinearity,
        adc_calibration,
        adc00_fixed_input_noise,
        adc_noise_density_grid,
        adc_pex_flavor_paths,
        adc_pex_cdac_settling,
        adc_noise_vs_rate,
        adc_code_distributions,
        adc_power_vs_rate,
        adc_rate_characterization,
        adc_noise_vs_comp_time,
        comp_system_common_mode,
        comp_system_sampling_noise,
        comp_candidate_sweep,
        cdac_system_cap_mismatch,
    )
}
AGGREGATE_TARGETS = {"adc_rate_characterization"}


def main() -> None:
    """Run one named analysis pipeline, or every target when none is named."""

    # TODO: Change analysis roots to build/analysis_<domain>/<short-datetime>.
    ANALYSIS_OUTPUT_BASE = BASE_PATH / "build/analysis"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, help="Completed campaign directory for adc_pex_flavor_paths")
    parser.add_argument(
        "target",
        nargs="?",
        choices=sorted(TARGETS),
        help="analysis-pipeline function to run; omit to run all targets",
    )
    args = parser.parse_args()
    if args.inputs is not None and args.target != "adc_pex_flavor_paths":
        parser.error("--inputs requires adc_pex_flavor_paths")
    run_all = args.target is None
    target_names = tuple(name for name in TARGETS if name not in AGGREGATE_TARGETS) if run_all else (args.target,)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M")
    output_dirs: dict[str, Path] = {}
    for target_name in target_names:
        # TODO: Route cdac_* targets to analysis_cdac instead of analysis_adc.
        domain = "comp" if target_name.startswith("comp_") else "adc"
        output_dir = output_dirs.get(domain)
        if output_dir is None:
            output_dir = ANALYSIS_OUTPUT_BASE / domain / timestamp
            output_dir.mkdir(parents=True, exist_ok=False)
            output_dirs[domain] = output_dir
            print(f"Analysis output: {output_dir}")
        start_time = perf_counter()
        try:
            if args.inputs is not None:
                artifacts = adc_pex_flavor_paths(output_dir, inputs=args.inputs)
            else:
                artifacts = TARGETS[target_name](output_dir)
        except FileNotFoundError as error:
            if not run_all:
                raise
            runtime_s = perf_counter() - start_time
            print(f"Skipped {target_name}: missing {error.filename} after {runtime_s:.2f} s")
            continue
        runtime_s = perf_counter() - start_time
        print(f"Completed {target_name}: {len(artifacts)} artifacts in {runtime_s:.2f} s")


if __name__ == "__main__":
    main()
