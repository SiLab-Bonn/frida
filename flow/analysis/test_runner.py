"""Software-only tests for the explicit analysis-pipeline command line."""

from __future__ import annotations

import ast
import dataclasses
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import hdl21 as h
import numpy as np
import pytest

import flow.analysis as analysis_api
import flow.analysis.cdac as cdac_analysis
from flow.analysis import runner
from flow.analysis.adc import analyze_adc_ramp
from flow.analysis.test_adc import adc_measurement, adc_ramp_measurement
from flow.analysis.types import AdcCalibrationMethod, AnalysisAdcCalibration


def test_root_api_exposes_domain_analyses_not_campaign_combiners() -> None:
    """Keep the package root focused on reusable measurement analyses."""

    assert hasattr(analysis_api, "analyze_adc_code_distribution")
    assert hasattr(analysis_api, "analyze_adc_nonlinearity")
    assert hasattr(analysis_api, "analyze_adc_ramp")
    assert hasattr(analysis_api, "analyze_cdac_cap_mismatch")
    assert not hasattr(analysis_api, "combine_adc_noise_comparison")
    assert not hasattr(analysis_api, "classify_comp_common_mode_validity")


def test_runner_exposes_only_named_orchestration_entry_points() -> None:
    """Keep the runner surface limited to explicit, user-invoked pipelines."""

    tree = ast.parse(Path(runner.__file__).read_text())
    private_functions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("_")
    ]

    assert private_functions == []
    assert not any(target_name.startswith("adc00_adc01_") for target_name in runner.TARGETS)
    assert "adc_transfer_curve" in runner.TARGETS
    assert "adc_ramp_nonlinearity" in runner.TARGETS
    assert "adc_calibration" in runner.TARGETS
    assert "adc00_fixed_input_noise" in runner.TARGETS
    assert "adc_noise_density_grid" in runner.TARGETS
    assert "adc_pex_flavor_paths" in runner.TARGETS
    assert "adc_pex_cdac_settling" in runner.TARGETS
    assert "adc00_pex_transfer" not in runner.TARGETS
    assert "adc_code_distributions" in runner.TARGETS
    assert "adc_code_diag" not in runner.TARGETS
    assert "cdac_system_cap_mismatch" in runner.TARGETS
    assert "cdac_cap_mismatch" not in runner.TARGETS
    assert all(len(name) <= 26 for name in runner.TARGETS)


def test_adc_calibration_runner_combines_three_common_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurement = adc_ramp_measurement(cycles=8)
    nominal_ramp = analyze_adc_ramp(measurement)
    nominal_weight = nominal_ramp.curves[0].weights.astype(np.float64)
    nominal_weight *= 4095.0 / np.sum(nominal_weight)

    def calibration(method: AdcCalibrationMethod) -> AnalysisAdcCalibration:
        weight = nominal_weight.copy()
        weight[0] *= {"calibration1": 1.01, "calibration2": 0.99, "calibration3": 1.02}[method]
        weight *= 4095.0 / np.sum(weight)
        return AnalysisAdcCalibration(
            adc_index=0,
            method=method,
            label=method,
            code_max=4095,
            nominal_weights=nominal_weight,
            calibrated_weights=weight,
            measured_weight_mask=np.ones(17, dtype=np.bool_),
            training_sample_count=100,
            validation_sample_count=50,
            output_gain=1.0,
            output_offset_lsb=0.0,
        )

    ramp_dir = tmp_path / "build/scan_adc/20260812_011910"
    ramp_dir.mkdir(parents=True)
    ramp_path = ramp_dir / (
        "0000_00_adc00_160mbd_pwl10hz_m1000top1000mv_logicp0sym_vcm600mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5"
    )
    ramp_path.touch()
    cdac_dir = tmp_path / "build/scan_cdac/20260804_171234"
    cdac_dir.mkdir(parents=True)
    cdac_path = cdac_dir / "adc00.h5"
    cdac_path.touch()
    fake_ramp = SimpleNamespace(
        param=SimpleNamespace(campaign="adc_ramp", observed_adc=0, board_id="00"),
        daq=SimpleNamespace(dout=range(4_000_000)),
        info=SimpleNamespace(readbacks={}),
    )
    fake_cdac = SimpleNamespace(param=SimpleNamespace(board_id="00", observed_adc=0))
    cdac_measurements = (fake_cdac,)
    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "MeasAdcExt", SimpleNamespace)
    monkeypatch.setattr(runner, "MeasCdacExt", SimpleNamespace)
    monkeypatch.setattr(runner, "read_measurement", lambda path: fake_ramp if path == ramp_path else fake_cdac)
    monkeypatch.setattr(
        runner,
        "analyze_cdac_cap_mismatch_campaign",
        lambda *_args, **_kwargs: ((cdac_measurements,), (object(),)),
    )
    monkeypatch.setattr(
        runner,
        "load_board_map",
        lambda: {"boards": {"00": {"comparator_calibration": {0: {"offset_v": 0.0}}}}},
    )
    monkeypatch.setattr(
        runner,
        "analyze_calibration1",
        lambda _measurements, *, comparator_offset_v: calibration("calibration1"),
    )
    monkeypatch.setattr(
        runner,
        "analyze_calibration2",
        lambda _measurement, _ramp: calibration("calibration2"),
    )
    monkeypatch.setattr(
        runner,
        "analyze_calibration3",
        lambda _measurement, _ramp: calibration("calibration3"),
    )
    monkeypatch.setattr(
        runner,
        "analyze_adc_ramp",
        lambda _measurement, *, calibrations=(): analyze_adc_ramp(measurement, calibrations=calibrations),
    )
    for name in (
        "plot_adc_calibration_weights",
        "plot_adc_ramp_transfer",
        "plot_adc_ramp_histogram",
        "plot_adc_ramp_nonlinearity",
    ):
        monkeypatch.setattr(runner, name, lambda *_args, **_kwargs: ())

    artifacts = runner.adc_calibration(tmp_path)

    assert [path.name for path in artifacts] == [
        "adc00_calibration_metrics.csv",
        "adc00_calibration_weights.csv",
    ]
    metrics = artifacts[0].read_text()
    weights = artifacts[1].read_text()
    assert all(method in metrics for method in ("calibration1", "calibration2", "calibration3"))
    assert "ideal_weight_lsb" in weights
    assert len(weights.splitlines()) == 18


def test_cdac_analysis_replaces_whole_curves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never concatenate analog points from two acquisition sessions."""

    measurement_runs = [[], [], []]
    for adc_index in range(4):
        for side in ("p", "n"):
            for element in range(16):
                for direction in ("1to0", "0to1"):
                    for diffcaps in (0, 1):
                        measurement_runs[0].append(
                            SimpleNamespace(
                                param=SimpleNamespace(
                                    campaign="cdac_ab",
                                    board_id="test_board",
                                    observed_adc=adc_index,
                                    cdac_side=side,
                                    cdac_element=element,
                                    cdac_direction=direction,
                                    tb=SimpleNamespace(dac_diffcaps=diffcaps),
                                ),
                                info=SimpleNamespace(backend="spice", readbacks={}),
                                point_index=0,
                            )
                        )
    for point_index in (2, 3, 4):
        measurement_runs[1].append(
            SimpleNamespace(
                param=SimpleNamespace(
                    campaign="cdac_ab",
                    board_id="test_board",
                    observed_adc=0,
                    cdac_side="p",
                    cdac_element=0,
                    cdac_direction="1to0",
                    tb=SimpleNamespace(dac_diffcaps=0),
                ),
                info=SimpleNamespace(backend="spice", readbacks={}),
                point_index=point_index,
            )
        )

    analyzed_measurements = {}

    def analyze(measurements, *, comparator_offset_v):
        assert comparator_offset_v == 8e-3
        analyzed_measurements[measurements[0].param.observed_adc] = measurements
        return SimpleNamespace(adc_index=measurements[0].param.observed_adc)

    monkeypatch.setattr(cdac_analysis, "MeasCdacExt", SimpleNamespace)
    monkeypatch.setattr(cdac_analysis, "analyze_cdac_cap_mismatch", analyze)
    groups, _analyses = cdac_analysis.analyze_cdac_cap_mismatch_campaign(
        measurement_runs,
        adc_indices=(0, 1, 2, 3),
        board_id="test_board",
        comparator_offset_v_by_adc={adc_index: 8e-3 for adc_index in range(4)},
    )
    replaced_curve = [
        measurement
        for measurement in groups[0]
        if (
            measurement.param.cdac_side,
            measurement.param.cdac_element,
            measurement.param.cdac_direction,
            measurement.param.tb.dac_diffcaps,
        )
        == ("p", 0, "1to0", 0)
    ]
    assert [cast(Any, measurement).point_index for measurement in replaced_curve] == [2, 3, 4]


def test_adc_transfer_curve_loads_pinned_directory_without_reconstructing_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate transfer-coordinate validation to the typed analysis."""

    meas_read_dir = tmp_path / "build/scan_adc/20260818_135848"
    meas_read_dir.mkdir(parents=True)
    measurements = {}
    for point_index, input_v in enumerate((-0.75, 0.0, 0.75)):
        path = meas_read_dir / f"{point_index:04d}_adc00.h5"
        path.touch()
        measurement = adc_measurement(
            np.full(7, 2048),
            vin_diff_v=input_v,
            sample_rate_hz=3.0e6,
            observed_adc=0,
        )
        measurements[path] = dataclasses.replace(
            measurement,
            info=dataclasses.replace(measurement.info, backend="physical"),
            param=dataclasses.replace(
                measurement.param,
                campaign="adc_transfer",
                board_id="fixture_board",
                tb=dataclasses.replace(
                    measurement.param.tb,
                    conversions=7,
                    vin_cm=h.Vdc.Params(dc=0.615),
                    vin_diff=h.Vdc.Params(dc=input_v),
                ),
            ),
        )
    monkeypatch.setattr(runner, "read_measurement", measurements.__getitem__)
    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "analyze_adc_transfer", lambda measurements: measurements)
    monkeypatch.setattr(
        runner,
        "plot_adc_transfer",
        lambda _measurements, _analysis, *, output_path: (output_path.with_suffix(".png"),),
    )

    assert runner.adc_transfer_curve(tmp_path / "output") == (tmp_path / "output/adc00_transfer_curve.png",)


def test_adc_ramp_runner_reuses_accepted_cdac_analysis_and_completed_ramp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep accepted ramp selection in the runner and plotting measurement-free."""

    measurements = {}
    ramp_dir = tmp_path / "build/scan_adc/20260812_011910"
    ramp_dir.mkdir(parents=True)
    cdac_dir = tmp_path / "build/scan_cdac/20260804_171234"
    cdac_dir.mkdir(parents=True)
    cdac_path = cdac_dir / "adc00.h5"
    cdac_path.touch()
    cdac_groups = []
    cdac_analyses = []
    for adc_index in range(4):
        ramp_path = ramp_dir / (
            f"{adc_index:04d}_00_adc{adc_index:02d}_160mbd_pwl10hz_m1000top1000mv_logicp0sym_"
            "vcm600mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5"
        )
        ramp_path.touch()
        ramp_measurement = SimpleNamespace(
            param=SimpleNamespace(campaign="adc_ramp", observed_adc=adc_index, board_id="00"),
            daq=SimpleNamespace(dout=range(4_000_000)),
            info=SimpleNamespace(readbacks={}),
        )
        measurements[ramp_path] = ramp_measurement
        cdac_groups.append(
            (
                SimpleNamespace(
                    param=SimpleNamespace(observed_adc=adc_index, board_id="00"),
                ),
            )
        )
        cdac_analyses.append(
            SimpleNamespace(
                adc_index=adc_index,
            )
        )
    measurements[cdac_path] = SimpleNamespace(param=SimpleNamespace(observed_adc=0, board_id="00"))
    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "MeasAdcExt", SimpleNamespace)
    monkeypatch.setattr(runner, "MeasCdacExt", SimpleNamespace)
    monkeypatch.setattr(runner, "read_measurement", measurements.__getitem__)
    monkeypatch.setattr(
        runner,
        "analyze_cdac_cap_mismatch_campaign",
        lambda *_args, **_kwargs: (tuple(cdac_groups), tuple(cdac_analyses)),
    )
    monkeypatch.setattr(
        runner,
        "load_board_map",
        lambda: {
            "boards": {"00": {"comparator_calibration": {adc_index: {"offset_v": 0.0} for adc_index in range(4)}}}
        },
    )
    monkeypatch.setattr(
        runner,
        "analyze_calibration1",
        lambda measurements, *, comparator_offset_v: SimpleNamespace(
            adc_index=measurements[0].param.observed_adc,
            comparator_offset_v=comparator_offset_v,
        ),
    )
    monkeypatch.setattr(
        runner,
        "plot_adc_ramp_transfer",
        lambda _analysis, *, output_path: (output_path.with_suffix(".png"),),
    )
    monkeypatch.setattr(
        runner,
        "plot_adc_ramp_histogram",
        lambda _analysis, *, output_path: (output_path.with_suffix(".png"),),
    )
    monkeypatch.setattr(
        runner,
        "plot_adc_ramp_weights",
        lambda _analysis, *, output_path: (output_path.with_suffix(".png"),),
    )
    monkeypatch.setattr(
        runner,
        "plot_adc_ramp_nonlinearity",
        lambda _analysis, *, output_path: (output_path.with_suffix(".png"),),
    )

    class Curve:
        decoding = "uncalibrated_dout"
        maximum_abs_dnl = 0.1
        maximum_abs_inl = 0.2
        missing_codes = 0
        maximum_transfer_reversal_dout = 0.5

    monkeypatch.setattr(
        runner,
        "analyze_adc_ramp",
        lambda measurement, *, calibrations: SimpleNamespace(
            adc_index=measurement.param.observed_adc,
            sample_count=4_000_000,
            retained_sample_count=3_999_488,
            reset_excluded_sample_count=512,
            sample_rate_hz=6.25e6,
            ramp_frequency_hz=1e3,
            curves=(Curve(),),
            calibration_adc_index=calibrations[0].adc_index,
        ),
    )

    artifacts = runner.adc_ramp_nonlinearity(tmp_path / "output")

    assert len(artifacts) == 17
    assert artifacts[0] == tmp_path / "output/adc00_ramp_transfer.png"
    assert artifacts[-2] == tmp_path / "output/adc03_ramp_nonlinearity.png"
    assert artifacts[-1] == tmp_path / "output/adc00_adc03_ramp_metrics.csv"


def test_adc_ramp_runner_rejects_wrong_measurement_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the runner boundary at the typed measurement object."""

    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "read_measurement", lambda _path: object())

    with pytest.raises(TypeError, match="expected MeasAdcExt"):
        runner.adc_ramp_nonlinearity(tmp_path / "output")


def test_adc00_fixed_input_noise_adds_external_activity_and_supply_noise_trajectories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plot the external-input capture and matching simulations beside ADC00."""

    measurements = {}
    physical_dir = tmp_path / "build/scan_adc/20260819_113714"
    for point_index, rate_mbd in enumerate((320, 960, 1600)):
        measurements[
            physical_dir
            / f"{point_index:04d}_00_adc00_{rate_mbd}mbd_dcp50mv_logicp2sym_vcm700mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5"
        ] = adc_measurement([0])
    external_dir = tmp_path / "build/scan_adc/20260821_173944"
    for point_index, rate_mbd in enumerate((320, 960, 1600)):
        measurements[
            external_dir
            / f"{point_index:04d}_00_adc00_{rate_mbd}mbd_dcp50mv_logicp2sym_vcm700mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5"
        ] = adc_measurement([0])
    all_active_dir = tmp_path / "build/scan_adc/20260822_144348"
    for point_index, rate_mbd in enumerate((320, 960, 1600)):
        measurements[
            all_active_dir
            / f"{point_index:04d}_00_adc00_{rate_mbd}mbd_dcp50mv_logicp2sym_vcm700mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5"
        ] = adc_measurement([0])
    for run_name in ("20260820_005128", "20260820_005122"):
        for rate_msps in (2, 6, 10):
            path = tmp_path / "build/sim/adc" / run_name / f"{rate_msps}msps_cm700mv_dc50mv/result.h5"
            measurements[path] = adc_measurement([0], internal=True)
    supply_noise_dir = tmp_path / "build/sim/adc/frida65a_supply_noise_vs_rate/20260821_182756"
    noise_by_name = {
        "none": (0.0, 0.0, 0.0),
        "vdda": (1e-3, 0.0, 0.0),
        "vddd": (0.0, 1e-3, 0.0),
        "vddac": (0.0, 0.0, 1e-3),
        "all": (1e-3, 1e-3, 1e-3),
    }
    for noise_name in ("none", "vdda", "vddd", "vddac", "all"):
        for rate_msps in (2, 6, 10):
            path = supply_noise_dir / f"{rate_msps}msps_{noise_name}/result.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            measurement = adc_measurement([0], internal=True)
            measurements[path] = dataclasses.replace(
                measurement,
                param=dataclasses.replace(
                    measurement.param,
                    symbol_rate=rate_msps * 160e6,
                    supply_noise_rms_v=noise_by_name[noise_name],
                ),
            )

    density_outputs = []
    distribution_outputs = []
    noise_outputs = []

    def plot_density(_measurement, _analysis, *, output_path):
        density_outputs.append(output_path.name)
        return (output_path.with_suffix(".png"),)

    def plot_distribution(_measurements, _analysis, *, output_path):
        distribution_outputs.append(output_path.name)
        return (output_path.with_suffix(".png"),)

    def plot_noise(_measurements, _analysis, *, output_path):
        noise_outputs.append(output_path.name)
        return (output_path.with_suffix(".png"),)

    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "read_measurement", measurements.__getitem__)
    monkeypatch.setattr(
        runner,
        "analyze_adc_noise_sweep",
        lambda _measurements: SimpleNamespace(active_conversion_rate_hz=np.asarray((2e6, 6e6, 10e6))),
    )
    monkeypatch.setattr(runner, "analyze_adc_decision_paths", lambda _measurement, *, selection: selection)
    monkeypatch.setattr(runner, "plot_adc_noise_sweep", plot_noise)
    monkeypatch.setattr(runner, "plot_adc_noise_distribution_sweep", plot_distribution)
    monkeypatch.setattr(runner, "plot_adc_decision_path_density", plot_density)

    artifacts = runner.adc00_fixed_input_noise(tmp_path / "output")

    assert len(artifacts) == 43
    assert noise_outputs == [
        "adc00_50mv_noise_vs_conversion_rate",
        "adc00_external_50mv_noise_vs_conversion_rate",
        "adc00_all_active_50mv_noise_vs_conversion_rate",
    ]
    assert distribution_outputs == [
        "adc00_50mv_output_code_distributions",
        "adc00_external_50mv_output_code_distributions",
        "adc00_all_active_50mv_output_code_distributions",
        "spice_hdl21gen_50mv_output_code_distributions",
        "spice_frida65a_pex_50mv_output_code_distributions",
        *(
            f"spice_frida65a_pex_supply_{noise_name}_50mv_output_code_distributions"
            for noise_name in ("none", "vdda", "vddd", "vddac", "all")
        ),
    ]
    assert density_outputs == [
        *(f"adc00_50mv_{rate}msps_decision_path_density" for rate in (2, 6, 10)),
        *(f"adc00_external_50mv_{rate}msps_decision_path_density" for rate in (2, 6, 10)),
        *(f"adc00_all_active_50mv_{rate}msps_decision_path_density" for rate in (2, 6, 10)),
        *(f"spice_hdl21gen_50mv_{rate}msps_decision_path_density" for rate in (2, 6, 10)),
        *(f"spice_frida65a_pex_50mv_{rate}msps_decision_path_density" for rate in (2, 6, 10)),
        *(
            f"spice_frida65a_pex_supply_{noise_name}_50mv_{rate}msps_decision_path_density"
            for noise_name in ("none", "vdda", "vddd", "vddac", "all")
            for rate in (2, 6, 10)
        ),
    ]


def test_adc_noise_density_grid_uses_final_manual_supply_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analyze both pinned campaigns through the same metadata-driven pipeline."""

    read_paths = []
    analyzed_groups = []
    plotted = []
    measurements_by_path = {}
    campaigns = (
        ("20260824_165039", 0.0, 0.6),
        ("20260824_234702", 0.05, 0.7),
    )
    for run_name, input_v, common_mode_v in campaigns:
        meas_read_dir = tmp_path / "build/scan_adc" / run_name
        meas_read_dir.mkdir(parents=True)
        for adc_index in range(16):
            for rate_index in range(3):
                path = meas_read_dir / f"{3 * adc_index + rate_index:04d}_fixture.h5"
                path.touch()
                measurements_by_path[path] = SimpleNamespace(
                    param=SimpleNamespace(
                        observed_adc=adc_index,
                        tb=SimpleNamespace(
                            vin_diff=h.Vdc.Params(dc=input_v),
                            vin_cm=h.Vdc.Params(dc=common_mode_v),
                        ),
                    )
                )

    def read(path: Path) -> SimpleNamespace:
        read_paths.append(path)
        return measurements_by_path[path]

    def analyze(measurements):
        analyzed_groups.append(measurements)
        return f"analysis-{len(analyzed_groups) - 1}"

    def plot(measurements, analyses, *, output_path):
        plotted.append((measurements, analyses, output_path))
        return (output_path.with_suffix(".pdf"),)

    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "MeasAdcExt", SimpleNamespace)
    monkeypatch.setattr(runner, "read_measurement", read)
    monkeypatch.setattr(runner, "analyze_adc_noise_sweep", analyze)
    monkeypatch.setattr(runner, "plot_adc_noise_distribution_grid", plot)

    output_dir = tmp_path / "output"
    artifacts = runner.adc_noise_density_grid(output_dir)

    assert len(read_paths) == 96
    assert tuple(len(group) for group in analyzed_groups) == (3,) * 32
    assert tuple(group[0].param.observed_adc for group in analyzed_groups) == 2 * tuple(range(16))
    assert [output_path.name for _measurements, _analyses, output_path in plotted] == [
        "adc00_adc15_0mv_600mv_output_code_density_grid",
        "adc00_adc15_50mv_700mv_output_code_density_grid",
    ]
    assert artifacts == tuple(output_path.with_suffix(".pdf") for _measurements, _analyses, output_path in plotted)


def test_adc_pex_flavor_runners_use_h5_flavors_and_rates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plot every extracted flavor and rate using persisted simulation metadata."""

    meas_read_dir = tmp_path / "build/sim/adc/frida65a_noise_vs_rate/20260827_165917"
    measurements = {}
    flavors = (
        "adc_1layer_radix17",
        "adc_1layer_radix20",
        "adc_2layer_radix17",
        "adc_2layer_radix20",
    )
    for flavor in flavors:
        for rate_msps in (2, 6, 10):
            path = meas_read_dir / flavor / f"{rate_msps}msps_cm700mv_dc50mv/result.h5"
            path.parent.mkdir(parents=True)
            path.touch()
            measurement = adc_measurement([0], internal=True)
            measurements[path] = dataclasses.replace(
                measurement,
                param=dataclasses.replace(
                    measurement.param,
                    pex_cell=flavor,
                    symbol_rate=rate_msps * 160e6,
                    vin_diff=h.Vdc.Params(dc=0.05),
                ),
            )

    outputs = []
    selections = []

    def analyze_noise(measurement_group):
        return SimpleNamespace(
            active_conversion_rate_hz=np.asarray(
                [float(measurement.param.symbol_rate) / 160.0 for measurement in measurement_group]
            )
        )

    def analyze_paths(_measurement, *, selection):
        selections.append(selection)
        return object()

    def plot_paths(_measurement, _analysis, *, output_path):
        outputs.append(output_path.name)
        return (output_path.with_suffix(".pdf"),)

    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "read_measurement", measurements.__getitem__)
    monkeypatch.setattr(runner, "analyze_adc_noise_sweep", analyze_noise)
    monkeypatch.setattr(runner, "analyze_adc_decision_paths", analyze_paths)
    monkeypatch.setattr(runner, "plot_adc_decision_path_density", plot_paths)

    settling_outputs = []

    def analyze_settling(measurement):
        return SimpleNamespace(active_conversion_rate_hz=float(measurement.param.symbol_rate) / 160.0)

    def plot_settling(measurement, analysis, *, output_path):
        settling_outputs.append((measurement, analysis, output_path.name))
        return (output_path.with_suffix(".pdf"),)

    from flow.analysis.test_adc import adc_sampling_measurement

    sampling_outputs = []
    sampling_analysis = runner.analyze_adc_sampling_noise(adc_sampling_measurement())

    def plot_sampling(analyses, *, labels, output_path):
        sampling_outputs.append((len(analyses), labels))
        return (output_path.with_suffix(".pdf"),)

    monkeypatch.setattr(runner, "plot_waveforms", lambda *args, **kwargs: pytest.fail("no generic key-node plot"))
    monkeypatch.setattr(runner, "analyze_adc_sampling_noise", lambda _: sampling_analysis)
    monkeypatch.setattr(runner, "plot_adc_sampling_noise", plot_sampling)
    monkeypatch.setattr(runner, "analyze_adc_cdac_settling", analyze_settling)
    monkeypatch.setattr(runner, "plot_adc_cdac_settling", plot_settling)

    artifacts = runner.adc_pex_flavor_paths(tmp_path / "output")

    assert len(artifacts) == 39
    assert len(settling_outputs) == 12
    assert sampling_outputs[0][0] == 12
    assert len((tmp_path / "output/adc_sampling_levels.csv").read_text().splitlines()) == 37
    assert len((tmp_path / "output/adc_sampling_noise.csv").read_text().splitlines()) == 13
    assert selections == ["all"] * 12
    assert outputs == [
        f"spice_{flavor}_{rate_msps}msps_cm700mv_dc50mv_50mv_{rate_msps}msps_decision_path_density"
        for flavor in flavors
        for rate_msps in (10, 2, 6)
    ]

    settling_outputs.clear()

    settling_artifacts = runner.adc_pex_cdac_settling(tmp_path / "output")

    assert len(settling_artifacts) == 12
    assert [name for _measurement, _analysis, name in settling_outputs] == [
        f"spice_{flavor}_50mv_{rate_msps}msps_cdac_settling" for flavor in flavors for rate_msps in (10, 2, 6)
    ]
    assert [float(measurement.param.symbol_rate) for measurement, _analysis, _name in settling_outputs] == [
        rate_msps * 160e6 for _flavor in flavors for rate_msps in (10, 2, 6)
    ]

    # All timing recipes retain the flavor in labels and artifact filenames.
    timing_dir = tmp_path / "timing_campaign"
    for timing in ("original", "extended_comp", "continuous_100ns"):
        path = timing_dir / "frida2_3layer_radix17" / timing / "result.h5"
        path.parent.mkdir(parents=True)
        path.touch()
        measurements[path] = next(iter(measurements.values()))
    artifacts = runner.adc_pex_flavor_paths(tmp_path / "timing_output", inputs=timing_dir)
    assert len(artifacts) == len(set(artifacts)) == 12
    assert all("FRIDA-2 3L R17" in label for label in sampling_outputs[-1][1])
    assert {label.split("\n")[0] for label in sampling_outputs[-1][1]} == {
        "FRIDA-2 3L R17 original",
        "FRIDA-2 3L R17 extended comp",
        "FRIDA-2 3L R17 continuous 100ns",
    }


def test_adc_noise_vs_comp_time_runner_uses_configured_adc_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the configured ADC subset without assuming only ADC00 and ADC01 exist."""

    measurements_by_path = {}
    run_dirs = {
        0: tmp_path / "build/scan_adc/20260802_081407",
        1: tmp_path / "build/loopback_fastrx/20260729_181030",
    }
    for adc_index, run_dir in run_dirs.items():
        run_dir.mkdir(parents=True)
        for rate_mbd in range(80, 1601, 40):
            for logic_offset in range(-3, 4):
                if adc_index == 0:
                    path = run_dir / (
                        f"{rate_mbd}_{logic_offset + 3}_00_adc00_{rate_mbd}mbd_dcp50mv_"
                        f"logic{logic_offset}sym_vcm800mv_test.h5"
                    )
                else:
                    path = run_dir / f"adc01_{rate_mbd}mbd_logic{logic_offset}_rx0_tap0.h5"
                path.touch()
                measurements_by_path[path] = SimpleNamespace(
                    param=SimpleNamespace(
                        observed_adc=adc_index,
                        tb=SimpleNamespace(
                            symbol_rate=rate_mbd * 1e6,
                            seq_logic_phase_delay_symbols=logic_offset,
                            seq_comp_phase_delay_symbols=0,
                            vin_diff=h.Vdc.Params(dc=0.05),
                            vin_cm=h.Vdc.Params(dc=0.8),
                        ),
                    )
                )

    output_paths = []

    def plot(_measurements, _analysis, *, output_path):
        output_paths.append(output_path)
        return (output_path.with_suffix(".png"),)

    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "MeasAdcExt", SimpleNamespace)
    monkeypatch.setattr(runner, "read_measurement", lambda path: measurements_by_path[path])
    monkeypatch.setattr(runner, "analyze_adc_noise_sweep", lambda _measurements: SimpleNamespace())
    monkeypatch.setattr(runner, "plot_adc_noise_sweep", plot)

    artifacts = runner.adc_noise_vs_comp_time(tmp_path / "output")

    assert [path.name for path in output_paths] == (
        ["adc00_noise_vs_conversion_rate_and_logic_offset", "adc01_noise_vs_conversion_rate_and_logic_offset"]
    )
    assert len(artifacts) == 2


def test_adc_power_runner_combines_measured_and_separate_simulated_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readbacks = {f"{rail}_active_average_power_w": 10.0e-6 for rail in ("vdd_a", "vdd_d", "vdd_dac")}
    readbacks.update({f"{rail}_static_average_power_w": 2.0e-6 for rail in ("vdd_a", "vdd_d", "vdd_dac")})
    physical_by_adc = {
        adc_index: adc_measurement(
            [100, 101, 102],
            observed_adc=adc_index,
            readbacks=readbacks,
        )
        for adc_index in (0, 1)
    }
    simulated = adc_measurement([100, 101, 102], readbacks=readbacks, internal=True)
    measurements_by_path = {}
    sine_dir = tmp_path / "build/scan_adc/20260730_215145_complete"
    sine_dir.mkdir(parents=True)
    for adc_index in (0, 1):
        for rate_mbd in range(80, 1601, 40):
            path = sine_dir / (
                f"point_00_adc{adc_index:02d}_{rate_mbd}mbd_sin9998.77hz_p0mv_1000mvpp_"
                "logicp2sym_vcm600mv_vdda1200mv_vddd1200mv_vddac1200mv_t25c.h5"
            )
            path.touch()
            measurement = physical_by_adc[adc_index]
            measurements_by_path[path] = dataclasses.replace(
                measurement,
                param=dataclasses.replace(
                    measurement.param,
                    tb=dataclasses.replace(measurement.param.tb, symbol_rate=rate_mbd * 1e6),
                ),
            )
    for run_name in ("hdl21gen_noise_vs_rate/20260801_0821", "frida65a_noise_vs_rate/20260731_2353"):
        run_dir = tmp_path / "build/adc" / run_name
        for rate_msps in (2, 6, 10):
            path = run_dir / f"{rate_msps}msps_cm600mv_dc50mv/result.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            measurements_by_path[path] = simulated

    plot_calls = []

    def analyze_power(measurements):
        rates_hz = np.asarray((2.0e6, 6.0e6, 10.0e6)) if len(measurements) == 3 else np.asarray((10.0e6,))
        if len(measurements) > 3:
            rates_hz = np.asarray([measurement.param.tb.symbol_rate / 160 for measurement in measurements])
        return SimpleNamespace(active_conversion_rate_hz=rates_hz)

    def plot_power(_measurements, _analysis, *, output_path):
        plot_calls.append(("rate", output_path.name))
        return (output_path.with_suffix(".png"),)

    def plot_power_waveform(_analysis, *, output_path):
        plot_calls.append(("waveform", output_path.name))
        return (output_path.with_suffix(".png"),)

    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "read_measurement", measurements_by_path.__getitem__)
    monkeypatch.setattr(runner, "analyze_adc_power_sweep", analyze_power)
    monkeypatch.setattr(runner, "analyze_adc_power_waveform", lambda _measurement: object())
    monkeypatch.setattr(runner, "plot_adc_power_sweep", plot_power)
    monkeypatch.setattr(runner, "plot_adc_power_waveform", plot_power_waveform)
    monkeypatch.setattr(runner, "analyze_measurement_waveforms", lambda _measurement: object())
    monkeypatch.setattr(runner, "plot_waveforms", lambda *_args, output_path: (output_path,))
    monkeypatch.setattr(runner, "analyze_adc_dynamic", lambda _measurement: object())
    monkeypatch.setattr(runner, "plot_adc_dynamic", lambda *_args, output_path, **_kwargs: (output_path,))

    artifacts = runner.adc_power_vs_rate(tmp_path / "output")

    assert plot_calls == [
        ("rate", "adc_power_vs_conversion_rate_adc00"),
        ("rate", "adc_power_vs_conversion_rate_adc01"),
        ("rate", "spice_ideal_power_vs_conversion_rate"),
        ("waveform", "spice_ideal_10msps_supply_power"),
        ("rate", "spice_pex_power_vs_conversion_rate"),
        ("waveform", "spice_pex_10msps_supply_power"),
    ]
    assert len(artifacts) == 10
    assert [path.name for path in artifacts[-4:]] == [
        "adc00_80mbd_sine_waveforms",
        "adc00_80mbd_sine_fit_and_spectrum",
        "adc01_80mbd_sine_waveforms",
        "adc01_80mbd_sine_fit_and_spectrum",
    ]


def test_adc_noise_vs_rate_groups_measurements_by_h5_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurements_by_path = {}
    for run_name, input_v in (("20260801_194930", 0.050), ("20260802_021624", 0.025)):
        meas_read_dir = tmp_path / "build/scan_adc" / run_name
        meas_read_dir.mkdir(parents=True)
        for adc_index in (0, 1):
            path = meas_read_dir / f"adc{adc_index:02d}.h5"
            path.touch()
            measurements_by_path[path] = SimpleNamespace(
                param=SimpleNamespace(
                    observed_adc=adc_index,
                    tb=SimpleNamespace(vin_diff=h.Vdc.Params(dc=input_v)),
                )
            )
    sine_meas_read_dir = tmp_path / "build/scan_adc/20260730_215145_complete"
    sine_meas_read_dir.mkdir(parents=True)
    for adc_index in (0, 1):
        path = sine_meas_read_dir / f"adc{adc_index:02d}.h5"
        path.touch()
        measurements_by_path[path] = SimpleNamespace(param=SimpleNamespace(observed_adc=adc_index))
    for run_name in ("hdl21gen_noise_vs_rate/20260801_0821", "frida65a_noise_vs_rate/20260731_2353"):
        for rate_msps in (2, 6, 10):
            path = tmp_path / "build/adc" / run_name / f"{rate_msps}msps_cm600mv_dc50mv/result.h5"
            measurements_by_path[path] = SimpleNamespace(param=SimpleNamespace(observed_adc=0))

    comparisons = []
    output_paths = []

    def combine(dc_sweeps, sine_dynamic, simulated_sweeps, *, series_labels):
        comparisons.append((dc_sweeps, sine_dynamic, simulated_sweeps, series_labels))
        return SimpleNamespace()

    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "MeasAdcExt", SimpleNamespace)
    monkeypatch.setattr(runner, "MeasAdcInt", SimpleNamespace)
    monkeypatch.setattr(runner, "read_measurement", measurements_by_path.__getitem__)
    monkeypatch.setattr(runner, "analyze_adc_noise_sweep", lambda measurements: tuple(measurements))
    monkeypatch.setattr(runner, "analyze_adc_dynamic_sweep", lambda measurements: tuple(measurements))
    monkeypatch.setattr(runner, "combine_adc_noise_comparison", combine)
    monkeypatch.setattr(
        runner,
        "plot_adc_noise_sweep",
        lambda _measurements, _analysis, *, output_path: output_paths.append(output_path) or (output_path,),
    )

    artifacts = runner.adc_noise_vs_rate(tmp_path / "output")

    assert [path.name for path in output_paths] == [
        "adc00_noise_vs_conversion_rate",
        "adc01_noise_vs_conversion_rate",
    ]
    assert len(comparisons[0][2]) == 2
    assert comparisons[1][2] == []
    assert comparisons[0][3][1:3] == ["Measured (25 mV DC)", "Measured (50 mV DC)"]
    assert tuple(output_paths) == artifacts


def test_adc_code_distributions_derives_groups_and_selected_rate_names_from_h5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurements_by_path = {}
    for run_name, input_v in (("20260801_194930", 0.050), ("20260802_021624", 0.025)):
        meas_read_dir = tmp_path / "build/scan_adc" / run_name
        meas_read_dir.mkdir(parents=True)
        for point_index, rate_hz in enumerate((2e6, 6e6, 10e6)):
            path = meas_read_dir / f"{point_index:04d}_adc07.h5"
            path.touch()
            measurements_by_path[path] = SimpleNamespace(
                active_rate_hz=rate_hz,
                param=SimpleNamespace(
                    observed_adc=7,
                    tb=SimpleNamespace(vin_diff=h.Vdc.Params(dc=input_v)),
                ),
            )
    for rate_msps in (2, 6, 10):
        path = tmp_path / "build/adc/hdl21gen_noise_vs_rate/20260801_0821" / f"{rate_msps}msps_cm600mv_dc50mv/result.h5"
        measurements_by_path[path] = SimpleNamespace(
            active_rate_hz=rate_msps * 1e6,
            param=SimpleNamespace(observed_adc=0),
        )

    outputs: dict[str, list[str]] = {"distribution": [], "code": [], "paths": [], "density": []}

    def noise(measurements):
        return SimpleNamespace(active_conversion_rate_hz=np.asarray([item.active_rate_hz for item in measurements]))

    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "MeasAdcExt", SimpleNamespace)
    monkeypatch.setattr(runner, "MeasAdcInt", SimpleNamespace)
    monkeypatch.setattr(runner, "read_measurement", measurements_by_path.__getitem__)
    monkeypatch.setattr(runner, "analyze_adc_noise_sweep", noise)
    monkeypatch.setattr(runner, "analyze_adc_code_distribution", lambda measurements: tuple(measurements))
    monkeypatch.setattr(runner, "analyze_adc_decision_paths", lambda measurement, *, selection: measurement)
    monkeypatch.setattr(
        runner,
        "plot_adc_noise_distribution_sweep",
        lambda *_args, output_path: outputs["distribution"].append(output_path.name) or (output_path,),
    )
    monkeypatch.setattr(
        runner,
        "plot_adc_code_distribution",
        lambda *_args, output_path: outputs["code"].append(output_path.name) or (output_path,),
    )
    monkeypatch.setattr(
        runner,
        "plot_adc_decision_paths",
        lambda *_args, output_path: outputs["paths"].append(output_path.name) or (output_path,),
    )
    monkeypatch.setattr(
        runner,
        "plot_adc_decision_path_density",
        lambda *_args, output_path: outputs["density"].append(output_path.name) or (output_path,),
    )

    runner.adc_code_distributions(tmp_path / "output")

    assert outputs["distribution"] == [
        "adc07_25mv_dc_output_code_distributions",
        "adc07_50mv_dc_output_code_distributions",
    ]
    assert outputs["code"] == [
        "spice_hdl21gen_2msps_output_code_histogram",
        "spice_hdl21gen_6msps_output_code_histogram",
        "spice_hdl21gen_10msps_output_code_histogram",
    ]
    assert outputs["density"] == [
        "adc07_25mv_2msps_decision_path_density",
        "adc07_25mv_10msps_decision_path_density",
        "adc07_50mv_2msps_decision_path_density",
        "adc07_50mv_10msps_decision_path_density",
    ]


def test_comp_common_mode_groups_adc_and_common_mode_from_h5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meas_read_dir = tmp_path / "build/scan_comp/20260805_171216"
    meas_read_dir.mkdir(parents=True)
    measurements_by_path = {}
    for adc_index in (2, 5):
        for point_index, common_mode_v in enumerate((0.9, 0.7)):
            path = meas_read_dir / f"adc{adc_index:02d}_{point_index}.h5"
            path.touch()
            measurements_by_path[path] = SimpleNamespace(
                param=SimpleNamespace(
                    observed_adc=adc_index,
                    tb=SimpleNamespace(vin_cm=h.Vdc.Params(dc=common_mode_v)),
                )
            )

    plotted = []
    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "MeasCompExt", SimpleNamespace)
    monkeypatch.setattr(runner, "read_measurement", measurements_by_path.__getitem__)
    monkeypatch.setattr(runner, "analyze_comp_offset_noise", lambda group: tuple(group))
    monkeypatch.setattr(runner, "classify_comp_common_mode_validity", lambda _groups, analyses: analyses)
    monkeypatch.setattr(
        runner,
        "plot_comp_common_mode_campaign",
        lambda groups, _analyses, *, output_path: plotted.append((groups, output_path)) or (output_path,),
    )

    runner.comp_system_common_mode(tmp_path / "output")

    assert [path.name for _groups, path in plotted] == [
        "adc02_comparator_common_mode",
        "adc05_comparator_common_mode",
    ]
    assert [float(group[0].param.tb.vin_cm.dc) for group in plotted[0][0]] == [0.7, 0.9]


def test_comp_sampling_noise_replaces_exact_correction_curves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_meas_read_dir = tmp_path / "build/scan_comp/20260805_183915"
    correction_meas_read_dir = tmp_path / "build/scan_comp/20260805_192902"
    base_meas_read_dir.mkdir(parents=True)
    correction_meas_read_dir.mkdir(parents=True)
    measurements_by_path = {}

    def add(directory: Path, name: str, adc_index: int, coupling: float, mode: str, marker: str) -> None:
        path = directory / f"{name}.h5"
        path.touch()
        measurements_by_path[path] = SimpleNamespace(
            marker=marker,
            param=SimpleNamespace(
                observed_adc=adc_index,
                requested_dac_rail_percent=coupling,
                sampling_mode=mode,
            ),
        )

    add(base_meas_read_dir, "adc01_target_a", 1, 100.0, "track", "base-adc01-target-a")
    add(base_meas_read_dir, "adc01_target_b", 1, 100.0, "track", "base-adc01-target-b")
    add(base_meas_read_dir, "adc01_keep", 1, 50.0, "hold", "base-adc01-keep")
    add(base_meas_read_dir, "adc02_target_a", 2, 75.0, "track", "base-adc02-target-a")
    add(base_meas_read_dir, "adc02_target_b", 2, 75.0, "track", "base-adc02-target-b")
    add(base_meas_read_dir, "adc02_keep", 2, 25.0, "hold", "base-adc02-keep")
    add(correction_meas_read_dir, "adc01_a", 1, 100.0, "track", "correction-adc01-a")
    add(correction_meas_read_dir, "adc01_b", 1, 100.0, "track", "correction-adc01-b")
    add(correction_meas_read_dir, "adc02_a", 2, 75.0, "track", "correction-adc02-a")
    add(correction_meas_read_dir, "adc02_b", 2, 75.0, "track", "correction-adc02-b")

    analyzed_groups = {}

    def analyze(group):
        first = group[0]
        key = (
            first.param.observed_adc,
            first.param.requested_dac_rail_percent,
            first.param.sampling_mode,
        )
        analyzed_groups[key] = tuple(measurement.marker for measurement in group)
        return SimpleNamespace()

    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "MeasCompExt", SimpleNamespace)
    monkeypatch.setattr(runner, "read_measurement", measurements_by_path.__getitem__)
    monkeypatch.setattr(runner, "analyze_comp_offset_noise", analyze)
    monkeypatch.setattr(
        runner,
        "plot_comp_sampling_campaign",
        lambda *_args, output_path: (output_path,),
    )

    runner.comp_system_sampling_noise(tmp_path / "output")

    assert analyzed_groups[(1, 100.0, "track")] == ("correction-adc01-a", "correction-adc01-b")
    assert analyzed_groups[(2, 75.0, "track")] == ("correction-adc02-a", "correction-adc02-b")
    assert analyzed_groups[(1, 50.0, "hold")] == ("base-adc01-keep",)
    assert analyzed_groups[(2, 25.0, "hold")] == ("base-adc02-keep",)


def test_comp_candidate_sweep_delegates_validity_to_typed_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "build/comp/frida65_candidate_scurve_power/candidates/fixture/result.h5"
    path.parent.mkdir(parents=True)
    path.touch()
    enum_value = SimpleNamespace(name="fixture")
    comp = SimpleNamespace(
        comp_stages=enum_value,
        preamp_diff_xtors=enum_value,
        preamp_bias=enum_value,
        latch_inner_on_xtors=enum_value,
        latch_outer_on_xtors=enum_value,
        latch_inner_init_xtors=enum_value,
        latch_outer_init_xtors=enum_value,
        diffpair_w=1,
        diffpair_l=1,
        tail_w=1,
        tail_l=1,
        rst_w=1,
        rst_l=1,
        latch_on_w=1,
        latch_on_l=1,
        latch_init_w=1,
        latch_init_l=1,
        srlatch_n_w=1,
        srlatch_p_w=1,
    )
    measurement = SimpleNamespace(
        info=SimpleNamespace(readbacks={"candidate_id": "fixture"}),
        param=SimpleNamespace(comp=comp),
    )
    analysis = SimpleNamespace(
        candidate_id=("fixture",),
        candidate_label=("Fixture",),
        size_profile=("half",),
        topology_index=np.asarray([0]),
        total_width_units=np.asarray([1.0]),
        total_active_area_units=np.asarray([1.0]),
        total_active_area_um2=np.asarray([1.0]),
        device_count=np.asarray([1]),
        validity=("valid",),
        offset_v=np.asarray([0.0]),
        noise_sigma_v=np.asarray([1e-3]),
        average_power_w=np.asarray([1e-6]),
        energy_per_decision_j=np.asarray([1e-15]),
        maximum_clock_to_decision_s=np.asarray([1e-9]),
        maximum_settling_s=np.asarray([2e-9]),
        unresolved_fraction=np.asarray([0.0]),
    )
    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "MeasCompInt", SimpleNamespace)
    monkeypatch.setattr(runner, "read_measurement", lambda _path: measurement)
    monkeypatch.setattr(runner, "analyze_comp_candidate_sweep", lambda measurements: analysis)
    monkeypatch.setattr(runner, "plot_comp_candidate_sweep", lambda *_args, output_path: (output_path,))
    monkeypatch.setattr(runner, "plot_comp_noise_power_tradeoff", lambda *_args, output_path: (output_path,))

    artifacts = runner.comp_candidate_sweep(tmp_path / "output")

    assert [artifact.name for artifact in artifacts] == [
        "comp_candidate_noise_power_settling",
        "comp_candidate_noise_power_tradeoff",
        "comp_candidate_noise_power_settling.csv",
    ]
    assert "fixture" in artifacts[-1].read_text()


def test_cdac_runner_derives_board_and_adc_indices_from_h5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurement_runs = []
    measurements_by_path = {}
    for run_name in ("20260804_171234", "20260804_193030", "20260804_193631"):
        meas_read_dir = tmp_path / "build/scan_cdac" / run_name
        meas_read_dir.mkdir(parents=True)
        run_measurements = []
        for adc_index in (2, 7):
            path = meas_read_dir / f"adc{adc_index:02d}.h5"
            path.touch()
            measurement = SimpleNamespace(param=SimpleNamespace(observed_adc=adc_index, board_id="fixture"))
            measurements_by_path[path] = measurement
            run_measurements.append(measurement)
        measurement_runs.append(tuple(run_measurements))

    received = {}

    def analyze(runs, *, adc_indices, board_id, comparator_offset_v_by_adc):
        received.update(
            runs=tuple(runs),
            adc_indices=adc_indices,
            board_id=board_id,
            offsets=comparator_offset_v_by_adc,
        )
        groups = tuple(
            (measurements_by_path[next(path for path in measurements_by_path if f"adc{adc:02d}" in path.name)],)
            for adc in adc_indices
        )
        analyses = tuple(SimpleNamespace(adc_index=adc_index) for adc_index in adc_indices)
        return groups, analyses

    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "MeasCdacExt", SimpleNamespace)
    monkeypatch.setattr(runner, "read_measurement", measurements_by_path.__getitem__)
    monkeypatch.setattr(
        runner,
        "load_board_map",
        lambda: {"boards": {"fixture": {"comparator_calibration": {2: {"offset_v": 2e-3}, 7: {"offset_v": 7e-3}}}}},
    )
    monkeypatch.setattr(runner, "analyze_cdac_cap_mismatch_campaign", analyze)
    monkeypatch.setattr(runner, "plot_cdac_cap_mismatch", lambda *_args, output_path: (output_path,))
    monkeypatch.setattr(runner, "plot_cdac_cap_mismatch_comparison", lambda *_args, output_path: (output_path,))

    artifacts = runner.cdac_system_cap_mismatch(tmp_path / "output")

    assert received == {
        "runs": tuple(measurement_runs),
        "adc_indices": (2, 7),
        "board_id": "fixture",
        "offsets": {2: 2e-3, 7: 7e-3},
    }
    assert [path.name for path in artifacts] == [
        "adc02_cdac_cap_mismatch",
        "adc07_cdac_cap_mismatch",
        "adc00_adc03_cdac_cap_mismatch_comparison",
    ]


def test_main_runs_named_target_in_one_timestamped_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pass one shared derived-artifact directory to the selected target."""

    received_output_dirs: list[Path] = []

    def example_target(output_dir: Path) -> tuple[Path, ...]:
        received_output_dirs.append(output_dir)
        artifact = output_dir / "example.png"
        artifact.write_bytes(b"plot")
        return (artifact,)

    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(runner, "TARGETS", {"comp_example_target": example_target})
    monkeypatch.setattr(sys, "argv", ["flow.analysis.runner", "comp_example_target"])

    runner.main()

    assert len(received_output_dirs) == 1
    output_dir = received_output_dirs[0]
    assert output_dir.parent == tmp_path / "build/analysis/comp"
    assert re.fullmatch(r"\d{8}_\d{4}", output_dir.name)
    assert (output_dir / "example.png").read_bytes() == b"plot"
    output = capsys.readouterr().out
    assert f"Analysis output: {output_dir}" in output
    assert "Completed comp_example_target: 1 artifacts in " in output


def test_main_without_target_runs_all_in_registration_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run all targets in one output directory when no name is supplied."""

    calls: list[tuple[str, Path]] = []

    def first(output_dir: Path) -> tuple[Path, ...]:
        calls.append(("first", output_dir))
        return ()

    def second(output_dir: Path) -> tuple[Path, ...]:
        calls.append(("second", output_dir))
        return ()

    def missing(output_dir: Path) -> tuple[Path, ...]:
        calls.append(("missing", output_dir))
        raise FileNotFoundError(2, "missing input", "capture.h5")

    monkeypatch.setattr(runner, "BASE_PATH", tmp_path)
    monkeypatch.setattr(
        runner,
        "TARGETS",
        {"first": first, "missing": missing, "second": second},
    )
    monkeypatch.setattr(sys, "argv", ["flow.analysis.runner"])

    runner.main()

    assert [name for name, _output_dir in calls] == ["first", "missing", "second"]
    assert len({output_dir for _name, output_dir in calls}) == 1
    output = capsys.readouterr().out
    assert output.count("Completed ") == 2
    assert "Completed first: 0 artifacts in " in output
    assert "Completed second: 0 artifacts in " in output
    assert "Skipped missing: missing capture.h5 after " in output


def test_main_rejects_unknown_target(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject arbitrary function names before creating an output directory."""

    monkeypatch.setattr(runner, "TARGETS", {"known_target": lambda _output_dir: ()})
    monkeypatch.setattr(sys, "argv", ["flow.analysis.runner", "unknown_target"])

    with pytest.raises(SystemExit, match="2"):
        runner.main()

    assert "invalid choice: 'unknown_target'" in capsys.readouterr().err
