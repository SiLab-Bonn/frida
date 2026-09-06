"""Software-only tests for typed measurement and analysis plots."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colors as mcolors
from matplotlib.ticker import FixedLocator

import flow.analysis.plots as analysis_plots
from flow.analysis.adc import (
    analyze_adc_cdac_settling,
    analyze_adc_code_distribution,
    analyze_adc_decision_paths,
    analyze_adc_dynamic,
    analyze_adc_dynamic_sweep,
    analyze_adc_noise_sweep,
    analyze_adc_nonlinearity,
    analyze_adc_power_sweep,
    analyze_adc_power_waveform,
    analyze_adc_ramp,
    analyze_adc_transfer,
)
from flow.analysis.cdac import _expected_cdac_effective_fraction
from flow.analysis.comp import analyze_comp_offset_noise
from flow.analysis.plots import (
    CURVE_COLORS,
    DENSITY_COLOR_MAP,
    GRID_MAJOR_COLOR,
    INFO_BOX_FONT_SIZE,
    LEGEND_FACE_COLOR,
    NORD_BLUE,
    NORD_GREEN,
    NORD_LIGHT_BLUE,
    NORD_ORANGE,
    NORD_YELLOW,
    PLOT_STYLE,
    SPECTRUM_COLOR_MAP,
    SPINE_COLOR,
    TEXT_COLOR,
    plot_adc_cdac_settling,
    plot_adc_code_distribution,
    plot_adc_decision_path_density,
    plot_adc_decision_paths,
    plot_adc_dynamic,
    plot_adc_dynamic_sweep,
    plot_adc_noise_distribution_grid,
    plot_adc_noise_distribution_sweep,
    plot_adc_noise_sweep,
    plot_adc_power_sweep,
    plot_adc_power_waveform,
    plot_adc_ramp_histogram,
    plot_adc_ramp_nonlinearity,
    plot_adc_ramp_transfer,
    plot_adc_ramp_weights,
    plot_adc_static_nonlinearity,
    plot_adc_transfer,
    plot_cdac_cap_mismatch,
    plot_cdac_cap_mismatch_comparison,
    plot_comp_common_mode_campaign,
    plot_comp_sampling_campaign,
    plot_waveforms,
    style_adc_code_dispersion_lsb,
    style_grid,
)
from flow.analysis.test_adc import adc_cdac_settling_measurement, adc_measurement, adc_ramp_measurement
from flow.analysis.test_comp import comparator_measurement
from flow.analysis.test_types import all_measurements
from flow.analysis.types import (
    AnalysisAdcNoiseComparison,
    AnalysisCdacCapMismatch,
    CompDaq,
    MeasAdcExt,
    MeasAdcInt,
    MeasCompExt,
)
from flow.analysis.waveform import analyze_measurement_waveforms
from flow.scans.scan_cdac import _build_cdac_params
from flow.scans.scan_comp import _build_comp_params


def test_sampling_noise_histogram_has_common_voltage_bins_and_axes(tmp_path, monkeypatch) -> None:
    from flow.analysis.adc import analyze_adc_sampling_noise
    from flow.analysis.test_adc import adc_sampling_measurement

    analysis = analyze_adc_sampling_noise(adc_sampling_measurement())
    shifted = replace(analysis, held_p_v=analysis.held_p_v + 0.01)
    captured = []

    def capture(fig, output_path):
        captured.append(fig)
        return (output_path.with_suffix(".pdf"),)

    monkeypatch.setattr(analysis_plots, "save_figure", capture)
    analysis_plots.plot_adc_sampling_noise(
        [analysis, shifted, analysis, analysis, analysis, analysis, analysis],
        labels=[f"Design {index}" for index in range(7)],
        output_path=tmp_path / "sampling",
    )
    fig = captured[0]
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(axes) == 7
    assert len({ax.get_xlim() for ax in axes}) == len({ax.get_ylim() for ax in axes}) == 1
    for ax in axes:
        assert sum(patch.get_height() for patch in ax.patches) == pytest.approx(1.0)
        assert all(patch.get_width() == pytest.approx(25.0) for patch in ax.patches)
    assert [patch.get_height() for patch in axes[0].patches] == [patch.get_height() for patch in axes[1].patches]
    assert "µV" in fig._supxlabel.get_text()
    assert "LSB" not in fig._supxlabel.get_text()
    plt.close(fig)


def assert_plot_formats(paths: tuple[Path, ...]) -> None:
    assert tuple(path.suffix for path in paths) == (".png", ".svg", ".pdf")
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0
    assert plt.imread(paths[0]).shape[:2] == (2700, 4800)


def read_svg(paths: tuple[Path, ...]) -> str:
    return next(path for path in paths if path.suffix == ".svg").read_text()


@pytest.fixture(autouse=True)
def enable_all_plot_formats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise every shared output switch without changing production defaults."""

    monkeypatch.setattr(analysis_plots, "PLOT_PNGS", True)
    monkeypatch.setattr(analysis_plots, "PLOT_PDFS", True)
    monkeypatch.setattr(analysis_plots, "PLOT_SVGS", True)


def test_shared_plot_style_uses_computer_modern_and_nord() -> None:
    """Keep typography, palette, axes, grids, and legends consistent."""

    with mpl.rc_context(PLOT_STYLE):
        assert plt.rcParams["mathtext.fontset"] == "cm"
        assert plt.rcParams["font.family"] == ["serif"]
        assert plt.rcParams["font.size"] == 10.0
        assert plt.rcParams["axes.prop_cycle"].by_key()["color"] == list(CURVE_COLORS)
        assert plt.rcParams["figure.figsize"] == [9.6, 5.4]
        assert plt.rcParams["figure.constrained_layout.use"] is True
        assert plt.rcParams["savefig.dpi"] == 500
        assert plt.rcParams["axes.titlesize"] == 12.0
        assert plt.rcParams["figure.titlesize"] == 12.0
        assert plt.rcParams["axes.labelsize"] == 10.0
        assert plt.rcParams["xtick.labelsize"] == 10.0
        assert plt.rcParams["ytick.labelsize"] == 10.0
        assert plt.rcParams["xtick.major.size"] == plt.rcParams["ytick.major.size"] == 2.5
        assert plt.rcParams["xtick.minor.size"] == plt.rcParams["ytick.minor.size"] == 1.5
        assert plt.rcParams["legend.fontsize"] == 10.0
        assert plt.rcParams["legend.title_fontsize"] == 10.0
        assert plt.rcParams["legend.linewidth"] == 0.8
        assert plt.rcParams["lines.linewidth"] == 1.0
        assert plt.rcParams["lines.markersize"] == 4.0
        assert plt.rcParams["text.color"] == "black"
        assert plt.rcParams["axes.labelcolor"] == TEXT_COLOR
        assert plt.rcParams["axes.edgecolor"] == SPINE_COLOR
        assert plt.rcParams["axes.grid"] is False
        assert plt.rcParams["xtick.color"] == TEXT_COLOR
        assert plt.rcParams["ytick.color"] == TEXT_COLOR
        assert mcolors.to_hex(SPECTRUM_COLOR_MAP(0.0)) == NORD_BLUE.lower()
        assert mcolors.to_hex(SPECTRUM_COLOR_MAP(0.5)) == NORD_ORANGE.lower()
        assert mcolors.to_hex(SPECTRUM_COLOR_MAP(1.0)) == NORD_YELLOW.lower()
        assert mcolors.to_hex(DENSITY_COLOR_MAP(0.0)) == mcolors.to_hex(SPECTRUM_COLOR_MAP(0.2))
        assert mcolors.to_hex(DENSITY_COLOR_MAP(0.0)) != NORD_BLUE.lower()

        fig, ax = plt.subplots()
        ax.plot((0, 1), (0, 1), label="trace")
        scatter = ax.scatter((0.5,), (0.5,))
        assert np.array_equal(scatter.get_sizes(), np.asarray([16.0]))
        quarter_ticks = np.arange(0.0, 1.01, 0.25)
        ax.set_xticks(quarter_ticks, minor=True)
        style_grid(ax)
        ax.legend()
        colorbar = fig.colorbar(mpl.cm.ScalarMappable(), ax=ax)
        colorbar.set_label("Scale")
        fig.canvas.draw()
        assert ax.spines["left"].get_edgecolor() == mcolors.to_rgba(SPINE_COLOR)
        assert ax.xaxis.label.get_color() == TEXT_COLOR
        assert ax.get_xticklabels()[0].get_color() == TEXT_COLOR
        assert colorbar.outline.get_edgecolor() == mcolors.to_rgba(SPINE_COLOR)
        assert colorbar.ax.get_yticklabels()[0].get_color() == TEXT_COLOR
        assert colorbar.ax.yaxis.label.get_color() == TEXT_COLOR
        assert ax.get_xgridlines()[0].get_color() == GRID_MAJOR_COLOR
        assert isinstance(ax.xaxis.get_minor_locator(), FixedLocator)
        assert np.array_equal(ax.get_xticks(minor=True), quarter_ticks[1:-1])
        assert ax.get_axisbelow() is True
        assert ax.lines[0].get_alpha() in (None, 1.0)
        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_frame().get_facecolor()[:3] == mcolors.to_rgb(LEGEND_FACE_COLOR)
        assert legend.get_frame().get_linewidth() == 0.8
        plt.close(fig)


def test_save_figure_preserves_explicit_layout_across_formats(tmp_path: Path) -> None:
    """Do not let later output backends replace a manually positioned layout."""

    with mpl.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(layout="none")
        figure.subplots_adjust(left=0.2, right=0.7, bottom=0.25, top=0.8)
        expected_position = axis.get_position().bounds
        paths = analysis_plots.save_figure(figure, tmp_path / "explicit_layout")

    assert_plot_formats(paths)
    np.testing.assert_allclose(axis.get_position().bounds, expected_position)


def test_code_dispersion_text_distinguishes_single_bin_and_small_spread() -> None:
    assert style_adc_code_dispersion_lsb(0.0, single_code=True) == "<1.0"
    assert style_adc_code_dispersion_lsb(0.0063245, single_code=False) == "0.0063"
    assert style_adc_code_dispersion_lsb(0.68, single_code=False) == "0.7"


def test_waveform_plot_uses_typed_signal_names_and_scaled_time(tmp_path: Path) -> None:
    msmt = adc_measurement([1, 2, 3], internal=True)
    paths = plot_waveforms(
        analyze_measurement_waveforms(
            msmt,
            signal_names=("vin_diff_v", "dac_botplate_p_c0_v"),
        ),
        output_path=tmp_path / "wave",
    )
    assert_plot_formats(paths)
    svg = read_svg(paths)
    assert "vin_diff_v" in svg
    assert "dac_botplate_p_c0_v" in svg
    assert "Time (" in svg
    assert "Source: SPICE" in svg
    assert "Rate: 1.6 Msps" in svg
    assert "CDAC init: h'5555" in svg
    assert "Datetime:" not in svg
    assert "LOGIC offset:" not in svg


def test_cdac_settling_plot_separates_saved_bit_cycles_and_uses_millivolts(tmp_path: Path, monkeypatch) -> None:
    measurement = adc_cdac_settling_measurement()
    measurement = replace(
        measurement,
        param=replace(
            measurement.param,
            pex_cell="adc_1layer_radix17",
            symbol_rate=1.6e9,
        ),
    )
    analysis = replace(analyze_adc_cdac_settling(measurement), active_conversion_rate_hz=10e6)
    save = analysis_plots.save_figure

    def check_dotted_nodes(fig, output_path):
        for column, ax in enumerate(fig.axes[:3]):
            dotted = [line for line in ax.lines if line.get_linestyle() == ":"]
            assert [line.get_color() for line in dotted] == [NORD_BLUE, NORD_ORANGE]
            assert analysis.comp_latch_p_v is not None and analysis.comp_latch_n_v is not None
            np.testing.assert_allclose(dotted[0].get_ydata(), analysis.comp_latch_p_v[column])
            np.testing.assert_allclose(dotted[1].get_ydata(), analysis.comp_latch_n_v[column])
        return save(fig, output_path)

    monkeypatch.setattr(analysis_plots, "save_figure", check_dotted_nodes)

    paths = plot_adc_cdac_settling(
        measurement,
        analysis,
        output_path=tmp_path / "cdac_settling",
    )

    assert_plot_formats(paths)
    svg = read_svg(paths)
    assert "C0 (cycle 0)" in svg
    assert "C7 (cycle 7)" in svg
    assert "C15 (cycle 15)" in svg
    assert "VDAC residual (mV)" in svg
    assert "internal_p" in svg
    assert "internal_n" in svg
    for label in (
        "clk_comp",
        "out_p",
        "out_n",
        "clk_logic",
        "state_p",
        "state_n",
        "bot_p",
        "bot_n",
        "top_p",
        "top_n",
    ):
        assert label in svg


def test_comparator_campaign_and_cdac_ab_plots_are_separate_per_adc(tmp_path: Path) -> None:
    comparator_groups = []
    comparator_analyses = []
    for vin_cm_v in (0.6, 0.8):
        group = []
        for vin_diff_v, ones in ((-1e-3, 100), (0.0, 50), (1e-3, 0)):
            base = comparator_measurement()
            group.append(
                MeasCompExt(
                    info=replace(base.info, measurement_type="MeasCompExt", backend="physical"),
                    param=_build_comp_params(
                        adc_index=0,
                        campaign="comp_common_mode",
                        sampling_mode="track",
                        sweep_stage="fixed",
                        vin_cm_v=vin_cm_v,
                        vin_diff_v=vin_diff_v,
                        conversions=100,
                    ),
                    daq=CompDaq(
                        trial_index=np.arange(100),
                        vin_diff_v=np.full(100, vin_diff_v),
                        vin_cm_v=np.full(100, vin_cm_v),
                        decision=np.concatenate((np.ones(ones, dtype=np.uint8), np.zeros(100 - ones, dtype=np.uint8))),
                    ),
                    wave=None,
                )
            )
        comparator_groups.append(group)
        comparator_analyses.append(analyze_comp_offset_noise(group))
    comparator_paths = plot_comp_common_mode_campaign(
        comparator_groups,
        comparator_analyses,
        output_path=tmp_path / "comp_campaign",
    )
    assert comparator_paths[0].is_file()
    assert plt.imread(comparator_paths[0]).shape[:2] == (2700, 4800)

    params = _build_cdac_params(
        adc_index=0,
        side="p",
        element=0,
        direction="1to0",
        dac_diffcaps=0,
        vin_diff_v=0.3,
        conversions=1,
        sweep_stage="fixed",
    )
    cdac_measurement = replace(all_measurements()[4], param=params)
    cdac_analysis = AnalysisCdacCapMismatch(
        adc_index=0,
        expected_effective_fraction=np.full(16, 0.015),
        main_fraction=np.full((2, 16), 0.02),
        diff_fraction=np.full((2, 16), 0.005),
        effective_fraction=np.full((2, 16), 0.015),
        effective_fraction_by_direction=np.full((2, 16, 2), 0.015),
        direction_bias=np.zeros((2, 16, 2)),
    )
    cdac_paths = plot_cdac_cap_mismatch(
        [cdac_measurement],
        cdac_analysis,
        output_path=tmp_path / "cdac_ab",
    )
    assert cdac_paths[0].is_file()
    assert plt.imread(cdac_paths[0]).shape[:2] == (2700, 4800)

    comparison_groups = []
    comparison_analyses = []
    for adc_index in range(4):
        adc_params = _build_cdac_params(
            adc_index=adc_index,
            side="p",
            element=0,
            direction="1to0",
            dac_diffcaps=0,
            vin_diff_v=0.3,
            conversions=1,
            sweep_stage="fixed",
        )
        comparison_groups.append([replace(cdac_measurement, param=adc_params)])
        comparison_analyses.append(replace(cdac_analysis, adc_index=adc_index))
    comparison_paths = plot_cdac_cap_mismatch_comparison(
        comparison_groups,
        comparison_analyses,
        output_path=tmp_path / "cdac_ab_comparison",
    )
    assert comparison_paths[0].is_file()
    assert plt.imread(comparison_paths[0]).shape[:2] == (2700, 4800)


def test_comparator_common_mode_crop_and_sampling_noise_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    figures = []

    def capture_figure(fig, *_args, **_kwargs):
        figures.append(fig)
        return ()

    monkeypatch.setattr(analysis_plots, "save_figure", capture_figure)

    def group(
        campaign: str,
        mode: str,
        vin_cm_v: float,
        *,
        center_v: float = 0.0,
        coupling_percent: float | None = None,
    ):
        measurements = []
        for vin_diff_v, ones in (
            (center_v - 1.0e-3, 0),
            (center_v, 50),
            (center_v + 1.0e-3, 100),
        ):
            base = comparator_measurement()
            measurements.append(
                MeasCompExt(
                    info=replace(base.info, measurement_type="MeasCompExt", backend="physical"),
                    param=_build_comp_params(
                        adc_index=0,
                        campaign=campaign,
                        sampling_mode=mode,
                        sweep_stage="fixed",
                        vin_cm_v=vin_cm_v,
                        vin_diff_v=vin_diff_v,
                        conversions=100,
                        requested_dac_rail_percent=coupling_percent,
                    ),
                    daq=CompDaq(
                        trial_index=np.arange(100),
                        vin_diff_v=np.full(100, vin_diff_v),
                        vin_cm_v=np.full(100, vin_cm_v),
                        decision=np.concatenate((np.ones(ones, dtype=np.uint8), np.zeros(100 - ones, dtype=np.uint8))),
                    ),
                    wave=None,
                )
            )
        return measurements

    common_groups = [
        group("comp_common_mode", "track", vin_cm_v, center_v=10.0e-3) for vin_cm_v in (0.6, 0.7, 0.8, 1.0)
    ]
    plot_comp_common_mode_campaign(
        common_groups,
        [analyze_comp_offset_noise(values) for values in common_groups],
        output_path=Path("unused_common"),
    )
    common_figure = figures[-1]
    assert [ax.get_title() for ax in common_figure.axes] == [
        "Comparator S-curve (CDF)",
        "Gaussian fit of μ (threshold) and σ (noise)",
    ]
    assert common_figure.axes[0].get_xlim() == pytest.approx((0.0, 25.0))
    assert common_figure.axes[0].get_xlabel() == "Differential input (mV)"
    assert common_figure.axes[1].get_ylim() == pytest.approx((0.0, 25.0))
    assert [
        line.get_label() for line in common_figure.axes[0].get_lines() if line.get_label().startswith("Vin_cm")
    ] == [
        "Vin_cm = 0.7 V",
        "Vin_cm = 0.8 V",
        "Vin_cm = 1 V",
    ]
    assert common_figure.axes[1].get_xticks() == pytest.approx((0.7, 0.8, 1.0))
    assert common_figure.axes[1].get_xlim() == pytest.approx((0.65, 1.05))
    assert common_figure.axes[1].get_xlabel() == "Common-mode input (V)"
    assert common_figure.axes[1].get_ylabel() == "Input error (mV)"
    common_fit_lines = [line for line in common_figure.axes[0].get_lines() if line.get_label().startswith("Vin_cm")]
    assert len(common_figure.axes[0].collections) == 3
    assert all(len(line.get_xdata()) == 1_001 for line in common_fit_lines)
    expected_common_mode_colors = [SPECTRUM_COLOR_MAP((vin_cm_v - 0.7) / (1.2 - 0.7)) for vin_cm_v in (0.7, 0.8, 1.0)]
    for line, expected_color in zip(common_fit_lines, expected_common_mode_colors, strict=True):
        np.testing.assert_allclose(mcolors.to_rgba(line.get_color()), expected_color)
    for violin, expected_color in zip(
        common_figure.axes[1].collections,
        expected_common_mode_colors,
        strict=True,
    ):
        np.testing.assert_allclose(violin.get_facecolor()[0, :3], expected_color[:3], atol=0.01)
        assert violin.get_facecolor()[0, 3] == pytest.approx(1.0)

    sampling_groups = [
        group(
            "comp_sampling_noise",
            mode,
            0.7,
            center_v=10.0e-3 + coupling_percent * 5.0e-6 + (0.2e-3 if mode == "hold" else 0.0),
            coupling_percent=coupling_percent,
        )
        for coupling_percent in (0.0, 25.0, 50.0, 75.0, 100.0)
        for mode in ("track", "hold")
    ]
    plot_comp_sampling_campaign(
        sampling_groups,
        [analyze_comp_offset_noise(values) for values in sampling_groups],
        output_path=Path("unused_sampling"),
    )
    sampling_figure = figures[-1]
    assert [ax.get_title() for ax in sampling_figure.axes] == [
        "Comparator S-curves (CDF)",
        "Gaussian fit of μ (threshold) and σ (noise)",
    ]
    assert sampling_figure.axes[0].get_xlim() == pytest.approx((0.0, 25.0))
    curve_labels = [text.get_text() for text in sampling_figure.axes[0].get_legend().get_texts()]
    assert curve_labels == [
        "P/N = 0/100%",
        "P/N = 25/75%",
        "P/N = 50/50%",
        "P/N = 75/25%",
        "P/N = 100/0%",
    ]
    sampling_fit_lines = [line for line in sampling_figure.axes[0].get_lines() if len(line.get_xdata()) == 1_001]
    assert len(sampling_fit_lines) == 10
    assert len(sampling_figure.axes[0].collections) == 10
    assert all(len(line.get_xdata()) == 1_001 for line in sampling_fit_lines)
    distribution_ax = sampling_figure.axes[1]
    assert distribution_ax.get_xlim() == pytest.approx((-8.0, 108.0))
    assert distribution_ax.get_ylim() == pytest.approx((0.0, 25.0))
    assert distribution_ax.get_xticks() == pytest.approx((0.0, 25.0, 50.0, 75.0, 100.0))
    assert distribution_ax.get_xlabel() == "VDAC coupling (P/N % of VDD_DAC)"
    assert distribution_ax.get_ylabel() == "Input error (mV)"
    distribution_labels = [text.get_text() for text in distribution_ax.get_legend().get_texts()]
    assert distribution_labels == ["Track", "Hold"]
    assert len(distribution_ax.collections) == 10
    assert not distribution_ax.texts
    assert sampling_figure._suptitle.get_text() == (
        "Comparator threshold and input-referred noise versus VDAC coupling"
    )

    for fig in figures:
        plt.close(fig)


def test_cdac_pex_expectation_includes_recorded_topplate_parasitic() -> None:
    params = _build_cdac_params(
        adc_index=0,
        side="p",
        element=0,
        direction="1to0",
        dac_diffcaps=0,
        vin_diff_v=0.3,
        conversions=1,
        sweep_stage="fixed",
    )
    base = replace(all_measurements()[4], param=params)
    measurement = replace(
        base,
        info=replace(base.info, readbacks={"cdac_topplate_parasitic_weight": 100.0}),
    )
    weights = np.asarray(params.tb.dut.cdac.weights, dtype=np.float64)
    expected = weights / (np.sum(65.0 * np.ceil(weights / 64.0)) + 100.0)
    np.testing.assert_allclose(
        _expected_cdac_effective_fraction([measurement]),
        expected,
    )

    inconsistent = replace(
        measurement,
        info=replace(measurement.info, readbacks={"cdac_topplate_parasitic_weight": 200.0}),
    )
    with pytest.raises(ValueError, match="inconsistent"):
        _expected_cdac_effective_fraction([measurement, inconsistent])


def test_adc_transfer_noise_and_linearity_plots(tmp_path: Path) -> None:
    msmt = adc_measurement(
        np.repeat(np.arange(16), 8),
        vin_diff_v=np.repeat(np.linspace(-0.6, 0.6, 16), 8),
        internal=True,
    )
    outputs = (
        plot_adc_transfer(
            [msmt],
            analyze_adc_transfer([msmt]),
            output_path=tmp_path / "transfer",
        ),
        plot_adc_code_distribution(
            [msmt],
            analyze_adc_code_distribution([msmt]),
            output_path=tmp_path / "noise",
        ),
        plot_adc_static_nonlinearity(
            msmt,
            analyze_adc_nonlinearity(msmt, method="code_density", code_range=(1, 14)),
            output_path=tmp_path / "nonlin",
        ),
    )
    for paths in outputs:
        assert_plot_formats(paths)


def test_adc_ramp_plots_render_completed_analysis(tmp_path: Path) -> None:
    """Keep ramp plotters independent of measurements and CDAC fitting."""

    analysis = analyze_adc_ramp(adc_ramp_measurement())
    nominal = analysis.curves[0]
    analysis = replace(
        analysis,
        curves=(
            nominal,
            replace(
                nominal,
                decoding="calibration1",
                label="CDAC S-curve weights",
                transfer_mean_dout=nominal.transfer_mean_dout + 1.0,
            ),
        ),
    )
    outputs = (
        plot_adc_ramp_transfer(analysis, output_path=tmp_path / "ramp_transfer"),
        plot_adc_ramp_histogram(analysis, output_path=tmp_path / "ramp_histogram"),
        plot_adc_ramp_weights(analysis, output_path=tmp_path / "ramp_weights"),
        plot_adc_ramp_nonlinearity(analysis, output_path=tmp_path / "ramp_nonlinearity"),
    )
    for paths in outputs:
        assert_plot_formats(paths)
        svg = read_svg(paths)
        if "weights" in paths[0].stem:
            assert "Ideal" in svg
            assert "Direction-matched measured" in svg
        else:
            assert "Uncalibrated DOUT" in svg
            assert "CDAC S-curve weights" in svg
        assert plt.imread(paths[0]).shape[:2] == (2700, 4800)
    histogram_svg = read_svg(outputs[1])
    assert "Mean samples per code in bin" in histogram_svg
    assert "missing codes" not in histogram_svg


def test_dynamic_sweep_and_decision_path_plots(tmp_path: Path) -> None:
    measurements = []
    for index, frequency_hz in enumerate((1_000.0, 8_000.0)):
        sample_rate_hz = 100_000.0
        time_s = np.arange(4_096) / sample_rate_hz
        samples = np.rint(2_048.0 + 1_200.0 * np.sin(2.0 * np.pi * frequency_hz * time_s + 0.2))
        measurements.append(
            adc_measurement(
                samples,
                sample_rate_hz=sample_rate_hz,
                input_frequency_hz=frequency_hz,
                logic_phase_delay_symbols=index,
            )
        )
    dynamic = analyze_adc_dynamic(measurements[0])
    sweep = analyze_adc_dynamic_sweep(measurements)
    assert_plot_formats(
        plot_adc_dynamic(
            measurements[0],
            dynamic,
            output_path=tmp_path / "dynamic",
        )
    )
    assert_plot_formats(
        plot_adc_dynamic_sweep(
            measurements,
            sweep,
            output_path=tmp_path / "sweep",
        )
    )

    decisions = analyze_adc_decision_paths(measurements[0], selection="single")
    paths = plot_adc_decision_paths(
        measurements[0],
        decisions,
        output_path=tmp_path / "decisions",
    )
    assert_plot_formats(paths)
    decision_svg = read_svg(paths)
    assert "ADC decision paths" in decision_svg
    assert GRID_MAJOR_COLOR.lower() not in decision_svg.lower()

    all_decisions = analyze_adc_decision_paths(measurements[0], selection="all")
    density_paths = plot_adc_decision_path_density(
        measurements[0],
        all_decisions,
        output_path=tmp_path / "decision_density",
    )
    assert_plot_formats(density_paths)
    density_svg = read_svg(density_paths)
    assert "decision-path density" in density_svg
    assert "Full trajectory" in density_svg
    assert "Final trajectory" in density_svg
    assert "Code density" in density_svg
    assert "Successive approximation code (LSB)" in density_svg
    assert "Running estimate (LSB)" not in density_svg
    assert "N:" in density_svg
    assert "μ:" in density_svg
    assert "σ:" in density_svg
    assert "Count / N" in density_svg
    assert "Conversions per path" in density_svg
    assert GRID_MAJOR_COLOR.lower() not in density_svg.lower()
    assert analysis_plots.GRID_MINOR_COLOR.lower() not in density_svg.lower()
    assert plt.imread(density_paths[0]).shape[:2] == (2700, 4800)


def test_decision_path_density_holds_each_discrete_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not invent linearly interpolated SAR estimates between decisions."""

    msmt = adc_measurement([100, 101, 102])
    analysis = analyze_adc_decision_paths(msmt, selection="all")
    original_histogram2d = np.histogram2d
    sampled_estimates = []
    rendered_polygons = []

    def record_histogram2d(x, y, *args, **kwargs):
        sampled_estimates.append(np.asarray(y))
        return original_histogram2d(x, y, *args, **kwargs)

    original_poly_collection = analysis_plots.PolyCollection

    def record_poly_collection(vertices, *args, **kwargs):
        rendered_polygons.extend(np.asarray(vertices, dtype=np.float64))
        return original_poly_collection(vertices, *args, **kwargs)

    monkeypatch.setattr(np, "histogram2d", record_histogram2d)
    monkeypatch.setattr(analysis_plots, "PolyCollection", record_poly_collection)
    captured = {}

    def save(fig, output_path):
        captured["figure"] = fig
        captured["output_path"] = output_path
        return ()

    monkeypatch.setattr(analysis_plots, "save_figure", save)
    plot_adc_decision_path_density(
        msmt,
        analysis,
        output_path=tmp_path / "held_decision_density",
    )

    figure = captured["figure"]
    assert captured["output_path"] == tmp_path / "held_decision_density"
    assert all(ax.get_facecolor()[:3] == mcolors.to_rgb(NORD_LIGHT_BLUE) for ax in figure.axes[:3])
    info_boxes = (figure.axes[0].artists[0], figure.axes[2].artists[0])
    assert all(box.get_child().get_children()[0].get_fontsize() == INFO_BOX_FONT_SIZE for box in info_boxes)
    assert all(type(box.patch.get_boxstyle()).__name__ == "Round" for box in info_boxes)

    sampled = sampled_estimates[0].reshape(len(analysis.estimate_dout), -1)
    expected = np.repeat(analysis.estimate_dout, 8, axis=1)
    np.testing.assert_array_equal(sampled, expected)

    # The largest jump in one representative path must be connected at the
    # exact integer decision boundary and at its true endpoint values.
    cycle = int(np.argmax(np.abs(np.diff(analysis.estimate_dout[0])))) + 1
    previous = analysis.estimate_dout[0, cycle - 1]
    current = analysis.estimate_dout[0, cycle]
    previous_box_code = np.floor(previous + 0.5)
    current_box_code = np.floor(current + 0.5)
    lower_edge = min(previous_box_code, current_box_code) - 0.5
    upper_edge = max(previous_box_code, current_box_code) + 0.5
    expected_segment = np.asarray(
        (
            (float(cycle) - 0.05, lower_edge),
            (float(cycle) + 0.05, lower_edge),
            (float(cycle) + 0.05, upper_edge),
            (float(cycle) - 0.05, upper_edge),
        )
    )
    assert abs(current - previous) > 1
    assert any(np.allclose(segment, expected_segment) for segment in rendered_polygons)


def test_decision_path_density_marks_unresolved_code_dispersion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report an all-one-code capture as unresolved rather than noiseless."""

    measurement = adc_measurement([2_048] * 10)
    analysis = analyze_adc_decision_paths(measurement, selection="all")
    captured = {}

    def save(fig, output_path):
        captured["figure"] = fig
        captured["output_path"] = output_path
        return ()

    monkeypatch.setattr(analysis_plots, "save_figure", save)
    plot_adc_decision_path_density(
        measurement,
        analysis,
        output_path=tmp_path / "unresolved_decision_density",
    )

    figure = captured["figure"]
    statistics_box = figure.axes[2].artists[0]
    statistics_text = statistics_box.get_child().get_children()[0].get_text()
    assert captured["output_path"] == tmp_path / "unresolved_decision_density"
    assert statistics_text == "μ: 0\nσ: <1.0 LSB"
    plt.close(figure)


def test_noise_rate_and_power_sweep_plots(tmp_path: Path) -> None:
    measurements = []
    for adc_index, sample_rate_hz in ((0, 100_000.0), (1, 200_000.0)):
        time_s = np.arange(4_096) / sample_rate_hz
        samples = np.rint(2_048.0 + 1_200.0 * np.sin(2.0 * np.pi * 1_000.0 * time_s))
        readbacks = {}
        for rail, current_a in (("vdd_a", 2e-6), ("vdd_d", 40e-6), ("vdd_dac", 20e-6)):
            readbacks[f"{rail}_measured_voltage_v"] = 1.2
            readbacks[f"{rail}_measured_current_a"] = 0.5 * current_a
            readbacks[f"{rail}_active_average_current_a"] = current_a
            readbacks[f"{rail}_active_average_power_w"] = 1.2 * current_a
        measurements.append(
            adc_measurement(
                samples,
                sample_rate_hz=sample_rate_hz,
                input_frequency_hz=1_000.0,
                observed_adc=adc_index,
                readbacks=readbacks,
            )
        )

    noise = analyze_adc_noise_sweep(measurements)
    dynamic_paths = plot_adc_noise_sweep(
        measurements,
        AnalysisAdcNoiseComparison(
            active_conversion_rate_hz=noise.active_conversion_rate_hz,
            input_lsb_v=noise.input_lsb_v,
            input_referred_noise_rms_v=noise.input_referred_noise_rms_v,
            noise_valid=noise.noise_valid,
            series_label=("ADC00", "ADC01"),
        ),
        output_path=tmp_path / "dynamic_rate",
    )
    assert_plot_formats(dynamic_paths)
    assert plt.imread(dynamic_paths[0]).shape[:2] == (2700, 4800)
    dynamic_svg = read_svg(dynamic_paths).lower()
    assert "snr (db)" in dynamic_svg
    assert "enob (bit)" in dynamic_svg
    assert "input-referred noise (lsb rms)" in dynamic_svg
    assert "input-referred noise (mv rms)" in dynamic_svg
    assert "time per decision cycle (ns)" in dynamic_svg
    power_outputs = [
        plot_adc_power_sweep(
            (measurement,),
            analyze_adc_power_sweep((measurement,)),
            output_path=tmp_path / f"power_adc{measurement.param.observed_adc:02d}",
        )
        for measurement in measurements
    ]
    for power_paths in power_outputs:
        assert_plot_formats(power_paths)
    for power_paths in power_outputs:
        power_svg = read_svg(power_paths)
        assert "static and dynamic supply power" in power_svg
        component_labels = (
            "Digital static",
            "DAC static",
            "Analog static",
            "Digital dynamic",
            "DAC dynamic",
            "Analog dynamic",
        )
        assert [power_svg.index(label) for label in component_labels] == sorted(
            power_svg.index(label) for label in component_labels
        )
        assert [power_svg.rindex(label) for label in component_labels] == sorted(
            power_svg.rindex(label) for label in component_labels
        )
        assert "Total:" not in power_svg


def test_spice_power_rate_and_instantaneous_waveform_plots(tmp_path: Path) -> None:
    readbacks = {
        "vdd_a_active_average_power_w": 12.0e-6,
        "vdd_d_active_average_power_w": 24.0e-6,
        "vdd_dac_active_average_power_w": 36.0e-6,
    }
    measurement = adc_measurement(
        [100, 101, 102],
        readbacks=readbacks,
        internal=True,
        waveform_sample_count=201,
    )
    assert isinstance(measurement, MeasAdcInt)
    time_s = measurement.wave.time_s
    seq_init_v = np.zeros_like(measurement.wave.seq_init_v)
    seq_init_v[0, (time_s >= 25.0e-9) & (time_s <= 50.0e-9)] = 1.2
    seq_samp_v = np.zeros_like(seq_init_v)
    seq_samp_v[0, (time_s >= 75.0e-9) & (time_s <= 100.0e-9)] = 1.2
    seq_comp_v = np.zeros_like(seq_init_v)
    seq_comp_v[0, (time_s >= 125.0e-9) & (time_s <= 150.0e-9)] = 1.2
    seq_logic_v = np.zeros_like(seq_init_v)
    seq_logic_v[0, (time_s >= 175.0e-9) & (time_s <= 200.0e-9)] = 1.2
    active_stop_s = 650.0e-9
    currents = {}
    for rail, static_current_a, active_current_a in (
        ("vdd_a", 2.0e-6, 10.0e-6),
        ("vdd_d", 4.0e-6, 20.0e-6),
        ("vdd_dac", 6.0e-6, 30.0e-6),
    ):
        current_a = np.full_like(seq_init_v, active_current_a)
        current_a[0, time_s > active_stop_s] = static_current_a
        currents[f"{rail}_i"] = current_a
    measurement = replace(
        measurement,
        wave=replace(
            measurement.wave,
            seq_init_v=seq_init_v,
            seq_samp_v=seq_samp_v,
            seq_comp_v=seq_comp_v,
            seq_logic_v=seq_logic_v,
            **currents,
        ),
    )
    analysis = analyze_adc_power_sweep((measurement,))

    rate_paths = plot_adc_power_sweep(
        (measurement,),
        analysis,
        output_path=tmp_path / "spice_ideal_power_vs_conversion_rate",
    )
    waveform_paths = plot_adc_power_waveform(
        analyze_adc_power_waveform(measurement),
        output_path=tmp_path / "spice_ideal_10msps_supply_power",
    )

    assert rate_paths[0].name == "spice_ideal_power_vs_conversion_rate.png"
    assert_plot_formats(rate_paths)
    assert_plot_formats(waveform_paths)
    waveform_svg = read_svg(waveform_paths)
    assert "Analog (µW)" in waveform_svg
    assert "Digital (µW)" in waveform_svg
    assert "DAC (µW)" in waveform_svg
    assert "Static average" in waveform_svg
    assert "Active average" in waveform_svg
    assert "Sequencer" in waveform_svg
    assert "INIT" in waveform_svg
    assert "SAMP" in waveform_svg
    assert "COMP" in waveform_svg
    assert "LOGIC" in waveform_svg
    for tick in ("0", "125", "250", "375", "500", "625"):
        assert f"<!-- {tick} -->" in waveform_svg


def test_noise_sweep_plot_uses_stable_timing_colors(tmp_path: Path) -> None:
    measurements = [
        adc_measurement(
            [100, 100, 100, 100] if offset == -3 else [100, 101, 99, 100],
            sample_rate_hz=1.0e6,
            logic_phase_delay_symbols=offset,
        )
        for offset in range(-3, 4)
    ]
    paths = plot_adc_noise_sweep(
        measurements,
        analyze_adc_noise_sweep(measurements),
        output_path=tmp_path / "noise_sweep",
    )
    assert_plot_formats(paths)
    assert plt.imread(paths[0]).shape[:2] == (2700, 4800)
    svg = read_svg(paths).lower()
    assert "snr (db)" in svg
    assert "enob (bit)" in svg
    assert "input-referred noise (lsb rms)" in svg
    assert "input-referred noise (mv rms)" in svg
    assert "as % of decision cycle" in svg
    assert "logic offsets:" not in svg
    assert "#eceff4" in svg
    for color in ("#d08770", "#a3be8c", "#b48ead", "#ebcb8b", "#bf616a", "#88c0d0"):
        assert color in svg


def test_noise_distribution_sweep_uses_one_count_scale(tmp_path: Path) -> None:
    measurements = [
        adc_measurement(
            [100, 100, 101, 101, 101, 102],
            sample_rate_hz=sample_rate_hz,
            observed_adc=0,
            logic_phase_delay_symbols=2,
        )
        for sample_rate_hz in (1.0e6, 2.0e6, 3.0e6)
    ]
    paths = plot_adc_noise_distribution_sweep(
        measurements,
        analyze_adc_noise_sweep(measurements),
        output_path=tmp_path / "noise_distributions",
    )

    assert_plot_formats(paths)
    assert plt.imread(paths[0]).shape[:2] == (2700, 4800)
    svg = read_svg(paths)
    assert "ADC fixed-input output-code distributions" in svg
    assert "ADC: 00" in svg
    assert "CDAC init: h'5555" in svg
    assert "Global histogram scale" not in svg
    assert "Mean ±1σ" in svg


@pytest.mark.parametrize(
    ("code_center", "expected_y_limits"),
    ((2048, (2010.0, 2100.0)), (2200, (2160.0, 2250.0))),
)
def test_noise_distribution_grid_shares_axes_across_all_adcs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_center: int,
    expected_y_limits: tuple[float, float],
) -> None:
    measurement_groups = tuple(
        tuple(
            cast(
                MeasAdcExt,
                adc_measurement(
                    (
                        [code_center + adc_index] * 5
                        if adc_index == 0 and rate_hz == 10.0e6
                        else [
                            code_center - 2 + adc_index,
                            code_center - 1 + adc_index,
                            code_center + adc_index,
                            code_center + adc_index,
                            code_center + 1 + adc_index,
                        ]
                    ),
                    sample_rate_hz=rate_hz / 1.6,
                    observed_adc=adc_index,
                    logic_phase_delay_symbols=2,
                ),
            )
            for rate_hz in (2.0e6, 6.0e6, 10.0e6)
        )
        for adc_index in range(16)
    )
    analyses = tuple(analyze_adc_noise_sweep(measurements) for measurements in measurement_groups)
    captured = {}

    def save(fig, output_path):
        captured["figure"] = fig
        captured["output_path"] = output_path
        return ()

    monkeypatch.setattr(analysis_plots, "save_figure", save)
    paths = plot_adc_noise_distribution_grid(
        measurement_groups,
        analyses,
        output_path=tmp_path / "noise_distribution_grid",
    )

    assert paths == ()
    assert captured["output_path"] == tmp_path / "noise_distribution_grid"
    figure = captured["figure"]
    axes = figure.axes[:16]
    np.testing.assert_allclose(figure.get_size_inches(), (9.6, 5.4))
    assert len(figure.axes) == 17
    assert figure._suptitle.get_text() == "Code density vs sampling rate for fixed input"
    assert figure._suptitle.get_position()[0] == 0.5
    assert figure._suptitle.get_horizontalalignment() == "center"
    assert all(not ax.get_title() for ax in axes)
    assert all(ax.get_facecolor()[:3] == mcolors.to_rgb(NORD_LIGHT_BLUE) for ax in axes)
    assert len({ax.get_xlim() for ax in axes}) == 1
    assert len({ax.get_ylim() for ax in axes}) == 1
    assert axes[0].get_ylim() == expected_y_limits
    assert all(tuple(ax.get_xticks()) == (2.0, 6.0, 10.0) for ax in axes)
    assert len({tuple(ax.get_yticks()) for ax in axes}) == 1
    assert all(np.allclose(np.diff(ax.get_yticks()), 25.0) for ax in axes)
    assert all(not ax.get_xlabel() and not ax.get_ylabel() for ax in axes)
    assert all(len(ax.patches) > 0 and len(ax.lines) >= 6 for ax in axes)
    assert all(any(line.get_linestyle() == ":" for line in ax.lines) for ax in axes)
    assert all(not ax.texts for ax in axes)
    assert all(len(ax.artists) == 1 for ax in axes)
    for adc_index, (ax, analysis) in enumerate(zip(axes, analyses, strict=True)):
        order = np.argsort(analysis.active_conversion_rate_hz)
        means = analysis.mean_dout[order]
        standard_deviations = analysis.std_dout[order]
        summary_box = ax.artists[0]
        summary_texts = tuple(text_area.get_children()[0] for text_area in summary_box.get_child().get_children())
        dispersion_range_text = (
            "σ:"
            f"{'<1.0' if np.count_nonzero(analysis.count[order][0]) == 1 else f'{standard_deviations[0]:.1f}'}→"
            f"{'<1.0' if np.count_nonzero(analysis.count[order][-1]) == 1 else f'{standard_deviations[-1]:.1f}'} LSB"
        )
        assert tuple(text.get_text() for text in summary_texts) == (
            f"ADC:{adc_index:02d}",
            f"μ:{means[0]:.0f}→{means[-1]:.0f}",
            dispersion_range_text,
        )
        assert tuple(text.get_color() for text in summary_texts) == (
            TEXT_COLOR,
            NORD_ORANGE,
            NORD_GREEN,
        )
        assert all(text.get_fontsize() == INFO_BOX_FONT_SIZE for text in summary_texts)
        expected_location = 3 if float(np.mean(means)) > float(np.mean(expected_y_limits)) else 2
        assert summary_box.loc == expected_location
        assert summary_box.pad == mpl.rcParamsDefault["legend.borderpad"]
        assert type(summary_box.patch.get_boxstyle()).__name__ == "Round"
        assert summary_box.patch.get_boxstyle().pad == mpl.rcParamsDefault["legend.borderpad"]
    assert not figure.legends
    assert len(figure.artists) == 1
    np.testing.assert_allclose(figure.axes[-1].get_position().bounds, (0.84, 0.10, 0.02, 0.45))
    legend_box = figure.artists[0]
    legend_children = legend_box.get_child().get_children()
    legend_rows = legend_children[0].get_children()
    legend_texts = tuple(row.get_children()[0].get_children()[0] for row in legend_rows)
    legend_handles = tuple(row.get_children()[1].get_children()[0] for row in legend_rows)
    assert tuple(text.get_text() for text in legend_texts) == (
        "Gaussian fit",
        "Average (μ)",
        "Dispersion (σ)",
    )
    system_info = legend_children[1].get_children()[0]
    system_info_text = system_info.get_text()
    assert "ADCs: 00-15" in system_info_text
    assert "Board: test_board" in system_info_text
    assert "CDAC init: h'5555" in system_info_text
    assert "N: 5" in system_info_text
    assert legend_box.pad == mpl.rcParamsDefault["legend.borderpad"]
    assert type(legend_box.patch.get_boxstyle()).__name__ == "Round"
    assert legend_box.patch.get_boxstyle().pad == mpl.rcParamsDefault["legend.borderpad"]
    mean_line = axes[1].lines[-1]
    lower_deviation_line = axes[1].lines[-3]
    upper_deviation_line = axes[1].lines[-2]
    analysis = analyses[1]
    np.testing.assert_allclose(mean_line.get_ydata(), analysis.mean_dout)
    np.testing.assert_allclose(lower_deviation_line.get_ydata(), analysis.mean_dout - analysis.std_dout)
    np.testing.assert_allclose(upper_deviation_line.get_ydata(), analysis.mean_dout + analysis.std_dout)
    assert np.all(mean_line.get_xdata() < analysis.active_conversion_rate_hz / 1e6)
    assert np.all(lower_deviation_line.get_xdata() < analysis.active_conversion_rate_hz / 1e6)
    assert mean_line.get_marker() == lower_deviation_line.get_marker() == upper_deviation_line.get_marker() == "None"
    assert (
        mean_line.get_linestyle() == lower_deviation_line.get_linestyle() == upper_deviation_line.get_linestyle() == ":"
    )
    assert (
        mean_line.get_linewidth() == lower_deviation_line.get_linewidth() == upper_deviation_line.get_linewidth() == 1.0
    )
    zero_sigma_baseline = axes[0].lines[2]
    zero_sigma_impulse = axes[0].lines[3]
    np.testing.assert_allclose(zero_sigma_baseline.get_xdata(), (10.0, 10.0))
    np.testing.assert_allclose(zero_sigma_baseline.get_ydata(), expected_y_limits)
    np.testing.assert_allclose(zero_sigma_impulse.get_xdata(), (10.0, 8.0))
    np.testing.assert_allclose(zero_sigma_impulse.get_ydata(), (code_center, code_center))
    assert zero_sigma_baseline.get_linestyle() == zero_sigma_impulse.get_linestyle() == ":"
    assert figure._supxlabel.get_text() == "Active conversion rate (MS/s)"
    assert figure._supylabel.get_text() == "Output code (LSB)"
    assert all(text.get_fontsize() == 10.0 for text in (*legend_texts, system_info))
    assert legend_handles[1].get_marker() == legend_handles[2].get_marker() == "None"
    assert all(handle.get_linestyle() == ":" for handle in legend_handles)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    assert len({round(text.get_window_extent(renderer).x0, 6) for text in legend_texts}) == 1
    assert all(
        text.get_window_extent(renderer).x1 < handle.get_window_extent(renderer).x0
        for text, handle in zip(legend_texts, legend_handles, strict=True)
    )
    plt.close(figure)
