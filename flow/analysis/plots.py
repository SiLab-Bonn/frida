"""Plots of typed FRIDA measurements and analysis results."""

from __future__ import annotations

import os
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator, NullLocator, StrMethodFormatter
from scipy.special import ndtr

from flow.analysis.types import (
    AnalysisAdcCalibration,
    AnalysisAdcCdacSettling,
    AnalysisAdcCodeDistribution,
    AnalysisAdcDecisionPaths,
    AnalysisAdcDynamic,
    AnalysisAdcDynamicSweep,
    AnalysisAdcNoiseComparison,
    AnalysisAdcNoiseSweep,
    AnalysisAdcNonlinearity,
    AnalysisAdcPowerSweep,
    AnalysisAdcPowerWaveform,
    AnalysisAdcRamp,
    AnalysisAdcSamplingNoise,
    AnalysisAdcScopeBits,
    AnalysisAdcTransfer,
    AnalysisCdacCapMismatch,
    AnalysisCompCandidateSweep,
    AnalysisCompOffsetNoise,
    AnalysisCompPower,
    AnalysisCompTiming,
    AnalysisDiffampNoise,
    AnalysisWaveform,
    MeasAdc,
    MeasAdcExt,
    MeasAdcInt,
    MeasCdacExt,
    MeasCompExt,
    MeasCompInt,
    Measurement,
)
from flow.analysis.waveform import style_measurement_text

PLOT_PNGS = False
PLOT_SVGS = False
PLOT_PDFS = True
PNG_DPI = 500
INFO_BOX_FONT_SIZE = 7.0

# Nord presentation colors. The ordering gives all plots a stable semantic
# sequence instead of inheriting Matplotlib's version-dependent default cycle.
PLOT_FACE_COLOR = "white"
TEXT_COLOR = "black"
SPINE_COLOR = "#4C566A"
LEGEND_FACE_COLOR = "#ECEFF4"
GRID_MAJOR_COLOR = "#D8DEE9"
GRID_MINOR_COLOR = "#E5E9F0"
NORD_BLUE = "#5E81AC"
NORD_RED = "#BF616A"
NORD_GREEN = "#A3BE8C"
NORD_ORANGE = "#D08770"
NORD_PURPLE = "#B48EAD"
NORD_CYAN = "#88C0D0"
NORD_YELLOW = "#EBCB8B"
NORD_TEAL = "#8FBCBB"
NORD_LIGHT_BLUE = "#81A1C1"
NORD_DARK = "#4C566A"
CURVE_COLORS = (
    NORD_BLUE,
    NORD_ORANGE,
    NORD_GREEN,
    NORD_PURPLE,
    NORD_YELLOW,
    NORD_RED,
    NORD_CYAN,
    NORD_TEAL,
    NORD_LIGHT_BLUE,
    NORD_DARK,
)
SPECTRUM_COLOR_MAP = LinearSegmentedColormap.from_list(
    "nord_blue_orange_yellow",
    (NORD_BLUE, NORD_ORANGE, NORD_YELLOW),
)
DENSITY_COLOR_MAP = LinearSegmentedColormap.from_list(
    "nord_purple_orange_yellow",
    SPECTRUM_COLOR_MAP(np.linspace(0.2, 1.0, 256)),
)
PLOT_STYLE = mpl.RcParams(
    {
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "font.size": 10.0,
        "figure.figsize": (9.6, 5.4),
        "figure.constrained_layout.use": True,
        "axes.titlesize": 12.0,
        "axes.titlecolor": TEXT_COLOR,
        "axes.titleweight": "normal",
        "axes.labelsize": 10.0,
        "axes.labelcolor": TEXT_COLOR,
        "axes.edgecolor": SPINE_COLOR,
        "axes.linewidth": 0.8,
        "axes.facecolor": PLOT_FACE_COLOR,
        "axes.grid": False,
        "axes.prop_cycle": cycler(color=CURVE_COLORS),
        "xtick.color": TEXT_COLOR,
        "xtick.direction": "in",
        "xtick.labelsize": 10.0,
        "xtick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "xtick.top": True,
        "ytick.color": TEXT_COLOR,
        "ytick.direction": "in",
        "ytick.labelsize": 10.0,
        "ytick.major.size": 2.5,
        "ytick.minor.size": 1.5,
        "ytick.right": True,
        "text.color": TEXT_COLOR,
        "figure.facecolor": PLOT_FACE_COLOR,
        "figure.titlesize": 12.0,
        "figure.titleweight": "normal",
        "savefig.facecolor": PLOT_FACE_COLOR,
        "savefig.dpi": PNG_DPI,
        "savefig.bbox": None,
        "legend.loc": "best",
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.facecolor": LEGEND_FACE_COLOR,
        "legend.edgecolor": SPINE_COLOR,
        "legend.framealpha": 0.9,
        "legend.labelcolor": TEXT_COLOR,
        "legend.linewidth": 0.8,
        "legend.borderpad": 0.4,
        "legend.borderaxespad": 0.5,
        "legend.fontsize": 10.0,
        "legend.title_fontsize": 10.0,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_grid(ax: plt.Axes) -> None:
    """Apply the shared light grid."""

    # rcParams cannot give major and minor grid lines different appearances.
    # Retain explicitly selected minor-tick intervals, such as the 0.25 MSPS
    # measurement spacing. ``minorticks_on`` would replace them with an
    # AutoMinorLocator and produce misleading 0.20 MSPS tick marks.
    if isinstance(ax.xaxis.get_minor_locator(), NullLocator):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if isinstance(ax.yaxis.get_minor_locator(), NullLocator):
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        color=GRID_MAJOR_COLOR,
        alpha=0.95,
        linewidth=0.8,
    )
    ax.grid(
        True,
        which="minor",
        color=GRID_MINOR_COLOR,
        alpha=1.0,
        linewidth=0.5,
    )


def style_frequency_text(value: float) -> str:
    """Format one frequency with a compact SI prefix."""

    # rcParams does not format domain values embedded in labels and legends.
    for scale, suffix in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if abs(value) >= scale:
            return f"{value / scale:g} {suffix}"
    return f"{value:g} Hz"


def style_adc_code_dispersion_lsb(value: float, *, single_code: bool) -> str:
    """Format unresolved single-code dispersion as a one-LSB upper bound."""

    if single_code:
        return "<1.0"
    rounded = f"{value:.1f}"
    return f"{value:.2g}" if rounded == "0.0" else rounded


def style_info_box(
    ax: plt.Axes,
    lines: Sequence[str],
    *,
    location: str = "upper right",
    line_colors: Sequence[str] | None = None,
) -> None:
    """Add measurement setup using the same spacing and frame as a legend."""

    # Legend rcParams do not apply automatically to anchored offset boxes.
    if not lines:
        return
    if line_colors is not None and len(line_colors) != len(lines):
        raise ValueError("information-box line colors must match its lines")
    if line_colors is None:
        child = TextArea("\n".join(lines), textprops={"size": INFO_BOX_FONT_SIZE})
    else:
        child = VPacker(
            children=[
                TextArea(line, textprops={"color": color, "size": INFO_BOX_FONT_SIZE})
                for line, color in zip(lines, line_colors, strict=True)
            ],
            align="left",
            pad=0.0,
            sep=1.0,
        )
    box = AnchoredOffsetbox(
        loc=location,
        child=child,
        frameon=True,
        pad=mpl.rcParams["legend.borderpad"],
        borderpad=mpl.rcParams["legend.borderaxespad"],
    )
    box.patch.set_boxstyle(f"round,pad={mpl.rcParams['legend.borderpad']}")
    box.patch.set_facecolor(mpl.rcParams["legend.facecolor"])
    box.patch.set_edgecolor(mpl.rcParams["legend.edgecolor"])
    box.patch.set_alpha(mpl.rcParams["legend.framealpha"])
    box.patch.set_linewidth(mpl.rcParams["legend.linewidth"])
    ax.add_artist(box)


def style_measurement_group_text(msmt_list: Sequence[Measurement]) -> tuple[str, ...]:
    """Return only setup lines shared by every measurement in a group."""

    # rcParams cannot derive display text from typed measurement metadata.
    if not msmt_list:
        return ()
    shared = list(style_measurement_text(msmt_list[0]))
    for msmt in msmt_list[1:]:
        current = set(style_measurement_text(msmt))
        shared = [line for line in shared if line in current]
    adc_indices = sorted(
        {int(adc_index) for msmt in msmt_list if (adc_index := getattr(msmt.param, "observed_adc", None)) is not None}
    )
    shared = [line for line in shared if not line.startswith(("ADC: ", "ADCs: "))]
    if len(adc_indices) == 1:
        shared.insert(0, f"ADC: {adc_indices[0]:02d}")
    elif adc_indices:
        shared.insert(0, "ADCs: " + ", ".join(f"{adc_index:02d}" for adc_index in adc_indices))
    return tuple(shared)


def style_time_units(time_s: np.ndarray) -> tuple[float, str]:
    # rcParams cannot choose a readable unit from the plotted data extent.
    maximum = float(np.max(np.abs(time_s)))
    if maximum < 1e-9:
        return 1e12, "ps"
    if maximum < 1e-6:
        return 1e9, "ns"
    if maximum < 1e-3:
        return 1e6, "µs"
    if maximum < 1.0:
        return 1e3, "ms"
    return 1.0, "s"


def save_figure(
    fig: plt.Figure,
    output_path: Path,
) -> tuple[Path, ...]:
    output_path = Path(output_path)
    if output_path.suffix:
        raise ValueError("plot output_path must be a suffixless artifact stem")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    formats = tuple(
        output_format
        for output_format, enabled in (
            ("png", PLOT_PNGS),
            ("svg", PLOT_SVGS),
            ("pdf", PLOT_PDFS),
        )
        if enabled
    )
    if not formats:
        raise RuntimeError("at least one plot output format must be enabled")
    # Preserve figures that explicitly disable automatic layout. Matplotlib can
    # otherwise install the rcParam-selected constrained-layout engine after the
    # first backend renders, causing later formats to move manually positioned
    # axes underneath legends and colorbars.
    with mpl.rc_context({"figure.constrained_layout.use": False}):
        for output_format in formats:
            path = output_path.with_suffix(f".{output_format}")
            fig.savefig(path)
            paths.append(path)
    plt.close(fig)
    return tuple(paths)


@mpl.rc_context(PLOT_STYLE)
def plot_waveforms(
    analysis: AnalysisWaveform,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot one completed typed waveform analysis."""

    scale, unit = style_time_units(analysis.time_s)
    fig, axes = plt.subplots(
        len(analysis.signal_names),
        1,
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for ax, name, signal_unit, values in zip(
        axes,
        analysis.signal_names,
        analysis.signal_units,
        analysis.signal_values,
        strict=True,
    ):
        ax.plot(analysis.time_s * scale, values)
        suffix = f" ({signal_unit})" if signal_unit else ""
        ax.set_ylabel(f"{name}{suffix}")
        style_grid(ax)
    axes[-1].set_xlabel(f"Time ({unit})")
    style_info_box(axes[0], analysis.setup_lines)
    fig.suptitle(analysis.title)
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_diffamp_noise(
    analysis: AnalysisDiffampNoise,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot differential-amplifier noise distribution and spectrum."""

    centered_mv = analysis.centered_v * 1e3
    noise_rms_mv = analysis.noise_rms_v * 1e3
    gaussian_x_mv = np.linspace(-5.0 * noise_rms_mv, 5.0 * noise_rms_mv, 1001)
    gaussian_density_per_mv = np.exp(-0.5 * (gaussian_x_mv / noise_rms_mv) ** 2) / (noise_rms_mv * np.sqrt(2.0 * np.pi))
    fig, (histogram_ax, spectrum_ax) = plt.subplots(2, 1)
    histogram_ax.hist(
        centered_mv,
        bins=120,
        density=True,
        color=NORD_BLUE,
        edgecolor=PLOT_FACE_COLOR,
        linewidth=0.35,
        label="Measured samples",
    )
    histogram_ax.plot(
        gaussian_x_mv,
        gaussian_density_per_mv,
        color=CURVE_COLORS[1],
        linestyle=":",
        label=(f"Gaussian fit (raw µ = {analysis.mean_v * 1e3:.3f} mV subtracted; σ = {noise_rms_mv:.3f} mV)"),
    )
    histogram_ax.set_xlabel("Differential output noise about its mean (mV)")
    histogram_ax.set_ylabel("Density (mV⁻¹)")
    histogram_ax.legend()

    positive = analysis.spectrum_frequency_hz > 0.0
    spectrum_ax.loglog(
        analysis.spectrum_frequency_hz[positive],
        analysis.spectrum_amplitude_density_v_per_sqrt_hz[positive] * 1e6,
        color=NORD_BLUE,
        label=f"Measured spectrum (integrated RMS = {analysis.integrated_fft_noise_rms_v * 1e3:.3f} mV)",
    )
    spectrum_ax.axvline(
        analysis.measurement_bandwidth_hz,
        color=CURVE_COLORS[1],
        linestyle=":",
        label=f"Measurement bandwidth = {analysis.measurement_bandwidth_hz / 1e6:g} MHz",
    )
    spectrum_ax.set_xlabel("Frequency (Hz)")
    spectrum_ax.set_ylabel("ASD (µV/√Hz)")
    spectrum_ax.legend()
    for ax in (histogram_ax, spectrum_ax):
        style_grid(ax)
    fig.suptitle("Differential-amplifier output noise")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_fastrx_scope_comparison(
    msmt: MeasAdcExt,
    analysis: AnalysisAdcScopeBits,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot CH2--CH4 and aligned Scope/FastRX decision streams."""

    wave = msmt.wave
    if wave is None:
        raise ValueError("scope/FastRX comparison plot requires a captured scope waveform")
    params = msmt.param.tb
    decision_period_s = 8.0 / float(params.symbol_rate)
    edge_times_s = analysis.comp_edge_times_s
    sample_times_s = analysis.sample_times_s
    decision_end_times_s = np.concatenate((edge_times_s[1:], [edge_times_s[-1] + decision_period_s]))
    decision_widths_s = decision_end_times_s - edge_times_s

    time_s = wave.time_s
    scale, unit = style_time_units(time_s)
    scaled_time = time_s * scale
    scaled_edges = edge_times_s * scale
    scaled_ends = decision_end_times_s * scale
    scaled_widths = decision_widths_s * scale
    scaled_samples = sample_times_s * scale
    display_start_s = max(float(time_s[0]), float(edge_times_s[0] - 4.0 * decision_period_s))
    display_end_s = min(float(time_s[-1]), float(decision_end_times_s[-1]))

    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.0, 1.35, 1.25)},
    )
    waveform_rows = (
        ("COMP", wave.seq_comp_v[0]),
        ("LOGIC", wave.seq_logic_v[0]),
        ("COMP_OUT", wave.comp_out_v[0]),
    )
    for ax, (label, values), color in zip(axes[:3], waveform_rows, CURVE_COLORS, strict=False):
        ax.plot(scaled_time, values, color=color)
        for edge in scaled_edges:
            ax.axvline(edge, color=SPINE_COLOR, alpha=0.18)
        ax.set_ylabel(f"{label} (V)")
        style_grid(ax)

    axes[2].scatter(
        scaled_samples,
        analysis.sample_values_v,
        marker="o",
        color=CURVE_COLORS[3],
        edgecolor=PLOT_FACE_COLOR,
        linewidth=0.5,
        zorder=5,
        label="Scope decode sample",
    )
    axes[2].legend()

    scope_values = analysis.scope_bits.astype(np.uint8)
    fastrx_values = analysis.fastrx_bits.astype(np.uint8)
    mismatches = analysis.mismatch_mask
    decision_ax = axes[3]
    for decision_index, (start, end, width) in enumerate(zip(scaled_edges, scaled_ends, scaled_widths, strict=True)):
        for y, values in ((1.0, scope_values), (0.0, fastrx_values)):
            bit = int(values[decision_index])
            decision_ax.barh(
                y,
                width,
                left=start,
                height=0.56,
                align="center",
                color=NORD_GREEN if bit else NORD_BLUE,
                edgecolor=NORD_RED if mismatches[decision_index] else PLOT_FACE_COLOR,
                linewidth=1.5 if mismatches[decision_index] else 0.8,
            )
            decision_ax.text(
                start + width / 2.0,
                y,
                str(bit),
                ha="center",
                va="center",
                fontweight="bold",
            )
    decision_ax.set_yticks((1.0, 0.0), labels=("Scope decode", "FastRX decode"))
    decision_ax.set_ylim(-0.6, 1.6)
    decision_ax.set_ylabel("Decision stream")
    decision_ax.set_xlabel(f"Time ({unit})")
    decision_ax.set_title("Decoded decision streams")
    style_grid(decision_ax)

    axes[0].set_xlim(display_start_s * scale, display_end_s * scale)
    setup_lines = (
        *(line for line in style_measurement_text(msmt) if line.startswith("ADC:")),
        f"Symbol rate: {float(params.symbol_rate) / 1e6:g} MBd",
        (
            "COMP→LOGIC: "
            f"{float(params.seq_logic_phase_delay_symbols) - float(params.seq_comp_phase_delay_symbols):+g} symbols"
        ),
    )
    style_info_box(axes[0], setup_lines)
    fig.suptitle("ADC scope and FastRX decision comparison")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_transfer(
    msmt_list: Sequence[MeasAdc],
    analysis: AnalysisAdcTransfer,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot individual ADC conversions and the mean static transfer."""

    fig, ax = plt.subplots()
    inputs = np.concatenate([msmt.daq.vin_diff_v for msmt in msmt_list])
    dout = np.concatenate([msmt.daq.dout for msmt in msmt_list])
    ax.scatter(inputs * 1e3, dout, label="Conversions")
    ax.errorbar(
        analysis.vin_diff_v * 1e3,
        analysis.mean_dout,
        yerr=analysis.std_dout,
        marker="o",
        capsize=2,
        label="Mean ± 1σ",
    )
    ax.set_xlabel("Differential input (mV)")
    ax.set_ylabel("ADC output (LSB)")
    ax.set_title("ADC static transfer")
    style_grid(ax)
    ax.legend()
    style_info_box(ax, style_measurement_group_text(msmt_list), location="lower right")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_ramp_transfer(
    analysis: AnalysisAdcRamp,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot the phase-reconstructed ramp transfer for every decoded curve."""

    fig, ax = plt.subplots()
    for curve in analysis.curves:
        ax.plot(
            curve.transfer_vin_diff_v * 1e3,
            curve.transfer_mean_dout,
            label=curve.label,
        )
    ax.set_xlabel("Inferred differential input (mV)")
    ax.set_ylabel("Mean ADC output (LSB)")
    ax.set_title("ADC ramp transfer")
    style_grid(ax)
    ax.legend()
    style_info_box(
        ax,
        (
            f"ADC: {analysis.adc_index:02d}",
            f"Sample rate: {analysis.sample_rate_hz / 1e6:.6g} MS/s",
            f"Ramp: {analysis.ramp_frequency_hz:.6g} Hz",
        ),
        location="lower right",
    )
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_ramp_histogram(
    analysis: AnalysisAdcRamp,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot overlaid code-density histograms from completed ramp analyses."""

    fig, ax = plt.subplots()
    number_bins = 128
    first_code = int(analysis.curves[0].code[0]) + 1
    last_code = int(analysis.curves[0].code[-1])
    bin_edges = np.linspace(first_code, last_code, number_bins + 1, dtype=np.int64)
    bin_edges = np.unique(bin_edges)
    for curve, color in zip(analysis.curves, CURVE_COLORS, strict=False):
        average_count = np.asarray([np.mean(curve.count[lower:upper]) for lower, upper in pairwise(bin_edges)])
        ax.stairs(
            average_count,
            bin_edges,
            baseline=0.0,
            fill=False,
            color=color,
            label=curve.label,
        )
    ax.set_xlabel("Output code")
    ax.set_ylabel("Mean samples per code in bin")
    ax.set_title("ADC ramp code density")
    style_grid(ax)
    ax.legend(ncols=2)
    style_info_box(
        ax,
        (
            f"ADC: {analysis.adc_index:02d}",
            f"Ramp: {analysis.ramp_frequency_hz:.6g} Hz",
        ),
        location="lower right",
    )
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_ramp_weights(
    analysis: AnalysisAdcRamp,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Compare nominal and direction-matched physical decision weights."""

    nominal, measured = analysis.curves[:2]
    code_max = len(nominal.code) - 1
    nominal_weights = nominal.weights * code_max / np.sum(nominal.weights)
    measured_weights = measured.weights * code_max / np.sum(measured.weights)
    stages = np.arange(16)
    relative_error_percent = 100.0 * (measured_weights[:-1] / nominal_weights[:-1] - 1.0)

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    axes[0].plot(stages, nominal_weights[:-1], "o-", color=NORD_DARK, label="Ideal")
    axes[0].plot(
        stages,
        measured_weights[:-1],
        "o-",
        color=NORD_BLUE,
        label="Direction-matched measured",
    )
    axes[0].set_yscale("log", base=2)
    axes[0].set_ylabel("Decision weight (LSB)")
    axes[0].legend()
    axes[1].axhline(0.0, color=SPINE_COLOR)
    axes[1].bar(stages, relative_error_percent, color=NORD_BLUE, width=0.7)
    axes[1].set_ylabel("Error (%)")
    axes[1].set_xlabel("Conversion stage")
    axes[1].set_xticks(stages)
    axes[1].set_xticklabels([f"C{stage}" for stage in stages])
    for ax in axes:
        style_grid(ax)
    style_info_box(axes[1], (f"ADC: {analysis.adc_index:02d}",), location="lower right")
    fig.suptitle("Ideal and extracted ADC decision weights")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_calibration_weights(
    analysis_list: Sequence[AnalysisAdcCalibration],
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Compare any calibration methods against one shared ideal weight set."""

    nominal = analysis_list[0].nominal_weights

    decision = np.arange(17)
    labels = [f"B{stage}\nC{stage}" for stage in range(16)] + ["B16\nterminal"]
    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    axes[0].plot(decision, nominal, "o-", color=NORD_DARK, label="Ideal")
    axes[1].axhline(0.0, color=SPINE_COLOR)
    for calibration, color in zip(analysis_list, CURVE_COLORS, strict=False):
        axes[0].plot(
            decision,
            calibration.calibrated_weights,
            "o-",
            color=color,
            label=calibration.label,
        )
        error_percent = 100.0 * (calibration.calibrated_weights / nominal - 1.0)
        axes[1].plot(
            decision,
            error_percent,
            "o-",
            color=color,
            label=calibration.label,
        )
        inferred = ~calibration.measured_weight_mask
        if np.any(inferred):
            axes[0].scatter(
                decision[inferred],
                calibration.calibrated_weights[inferred],
                marker="x",
                color=color,
                zorder=5,
            )
    axes[0].set_yscale("log", base=2)
    axes[0].set_ylabel("BOUT weight (LSB)")
    axes[0].set_title("Ideal and calibrated digital weights")
    axes[1].set_ylabel("Difference from ideal (%)")
    axes[1].set_xlabel("BOUT decision coefficient")
    axes[1].set_xticks(decision)
    axes[1].set_xticklabels(labels)
    axes[0].legend(ncols=2)
    for ax in axes:
        style_grid(ax)
    style_info_box(axes[1], (f"ADC: {analysis_list[0].adc_index:02d}",))
    fig.suptitle("ADC digital calibration weights")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_static_nonlinearity(
    msmt: MeasAdc,
    analysis: AnalysisAdcNonlinearity,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot static ADC DNL and INL from one completed analysis."""

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(analysis.code, analysis.dnl)
    axes[1].plot(analysis.code, analysis.inl)
    axes[0].axhline(0.0, color=SPINE_COLOR)
    axes[0].set_ylabel("DNL (LSB)")
    axes[1].axhline(0.0, color=SPINE_COLOR)
    axes[1].set_ylabel("INL (LSB)")
    axes[1].set_xlabel("Output code")
    for ax in axes:
        style_grid(ax)
    style_info_box(axes[0], style_measurement_text(msmt))
    fig.suptitle(f"ADC {analysis.method.replace('_', '-')} nonlinearity")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_ramp_nonlinearity(
    analysis: AnalysisAdcRamp,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot overlaid ramp DNL and INL for every completed decoding."""

    fig, axes = plt.subplots(2, 1, sharex=True)
    for curve in analysis.curves:
        axes[0].plot(curve.linearity_code, curve.dnl, label=curve.label)
        axes[1].plot(curve.linearity_code, curve.inl, label=curve.label)
    axes[0].axhline(0.0, color=SPINE_COLOR)
    axes[0].set_ylabel("DNL (LSB)")
    axes[1].axhline(0.0, color=SPINE_COLOR)
    axes[1].set_ylabel("INL (LSB)")
    axes[1].set_xlabel("Output code")
    for ax in axes:
        style_grid(ax)
    axes[0].legend()
    style_info_box(
        axes[1],
        (f"ADC: {analysis.adc_index:02d}", f"Ramp: {analysis.ramp_frequency_hz:.6g} Hz"),
    )
    fig.suptitle("ADC ramp nonlinearity")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_sampling_noise(
    analyses: Sequence[AnalysisAdcSamplingNoise],
    *,
    labels: Sequence[str],
    output_path: Path,
) -> tuple[Path, ...]:
    """Compare mean-centered held voltages using common microvolt bins and axes."""

    if not analyses or len(analyses) != len(labels):
        raise ValueError("sampling noise requires one label per analysis")
    errors_uv = [(item.held_diff_v - np.mean(item.held_diff_v)) * 1e6 for item in analyses]
    limit_uv = max(25.0, np.ceil(max(np.max(np.abs(values)) for values in errors_uv) / 25.0) * 25.0)
    bins_uv = np.arange(-limit_uv, limit_uv + 25.0, 25.0)
    columns = min(4, len(analyses))
    rows = (len(analyses) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
    if rows > 2:
        fig.set_size_inches(9.6, 2.7 * rows)
    for index, (analysis, label, error_uv) in enumerate(zip(analyses, labels, errors_uv, strict=True)):
        ax = axes.flat[index]
        ax.hist(
            error_uv,
            bins=bins_uv,
            weights=np.full(len(error_uv), 1.0 / len(error_uv)),
            color=CURVE_COLORS[index % len(CURVE_COLORS)],
            edgecolor=SPINE_COLOR,
            linewidth=0.4,
            alpha=0.85,
        )
        ax.set_title(label, fontsize=9)
        ax.set_xlim(-limit_uv, limit_uv)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        if index + columns >= len(analyses):
            ax.tick_params(labelbottom=True)
        style_grid(ax)
        style_info_box(ax, (f"SD = {analysis.sigma_v * 1e6:.1f} µV", f"N = {len(error_uv)}"))
    for ax in axes.flat[len(analyses) :]:
        ax.set_visible(False)
    fig.supxlabel("Held VDAC_P − VDAC_N, relative to each design's mean (µV)")
    fig.supylabel("Fraction of conversions per 25 µV bin")
    fig.suptitle("Held sampling-level noise · before the first comparator decision")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_code_distribution(
    msmt_list: Sequence[MeasAdc],
    analysis: AnalysisAdcCodeDistribution,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot code histograms and standard deviation at static input points."""

    fig, axes = plt.subplots(2, 1)
    for index, vin_diff_v in enumerate(analysis.vin_diff_v):
        active = analysis.count[index] > 0
        axes[0].step(
            analysis.code[active],
            analysis.count[index, active],
            where="mid",
            label=f"{vin_diff_v * 1e3:g} mV",
        )
    axes[0].set_xlabel("Output code")
    axes[0].set_ylabel("Count")
    axes[0].legend(ncols=2)
    axes[1].plot(analysis.vin_diff_v * 1e3, analysis.std_dout, marker="o")
    axes[1].set_xlabel("Differential input (mV)")
    axes[1].set_ylabel("Standard deviation (LSB)")
    for ax in axes:
        style_grid(ax)
    style_info_box(axes[1], style_measurement_group_text(msmt_list))
    fig.suptitle("ADC output-code distribution")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_noise_sweep(
    msmt_list: Sequence[MeasAdc],
    analysis: AnalysisAdcNoiseSweep | AnalysisAdcNoiseComparison,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot noise, equivalent full-scale SNR, and ENOB on one rate panel."""

    # The SNR and ENOB axes use a full-scale sine whose peak-to-peak
    # range is the ADC input range represented by all output codes.
    first = msmt_list[0]
    params = first.param.tb if isinstance(first, MeasAdcExt) else first.param
    adc_bits = params.dut.adc_bits
    full_scale_rms_lsb = ((1 << adc_bits) - 1) / (2.0 * np.sqrt(2.0))
    noise_rms_v = np.asarray(analysis.input_referred_noise_rms_v)
    noise_rms_lsb = noise_rms_v / analysis.input_lsb_v

    fig, ax = plt.subplots()
    conversion_rate_msps = analysis.active_conversion_rate_hz / 1e6
    if isinstance(analysis, AnalysisAdcNoiseSweep):
        timing_values = np.unique(analysis.comparator_time_percent)
        labels = tuple(f"{value:g}%" for value in timing_values)
        selections = tuple(analysis.comparator_time_percent == value for value in timing_values)
        colors = tuple(CURVE_COLORS[index % len(CURVE_COLORS)] for index in range(len(labels)))
    else:
        labels = tuple(dict.fromkeys(analysis.series_label))
        label_values = np.asarray(analysis.series_label)
        selections = tuple(label_values == label for label in labels)
        colors = tuple(CURVE_COLORS[index % len(CURVE_COLORS)] for index in range(len(labels)))

    for label, selected, color in zip(labels, selections, colors, strict=True):
        selected = selected & analysis.noise_valid
        if not np.any(selected):
            continue
        order = np.argsort(conversion_rate_msps[selected])
        selected_rate = conversion_rate_msps[selected][order]
        selected_noise_lsb = noise_rms_lsb[selected][order]
        if label == "Input stimulus noise":
            ax.hlines(
                float(np.mean(selected_noise_lsb)),
                float(selected_rate[0]),
                float(selected_rate[-1]),
                color=color,
                linestyle=":",
                label=label,
            )
        else:
            ax.plot(
                selected_rate,
                selected_noise_lsb,
                marker="o",
                color=color,
                label=label,
            )
    ax.set_xlabel("Active conversion rate (Msps)")
    ax.set_ylabel("Input-referred noise (LSB RMS)")
    ax.invert_yaxis()
    ax.set_ylim(9.0, 0.0)
    ax.set_yticks(np.arange(0.0, 10.0, 1.0))
    ax.set_xticks(np.arange(0.0, 11.0, 1.0))
    ax.set_xlim(0.0, 10.25)
    ax.set_xticks(np.arange(0.0, 10.251, 0.25), minor=True)
    ax.set_title("ADC noise performance vs conversion rate")
    ax.tick_params(which="both", right=False)
    style_grid(ax)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(
            ncols=4 if isinstance(analysis, AnalysisAdcNoiseSweep) else 1,
            title=(
                "COMP→LOGIC interval\n(as % of decision cycle)" if isinstance(analysis, AnalysisAdcNoiseSweep) else None
            ),
        )
    noise_mv_axis = ax.secondary_yaxis(
        "left",
        functions=(
            lambda noise_lsb: np.asarray(noise_lsb) * analysis.input_lsb_v * 1e3,
            lambda noise_mv: np.asarray(noise_mv) / (analysis.input_lsb_v * 1e3),
        ),
    )
    noise_mv_axis.spines["left"].set_position(("outward", 58))
    noise_mv_axis.set_ylabel("Input-referred noise (mV RMS)")
    noise_mv_axis.tick_params(which="both", left=True, right=False)

    enob_axis = ax.secondary_yaxis(
        "right",
        functions=(
            lambda noise_lsb: (
                (
                    20.0
                    * (
                        np.log10(full_scale_rms_lsb)
                        - np.log10(np.maximum(np.asarray(noise_lsb), np.finfo(np.float64).tiny))
                    )
                    - 1.76
                )
                / 6.02
            ),
            lambda enob_bits: full_scale_rms_lsb * np.power(10.0, -(6.02 * np.asarray(enob_bits) + 1.76) / 20.0),
        ),
    )
    enob_axis.set_ylabel("ENOB (bit)")
    enob_axis.set_yticks(np.arange(7.0, 13.0, 1.0))
    enob_axis.tick_params(which="both", left=False, right=True)

    snr_axis = ax.secondary_yaxis(
        "right",
        functions=(
            lambda noise_lsb: (
                20.0
                * (
                    np.log10(full_scale_rms_lsb)
                    - np.log10(np.maximum(np.asarray(noise_lsb), np.finfo(np.float64).tiny))
                )
            ),
            lambda snr_db: full_scale_rms_lsb * np.power(10.0, -np.asarray(snr_db) / 20.0),
        ),
    )
    snr_axis.spines["right"].set_position(("outward", 58))
    snr_axis.set_ylabel("SNR (dB)")
    snr_axis.set_yticks(np.arange(45.0, 71.0, 5.0))
    snr_axis.tick_params(which="both", left=False, right=True)

    decision_time_axis = ax.twiny()
    decision_time_axis.set_xlim(ax.get_xlim())
    decision_time_axis.xaxis.set_ticks_position("bottom")
    decision_time_axis.xaxis.set_label_position("bottom")
    decision_time_axis.spines["bottom"].set_position(("outward", 38))
    decision_time_axis.spines["top"].set_visible(False)
    decision_time_axis.set_xlabel("Time per decision cycle (ns)")
    labeled_rates_msps = np.arange(1.0, 11.0)
    decision_cycle_ns = 50.0 / labeled_rates_msps
    decision_time_axis.set_xticks(labeled_rates_msps)
    decision_time_axis.set_xticklabels(tuple(f"{interval:.3g}" for interval in decision_cycle_ns))
    decision_time_axis.tick_params(
        which="both",
        top=False,
        bottom=True,
    )
    style_info_box(ax, style_measurement_group_text(msmt_list), location="lower left")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_noise_distribution_sweep(
    msmt_list: Sequence[MeasAdc],
    analysis: AnalysisAdcNoiseSweep,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot left-facing output-code histograms along the conversion-rate axis."""

    code = analysis.code
    count = analysis.count
    order = np.argsort(analysis.active_conversion_rate_hz)
    rates_msps = analysis.active_conversion_rate_hz[order] / 1e6
    counts = count[order]
    populated = np.flatnonzero(np.any(counts > 0, axis=0))
    first_code = max(0, int(populated[0]) - 2)
    last_code = min(len(code) - 1, int(populated[-1]) + 2)
    codes = code[first_code : last_code + 1]
    visible_counts = counts[:, first_code : last_code + 1]

    fig, ax = plt.subplots()
    maximum_count = int(np.max(visible_counts))
    histogram_scale = int(np.ceil(maximum_count / 10_000.0) * 10_000)
    if len(rates_msps) == 1:
        maximum_width_msps = 0.2
    else:
        maximum_width_msps = min(0.2, 0.8 * float(np.min(np.diff(rates_msps))))
    for rate_msps, histogram in zip(rates_msps, visible_counts, strict=True):
        populated_codes = histogram > 0
        widths = maximum_width_msps * histogram[populated_codes] / histogram_scale
        ax.barh(
            codes[populated_codes],
            widths,
            left=rate_msps - widths / 2.0,
            height=1.0,
            facecolor=NORD_BLUE,
            edgecolor=NORD_BLUE,
            linewidth=0.45,
        )
    mean = analysis.mean_dout[order]
    std = analysis.std_dout[order]
    ax.plot(rates_msps, mean, color=CURVE_COLORS[1], marker="o", label="Mean")
    ax.plot(
        rates_msps,
        mean - std,
        color=CURVE_COLORS[2],
        linestyle="--",
        label="Mean ±1σ",
    )
    ax.plot(rates_msps, mean + std, color=CURVE_COLORS[2], linestyle="--")
    ax.set_xlabel("Active conversion rate (Msps)")
    ax.set_ylabel("ADC output code (LSB)")
    ax.set_xticks(np.arange(0.0, 11.0, 1.0))
    ax.set_xticks(np.arange(0.0, 10.251, 0.25), minor=True)
    ax.set_xlim(0.0, 10.25)
    ax.set_ylim(mean[0] - 3.0 * std[0], mean[0] + 3.0 * std[0])
    ax.set_title("ADC fixed-input output-code distributions")
    style_grid(ax)
    ax.legend()
    style_info_box(ax, style_measurement_group_text(msmt_list), location="lower left")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_noise_distribution_grid(
    msmt_groups: Sequence[Sequence[MeasAdcExt]],
    analyses: Sequence[AnalysisAdcNoiseSweep],
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot three rate-indexed code densities for each ADC in a 4-by-4 grid."""

    if len(msmt_groups) != 16 or len(analyses) != 16:
        raise ValueError("ADC noise-distribution grid requires 16 measurement groups and analyses")
    groups_by_adc = {}
    for measurements, analysis in zip(msmt_groups, analyses, strict=True):
        adc_indices = {measurement.param.observed_adc for measurement in measurements}
        if len(adc_indices) != 1:
            raise ValueError("each ADC noise-distribution group must contain one observed ADC")
        adc_index = adc_indices.pop()
        if adc_index is None or adc_index in groups_by_adc:
            raise ValueError("ADC noise-distribution groups require unique observed ADC indices")
        if len(analysis.active_conversion_rate_hz) != 3:
            raise ValueError("each ADC noise-distribution grid analysis requires three conversion rates")
        groups_by_adc[adc_index] = analysis
    if set(groups_by_adc) != set(range(16)):
        raise ValueError("ADC noise-distribution grid requires ADC00 through ADC15")

    means = np.concatenate(tuple(analysis.mean_dout for analysis in analyses))
    y_span_lsb = 90.0
    y_center_lsb = 5.0 * np.round((float(np.min(means)) + float(np.max(means))) / 10.0)
    y_limits = (y_center_lsb - y_span_lsb / 2.0, y_center_lsb + y_span_lsb / 2.0)
    all_measurements = tuple(measurement for group in msmt_groups for measurement in group)
    shared_setup_lines = style_measurement_group_text(all_measurements)
    cdac_setup_lines = tuple(line for line in shared_setup_lines if line.startswith("CDAC init: "))
    shared_setup_lines = tuple(
        line for line in shared_setup_lines if not line.startswith(("ADC: ", "ADCs: ", "CDAC init: "))
    )
    campaign_sample_counts = np.unique(np.concatenate(tuple(analysis.sample_count for analysis in analyses)))
    if len(campaign_sample_counts) == 1:
        sample_count_text = f"{int(campaign_sample_counts[0]):.3g}"
    else:
        sample_count_text = f"{int(np.min(campaign_sample_counts)):.3g}-{int(np.max(campaign_sample_counts)):.3g}"
    system_info_lines = (
        f"ADCs: {min(groups_by_adc):02d}-{max(groups_by_adc):02d}",
        *shared_setup_lines,
        *cdac_setup_lines,
        f"N: {sample_count_text}",
    )
    system_info_text = "\n".join(system_info_lines)
    maximum_sample_count = max(int(np.max(analysis.sample_count)) for analysis in analyses)
    density_norm = LogNorm(vmin=1, vmax=max(2, maximum_sample_count))
    maximum_width_msps = 2.0

    fig, axes = plt.subplots(
        4,
        4,
        sharex=True,
        sharey=True,
        layout="none",
    )
    fig.subplots_adjust(left=0.075, right=0.79, bottom=0.105, top=0.94, wspace=0.05, hspace=0.05)
    fig.suptitle("Code density vs sampling rate for fixed input", y=0.985)
    for adc_index, ax in enumerate(axes.flat):
        analysis = groups_by_adc[adc_index]
        order = np.argsort(analysis.active_conversion_rate_hz)
        rates_msps = analysis.active_conversion_rate_hz[order] / 1e6
        counts = analysis.count[order]
        means = analysis.mean_dout[order]
        standard_deviations = analysis.std_dout[order]
        sample_counts = analysis.sample_count[order]
        mean_curve_x = []
        standard_deviation_curve_x = []
        for rate_msps, histogram, sample_count, mean, standard_deviation in zip(
            rates_msps,
            counts,
            sample_counts,
            means,
            standard_deviations,
            strict=True,
        ):
            populated = histogram > 0
            fractions = histogram[populated] / sample_count
            peak_fraction_per_lsb = float(np.max(fractions))
            widths = maximum_width_msps * fractions / peak_fraction_per_lsb
            ax.barh(
                analysis.code[populated],
                widths,
                left=rate_msps - widths,
                height=1.0,
                color=DENSITY_COLOR_MAP(density_norm(histogram[populated])),
                edgecolor="none",
                rasterized=True,
                zorder=2,
            )
            if standard_deviation > 0.0:
                fit_code = np.linspace(*y_limits, 501)
                fit_fraction_per_lsb = np.exp(-0.5 * ((fit_code - mean) / standard_deviation) ** 2) / (
                    standard_deviation * np.sqrt(2.0 * np.pi)
                )
                fit_fraction_per_lsb = np.minimum(fit_fraction_per_lsb, peak_fraction_per_lsb)
                fit_x = rate_msps - maximum_width_msps * fit_fraction_per_lsb / peak_fraction_per_lsb
                ax.plot(fit_x, fit_code, color=TEXT_COLOR, linestyle=":", zorder=3)
                gaussian_peak = 1.0 / (standard_deviation * np.sqrt(2.0 * np.pi))
                mean_curve_x.append(
                    rate_msps - maximum_width_msps * min(gaussian_peak, peak_fraction_per_lsb) / peak_fraction_per_lsb
                )
                standard_deviation_curve_x.append(
                    rate_msps
                    - maximum_width_msps
                    * min(gaussian_peak * np.exp(-0.5), peak_fraction_per_lsb)
                    / peak_fraction_per_lsb
                )
            else:
                ax.plot(
                    (rate_msps, rate_msps),
                    y_limits,
                    color=TEXT_COLOR,
                    linestyle=":",
                    zorder=3,
                )
                ax.plot(
                    (rate_msps, rate_msps - maximum_width_msps),
                    (mean, mean),
                    color=TEXT_COLOR,
                    linestyle=":",
                    zorder=3,
                )
                mean_curve_x.append(rate_msps - maximum_width_msps)
                standard_deviation_curve_x.append(rate_msps - maximum_width_msps)

        ax.plot(
            standard_deviation_curve_x,
            means - standard_deviations,
            color=NORD_GREEN,
            linestyle=":",
            zorder=4,
        )
        ax.plot(
            standard_deviation_curve_x,
            means + standard_deviations,
            color=NORD_GREEN,
            linestyle=":",
            zorder=4,
        )
        ax.plot(mean_curve_x, means, color=NORD_ORANGE, linestyle=":", zorder=5)
        summary_location = "lower left" if float(np.mean(means)) > float(np.mean(y_limits)) else "upper left"
        dispersion_range_text = (
            "σ:"
            f"{style_adc_code_dispersion_lsb(float(standard_deviations[0]), single_code=np.count_nonzero(counts[0]) == 1)}→"
            f"{style_adc_code_dispersion_lsb(float(standard_deviations[-1]), single_code=np.count_nonzero(counts[-1]) == 1)} LSB"
        )
        style_info_box(
            ax,
            (
                f"ADC:{adc_index:02d}",
                f"μ:{means[0]:.0f}→{means[-1]:.0f}",
                dispersion_range_text,
            ),
            location=summary_location,
            line_colors=(TEXT_COLOR, NORD_ORANGE, NORD_GREEN),
        )
        ax.set_facecolor(NORD_LIGHT_BLUE)
        ax.set_xlim(-0.25, 10.5)
        ax.set_ylim(*y_limits)
        ax.set_xticks((2.0, 6.0, 10.0))
        ax.yaxis.set_major_locator(MultipleLocator(25.0))

    colorbar_ax = fig.add_axes((0.84, 0.10, 0.02, 0.45))
    colorbar = fig.colorbar(
        ScalarMappable(norm=density_norm, cmap=DENSITY_COLOR_MAP),
        cax=colorbar_ax,
    )
    colorbar.set_label("Conversions per code")
    legend_rows: list[Artist] = []
    for label, color in (
        ("Gaussian fit", TEXT_COLOR),
        ("Average (μ)", NORD_ORANGE),
        ("Dispersion (σ)", NORD_GREEN),
    ):
        handle = DrawingArea(22.0, 10.0)
        handle.add_artist(Line2D((0.0, 22.0), (5.0, 5.0), color=color, linestyle=":"))
        legend_rows.append(
            HPacker(
                children=[TextArea(label, textprops={"size": mpl.rcParams["legend.fontsize"]}), handle],
                align="center",
                pad=0.0,
                sep=5.0,
            )
        )
    legend_box = AnchoredOffsetbox(
        loc="upper left",
        bbox_to_anchor=(0.80, 0.92),
        bbox_transform=fig.transFigure,
        child=VPacker(
            children=[
                VPacker(children=legend_rows, align="left", pad=0.0, sep=2.0),
                TextArea(
                    system_info_text,
                    textprops={"multialignment": "left", "size": mpl.rcParams["legend.fontsize"]},
                ),
            ],
            align="left",
            pad=0.0,
            sep=6.0,
        ),
        frameon=True,
        pad=mpl.rcParams["legend.borderpad"],
        borderpad=mpl.rcParams["legend.borderaxespad"],
    )
    legend_box.patch.set_boxstyle(f"round,pad={mpl.rcParams['legend.borderpad']}")
    legend_box.patch.set_facecolor(mpl.rcParams["legend.facecolor"])
    legend_box.patch.set_edgecolor(mpl.rcParams["legend.edgecolor"])
    legend_box.patch.set_alpha(mpl.rcParams["legend.framealpha"])
    legend_box.patch.set_linewidth(mpl.rcParams["legend.linewidth"])
    fig.add_artist(legend_box)
    fig.supxlabel("Active conversion rate (MS/s)")
    fig.supylabel("Output code (LSB)", x=0.01)
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_dynamic(
    msmt: MeasAdc,
    analysis: AnalysisAdcDynamic,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot sine fit, residual, and spectrum for one dynamic acquisition."""

    time_scale, time_unit = style_time_units(analysis.time_s)
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(analysis.time_s * time_scale, analysis.measured_dout, label="Measured")
    axes[0].plot(analysis.time_s * time_scale, analysis.fitted_dout, label="Sine fit")
    axes[0].set_xlabel(f"Time ({time_unit})")
    axes[0].set_ylabel("ADC output (LSB)")
    axes[0].legend()
    axes[1].plot(analysis.time_s * time_scale, analysis.residual_dout)
    axes[1].set_xlabel(f"Time ({time_unit})")
    axes[1].set_ylabel("Residual (LSB)")
    positive = analysis.spectrum_frequency_hz > 0
    axes[2].semilogx(
        analysis.spectrum_frequency_hz[positive],
        analysis.spectrum_dbfs[positive],
        label=f"SNDR {analysis.spectral_sndr_db:.2f} dB; ENOB {analysis.spectral_enob_bits:.2f} bit",
    )
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Amplitude (dBFS)")
    for ax in axes:
        style_grid(ax)
    axes[2].legend()
    style_info_box(axes[1], style_measurement_text(msmt), location="lower right")
    fig.suptitle("ADC dynamic performance")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_dynamic_sweep(
    msmt_list: Sequence[MeasAdc],
    analysis: AnalysisAdcDynamicSweep,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot ENOB and SNDR versus input frequency for each conversion rate."""

    fig, axes = plt.subplots(2, 1, sharex=True)
    groups = np.unique(analysis.sample_rate_hz)
    for sample_rate_hz in groups:
        selected = analysis.sample_rate_hz == sample_rate_hz
        order = np.argsort(analysis.input_frequency_hz[selected])
        frequency = analysis.input_frequency_hz[selected][order]
        label = style_frequency_text(sample_rate_hz)
        axes[0].semilogx(frequency, analysis.spectral_enob_bits[selected][order], marker="o", label=label)
        axes[1].semilogx(frequency, analysis.spectral_sndr_db[selected][order], marker="o", label=label)
    axes[0].set_ylabel("ENOB (bit)")
    axes[1].set_ylabel("SNDR (dB)")
    axes[1].set_xlabel("Input frequency (Hz)")
    for ax in axes:
        style_grid(ax)
    axes[0].legend(title="Conversion rate")
    style_info_box(axes[1], style_measurement_group_text(msmt_list), location="lower right")
    fig.suptitle("ADC dynamic performance sweep")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_power_sweep(
    msmt_list: Sequence[MeasAdc],
    analysis: AnalysisAdcPowerSweep,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot one source's static and dynamic rail power versus rate."""

    order = np.argsort(analysis.active_conversion_rate_hz)
    rate_msps = analysis.active_conversion_rate_hz[order] / 1e6
    component_labels = (
        "Digital static",
        "DAC static",
        "Analog static",
        "Digital dynamic",
        "DAC dynamic",
        "Analog dynamic",
    )
    component_power_uw = tuple(
        values[order] * 1e6
        for values in (
            analysis.vdd_d_static_power_w,
            analysis.vdd_dac_static_power_w,
            analysis.vdd_a_static_power_w,
            analysis.vdd_d_dynamic_power_w,
            analysis.vdd_dac_dynamic_power_w,
            analysis.vdd_a_dynamic_power_w,
        )
    )
    analog_color, digital_color, dac_color = CURVE_COLORS[:3]
    component_colors = (
        digital_color,
        dac_color,
        analog_color,
        digital_color,
        dac_color,
        analog_color,
    )
    fig, ax = plt.subplots()
    collections = ax.stackplot(
        rate_msps,
        *component_power_uw,
        labels=component_labels,
        colors=component_colors,
    )
    for index, (collection, color) in enumerate(zip(collections, component_colors, strict=True)):
        collection.set_edgecolor(color)
        collection.set_linewidth(0.7)
        if index >= 3:
            collection.set_hatch("///")
    total_power_uw = analysis.total_power_w[order] * 1e6
    ax.plot(rate_msps, total_power_uw, color=TEXT_COLOR)
    ax.set_ylabel("Supply power (µW)")
    ax.set_xlabel("Active conversion rate (Msps)")
    ax.set_xlim(0.0, float(np.max(rate_msps)) + 0.25)
    if np.max(rate_msps) >= 1.0:
        ax.set_xticks(np.arange(1.0, np.floor(np.max(rate_msps)) + 1.0))
    ax.set_xticks(np.arange(0.0, float(np.max(rate_msps)) + 0.251, 0.25), minor=True)
    ax.set_ylim(0.0, max(float(np.max(total_power_uw)) * 1.25, 1.0))
    style_grid(ax)
    ax.legend()
    style_info_box(ax, style_measurement_group_text(msmt_list), location="lower right")
    ax.set_title("ADC static and dynamic supply power")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_cdac_settling(
    msmt: MeasAdcInt,
    analysis: AnalysisAdcCdacSettling,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot CDAC-update intervals, with saved regenerative nodes dotted."""

    time_scale, time_unit = style_time_units(analysis.time_s)
    fig, axes = plt.subplots(
        3,
        3,
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": (1.0, 1.0, 1.25)},
    )
    scaled_time = analysis.time_s * time_scale
    stage_cycles = tuple(dict.fromkeys(zip(analysis.stage_index, analysis.cycle_index, strict=True)))
    for column, (stage_index, cycle_index) in enumerate(stage_cycles):
        selected = analysis.stage_index == stage_index
        if not np.any(selected):
            raise ValueError(f"ADC CDAC settling plot is missing saved stage C{stage_index}")
        comparator_ax, drive_ax, settling_ax = axes[:, column]
        comparator_ax.plot(scaled_time, analysis.clk_comp_v[selected][0], color=NORD_DARK)
        for values in analysis.comp_out_p_v[selected]:
            comparator_ax.plot(scaled_time, values, color=NORD_BLUE, alpha=0.35)
        for values in analysis.comp_out_n_v[selected]:
            comparator_ax.plot(scaled_time, values, color=NORD_ORANGE, alpha=0.35)
        for name, color in (("comp_latch_p_v", NORD_BLUE), ("comp_latch_n_v", NORD_ORANGE)):
            if (internal := getattr(analysis, name)) is not None:
                for values in internal[selected]:
                    comparator_ax.plot(scaled_time, values, color=color, linestyle=":", alpha=0.35)

        drive_ax.plot(scaled_time, analysis.seq_logic_v[selected][0], color=NORD_DARK)
        for values in analysis.dac_state_p_v[selected]:
            drive_ax.plot(scaled_time, values, color=NORD_BLUE, linestyle="--", alpha=0.35)
        for values in analysis.dac_state_n_v[selected]:
            drive_ax.plot(scaled_time, values, color=NORD_ORANGE, linestyle="--", alpha=0.35)
        for values in analysis.dac_botplate_p_v[selected]:
            drive_ax.plot(scaled_time, values, color=NORD_BLUE, alpha=0.35)
        for values in analysis.dac_botplate_n_v[selected]:
            drive_ax.plot(scaled_time, values, color=NORD_ORANGE, alpha=0.35)

        for values in analysis.vdac_p_settling_error_v[selected]:
            settling_ax.plot(scaled_time, values * 1e3, color=NORD_BLUE, alpha=0.35)
        for values in analysis.vdac_n_settling_error_v[selected]:
            settling_ax.plot(scaled_time, values * 1e3, color=NORD_ORANGE, alpha=0.35)
        settling_ax.axhline(0.0, color=NORD_DARK, linestyle="--")
        settling_ax.set_ylim(-25.0, 25.0)
        settling_ax.set_xlabel(f"Time ({time_unit})")
        comparator_ax.set_title(f"C{int(stage_index)} (cycle {int(cycle_index)})")
        for ax in (comparator_ax, drive_ax, settling_ax):
            ax.set_xlim(float(scaled_time[0]), float(scaled_time[-1]))
            style_grid(ax)

    digital_limit_v = float(msmt.param.vdd_d.dc)
    axes[0, 0].set_ylim(-0.08 * digital_limit_v, 1.08 * digital_limit_v)
    axes[1, 0].set_ylim(-0.08 * digital_limit_v, 1.08 * digital_limit_v)
    axes[0, 0].set_ylabel("Comparator (V)")
    axes[1, 0].set_ylabel("CDAC drive (V)")
    axes[2, 0].set_ylabel("VDAC residual (mV)")
    axes[0, 0].legend(
        handles=(
            Line2D((), (), color=NORD_DARK, label="clk_comp"),
            Line2D((), (), color=NORD_BLUE, label="out_p"),
            Line2D((), (), color=NORD_ORANGE, label="out_n"),
            *(
                Line2D((), (), color=color, linestyle=":", label=label)
                for name, color, label in (
                    ("comp_latch_p_v", NORD_BLUE, "internal_p"),
                    ("comp_latch_n_v", NORD_ORANGE, "internal_n"),
                )
                if getattr(analysis, name) is not None
            ),
        ),
        loc="lower left",
        ncols=2,
        fontsize=INFO_BOX_FONT_SIZE,
    )
    axes[1, 0].legend(
        handles=(
            Line2D((), (), color=NORD_DARK, label="clk_logic"),
            Line2D((), (), color=NORD_BLUE, linestyle="--", label="state_p"),
            Line2D((), (), color=NORD_ORANGE, linestyle="--", label="state_n"),
            Line2D((), (), color=NORD_BLUE, label="bot_p"),
            Line2D((), (), color=NORD_ORANGE, label="bot_n"),
        ),
        loc="lower left",
        ncols=1,
        fontsize=INFO_BOX_FONT_SIZE,
    )
    axes[2, 0].legend(
        handles=(
            Line2D((), (), color=NORD_BLUE, label="top_p"),
            Line2D((), (), color=NORD_ORANGE, label="top_n"),
        ),
        loc="lower left",
        ncols=1,
        fontsize=INFO_BOX_FONT_SIZE,
    )
    flavor = (
        msmt.param.pex_cell.removeprefix("adc_").replace("_", " ").replace("layer", "-layer").replace("radix", "radix-")
    )
    setup_lines = [
        f"PEX: {flavor}",
        *style_measurement_text(msmt),
        f"Conversions: {len(np.unique(analysis.conversion_index))}",
    ]
    style_info_box(axes[0, -1], setup_lines, location="upper right")
    fig.suptitle("CDAC settling through representative SAR decisions")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_power_waveform(
    analysis: AnalysisAdcPowerWaveform,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot instantaneous rail power and sequencer timing for one conversion."""

    scale, unit = style_time_units(np.asarray((0.0, analysis.active_duration_s)))
    scaled_time = analysis.time_s * scale
    linear_threshold_uw = 1.0
    power_limit_uw = 2.0e3
    first_tick_decade = int(np.ceil(np.log10(linear_threshold_uw)))
    last_tick_decade = int(np.floor(np.log10(power_limit_uw)))
    positive_ticks = 10.0 ** np.arange(first_tick_decade, last_tick_decade + 1)
    power_ticks = np.concatenate((-positive_ticks[::-1], np.asarray([0.0]), positive_ticks))
    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.0, 1.0, 0.8)},
    )
    rail_labels = ("Analog", "Digital", "DAC")
    rail_power_w = (analysis.analog_power_w, analysis.digital_power_w, analysis.dac_power_w)
    for index, (ax, label, power_w, color) in enumerate(
        zip(axes[:3], rail_labels, rail_power_w, CURVE_COLORS[:3], strict=True)
    ):
        instantaneous_power_uw = power_w * 1e6
        static_power_uw = analysis.static_power_w[index] * 1e6
        active_power_uw = analysis.active_power_w[index] * 1e6
        ax.plot(scaled_time, instantaneous_power_uw, color=color, label="Instantaneous")
        ax.axhline(static_power_uw, color=NORD_DARK, linestyle=":", label="Static average")
        ax.axhline(active_power_uw, color=NORD_ORANGE, linestyle="--", label="Active average")
        ax.set_ylabel(f"{label} (µW)")
        ax.set_yscale("symlog", linthresh=linear_threshold_uw, linscale=0.6)
        ax.set_yticks(power_ticks)
        ax.set_ylim(-power_limit_uw, power_limit_uw)
        style_grid(ax)
        ax.legend(ncols=3)

    timing_ax = axes[3]
    timing_labels = ("INIT", "SAMP", "COMP", "LOGIC")
    timing_states = (analysis.init_high, analysis.samp_high, analysis.comp_high, analysis.logic_high)
    for row, (label, high, color) in enumerate(zip(timing_labels, timing_states, CURVE_COLORS, strict=False)):
        timing_ax.step(scaled_time, row + 0.72 * high, where="post", color=color)
    timing_ax.set_yticks(np.arange(len(timing_labels)) + 0.36, labels=timing_labels)
    timing_ax.set_ylim(-0.15, len(timing_labels) - 0.05)
    timing_ax.set_ylabel("Sequencer")
    timing_ax.set_xlabel(f"Time ({unit})")
    display_duration_scaled = analysis.active_duration_s * scale
    display_margin_scaled = 0.02 * display_duration_scaled
    timing_ax.set_xlim(-display_margin_scaled, display_duration_scaled + display_margin_scaled)
    timing_ax.set_xticks(np.linspace(0.0, display_duration_scaled, 6))
    style_grid(timing_ax)

    setup_lines = [f"Source: {analysis.backend.upper()}"]
    if analysis.adc_index >= 0:
        setup_lines.append(f"ADC: {analysis.adc_index:02d}")
    setup_lines.append(f"Rate: {analysis.active_conversion_rate_hz / 1e6:g} Msps")
    style_info_box(axes[0], setup_lines)
    fig.suptitle("ADC instantaneous supply power")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_decision_paths(
    msmt: MeasAdc,
    analysis: AnalysisAdcDecisionPaths,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot running SAR estimates for selected conversions."""

    fig, ax = plt.subplots()
    cycles = np.arange(analysis.estimate_dout.shape[1])
    for row, estimate in enumerate(analysis.estimate_dout):
        ax.plot(
            cycles,
            estimate,
            color=CURVE_COLORS[0],
            label="Running estimate" if row == 0 else None,
        )
        ax.axhline(
            analysis.final_dout[row],
            color=CURVE_COLORS[1],
            label="Final output" if row == 0 else None,
        )
    ax.set_xlabel("Decision cycle")
    ax.set_xticks(cycles)
    ax.set_xticklabels(("Init", *(f"B{decision}" for decision in range(len(cycles) - 1))))
    ax.set_ylabel("Running estimate (LSB)")
    ax.set_title("ADC decision paths")
    ax.legend()
    style_info_box(ax, style_measurement_text(msmt), location="lower right")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_adc_decision_path_density(
    msmt: MeasAdc,
    analysis: AnalysisAdcDecisionPaths,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot how frequently conversions follow each running SAR trajectory."""

    params = msmt.param.tb if isinstance(msmt, MeasAdcExt) else msmt.param
    paths = analysis.estimate_dout
    cycles = np.arange(analysis.estimate_dout.shape[1], dtype=np.float64)
    substeps_per_decision = 8
    cycle_step = 1.0 / substeps_per_decision
    horizontal_bins = len(cycles) * substeps_per_decision
    cycle_edges = np.arange(horizontal_bins + 1, dtype=np.float64) * cycle_step
    fine_cycles = cycle_edges[:-1] + cycle_step / 2.0
    normalized_code_max = (1 << params.dut.adc_bits) - 1
    code_edges = np.arange(-0.5, normalized_code_max + 1.5, 1.0)
    count = np.zeros((len(cycle_edges) - 1, len(code_edges) - 1), dtype=np.float64)
    for first_row in range(0, len(paths), 10_000):
        path_chunk = paths[first_row : first_row + 10_000]
        # A SAR estimate is a discrete state, not a continuously changing
        # voltage. Hold each estimate through its decision interval and jump
        # to the next value exactly at the following integer cycle.
        held = np.repeat(path_chunk, substeps_per_decision, axis=1)
        path_count, _, _ = np.histogram2d(
            np.broadcast_to(fine_cycles, held.shape).ravel(),
            held.ravel(),
            bins=(cycle_edges, code_edges),
        )
        count += path_count

    # Reserve a narrow gutter between decision-state boxes. Each transition is
    # one filled vertical track spanning the gutter and the outside edges of
    # its source and destination code cells.
    transition_gutter_width = 0.10
    transition_half_gutter = transition_gutter_width / 2.0
    transition_tracks = []
    transition_occupancies = []
    for cycle in range(1, len(cycles)):
        transitions, occupancies = np.unique(
            paths[:, (cycle - 1, cycle)],
            axis=0,
            return_counts=True,
        )
        changed = transitions[:, 0] != transitions[:, 1]
        transitions = transitions[changed]
        occupancies = occupancies[changed]
        order = np.argsort(occupancies)
        for transition, occupancy in zip(transitions[order], occupancies[order], strict=True):
            source_code, destination_code = (float(code) for code in transition)
            source_box_code = np.floor(source_code + 0.5)
            destination_box_code = np.floor(destination_code + 0.5)
            lower_edge = min(source_box_code, destination_box_code) - 0.5
            upper_edge = max(source_box_code, destination_box_code) + 0.5
            transition_tracks.append(
                (
                    (float(cycle) - transition_half_gutter, lower_edge),
                    (float(cycle) + transition_half_gutter, lower_edge),
                    (float(cycle) + transition_half_gutter, upper_edge),
                    (float(cycle) - transition_half_gutter, upper_edge),
                )
            )
            transition_occupancies.append(float(occupancy))

    state_count = count[::substeps_per_decision]
    box_vertices = []
    box_occupancies = []
    for cycle, cycle_count in enumerate(state_count):
        for code in np.flatnonzero(cycle_count):
            box_vertices.append(
                (
                    (float(cycle) + transition_half_gutter, float(code) - 0.5),
                    (float(cycle + 1) - transition_half_gutter, float(code) - 0.5),
                    (float(cycle + 1) - transition_half_gutter, float(code) + 0.5),
                    (float(cycle) + transition_half_gutter, float(code) + 0.5),
                )
            )
            box_occupancies.append(float(cycle_count[code]))

    density_norm = LogNorm(vmin=1, vmax=max(2, len(paths)))

    final_mean_code = int(np.rint(np.mean(analysis.final_dout)))
    populated_min = int(np.floor(np.min(analysis.estimate_dout) + 0.5))
    populated_max = int(np.floor(np.max(analysis.estimate_dout) + 0.5))
    y_limits = (
        (
            max(-0.5, populated_min - 8.5),
            min(normalized_code_max + 0.5, populated_max + 8.5),
        ),
        (
            max(-0.5, final_mean_code - 25.5),
            min(normalized_code_max + 0.5, final_mean_code + 25.5),
        ),
    )

    fig, all_axes = plt.subplots(
        1,
        4,
        layout="constrained",
        gridspec_kw={"width_ratios": (1.0, 1.0, 0.30, 0.035), "wspace": 0.03},
    )
    axes = all_axes[:2]
    for panel_index, (ax, panel_title, y_limit) in enumerate(
        zip(
            axes,
            ("Full trajectory", "Final trajectory"),
            y_limits,
            strict=True,
        )
    ):
        boxes = PolyCollection(
            box_vertices,
            array=np.asarray(box_occupancies),
            cmap=DENSITY_COLOR_MAP,
            norm=density_norm,
            edgecolors="none",
            antialiaseds=False,
            rasterized=True,
            zorder=2,
        )
        ax.add_collection(boxes)
        if transition_tracks:
            connectors = PolyCollection(
                transition_tracks,
                array=np.asarray(transition_occupancies),
                cmap=DENSITY_COLOR_MAP,
                norm=density_norm,
                edgecolors="none",
                antialiaseds=False,
                rasterized=True,
                zorder=2,
            )
            ax.add_collection(connectors)
        ax.set_xlim(0.0, float(len(cycles) + 1))
        ax.set_ylim(*y_limit)
        labeled_cycles = cycles[::2].copy()
        if labeled_cycles[-1] != cycles[-1]:
            labeled_cycles[-1] = cycles[-1]
        ax.set_xticks(labeled_cycles)
        ax.set_xticklabels(("Init", *(f"B{int(cycle - 1)}" for cycle in labeled_cycles[1:])))
        ax.set_xticks(cycles, minor=True)
        ax.set_xlabel("Decision cycle")
        if panel_index == 0:
            ax.set_ylabel("Successive approximation code (LSB)")
        ax.set_title(panel_title)
        ax.set_facecolor(NORD_LIGHT_BLUE)
        if panel_index:
            ax.yaxis.set_minor_locator(MultipleLocator(1.0))

    histogram_ax = all_axes[2]
    sample_count = len(paths)
    final_count = state_count[-1]
    populated_final_codes = np.flatnonzero(final_count)
    populated_final_count = final_count[populated_final_codes]
    histogram_ax.barh(
        populated_final_codes,
        populated_final_count / sample_count,
        height=1.0,
        color=DENSITY_COLOR_MAP(density_norm(populated_final_count)),
        edgecolor="none",
        rasterized=True,
        zorder=2,
    )
    final_mean = float(np.mean(paths[:, -1]))
    final_std = float(np.std(paths[:, -1]))
    if final_std > 0.0:
        fit_code = np.linspace(*y_limits[1], 501)
        fit_fraction_per_lsb = np.exp(-0.5 * ((fit_code - final_mean) / final_std) ** 2) / (
            final_std * np.sqrt(2.0 * np.pi)
        )
        histogram_ax.plot(
            fit_fraction_per_lsb,
            fit_code,
            color=TEXT_COLOR,
            zorder=3,
        )
    histogram_ax.set_ylim(*y_limits[1])
    histogram_ax.set_xlim(left=0.0)
    histogram_ax.set_title("Code density")
    histogram_ax.set_xlabel("Count / N")
    histogram_ax.set_yticks([])
    histogram_ax.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    histogram_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    histogram_ax.set_facecolor(NORD_LIGHT_BLUE)
    style_info_box(
        histogram_ax,
        (
            f"μ: {final_mean:.0f}",
            f"σ: {style_adc_code_dispersion_lsb(final_std, single_code=len(populated_final_codes) == 1)} LSB",
        ),
    )

    colorbar = fig.colorbar(ScalarMappable(norm=density_norm, cmap=DENSITY_COLOR_MAP), cax=all_axes[3])
    colorbar.set_label("Conversions per path")
    style_info_box(axes[0], (*style_measurement_text(msmt), f"N: {sample_count}"))
    fig.suptitle("ADC decision-path density")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_comp_offset_noise(
    msmt_list: Sequence[MeasCompExt | MeasCompInt],
    analysis: AnalysisCompOffsetNoise,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot comparator decision probability versus differential input."""

    fig, ax = plt.subplots()
    ax.plot(
        analysis.vin_diff_v * 1e3,
        analysis.decision_probability,
        marker="o",
        color=CURVE_COLORS[0],
        label="Measured decisions",
    )
    ax.axhline(0.5, color=SPINE_COLOR)
    if np.isfinite(analysis.offset_v):
        ax.axvline(
            analysis.offset_v * 1e3,
            color=CURVE_COLORS[1],
            linestyle="--",
            label="50% threshold",
        )
    ax.set_xlabel("Differential input (mV)")
    ax.set_ylabel("Decision probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Comparator offset and input-referred noise")
    style_grid(ax)
    ax.legend()
    style_info_box(ax, style_measurement_group_text(msmt_list), location="lower right")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_comp_sampling_campaign(
    msmt_list2d: Sequence[Sequence[MeasCompExt | MeasCompInt]],
    analysis_list: Sequence[AnalysisCompOffsetNoise],
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot one ADC's matched track/ hold curves over VDAC coupling."""

    input_error_minimum_mv = 0.0
    input_error_maximum_mv = 25.0
    grouped_results = {
        (float(group[0].param.requested_dac_rail_percent), group[0].param.sampling_mode): (
            group,
            analysis,
        )
        for group, analysis in zip(msmt_list2d, analysis_list, strict=True)
    }
    coupling_percentages = (0.0, 25.0, 50.0, 75.0, 100.0)
    fig, (curve_ax, violin_ax) = plt.subplots(1, 2)
    coupling_colors = tuple(
        SPECTRUM_COLOR_MAP(index / (len(coupling_percentages) - 1)) for index in range(len(coupling_percentages))
    )
    mode_offsets = {"track": -2.6, "hold": 2.6}
    mode_linestyles = {"track": "-", "hold": "--"}
    mode_markers = {"track": "o", "hold": "s"}
    mode_hatches = {"track": None, "hold": "///"}
    for coupling_index, coupling_percent_p in enumerate(coupling_percentages):
        coupling_percent_n = 100.0 - coupling_percent_p
        for mode in ("track", "hold"):
            _group, analysis = grouped_results[(coupling_percent_p, mode)]
            threshold_mv = analysis.offset_v * 1e3
            noise_mv = analysis.noise_sigma_v * 1e3
            color = coupling_colors[coupling_index]
            curve_label = f"P/N = {coupling_percent_p:g}/{coupling_percent_n:g}%" if mode == "track" else None
            curve_ax.scatter(
                analysis.vin_diff_v * 1e3,
                analysis.decision_probability,
                color=color,
                marker=mode_markers[mode],
                edgecolors="none",
                zorder=2,
            )
            fit_input_v = np.linspace(
                float(np.min(analysis.vin_diff_v)),
                float(np.max(analysis.vin_diff_v)),
                1001,
            )
            fit_probability = ndtr(
                analysis.decision_polarity * (fit_input_v - analysis.offset_v) / analysis.noise_sigma_v
            )
            curve_ax.plot(
                fit_input_v * 1e3,
                fit_probability,
                color=color,
                linestyle=mode_linestyles[mode],
                label=curve_label,
                zorder=3,
            )

            distribution_mv = np.linspace(
                threshold_mv - 4.0 * noise_mv,
                threshold_mv + 4.0 * noise_mv,
                401,
            )
            density = np.exp(-0.5 * ((distribution_mv - threshold_mv) / noise_mv) ** 2)
            violin_center = coupling_percent_p + mode_offsets[mode]
            violin_half_width = 2.15 * density
            violin_ax.fill_betweenx(
                distribution_mv,
                violin_center - violin_half_width,
                violin_center + violin_half_width,
                color=color,
                linewidth=0.8,
                edgecolor=color,
                hatch=mode_hatches[mode],
            )
            violin_ax.plot(
                violin_center,
                threshold_mv,
                marker=mode_markers[mode],
                color=color,
            )

    curve_ax.axhline(0.5, color=SPINE_COLOR)
    curve_ax.set_xlim(input_error_minimum_mv, input_error_maximum_mv)
    curve_ax.set_xlabel("Differential input (mV)")
    curve_ax.set_ylabel("Decision probability")
    curve_ax.set_ylim(-0.02, 1.02)
    curve_ax.set_title("Comparator S-curves (CDF)")
    curve_ax.legend(title="VDAC coupling")

    violin_ax.set_xlim(-8.0, 108.0)
    violin_ax.set_ylim(input_error_minimum_mv, input_error_maximum_mv)
    violin_ax.set_xticks(
        coupling_percentages,
        [f"{value:g}/{100.0 - value:g}" for value in coupling_percentages],
    )
    violin_ax.set_xlabel("VDAC coupling (P/N % of VDD_DAC)")
    violin_ax.set_ylabel("Input error (mV)")
    violin_ax.set_title("Gaussian fit of μ (threshold) and σ (noise)")
    violin_ax.legend(
        handles=(
            Patch(facecolor=LEGEND_FACE_COLOR, edgecolor=SPINE_COLOR, label="Track"),
            Patch(facecolor=LEGEND_FACE_COLOR, edgecolor=SPINE_COLOR, hatch="///", label="Hold"),
        ),
    )

    for ax in (curve_ax, violin_ax):
        style_grid(ax)
    style_info_box(
        curve_ax,
        style_measurement_group_text(tuple(msmt for group in msmt_list2d for msmt in group)),
    )
    fig.suptitle("Comparator threshold and input-referred noise versus VDAC coupling")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_comp_common_mode_campaign(
    msmt_list2d: Sequence[Sequence[MeasCompExt | MeasCompInt]],
    analysis_list: Sequence[AnalysisCompOffsetNoise],
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot one ADC's comparator response over common-mode input."""

    common_mode_minimum_v = 0.7
    common_mode_maximum_v = 1.2
    input_error_minimum_mv = 0.0
    input_error_maximum_mv = 25.0
    selected_results = [
        (group, analysis)
        for group, analysis in zip(msmt_list2d, analysis_list, strict=True)
        if common_mode_minimum_v <= float(group[0].param.tb.vin_cm.dc) <= common_mode_maximum_v
    ]
    selected_results.sort(key=lambda result: float(result[0][0].param.tb.vin_cm.dc))
    fig, (curve_ax, violin_ax) = plt.subplots(1, 2)
    common_modes_v = []
    for group, analysis in selected_results:
        common_mode_v = float(group[0].param.tb.vin_cm.dc)
        threshold_mv = analysis.offset_v * 1e3
        noise_mv = analysis.noise_sigma_v * 1e3
        gradient_position = (common_mode_v - common_mode_minimum_v) / (common_mode_maximum_v - common_mode_minimum_v)
        color = SPECTRUM_COLOR_MAP(float(np.clip(gradient_position, 0.0, 1.0)))
        common_modes_v.append(common_mode_v)

        curve_ax.scatter(
            analysis.vin_diff_v * 1e3,
            analysis.decision_probability,
            color=color,
            edgecolors="none",
            zorder=2,
        )

        valid_fit = (
            analysis.validity == "valid" and np.isfinite(threshold_mv) and np.isfinite(noise_mv) and noise_mv > 0.0
        )
        if valid_fit:
            fit_input_v = np.linspace(
                float(np.min(analysis.vin_diff_v)),
                float(np.max(analysis.vin_diff_v)),
                1001,
            )
            fit_probability = ndtr(
                analysis.decision_polarity * (fit_input_v - analysis.offset_v) / analysis.noise_sigma_v
            )
            curve_ax.plot(
                fit_input_v * 1e3,
                fit_probability,
                color=color,
                label=f"Vin_cm = {common_mode_v:.3g} V",
                zorder=3,
            )
            distribution_mv = np.linspace(
                threshold_mv - 4.0 * noise_mv,
                threshold_mv + 4.0 * noise_mv,
                401,
            )
            density = np.exp(-0.5 * ((distribution_mv - threshold_mv) / noise_mv) ** 2)
            violin_half_width_v = 0.035 * density
            violin_ax.fill_betweenx(
                distribution_mv,
                common_mode_v - violin_half_width_v,
                common_mode_v + violin_half_width_v,
                color=color,
                linewidth=0.8,
                edgecolor=color,
            )
            violin_ax.plot(common_mode_v, threshold_mv, marker="o", color=color)
        else:
            curve_ax.plot(
                analysis.vin_diff_v * 1e3,
                analysis.decision_probability,
                color=color,
                label=f"Vin_cm = {common_mode_v:.3g} V (fit invalid)",
                zorder=3,
            )

    curve_ax.axhline(0.5, color=SPINE_COLOR)
    curve_ax.set_xlim(input_error_minimum_mv, input_error_maximum_mv)
    curve_ax.set_ylim(-0.02, 1.02)
    curve_ax.set_xlabel("Differential input (mV)")
    curve_ax.set_ylabel("Decision probability")
    curve_ax.set_title("Comparator S-curve (CDF)")
    curve_ax.legend(loc="lower right")

    violin_ax.set_xlim(
        min(common_modes_v) - 0.05,
        max(common_modes_v) + 0.05,
    )
    violin_ax.set_ylim(input_error_minimum_mv, input_error_maximum_mv)
    violin_ax.set_xticks(common_modes_v, [f"{value:.1f}" for value in common_modes_v])
    violin_ax.set_xlabel("Common-mode input (V)")
    violin_ax.set_ylabel("Input error (mV)")
    violin_ax.set_title("Gaussian fit of μ (threshold) and σ (noise)")

    for ax in (curve_ax, violin_ax):
        style_grid(ax)
    style_info_box(
        curve_ax,
        style_measurement_group_text(tuple(msmt for group in msmt_list2d for msmt in group)),
    )
    fig.suptitle("Comparator threshold and input-referred noise versus common mode")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_cdac_cap_mismatch(
    msmt_list: Sequence[MeasCdacExt],
    analysis: AnalysisCdacCapMismatch,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot one ADC's normalized A-to-B main/diff weights and diagnostics."""

    stages = np.arange(analysis.effective_fraction.shape[1])
    expected_effective = analysis.expected_effective_fraction
    fig, axes_grid = plt.subplots(2, 3, sharex=True)
    axes = axes_grid.ravel()
    for side, label, color in ((0, "P", CURVE_COLORS[0]), (1, "N", CURVE_COLORS[1])):
        axes[0].plot(
            stages,
            analysis.effective_fraction[side],
            "o-",
            color=color,
            label=label,
        )
        axes[1].plot(
            stages,
            analysis.effective_fraction[side] - expected_effective,
            "o-",
            color=color,
            label=label,
        )
        axes[3].plot(stages, analysis.main_fraction[side], "o-", color=color, label=f"{label} main")
        axes[3].plot(
            stages,
            analysis.diff_fraction[side],
            "s--",
            color=color,
            label=f"{label} diff",
        )
        axes[4].plot(
            stages,
            analysis.direction_bias[side, :, 0],
            marker="o",
            linestyle="-" if side == 0 else "--",
            color=color,
            label=f"{label}, main+diff",
        )
        axes[4].plot(
            stages,
            analysis.direction_bias[side, :, 1],
            marker="s",
            linestyle="-" if side == 0 else "--",
            color=color,
            label=f"{label}, main−diff",
        )
        axes[5].plot(
            stages,
            2.0 * analysis.diff_fraction[side],
            "o-",
            color=color,
            label=label,
        )
    axes[0].plot(stages, expected_effective, "k--", label="Ideal/PEX")
    axes[2].plot(
        stages,
        analysis.effective_fraction[0] - analysis.effective_fraction[1],
        "o-",
        color=NORD_PURPLE,
        label="P−N",
    )
    panel_titles = (
        "Effective fraction",
        "Residual from ideal/PEX",
        "P−N effective asymmetry",
        "Main and differential fractions",
        "Switching-direction bias",
        "Differential-capacitor separation",
    )
    ylabels = ("C/Ctotal", "Residual", "P−N", "C/Ctotal", "Half-difference", "Separation")
    tick_positions = (0, 3, 6, 9, 12, 15)
    tick_labels = ("C0", "C3", "C6", "C9", "C12", "C15")
    for ax, title, ylabel in zip(axes, panel_titles, ylabels, strict=True):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(tick_positions, tick_labels)
        style_grid(ax)
        ax.legend()
    style_info_box(axes[2], style_measurement_group_text(msmt_list), location="lower right")
    fig.supxlabel("Conversion stage")
    fig.suptitle("A-to-B CDAC capacitance")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_cdac_cap_mismatch_comparison(
    msmt_list2d: Sequence[Sequence[MeasCdacExt]],
    analysis_list: Sequence[AnalysisCdacCapMismatch],
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Compare normalized A-to-B CDAC extraction across ADC00–ADC03."""

    fig, axes_grid = plt.subplots(2, 2, sharex=True)
    axes = axes_grid.ravel()
    aligned = sorted(zip(msmt_list2d, analysis_list, strict=True), key=lambda item: item[1].adc_index)
    stages = np.arange(analysis_list[0].effective_fraction.shape[1])
    for group, analysis in aligned:
        expected = analysis.expected_effective_fraction
        effective_mean = (analysis.effective_fraction[0] + analysis.effective_fraction[1]) / 2.0
        diffcap_separation_mean = analysis.diff_fraction[0] + analysis.diff_fraction[1]
        label = f"ADC{analysis.adc_index:02d}"
        color = CURVE_COLORS[analysis.adc_index]
        axes[0].plot(stages, effective_mean, "o-", color=color, label=label)
        axes[1].plot(stages, effective_mean - expected, "o-", color=color, label=label)
        axes[2].plot(
            stages,
            analysis.effective_fraction[0] - analysis.effective_fraction[1],
            "o-",
            color=color,
            label=label,
        )
        axes[3].plot(stages, diffcap_separation_mean, "o-", color=color, label=label)

    panel_titles = (
        "Mean P/N effective fraction",
        "Residual from ideal/PEX",
        "P−N effective asymmetry",
        "Mean P/N diffcap separation",
    )
    ylabels = ("Fraction", "Residual", "Asymmetry", "Separation")
    tick_positions = (0, 3, 6, 9, 12, 15)
    tick_labels = ("C0", "C3", "C6", "C9", "C12", "C15")
    for ax, title, ylabel in zip(axes, panel_titles, ylabels, strict=True):
        ax.axhline(0.0, color=SPINE_COLOR)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(tick_positions, tick_labels)
        style_grid(ax)
    axes[0].legend()
    style_info_box(
        axes[3],
        style_measurement_group_text(tuple(msmt for group in msmt_list2d for msmt in group)),
        location="lower right",
    )
    fig.supxlabel("Conversion stage")
    fig.suptitle("A-to-B CDAC comparison")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_comp_timing(
    msmt_list: Sequence[MeasCompInt],
    analysis: AnalysisCompTiming,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot comparator delay, settling time, and unresolved outcomes."""

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(analysis.trial_index, analysis.clock_to_decision_s * 1e9, "o", label="Clock to decision")
    axes[0].plot(analysis.trial_index, analysis.settling_s * 1e9, "o", label="Settling")
    axes[0].set_ylabel("Time (ns)")
    axes[0].legend()
    axes[1].step(analysis.trial_index, analysis.unresolved, where="mid")
    axes[1].set_ylabel("Unresolved")
    axes[1].set_xlabel("Trial index")
    axes[1].set_yticks((0, 1))
    for ax in axes:
        style_grid(ax)
    style_info_box(axes[1], style_measurement_group_text(msmt_list), location="lower right")
    fig.suptitle("Comparator timing")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_comp_power(
    msmt_list: Sequence[MeasCompInt],
    analysis: AnalysisCompPower,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot comparator average power per measurement."""

    fig, ax = plt.subplots()
    labels = [str(index) for index in analysis.source_index]
    ax.bar(labels, analysis.average_power_w * 1e6)
    ax.set_ylabel("Average power (µW)")
    ax.set_xlabel("Measurement index")
    style_grid(ax)
    style_info_box(ax, style_measurement_group_text(msmt_list))
    fig.suptitle("Comparator power")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_comp_candidate_sweep(
    msmt_list: Sequence[MeasCompInt],
    analysis: AnalysisCompCandidateSweep,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot candidate noise, power, and settling on one area-ordered axis."""

    candidate_count = len(analysis.candidate_id)
    position = np.arange(candidate_count)
    colors = {
        "half": CURVE_COLORS[0],
        "double": CURVE_COLORS[1],
        "fabricated": CURVE_COLORS[2],
    }
    fig, axes = plt.subplots(3, 1, sharex=True)
    metrics = (
        (analysis.noise_sigma_v * 1e3, "Noise σ (mV)", "linear"),
        (analysis.average_power_w * 1e6, "Power (µW)", "log"),
        (analysis.maximum_settling_s * 1e9, "Settling (ns)", "linear"),
    )
    for ax, (values, ylabel, scale) in zip(axes, metrics, strict=True):
        for profile in ("half", "double", "fabricated"):
            selected = np.asarray(analysis.size_profile) == profile
            if not np.any(selected):
                continue
            ax.scatter(
                position[selected],
                values[selected],
                marker="o" if profile != "fabricated" else "*",
                color=colors[profile],
                label={"half": "0.5× FRIDA widths", "double": "2× FRIDA widths", "fabricated": "FRIDA baseline"}[
                    profile
                ],
                zorder=4 if profile == "fabricated" else 2,
            )
        ax.set_ylabel(ylabel)
        ax.set_yscale(scale)
        style_grid(ax)

    baseline = np.flatnonzero(np.asarray(analysis.size_profile) == "fabricated")
    for ax in axes:
        ax.axvline(baseline[0], color=CURVE_COLORS[2], linestyle="--")
    axes[0].legend(ncols=3)

    tick_count = min(12, candidate_count)
    tick_positions = np.unique(np.rint(np.linspace(0, candidate_count - 1, tick_count)).astype(int))
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tuple(f"{index}\n{analysis.total_active_area_um2[index]:.2f}" for index in tick_positions))
    axes[-1].set_xlabel("Area-ordered candidate index\n(total instantiated MOS Σ(W×L) in µm²)")
    axes[-1].set_xlim(-2, candidate_count + 1)
    style_info_box(axes[2], style_measurement_group_text(msmt_list), location="lower right")
    fig.suptitle("Comparator candidate noise, power, and settling")
    return save_figure(fig, output_path)


@mpl.rc_context(PLOT_STYLE)
def plot_comp_noise_power_tradeoff(
    analysis: AnalysisCompCandidateSweep,
    *,
    output_path: Path,
) -> tuple[Path, ...]:
    """Plot valid, resolved candidate noise against power, colored by settling."""

    noise_mv = analysis.noise_sigma_v * 1e3
    power_uw = analysis.average_power_w * 1e6
    settling_ns = analysis.maximum_settling_s * 1e9
    valid_scurve = np.asarray(analysis.validity) == "valid"
    finite_positive = (
        np.isfinite(noise_mv)
        & np.isfinite(power_uw)
        & np.isfinite(settling_ns)
        & (noise_mv > 0.0)
        & (power_uw > 0.0)
        & (settling_ns > 0.0)
    )
    resolved = analysis.unresolved_fraction == 0.0
    selected = valid_scurve & finite_positive & resolved
    selected_settling_ns = settling_ns[selected]
    color_min = float(np.min(selected_settling_ns))
    color_max = float(np.max(selected_settling_ns))
    if np.isclose(color_min, color_max):
        color_max = color_min + 1.0
    color_norm = Normalize(vmin=color_min, vmax=color_max)
    profiles = np.asarray(analysis.size_profile)

    fig, ax = plt.subplots()
    mappable = ScalarMappable(norm=color_norm, cmap=SPECTRUM_COLOR_MAP)
    for profile, marker, label in (
        ("half", "o", "0.5× FRIDA widths"),
        ("double", "s", "2× FRIDA widths"),
    ):
        profile_selected = selected & (profiles == profile)
        if not np.any(profile_selected):
            continue
        ax.scatter(
            noise_mv[profile_selected],
            power_uw[profile_selected],
            c=settling_ns[profile_selected],
            cmap=SPECTRUM_COLOR_MAP,
            norm=color_norm,
            marker=marker,
            edgecolors=SPINE_COLOR,
            linewidths=0.35,
            label=label,
            zorder=2,
        )
    baseline = np.flatnonzero(profiles == "fabricated")
    baseline_index = int(baseline[0])
    ax.scatter(
        noise_mv[baseline_index],
        power_uw[baseline_index],
        marker="*",
        color=CURVE_COLORS[2],
        edgecolors=TEXT_COLOR,
        linewidths=0.7,
        label="FRIDA65A fabricated baseline",
        zorder=5,
    )
    ax.annotate(
        "FRIDA65A",
        (noise_mv[baseline_index], power_uw[baseline_index]),
        xytext=(8, 7),
        textcoords="offset points",
    )

    colorbar = fig.colorbar(mappable, ax=ax, pad=0.02)
    colorbar.set_label("Worst settling time (ns)")

    ax.set_yscale("log")
    ax.set_xlabel("Input-referred noise σ (mV)")
    ax.set_ylabel("Average power (µW)")
    style_grid(ax)
    ax.legend()
    fig.suptitle("Comparator noise–power trade-off")
    return save_figure(fig, output_path)
