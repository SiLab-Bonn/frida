"""Run one shared ADC configuration through the behavioral model.

Run from /local/frida:

    uv run python -m flow.scans.scan_behavioral

The generated HDF5 file uses the same typed measurement schema as the
physical scan.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import hdl21 as h
import numpy as np

from flow.adc.behavioral import SAR_ADC
from flow.adc.sim import AdcTbParams
from flow.analysis.io import build_adc_interface_wave, write_measurement
from flow.analysis.types import AdcDaq, MeasAdcExt, MeasInfo
from flow.cdac import get_cdac_weights
from flow.scans.params import AdcScanParams
from flow.scans.scan_adc import (
    convert_dac_caps_to_adc_weights,
    convert_dout_to_normalized_dout,
)

ADC_INDEX = 0
PARAMS = AdcScanParams(
    tb=AdcTbParams(
        view="frida1",
        conversions=1,
        vin_cm=h.Vdc.Params(dc=0.600),
        vin_diff=h.Vdc.Params(dc=0.015),
    )
)
CAP_WEIGHTS = get_cdac_weights(PARAMS.tb.dut.cdac)
CODE_WEIGHTS = convert_dac_caps_to_adc_weights(CAP_WEIGHTS)
NUM_CAPTURE_BITS = len(CODE_WEIGHTS)
# TODO: Change this stable overwrite path to build/scan_behavioral/<short-datetime>.
SCAN_OUTDIR = Path(__file__).resolve().parents[2] / "build" / "behavioral_scan"

ADC_CLOCK_HZ = float(PARAMS.tb.symbol_rate) / len(PARAMS.tb.seq_init_pattern)
UNIT_CAPACITANCE = 1e-15
PARASITIC_RATIO = 1.0
APPLY_INPUT_ATTENUATION = False

COMPARATOR_NOISE = 0.0
COMPARATOR_OFFSET = 0.0
REFERENCE_NOISE = 0.0
SETTLING_TIME = 0.0
SWITCHING_STRAT = "monotonic"


def build_frida_params() -> dict[str, dict[str, object]]:
    """Build behavioral-model parameters matching FRIDA's ADC."""

    cdac_capacitance = sum(CAP_WEIGHTS) * UNIT_CAPACITANCE
    parasitic_capacitance = PARASITIC_RATIO * cdac_capacitance
    return {
        "ADC": {
            "sampling_frequency": ADC_CLOCK_HZ,
            "use_calibration": False,
            "resolution": PARAMS.tb.dut.adc_bits,
        },
        "COMP": {
            "offset_voltage": COMPARATOR_OFFSET,
            "common_mode_dependent_offset_gain": 0.0,
            "threshold_voltage_noise": COMPARATOR_NOISE,
            "capacitor_mismatch_error": 0.0,
        },
        "CDAC": {
            "positive_reference_voltage": 1.2,
            "negative_reference_voltage": 0.0,
            "reference_voltage_noise": REFERENCE_NOISE,
            "unit_capacitance": UNIT_CAPACITANCE,
            "use_individual_weights": True,
            "individual_weights": CAP_WEIGHTS,
            "parasitic_capacitance": parasitic_capacitance,
            "capacitor_mismatch_error": 0.0,
            "settling_time": SETTLING_TIME,
            "switching_strat": SWITCHING_STRAT,
            "array_size": len(CAP_WEIGHTS),
        },
    }


def input_attenuation(params: dict[str, dict[str, object]]) -> float:
    """Return optional sampled input gain from top-plate parasitic loading."""

    cdac = params["CDAC"]
    cdac_capacitance = sum(CAP_WEIGHTS) * cast(float, cdac["unit_capacitance"])
    parasitic_capacitance = cast(float, cdac["parasitic_capacitance"])
    return cdac_capacitance / (cdac_capacitance + parasitic_capacitance)


def convert_behavioral_to_bout_and_dout(
    adc: SAR_ADC,
    vin_p: float,
    vin_n: float,
) -> tuple[str, int, int]:
    """Run one conversion and return Bout plus raw and normalized Dout."""

    adc.sample_and_convert(
        vin_p,
        vin_n,
        do_calculate_energy=False,
        do_plot=False,
        do_normalize_result=False,
    )
    bits = [int(bit) for bit in adc.comp_result]
    if len(bits) != NUM_CAPTURE_BITS:
        raise RuntimeError(f"behavioral model produced {len(bits)} bits, expected {NUM_CAPTURE_BITS}")
    bout = "".join(str(bit) for bit in bits)
    dout_raw = sum(weight * bit for weight, bit in zip(CODE_WEIGHTS, bits, strict=True))
    dout = convert_dout_to_normalized_dout(
        dout_raw,
        CODE_WEIGHTS,
        PARAMS.tb.dut.adc_bits,
    )
    return bout, dout_raw, dout


def main() -> None:
    params = build_frida_params()
    adc = SAR_ADC(params)
    attenuation = input_attenuation(params) if APPLY_INPUT_ATTENUATION else 1.0
    if not isinstance(PARAMS.tb.vin_diff, h.Vdc.Params):
        raise TypeError("the behavioral adapter currently requires a DC vin_diff")
    vin_diff_v = float(PARAMS.tb.vin_diff.dc)
    vin_cm_v = float(PARAMS.tb.vin_cm.dc)
    vin_p = vin_cm_v + vin_diff_v / 2.0
    vin_n = vin_cm_v - vin_diff_v / 2.0
    sampled_vin_p = vin_cm_v + attenuation * (vin_p - vin_cm_v)
    sampled_vin_n = vin_cm_v + attenuation * (vin_n - vin_cm_v)

    cdac_capacitance = sum(CAP_WEIGHTS) * UNIT_CAPACITANCE
    cpar = cast(float, params["CDAC"]["parasitic_capacitance"])
    print("Behavioral FRIDA ADC configuration")
    print(f"Cap weights C0..C15: {CAP_WEIGHTS}")
    print(f"Decision weights B0..B16: {CODE_WEIGHTS}")
    print(f"Cdac={cdac_capacitance / 1e-15:.3f} fF, Cpar={cpar / 1e-15:.3f} fF")
    print(f"Sampled input attenuation={attenuation:.6g}")

    bout_values = np.empty((PARAMS.tb.conversions, NUM_CAPTURE_BITS), dtype=np.uint8)
    dout_raw_values = np.empty(PARAMS.tb.conversions, dtype=np.int64)
    dout_values = np.empty(PARAMS.tb.conversions, dtype=np.int64)
    for conversion_index in range(PARAMS.tb.conversions):
        bout, dout_raw, dout = convert_behavioral_to_bout_and_dout(
            adc,
            sampled_vin_p,
            sampled_vin_n,
        )
        bout_values[conversion_index] = np.fromiter(
            (int(bit) for bit in bout),
            dtype=np.uint8,
            count=NUM_CAPTURE_BITS,
        )
        dout_raw_values[conversion_index] = dout_raw
        dout_values[conversion_index] = dout
        print(f"conversion {conversion_index:02d}: Bout={bout} Dout_raw={dout_raw} Dout={dout}")

    measurement = MeasAdcExt(
        info=MeasInfo(
            schema_version=1,
            measurement_type="MeasAdcExt",
            backend="behavioral",
            timestamp_utc=datetime.now().astimezone(),
            instruments={"model": f"{SAR_ADC.__module__}.{SAR_ADC.__name__}"},
            readbacks={"input_attenuation": attenuation},
        ),
        param=PARAMS,
        daq=AdcDaq(
            conversion_index=np.arange(PARAMS.tb.conversions),
            bout=bout_values,
            dout_raw=dout_raw_values,
            dout=dout_values,
            vin_diff_v=np.full(PARAMS.tb.conversions, vin_diff_v),
        ),
        wave=build_adc_interface_wave(PARAMS.tb, bout_values[0]),
    )
    h5_path = SCAN_OUTDIR / f"adc_{ADC_INDEX:02d}.h5"
    write_measurement(h5_path, measurement)
    print(f"ADC {ADC_INDEX:02d}: saved typed measurement to {h5_path}")


if __name__ == "__main__":
    main()
