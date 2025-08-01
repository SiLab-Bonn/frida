import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import behavioral
import spice
import matplotlib.pyplot as plt
import math
import pandas as pd
import time

# eventually, need to pull the init code out, so that

params = {
    "ADC": {
        "resolution": 8,  # resolution of the ADC
        "sampling_frequency": 10.0e6,  # sampling rate in Hz
        "aperture_jitter": 0.0e-12,  # aperture jitter in seconds (TBD)
        "use_calibration": False,  # account for cap error when calculating re-analog results
    },
    "COMP": {
        "offset_voltage": 0.0e-3,  # offset voltage in Volts
        "common_mode_dependent_offset_gain": 0.0,  # common mode voltage gain
        "threshold_voltage_noise": 0.0e-3,  # RMS noise voltage in Volts
    },
    "CDAC": {
        "positive_reference_voltage": 1.2,  # reference voltage in Volts
        "negative_reference_voltage": 0.0,  # reference voltage in Volts
        "reference_voltage_noise": 0.0e-3,  # reference voltage noise in Volts
        "unit_capacitance": 0.8167e-15,  # unit capacitance in Farads (~0.8167 allows for Ctot per branch to ~100fF)
        "array_size": 8,    # NOTE: this param is N but get recomputed to M if radix != 2, and array_N_M_expansion = True
        "array_N_M_expansion": False,
        "use_individual_weights": False,  # use array values to build cap array
        "individual_weights": [],   # This can't be
        "parasitic_capacitance": 5.00e-14,  # in Farads at the output of the CDAC
        "radix": 1.80,  # for the cap values (use_individual_weights = False)
        "capacitor_mismatch_error": 0.0,  # mismatch error in percent of the unit cap
        "settling_time": 0.0e-9,  # TBD: individual settling errors per capacitor?
        "switching_strat": 'monotonic',     #used to determined initial starting voltages
    },
    "TESTBENCH": {
        'simulation_times':        [0,   6000e-6],  # starting and ending sim times, matching with bottom voltages to make pwl
        "positive_input_voltages": [0.2, 1.2],      # starting and ending voltages of the pwl voltage waveform
        "negative_input_voltages": [1.2, 0.2],
        "spicedir": None,   # Use this to write netlist from template
        "rawdir": None,     # Use this to set set SPICE output dir, and to read for parsing.
    },
}

# sets the caps from top to bottom, as fractions of 100fF
# This doesn't work because it has to be integers
# params["CDAC"]["individual_weights"] = [(100e-15 / 2 / (params["CDAC"]["radix"] ** i) ) for i in range(params["CDAC"]["array_size"])]

adc = behavioral.SAR_ADC(params)

start_time = time.time()
result = adc.sample_and_convert(
        0.792,
        0.6104,
        # 0.8,
        # 0.6,
        do_plot=False,
        do_calculate_energy=False,
        do_normalize_result=False,
)
end_time = time.time()
print(f"Function took {end_time - start_time:.6f} seconds")