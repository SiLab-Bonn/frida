"""Software-only checks for the native HDL21 comparator interface."""

import inspect

import hdl21 as h
import pytest

from . import sim


def test_default_comparator_sweep_is_complete() -> None:
    params = sim.CompTbParams()

    values = tuple(float(value) for value in params.vin_diff_values_v)
    assert values[0] == pytest.approx(-3e-3)
    assert values[-1] == pytest.approx(3e-3)
    assert len(values) == 61
    assert params.conversions == 100
    assert not hasattr(params, "temperature_c")


def test_comparator_testbench_consumes_explicit_sweep() -> None:
    params = sim.CompTbParams(vin_diff_values_v=(-1e-3, 0.0, 1e-3), conversions=2)
    tb = sim.CompTb(params)

    assert isinstance(tb, h.Module)
    assert tb.dut.of.name.startswith("Comp")
    assert len(tb.vvin_diff.of.params.wave.points) == 6


def test_comparator_testbench_rejects_repeated_values() -> None:
    with pytest.raises(ValueError, match="unique"):
        sim.CompTb(sim.CompTbParams(vin_diff_values_v=(0.0, 0.0)))


def test_main_owns_only_experiment_targets() -> None:
    source = inspect.getsource(sim.main)
    assert "hdl21_comp_perf_vs_size" in source
    assert "frida1_fixed_input_noise" in source
    assert "_check" not in source
    assert "TARGETS" not in vars(sim)
