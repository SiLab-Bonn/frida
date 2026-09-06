"""Software-only checks for the native HDL21 sampler interface."""

import inspect

import hdl21 as h
import pytest

from . import sim


def test_sampler_testbench_uses_typed_clock_parameters() -> None:
    params = sim.SampTbParams(clock_period_s=80e-9, clock_high_time_s=30e-9, input_voltage=0.7)
    tb = sim.SampTb(params)

    assert isinstance(tb, h.Module)
    assert float(tb.vclk.of.params.period) == pytest.approx(80e-9)
    assert float(tb.vclk.of.params.width) == pytest.approx(30e-9)
    assert float(tb.vdin.of.params.dc) == pytest.approx(0.7)
    assert not hasattr(params, "temperature_c")


def test_main_owns_only_experiment_targets() -> None:
    source = inspect.getsource(sim.main)
    assert "frida1_transient" in source
    assert "_check" not in source
    assert "TARGETS" not in vars(sim)
