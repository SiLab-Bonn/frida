"""Software-only checks for the native HDL21 CDAC interface."""

import inspect

import hdl21 as h
import pytest

from . import sim


def test_cdac_testbench_uses_typed_timing() -> None:
    params = sim.CdacTbParams(code_dwell_s=250e-9, transition_time_s=50e-12)
    tb = sim.CdacTb(params)

    assert isinstance(tb, h.Module)
    assert tb.dac_bits.width == params.cdac.n_dac + params.cdac.n_extra
    assert float(tb.vdac_15.of.params.wave.points[1][0]) == pytest.approx(250e-9)
    assert not hasattr(params, "temperature_c")


def test_cdac_testbench_packs_code_with_c0_as_highest_stage() -> None:
    tb = sim.CdacTb(sim.CdacTbParams())

    # The 11-bit baseline ramp leaves the five earliest/largest redundant
    # stages low and toggles C15 fastest.
    assert len(tb.vdac_0.of.params.wave.points) == 2
    assert len(tb.vdac_4.of.params.wave.points) == 2
    assert len(tb.vdac_5.of.params.wave.points) == 4
    assert len(tb.vdac_15.of.params.wave.points) == 4096


def test_main_owns_only_experiment_targets() -> None:
    source = inspect.getsource(sim.main)
    assert "frida1_transfer_curve" in source
    assert "_check" not in source
    assert "TARGETS" not in vars(sim)
