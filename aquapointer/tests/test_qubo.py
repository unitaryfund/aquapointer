import sys

import numpy as np
import pytest
from pulser import Pulse, Sequence
from pulser.devices import MockDevice
from pulser.waveforms import InterpolatedWaveform

sys.path.append("../aquapointer/")
from qubo_solution import (
    best_solution_from_samples,
    calculate_one_body_qubo_coeffs,
    generate_pulse_sequences,
    run_qubo,
    scale_detunings,
)


@pytest.mark.parametrize(
    "reg, best_solution", [(regs[0], "1000"), (regs[1], "00010000")]
)
def test_run_qubo(reg, best_solution):
    result = run_qubo(
        density,
        executor,
        MockDevice,
        reg,
        rescaled_pos,
        pos,
        variance,
        amplitude,
        brad,
        T,
        omega,
        max_det,
        num_samples,
    )
    assert str(result) == best_solution


@pytest.mark.parametrize("reg, result", [(regs[0], "1000"), (regs[1], "00010000")])
def test_best_solution(reg, best_solution):
    result = best_solution_from_samples(
        density,
        executor,
        MockDevice,
        reg,
        rescaled_pos,
        pos,
        variance,
        amplitude,
        brad,
        T,
        omega,
        max_det,
        num_samples,
    )
    assert str(result) == best_solution


@pytest.mark.parametrize("reg", [regs[0], regs[1]])
def test_generate_pulse_sequences(reg):
    pulse_seq = generate_pulse_sequences(MockDevice, reg, dets)
    seq = Sequence(reg, MockDevice)
    for i in range(n):
        # add an adiabatic pulse for every qubit
        seq.declare_channel(f"ch{i}", "rydberg_local")
        seq.target(i, f"ch{i}")
        pulse = Pulse(
            InterpolatedWaveform(T, [0, omega, 0]),
            InterpolatedWaveform(T, [-max_det, 0, dets[i]]),
            0,
        )
        seq.add(pulse, f"ch{i}")
    assert pulse_seq.to_abstract_repr() == seq.to_abstract_repr()


def test_scale_detunings():
    dets = np.array([item for item in gamma_list])
    for i in range(n):
        dets[i] -= np.mean(
            dsu.neighbouring_gamma_list(
                density, rescaled_pos, rescaled_pos[i], scale * brad, variance
            )
        )
    dets *= max_det / np.max(np.abs(dets))
    test_dets = scale_detunings(density, pos, rescaled_pos, brad, variance, max_det)
    assert dets == test_dets
