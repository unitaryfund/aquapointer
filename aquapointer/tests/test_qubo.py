
from re import M
from pulser.devices import MockDevice
from pulser import Sequence, Pulse
from pulser.waveforms import InterpolatedWaveform


import sys
sys.path.append('../aquapointer/')
import qubo_solution


def test_generate_pulse_sequences():
    pulse_seq = qubo_solution.generate_pulse_sequences(MockDevice, reg, dets)
    seq = Sequence(reg, MockDevice)
    for i in range(n):
        # add an adiabatic pulse for every qubit
        seq.declare_channel(f'ch{i}', 'rydberg_local')
        seq.target(i, f'ch{i}')
        pulse = Pulse(InterpolatedWaveform(T, [0, omega, 0]), InterpolatedWaveform(T, [-max_det, 0, dets[i]]), 0)
        seq.add(pulse, f'ch{i}')
    assert pulse_seq.to_abstract_repr() == seq.to_abstract_repr()


