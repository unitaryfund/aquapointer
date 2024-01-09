# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals. 
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
from pulser import Pulse, Register, Sequence
from pulser.waveforms import InterpolatedWaveform

sys.path.append("../aquapointer/")
import utils.detuning_scale_utils as dsu


def default_cost(rescaled_pos, density, variance, bitstring, brad, amp):
    return dsu.ising_energies(rescaled_pos, density, variance, [bitstring], brad, amp)


def scale_gaussian(xy_data, var, m_x, m_y, amp):
    x = xy_data[:, 0]
    y = xy_data[:, 1]
    return amp * dsu.gaussian(var, [m_x, m_y], x, y)


def scale_shift_gaussian(data, var, m_x, m_y, amp):
    x = data[:, 0]
    y = data[:, 1]
    d = data[:, 2]
    return d - amp * dsu.gaussian(var, [m_x, m_y], x, y)


def scale_gaussian_mixture(xy_data, var, amp, m_x, m_y):
    x = xy_data[:, 0]
    y = xy_data[:, 1]
    return amp * dsu.gaussian_mixture(x, y, var, [(m_x, m_y)])


def fit_gaussian_mixture(density):
    xy_data = np.argwhere(density)
    d_data = []
    for row in xy_data:
        d_data.append(density[row[0], row[1]])
    parameters, _ = scipy.optimize.curve_fit(scale_gaussian, xy_data, d_data)
    return parameters


def fit_shifted_gaussian(density):
    m, n = density.shape
    data = m * np.ones((m * n, 3))
    for i in range(m):  
        for j in range(n):
            data[i + m * j, 0] = i
            data[i + m * j, 1] = j
            data[i + m * j, 2] = density[i, j]
    parameters, _ = scipy.optimize.curve_fit(scale_shift_gaussian, data, data[:, 2])
    return parameters


def fit_gaussian(density):
    m, n = density.shape
    xy_data = m * np.ones((m * n, 2))
    d_data = np.ones((m * n,))
    for i in range(m):  
        for j in range(n):
            xy_data[i + m * j, 0] = i
            xy_data[i + m * j, 1] = j
            d_data[i + m * j] = density[i, j]
    parameters, _ = scipy.optimize.curve_fit(scale_gaussian, xy_data, d_data)
    return parameters


def calculate_one_body_qubo_coeffs(density, rescaled_pos, variance, pos):
    gamma_list = dsu.gamma_list(density, rescaled_pos, variance)
    distances_density = dsu.find_possible_distances(rescaled_pos)
    distances_register = dsu.find_possible_distances(pos)
    scale = distances_density[0] / distances_register[0]
    return gamma_list, scale


def scale_detunings(density, pos, rescaled_pos, brad, variance, max_det):
    gamma_list, scale = calculate_one_body_qubo_coeffs(
        density, rescaled_pos, variance, pos
    )
    dets = np.array([item for item in gamma_list])
    for i in range(len(pos)):
        # shift every value by the mean of neighboring detunings
        # a neighbor is defined as atoms within a blockade radius distance
        dets[i] -= np.mean(
            dsu.neighbouring_gamma_list(
                density, rescaled_pos, rescaled_pos[i], scale * brad, variance
            )
        )
    # rescale every detuning such that the maximum detuning is `max_det`
    dets *= max_det / np.max(np.abs(dets))
    return dets


def generate_pulse_sequences(device, register, dets, max_det, pulse_duration, omega):
    """Executes a pulse sequence and returns resulting bitstrings"""
    seq = Sequence(register, device)
    # add an adiabatic pulse for every qubit
    for i in range(len(dets)):
        seq.declare_channel(f"ch{i}", "rydberg_local")
        seq.target(i, f"ch{i}")
        pulse = Pulse(
            InterpolatedWaveform(pulse_duration, [0, omega, 0]),
            InterpolatedWaveform(pulse_duration, [-max_det, 0, dets[i]]),
            0,
        )
        seq.add(pulse, f"ch{i}")

    return seq


def run_qubo(
    density,
    executor,
    proc,
    variance,
    amplitude,
    qubo_cost=default_cost,
    num_samples=1000,
):
    pos = proc.pos[0]
    rescaled_pos = proc.scale_grid_to_register()
    device = proc.device
    register = proc.register
    brad = proc.pulse_settings.brad
    pulse_duration = proc.pulse_settings.pulse_duration
    omega = proc.pulse_settings.omega
    max_det = proc.pulse_settings.max_det

    dets = scale_detunings(density, pos, rescaled_pos, brad, variance, max_det)
    pulse_sequence = generate_pulse_sequences(
        device, register, dets, max_det, pulse_duration, omega
    )
    samples = executor(pulse_sequence, num_samples)
    solution = best_solution_from_samples(
        samples, rescaled_pos, density, brad, variance, amplitude, qubo_cost,
    )
    return solution


def best_solution_from_samples(
    samples, rescaled_pos, density, brad, var, amp, qubo_cost,
):
    best_solutions = []
    samplings = []
    quantum_solutions = sorted(samples.items(), key=lambda x: x[1], reverse=True)
    quantum_plus_classical_solutions = []

    for bitstring, count in quantum_solutions:
        # calculate QUBO cost of bitstring
        cost = qubo_cost(rescaled_pos, density, var, bitstring, brad, amp)
        # returns empty whenever the blockade constraint is not respected
        try:
            i_bit = cost[0][0]
            i_en = cost[0][1]
            quantum_plus_classical_solutions.append((bitstring, count, i_en))

        except IndexError:
            i_bit = bitstring
            i_en = 1e10
            quantum_plus_classical_solutions.append((bitstring, count, i_en))

    best_solution = sorted(
        quantum_plus_classical_solutions, key=lambda x: x[2], reverse=False
    )[0][0]
    best_solutions.append(best_solution)
    samplings.append(np.array(quantum_plus_classical_solutions, dtype=object))
    return best_solution
