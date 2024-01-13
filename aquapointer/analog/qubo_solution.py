# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Callable, List, Tuple

import numpy as np
import scipy
from numpy.typing import NDArray
from pulser import Pulse, Register, Sequence
from pulser.backend.qpu import QPUBackend
from pulser.waveforms import InterpolatedWaveform

import aquapointer.analog.utils.detuning_scale_utils as dsu


def default_cost(
    rescaled_pos: NDArray,
    density: NDArray,
    variance: float,
    bitstring: str,
    brad: float,
    amp: float,
) -> List[Tuple[float]]:
    return dsu.ising_energies(rescaled_pos, density, variance, [bitstring], brad, amp)


def scale_gaussian(
    xy_data: NDArray, var: float, m_x: float, m_y: float, amp: float
) -> float:
    x = xy_data[:, 0]
    y = xy_data[:, 1]
    return amp * dsu.gaussian(var, [m_x, m_y], x, y)


def fit_gaussian(density: NDArray) -> Tuple[float]:
    m, n = density.shape
    xy_data = np.zeros((m * n, 2))
    d_data = np.zeros(m * n)
    for i in range(m):
        for j in range(n):
            xy_data[n * i + j, 0] = i
            xy_data[n * i + j, 1] = j
            d_data[n * i + j] = density[i, j]

    parameters, _ = scipy.optimize.curve_fit(
        scale_gaussian,
        xy_data,
        d_data,
    )
    return parameters


def calculate_one_body_qubo_coeffs(
    density: NDArray, rescaled_pos: NDArray, variance: float, pos: NDArray
) -> Tuple[NDArray, float]:
    gamma_list = dsu.gamma_list(density, rescaled_pos, variance)
    distances_density = dsu.find_possible_distances(rescaled_pos)
    distances_register = dsu.find_possible_distances(pos)
    scale = distances_density[0] / distances_register[0]
    return gamma_list, scale


def scale_detunings(
    density: NDArray,
    pos: NDArray,
    rescaled_pos: NDArray,
    brad: float,
    variance: float,
    max_det: float,
) -> NDArray:
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


def generate_pulse_sequences(
    device: QPUBackend,
    register: Register,
    dets: List[float],
    max_det: float,
    pulse_duration: float,
    omega: float,
) -> Sequence:
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
    density: NDArray,
    executor: Callable[[Sequence, int], Any],
    proc: QPUBackend,
    variance: float,
    amplitude: float,
    qubo_cost: Callable[
        [NDArray, NDArray, float, str, float, float], float
    ] = default_cost,
    num_samples: int = 1000,
) -> str:
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
        samples,
        rescaled_pos,
        density,
        brad,
        variance,
        amplitude,
        qubo_cost,
    )
    return solution


def best_solution_from_samples(
    samples: int,
    rescaled_pos: NDArray,
    density: NDArray,
    brad: float,
    var: float,
    amp: float,
    qubo_cost: Callable[
        [NDArray, NDArray, float, str, float, float], float
    ] = default_cost,
) -> str:
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
