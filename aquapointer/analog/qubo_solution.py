# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import scipy
from numpy.typing import NDArray
from pulser import Pulse, Register, Sequence
from pulser.backend.qpu import QPUBackend
from pulser.waveforms import InterpolatedWaveform

from aquapointer.density_canvas import DensityCanvas
import aquapointer.analog.utils.detuning_scale_utils as dsu


def scale_gaussian(
    xy_data: NDArray, var: float, m_x: float, m_y: float, amp: float
) -> float:
    r"""Calculates the value at each point on a 2-D grid of a sum of isotropic
    normal distributions centered at ``m_x``, ``m_y``, ... with variance
    ``var`` and amplitude ``amp``.

    Args:
        xy_data: Array of indices of density array elements.
        var: Variance of the distribution.
        m_x: Center of the distribution along dimension 0 of the density array.
        m_y: Center of the distribution along dimension 1 of the density array.
        amp: Amplitude of the distribution.

    Returns:
        The value at each point on a 2-D grid of a sum of isotropic normal distributions.
    """
    x = xy_data[:, 0]
    y = xy_data[:, 1]
    return amp * dsu.gaussian(var, [m_x, m_y], x, y)


def fit_gaussian(density: NDArray) -> Tuple[float]:
    r"""Fits density data to a sum of isotropic normal distributions on a 2-D
    grid and returns the variance, center (in dimenions 0 and in dimension 1),
    and amplitude of the distribution.
    """
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


def generate_pulse_sequences(
    device: QPUBackend,
    register: Register,
    dets: List[float],
    max_det: float,
    pulse_duration: float,
    omega: float,
) -> Sequence:
    """Arranges detunings into a pulse sequence generated for a given device."""
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
    canvas: DensityCanvas.DensityCanvas,
    executor: Callable[[Sequence, int], Any],
    device: QPUBackend,
    pulse_settings: Dict[str, float],
    variance: float,
    amplitude: float,
    num_samples: int = 1000,
) -> str:
    r"""Obtain bitstring solving the QUBO problem for the input density slice.
    
    Args:
        canvas: Density canvas object containing density and coordinate info.
        executor: Function that executes a pulse sequence on a quantum backend.
        proc: ``Processor`` object storing settings for running on a quantum backend.

        qubo_cost: Cost function to be optimized in the QUBO.
        num_samples: Number of times to execute the quantum experiment or simulation on the backend.

    Returns:
        Bitstring solving the QUBO for the input density slice.
    """

    register = Register.from_coordinates(canvas._lattice._coords)
    brad = pulse_settings["brad"]
    pulse_duration = pulse_settings["pulse_duration"]
    omega = pulse_settings["omega"]
    max_det = pulse_settings["max_det"]

    detunings = canvas.calculate_detunings()
    pulse_sequence = generate_pulse_sequences(
        device, register, detunings, max_det, pulse_duration, omega
    )
    samples = executor(pulse_sequence, num_samples)
    solution = best_solution_from_samples(
        samples,
        canvas,
        brad,
        variance,
        amplitude
    )
    return solution


def best_solution_from_samples(
    samples: str,
    canvas: DensityCanvas.DensityCanvas,
    brad: float,
    var: float,
    amp: float,
) -> str:
    r"""Identify sampled bitstring with lowest QUBO cost for the input density slice.

    Args:
        samples: Bitstring samples obtained from executing the pulse sequence.
        density: Density slices of the protein cavity as a 2-D array of density values.
        brad: Blockade radius distance (in micrometers).
        var: Variance of the fitted density distribution.
        amp: Amplitude of the fitted density distribution.
        qubo_cost: Cost function to be optimized in the QUBO.

    Returns:
        Bitstring solving the QUBO for the input density slice.
    """
    best_solutions = []
    samplings = []
    quantum_solutions = sorted(samples.items(), key=lambda x: x[1], reverse=True)
    quantum_plus_classical_solutions = []

    for bitstring, count in quantum_solutions:
        # calculate QUBO cost of bitstring
        cost = canvas.calculate_bitstring_cost_from_coefficients(bitstring) 
        # returns empty whenever the blockade constraint is not respected
        try:
            quantum_plus_classical_solutions.append((bitstring, count, cost))

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
