# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import matplotlib.pyplot as plt
from numpy.typing import NDArray
from pulser.register import Register

from aquapointer.analog.utils import detuning_scale_utils as dsu


def visualize_registers(
    registers: List[Register],
    positions: List[float],
    rescaled_positions: List[NDArray],
    densities: List[NDArray],
) -> None:
    """Plot the register layout side-by-side with the density slice."""
    for k, density in enumerate(densities):
        print(f"Processing density {k+1} of {len(densities)}")
        print(f"Density {k+1} has {len(positions[k])} qubits")
        registers[k].draw()
        fig, ax = dsu.plot_density(
            density, rescaled_positions[k], title=f"Density slice {k}"
        )
    plt.show()


def rescaled_positions_to_3d_map(
    best_solutions: List[str], rescaled_positions: List[NDArray]
) -> List[int]:
    r"""Finds the locations of water molecules in a protein cavity from 2-D
    arrays of density values of the cavity.

    Args:
        best_solutions: List of QUBO solution bitstrings.
        rescaled_positions: List of arrays of rescaled register positions.

    Returns:
        List of 3-D coordinates of the locations of water molecules in the
            protein cavity.
    """
    qubit_indices = find_index_of_excited_qubits(best_solutions)
    qubit_rescaled_positions = find_positions_of_excited_qubits(
        qubit_indices, rescaled_positions
    )
    water_indices = []
    for i, res_pos in enumerate(qubit_rescaled_positions):
        ls = []
        for pos in res_pos:
            ls.append((int(pos[1]), int(pos[0])))
        water_indices.append(ls)
    return water_indices


def find_index_of_excited_qubits(best_solutions: List[str]) -> List[int]:
    r"""Finds the indices of excited qubits corresponding to each QUBO
    solution.
    """
    qubit_indices = []
    for bitstring in best_solutions:
        ls = []
        for i, b in enumerate(bitstring):
            if b == "1":
                ls.append(i)
        qubit_indices.append(ls)
    return qubit_indices


def find_positions_of_excited_qubits(
    qubit_indices: List[int], rescaled_positions: List[NDArray]
) -> List[float]:
    r"""Returns a list of coordinates of the excited qubits corresponding to
    each QUBO solution.
    """
    qubit_rescaled_positions = []
    for i, indices in enumerate(qubit_indices):
        ls = []
        for idx in indices:
            ls.append(rescaled_positions[i][idx])
        qubit_rescaled_positions.append(ls)
    return qubit_rescaled_positions
