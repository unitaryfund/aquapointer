import numpy as np
import matplotlib.pyplot as plt

import utils.detuning_scale_utils as dsu


def visualize_registers(registers, positions, rescaled_positions, densities):
    for k, density in enumerate(densities):
        print(f"Processing density {k+1} of {len(densities)}")
        print(f"Density {k+1} has {len(positions[k])} qubits")
        registers[k].draw()
        fig, ax = dsu.plot_density(density, rescaled_positions[k], title=f"Density slice {k}")
    plt.show()


def rescaled_positions_to_3d_map(best_solutions, rescaled_positions):
    qubit_indices = find_index_of_excited_qubits(best_solutions)
    qubit_rescaled_positions = find_positions_of_excited_qubits(qubit_indices, rescaled_positions)
    water_indices = []
    for i,res_pos in enumerate(qubit_rescaled_positions):
        ls = []
        for pos in res_pos:
            ls.append((int(pos[1]), int(pos[0])))
        water_indices.append(ls)
    return water_indices


def find_index_of_excited_qubits(best_solutions):
    qubit_indices = []
    for bitstring in best_solutions:
        ls = []
        for i,b in enumerate(bitstring):
            if b=='1':
                ls.append(i)
        qubit_indices.append(ls)
    return qubit_indices


def find_positions_of_excited_qubits(qubit_indices, rescaled_positions):
    qubit_rescaled_positions = []
    for i,indices in enumerate(qubit_indices):
        ls = []
        for idx in indices:
            ls.append(rescaled_positions[i][idx])
        qubit_rescaled_positions.append(ls)
    return qubit_rescaled_positions
