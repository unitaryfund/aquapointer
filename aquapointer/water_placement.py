import imp
import scipy

import sys
sys.path.append('../aquapointer/')
from density_mapping import find_index_of_excited_qubits, find_positions_of_excited_qubits, rescaled_positions_to_3d_map
from qubo_solution import default_cost, fit_gaussian, run_qubo


def water_placement(densities, points, executor, processor, qubo_cost=default_cost(), loc_from_bitstrings="k_means"):
    # Obtain Gaussian of first slice- subject to change 
    amp, var =  fit_gaussian(densities[0][0,:], densities[0][1,:])
    bitstrings = [run_qubo(d, executor, processor, qubo_cost, var, amp) for d in densities]
    indexes =  find_index_of_excited_qubits(bitstrings)
    qubit_rescaled_positions = find_positions_of_excited_qubits(indexes)
    water_indices = rescaled_positions_to_3d_map(qubit_rescaled_positions,     
                       loc_from_bitstrings)
    water_positions = []
    # from the indices find the water molecule positions in angstroms
    for i,slice in enumerate(water_indices):
        ls = []
        for idx_i, idx_j in slice:
            ls.append(points[i][idx_i,idx_j])
        water_positions.append(ls)
    if loc_from_bitstrings == "k_means":
        obs = scipy.cluster.vq.whiten(water_positions)
        k_or_guess = len(water_positions)  # placeholder
        final_positions = scipy.cluster.vq.kmeans(obs, k_or_guess)
    else:
	  # can use user-defined function
       final_positions = loc_from_bitstrings(water_positions)

    return final_positions
