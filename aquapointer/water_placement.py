from statistics import variance
import sys

import scipy

sys.path.append("../aquapointer/")
from density_mapping import rescaled_positions_to_3d_map
from qubo_solution import default_cost, fit_gaussian, run_qubo


def find_water_positions(
    densities,
    points,
    executor,
    processor_configs,
    num_samples=1000,
    qubo_cost=default_cost,
    location_clustering=None,
):
    params = fit_gaussian(densities[0])
    variance, amplitude = params[0], params[3]
    
    bitstrings = []
    for k, d in enumerate(densities):
        bitstrings.append(
            run_qubo(
                d,
                executor,
                processor_configs[k],
                variance,
                amplitude,
                qubo_cost,
                num_samples,
            )
        )

    water_indices = rescaled_positions_to_3d_map(
        bitstrings, [p.scale_grid_to_register() for p in processor_configs]
    )
    water_positions = []

    # from the indices find the water molecule positions in angstroms
    for i, slice in enumerate(water_indices):
        for idx_i, idx_j in slice:
            water_positions.append(points[i][idx_i, idx_j])
    if not location_clustering:
        return water_positions

    return location_clustering(water_positions)


def location_clustering_kmeans(water_positions):
    obs = scipy.cluster.vq.whiten(water_positions)
    k_or_guess = len(water_positions)  # placeholder
    final_positions = scipy.cluster.vq.kmeans(obs, k_or_guess)

    return final_positions
