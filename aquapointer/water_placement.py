import sys

import scipy

sys.path.append("../aquapointer/")
from density_mapping import (
    rescaled_positions_to_3d_map,
)
from qubo_solution import default_cost, fit_gaussian, run_qubo


def find_water_positions(
    densities,
    points,
    executor,
    processor_configs,
    num_samples=1000,
    qubo_cost=default_cost,
    loc_from_bitstrings="k_means",
):
    variance, amplitude = fit_gaussian(densities[0])

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
        ls = []
        for idx_i, idx_j in slice:
            ls.append(points[i][idx_i, idx_j])
        water_positions.append(ls)
    return water_positions
