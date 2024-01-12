# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, List, Optional, Any

import scipy
from aquapointer.analog.density_mapping import rescaled_positions_to_3d_map
from numpy.typing import NDArray
from aquapointer.analog_digital.processor import Processor
from pulser import Sequence
from aquapointer.analog.qubo_solution import default_cost, fit_gaussian, run_qubo


def find_water_positions(
    densities: List[NDArray],
    points: List[NDArray],
    executor: Callable[[Sequence, int], Any],
    processor_configs: List[Processor],
    num_samples: int = 1000,
    qubo_cost: Callable[[NDArray, NDArray, float, str, float, float], float] = default_cost,
    location_clustering: Optional[Callable[[List[List[float]]], List[Any]]] = None,
) -> List[List[float]]:
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


def location_clustering_kmeans(water_positions: List[List[float]]) -> List[List[float]]:
    obs = scipy.cluster.vq.whiten(water_positions)
    k_or_guess = len(water_positions)  # placeholder
    final_positions = scipy.cluster.vq.kmeans(obs, k_or_guess)

    return final_positions
