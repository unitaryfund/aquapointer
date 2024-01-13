# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Callable, List, Optional

import scipy
from numpy.typing import NDArray
from pulser import Sequence

from aquapointer.analog.density_mapping import rescaled_positions_to_3d_map
from aquapointer.analog.qubo_solution import default_cost, fit_gaussian, run_qubo
from aquapointer.analog_digital.processor import Processor


"""High-level tools for finding the locations of water molecules in a protein
cavity."""


def find_water_positions(
    densities: List[NDArray],
    points: List[NDArray],
    executor: Callable[[Sequence, int], Any],
    processor_configs: List[Processor],
    num_samples: int = 1000,
    qubo_cost: Callable[
        [NDArray, NDArray, float, str, float, float], float
    ] = default_cost,
    location_clustering: Optional[Callable[[List[List[float]]], List[Any]]] = None,
) -> List[List[float]]:
    r"""Finds the locations of water molecules in a protein cavity from 2-D
    arrays of density values of the cavity.

    Args:
        densities: List of density slices of the protein cavity as 2-D arrays
            of density values.
        points: List of arrays containing coordinates corresponding to each
            element of the density arrays.
        executor: Function that executes a pulse sequence on a quantum backend.
        processor_configs: List of ``Processor`` objects storing settings for
            running on a quantum backend.
        num_samples: Number of times to execute the quantum experiment or
            simulation on the backend.
        qubo_cost: Cost function to be optimized in the QUBO.
        location_clustering: Optional function for merging duplicate locations
            (typically identified in different layers).

    Returns:
        List of 3-D coordinates of the locations of water molecules in the
            protein cavity.
    """
    bitstrings = []
    for k, d in enumerate(densities):
        params = fit_gaussian(d)
        variance, amplitude = params[0], params[3]
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
    r"""Takes a list of 3-D coordinates of the locations of water molecules in the
    protein cavity and merges each set of duplicate locations into a
    single location.
    """
    obs = scipy.cluster.vq.whiten(water_positions)
    k_or_guess = len(water_positions)  # placeholder
    final_positions = scipy.cluster.vq.kmeans(obs, k_or_guess)

    return final_positions
