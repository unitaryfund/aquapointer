# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Callable, List, Optional

import numpy as np
import scipy
from numpy.typing import NDArray
from pulser import Sequence

from aquapointer.density_canvas.DensityCanvas import DensityCanvas
from aquapointer.analog.qubo_solution import fit_gaussian, run_qubo
from aquapointer.analog_digital.processor import Processor


"""High-level tools for finding the locations of water molecules in a protein
cavity."""


def find_water_positions(
    density_canvases: List[DensityCanvas],
    executor: Callable[[Sequence, int], Any],
    processor_configs: List[Processor],
    num_samples: int = 1000,
    location_clustering: Optional[Callable[[List[List[float]]], List[Any]]] = None,
) -> List[NDArray]:

    r"""Finds the locations of water molecules in a protein cavity from 2-D
    arrays of density values of the cavity.

    Args:
        density_canvases: List of density canvas objects containing density and geometry info of the protein cavity.
        executor: Function that executes a pulse sequence on a quantum backend.
        processor_configs: List of ``Processor`` objects storing settings for running on a quantum backend.
        num_samples: Number of times to execute the quantum experiment or simulation on the backend.
        qubo_cost: Cost function to be optimized in the QUBO.
        location_clustering: Optional function for merging duplicate locations (typically identified in different layers).


    Returns:
        List of 3-D coordinates of the locations of water molecules in the
            protein cavity.
    """
    bitstrings = []
    coords = []
    for k, d in enumerate(density_canvases):
        params = [58, 0, 0, 48.2]
        variance, amplitude = params[0], params[3]
        bitstring = run_qubo(
                d,
                executor,
                processor_configs[k],
                variance,
                amplitude,
                num_samples,
            )
        bitstrings.append(bitstring)
        if '1' not in bitstring:
            continue
        coords.append([c for i,c in enumerate(d._lattice._coords) if int(bitstring[i])])
        
    return np.array(sum(coords, []))



def location_clustering_kmeans(water_positions: List[List[float]]) -> List[List[float]]:
    r"""
    Takes a list of 3-D coordinates of the locations of water molecules in the
    protein cavity and merges each set of duplicate locations into a
    single location.
    """
    obs = scipy.cluster.vq.whiten(water_positions)
    k_or_guess = len(water_positions)  # placeholder
    final_positions = scipy.cluster.vq.kmeans(obs, k_or_guess)

    return final_positions
