# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import numpy as np
from gridData import Grid
from numpy.typing import NDArray


def density_slices_by_plane(
    filename: str, slicing_planes: List[Tuple[NDArray, NDArray]]
) -> List[NDArray]:
    density_grid = Grid(filename)
    density_3d_array = density_grid.grid
    coords = np.array(density_grid.centers())
    idx = [[] for _ in range(len(slicing_planes))]
    coordinates = [[] for _ in range(len(slicing_planes))]
    densities = [[] for _ in range(len(slicing_planes))]

    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            for k in range(coords.shape[2]):
                for s in range(len(slicing_planes) + 1):
                    if (
                        s == 0
                    ):  # one slicing plane in opposite direction of the normal
                        d = (coords[i, j, k] - slicing_planes[s - 1][0]).dot(
                            slicing_planes[s - 1][1]
                        )
                        if d >= 0:
                            idx[s].append(i, j, k)
                            coordinates[s].append(
                                coords[i, j, k] - d * slicing_planes[s - 1][1]
                            )
                            densities[s].append(density_3d_array[i, j, k])
                            break

                    elif s < len(slicing_planes):  # slice between two planes
                        d1 = (coords[i, j, k] - slicing_planes[s - 1][0]).dot(
                            slicing_planes[s - 1][1]
                        )
                        d2 = (coords[i, j, k] - slicing_planes[s][0]).dot(
                            slicing_planes[s][1]
                        )
                        if d1 < 0 and d2 >= 0:
                            idx[s].append(i, j, k)
                            coordinates[s].append(
                                coords[i, j, k]
                                - (
                                    d1 * slicing_planes[s - 1][1]
                                    + d2 * slicing_planes[s][1]    
                                )
                                / 2
                            )
                            densities[s].append(density_3d_array[i, j, k])
                            break

                    else:  # slice with one plane, in direction of the normal
                        d = (coords[i, j, k] - slicing_planes[s - 1][0]).dot(
                            slicing_planes[s - 1][1]
                        )
                        if d < 0:
                            idx[s].append(i, j, k)
                            coordinates[s].append(
                                coords[i, j, k] - d * slicing_planes[s - 1][1]
                            )
                            densities[s].append(density_3d_array[i, j, k])
                            break

    return coordinates, densities


def density_slices_by_axis(
    filename: str, axis: NDArray, steps: NDArray
) -> List[NDArray]:
    slicing_planes = generate_planes_along_axis(axis, steps)
    return density_slices_by_plane(filename, slicing_planes)


def generate_planes_along_axis(
    axis: NDArray, steps: NDArray
) -> List[Tuple[NDArray, NDArray]]:
    points = [np.array(axis * s) for s in steps]  # step from center along axis
    normals = [np.array(axis)] * len(points)
    return list(zip(points, normals))


def construct_plane(
    point: NDArray, normal: NDArray
) -> Tuple[NDArray, NDArray, NDArray]:
    c = -point.dot(normal)
    x, y = np.meshgrid(range(20), range(20))
    z = (-normal[0] * x - normal[1] * y - c) / normal[2]
    return x, y, z
