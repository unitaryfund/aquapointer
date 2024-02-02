# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from gridData import Grid
from numpy.typing import NDArray
from numpy.linalg import norm


def density_file_to_grid(filename: str) -> Grid:
    return Grid(filename)


def density_slices_by_axis(
    density_grid: Grid, axis: NDArray, steps: NDArray
) -> Tuple[List[NDArray]]:
    slicing_planes = generate_planes_by_axis(axis, steps, density_grid.origin)
    return density_slices_by_plane(density_grid, slicing_planes)


def density_slices_by_plane(
    density_grid: Grid, slicing_planes: List[Tuple[NDArray, NDArray]]
) -> Tuple[List[NDArray]]:
    density_3d_array = density_grid.grid
    centers = np.array(list(density_grid.centers()))
    idx = [[] for _ in range(len(slicing_planes))]
    coordinates = [[] for _ in range(len(slicing_planes))]
    densities = [[] for _ in range(len(slicing_planes))]

    for i in range(density_3d_array.shape[0]):
        for j in range(density_3d_array.shape[1]):
            for k in range(density_3d_array.shape[2]):
                for s in range(len(slicing_planes) + 1):
                    if s == 0:
                        # slice is in opposite direction of the normal
                        normal = slicing_planes[s - 1][1] / norm(
                            slicing_planes[s - 1][1]
                        )
                        d = (centers(i, j, k) - slicing_planes[s - 1][0]).dot(normal)
                        if d >= 0:
                            idx[s].append(i, j, k)
                            coordinates[s].append(centers(i, j, k) - d * normal)
                            densities[s].append(density_3d_array[i, j, k])
                            break

                    elif s < len(slicing_planes):
                        # slice between two planes
                        normal1 = slicing_planes[s - 1][1] / norm(
                            slicing_planes[s - 1][1]
                        )
                        d1 = (centers(i, j, k) - slicing_planes[s - 1][0]).dot(normal1)
                        normal2 = slicing_planes[s][1] / norm(slicing_planes[s][1])
                        d2 = (centers(i, j, k) - slicing_planes[s][0]).dot(normal2)
                        if d1 < 0 and d2 >= 0:
                            idx[s].append(i, j, k)
                            coordinates[s].append(
                                centers(i, j, k) - (d1 * normal1 + d2 * normal2) / 2
                            )
                            densities[s].append(density_3d_array[i, j, k])
                            break

                    else:
                        # slice with one plane, in direction of the normal
                        normal = slicing_planes[s - 1][1] / norm(
                            slicing_planes[s - 1][1]
                        )
                        d = (centers(i, j, k) - slicing_planes[s - 1][0]).dot(normal)
                        if d < 0:
                            idx[s].append(i, j, k)
                            coordinates[s].append(
                                centers(i, j, k) - d * slicing_planes[s - 1][1]
                            )
                            densities[s].append(density_3d_array[i, j, k])
                            break

    return coordinates, densities


def generate_planes_by_axis(
    axis: NDArray,
    steps: NDArray,
    origin: NDArray,
) -> List[Tuple[NDArray, NDArray]]:
    return [(origin + np.array(axis * s), np.array(axis)) for s in steps]


def find_density_origin(density_grid: Grid) -> NDArray:
    return density_grid.origin


def find_density_coordinate_boundaries(density_grid: Grid) -> List[NDArray]:
    centers = list(density_grid.centers())
    rel_coords = [(density_grid.origin - c) for c in centers]
    mins = [min(rel_coords)]
    maxes = [max(rel_coords)]
    return list(zip(mins, maxes))


def visualize_slicing_plane(point: NDArray, normal: NDArray) -> None:
    c = -point.dot(normal / norm(normal))
    x, y = np.meshgrid(range(20), range(20))
    z = (-normal[0] * x - normal[1] * y - c) / normal[2]

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    plt.show()
    return
