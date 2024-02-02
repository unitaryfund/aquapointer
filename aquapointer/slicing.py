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
    idx = [[] for _ in range(len(slicing_planes))]
    coordinates = [[] for _ in range(len(slicing_planes))]
    densities = [[] for _ in range(len(slicing_planes))]

    for ind in np.ndindex(density_grid.grid.shape):
        density = density_grid.grid[ind]
        center = density_grid.delta * np.array(ind) + density_grid.origin

        for s in range(len(slicing_planes) + 1):
            if s == 0:
                # slice is in opposite direction of the normal
                normal = slicing_planes[s][1] / norm(slicing_planes[s][1])
                d = (center - slicing_planes[s][0]).dot(normal)
                if d >= 0:
                    idx[s].append(ind)
                    coordinates[s].append(center - d * normal)
                    densities[s].append(density)
                    break

            elif s < len(slicing_planes):
                # slice between two planes
                normal1 = slicing_planes[s - 1][1] / norm(slicing_planes[s - 1][1])
                d1 = (center - slicing_planes[s - 1][0]).dot(normal1)
                normal2 = slicing_planes[s][1] / norm(slicing_planes[s][1])
                d2 = (center - slicing_planes[s][0]).dot(normal2)

                if d1 < 0 and d2 >= 0:
                    idx[s].append(ind)
                    coordinates[s].append(center - (d1 * normal1 + d2 * normal2) / 2)
                    densities[s].append(density)
                    break

            else:
                # slice with one plane, in direction of the normal
                normal = slicing_planes[s - 1][1] / norm(slicing_planes[s - 1][1])
                d = (center - slicing_planes[s - 1][0]).dot(normal)
                if d < 0:
                    idx[s].append(ind)
                    coordinates[s].append(center - d * slicing_planes[s - 1][1])
                    densities[s].append(density)
                    break

    return coordinates, densities


def generate_planes_by_axis(
    axis: NDArray,
    distances: NDArray,
    ref_point: NDArray,
) -> List[Tuple[NDArray, NDArray]]:
    return [(ref_point + axis * d, np.array(axis)) for d in distances]


def find_density_origin(density_grid: Grid) -> NDArray:
    return density_grid.origin


def find_density_coordinate_boundaries(density_grid: Grid) -> List[NDArray]:
    return density_grid.grid.shape * density_grid.delta


def visualize_slicing_plane(point: NDArray, normal: NDArray) -> None:
    c = -point.dot(normal / norm(normal))
    x, y = np.meshgrid(range(20), range(20))
    z = (-normal[0] * x - normal[1] * y - c) / normal[2]

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    plt.show()
    return
