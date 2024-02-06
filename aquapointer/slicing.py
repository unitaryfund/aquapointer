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
    density_grid: Grid, axis: NDArray, distances: NDArray
) -> Tuple[List[NDArray]]:
    slicing_planes = generate_planes_by_axis(axis, distances, density_grid.origin)
    return density_slices_by_plane(density_grid, slicing_planes)


def density_slices_by_plane(
    density_grid: Grid, slicing_planes: List[Tuple[NDArray, NDArray]]
) -> Tuple[List[NDArray]]:
    idx_lists = [[] for _ in range(len(slicing_planes) + 1)]
    point_lists= [[] for _ in range(len(slicing_planes) + 1)]
    density_lists = [[] for _ in range(len(slicing_planes) + 1)]
    normals = [s / norm(s) for s in list(zip(*slicing_planes))[1]]

    for ind in np.ndindex(density_grid.grid.shape):
        density = density_grid.grid[ind]
        center = density_grid.delta * np.array(ind) + density_grid.origin

        for s in range(len(slicing_planes) + 1):
            if s == 0:
                # slice is in opposite direction of the normal
                d = (center - slicing_planes[s][0]).dot(normals[s])
                if d < 0:
                    coords = center - d * normals[s] / 2
                    break

            elif s < len(slicing_planes):
                # slice between two planes
                d1 = (center - slicing_planes[s - 1][0]).dot(normals[s - 1])
                d2 = (center - slicing_planes[s][0]).dot(normals[s])
                if d1 > 0 and d2 <= 0:
                    coords = center - (d1 * normals[s - 1] + d2 * normals[s]) / 2
                    break

            else:
                # slice with one plane, in direction of the normal
                d = (center - slicing_planes[-1][0]).dot(normals[-1])
                if d >= 0:
                    coords = center - d * normals[-1] / 2

        idx_lists[s].append(ind)
        point_lists[s].append(coords)
        density_lists[s].append(density)

    return idx_lists, point_lists, density_lists


def generate_planes_by_axis(
    axis: NDArray,
    distances: NDArray,
    ref_point: NDArray,
) -> List[Tuple[NDArray, NDArray]]:
    return [(ref_point + axis * d, axis) for d in distances]


def find_density_origin(density_grid: Grid) -> NDArray:
    return density_grid.origin


def find_density_point_boundaries(density_grid: Grid) -> List[NDArray]:
    return density_grid.grid.shape * density_grid.delta


def visualize_slicing_plane(point: NDArray, normal: NDArray) -> None:
    c = -point.dot(normal / norm(normal))
    x, y = np.meshgrid(range(20), range(20))
    z = (-normal[0] * x - normal[1] * y - c) / normal[2]

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    plt.show()
    return
