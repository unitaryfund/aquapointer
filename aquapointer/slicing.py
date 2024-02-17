# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from itertools import groupby
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gridData import Grid
from numpy.linalg import norm
from numpy.typing import NDArray


def density_file_to_grid(filename: str) -> Grid:
    return Grid(filename)


def density_slices_by_axis(
    density_grid: Grid, axis: NDArray, distances: NDArray
) -> Tuple[List[NDArray]]:
    origin = density_grid.origin
    slicing_planes = generate_planes_by_axis(axis, distances, origin)
    return density_slices_by_plane(density_grid, slicing_planes, distances)


def density_slices_by_plane(
    density_grid: Grid,
    slicing_planes: List[Tuple[NDArray, NDArray]],
    distances: NDArray,
) -> Tuple[List[NDArray]]:
    idx_lists = [[] for _ in range(len(slicing_planes) + 1)]
    point_lists = [[] for _ in range(len(slicing_planes) + 1)]
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

    points = []
    densities = []
    for i in range(len(idx_lists)):
        points_array, density_array = shape_slice(
            point_lists[i], density_lists[i], normals[0]  # TODO: generalize
        )
        points.append(points_array)
        densities.append(density_array)

    return points, densities


def shape_slice(points: NDArray, density, normal: NDArray):
    # rotation angles between z' (normal) and the z-axis
    theta = np.arctan2(normal[1], normal[0])
    phi = np.arctan2(norm(normal[0:2]), normal[2])
    # rotate x and y to x' and y' respectively
    x_prime = np.array([np.cos(phi), np.sin(phi), 0])
    y_prime = np.array(
        [-np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta), np.sin(theta)]
    )
    # project points onto y' and group indices of projected points
    point_list = []
    density_list = []

    for _, group in groupby(points, lambda x: x.dot(y_prime)):
        # project points onto x' and sort indices of projected points
        idxs_by_xp = np.argsort([g.dot(x_prime) for g in group])
        point_list.append([points[i] for i in idxs_by_xp])
        density_list.append([density[i] for i in idxs_by_xp])

    m = max([len(p) for p in point_list])
    n = len(point_list)
    points_array = np.zeros((m, n, 3))
    density_array = np.zeros((m, n))

    for j, i in np.ndindex(n, m):  # i in range(n):
        points_array[i, j, :] = point_list[j][i]  # TODO: generalize
        density_array[i, j] = density_list[j][i]

    return points_array, density_array


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
