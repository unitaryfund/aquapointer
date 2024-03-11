# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from itertools import groupby, permutations
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gridData import Grid
from numpy.linalg import norm
from numpy.typing import NDArray


def density_file_to_grid(filename: str) -> Grid:
    """Load density file as a ``Grid`` object."""
    return Grid(filename)


def density_slices_by_axis(
    density_grid: Grid, axis: NDArray, distances: NDArray
) -> Tuple[List[NDArray]]:
    """Slice 3D density grid at specified intervals along a specified axis
    and flatten slices into 2D density arrays positioned at each midplane."""
    origin = density_origin(density_grid)
    slicing_planes = generate_planes_by_axis(axis, distances, origin)
    return density_slices_by_plane(density_grid, slicing_planes)


def density_slices_by_plane(
    density_grid: Grid,
    slicing_planes: List[Tuple[NDArray, NDArray]],
) -> Tuple[List[NDArray]]:
    """Slice 3D density grid by planes specified by a list of point and axis
    pairs and flatten slices into 2D density arrays positioned at each
    midplane."""
    idx_lists = [[] for _ in range(len(slicing_planes) + 1)]
    point_lists = [[] for _ in range(len(slicing_planes) + 1)]
    density_lists = [[] for _ in range(len(slicing_planes) + 1)]
    normals = [s / norm(s) for s in list(zip(*slicing_planes))[1]]
    origin = density_origin(density_grid)
    endpoint = density_point_boundaries(density_grid)

    midplane_points = (
        [(origin + slicing_planes[0][0]) / 2]
        + [
            (slicing_planes[s][0] + slicing_planes[s + 1][0]) / 2
            for s in range(len(slicing_planes) - 1)
        ]
        + [
            slicing_planes[-1][0]
            + slicing_planes[-1][1]
            * (endpoint - slicing_planes[-1][0]).dot(slicing_planes[-1][1])
            / 2
        ]
    )
    midplane_normals = (
        [normals[0]]
        + [
            np.mean(np.array(normals[n : n + 2]), axis=0)
            / norm(np.mean(np.array(normals[n : n + 2]), axis=0))
            for n in range(len(normals) - 1)
        ]
        + [normals[-1]]
    )

    for ind in np.ndindex(density_grid.grid.shape):
        density = density_grid.grid[ind]
        center = np.round(density_grid.delta, decimals=10) * np.array(
            ind
        ) + density_origin(density_grid)

        for s in range(len(slicing_planes) + 1):
            if s == 0:
                # slice is in opposite direction of the normal
                d = (center - slicing_planes[s][0]).dot(normals[s])
                if d < 0:
                    break

            elif 0 < s < len(slicing_planes):
                # slice between two planes
                d1 = (center - slicing_planes[s - 1][0]).dot(normals[s - 1])
                d2 = (center - slicing_planes[s][0]).dot(normals[s])
                if d1 >= 0 and d2 < 0:
                    break

            else:
                # slice with one plane, in direction of the normal
                d = (center - slicing_planes[s - 1][0]).dot(normals[s - 1])
                if d > 0:
                    break

        idx_lists[s].append(ind)
        point_lists[s].append(
            center
            - (center - midplane_points[s]).dot(midplane_normals[s])
            * midplane_normals[s]
        )
        density_lists[s].append(density)

    points = []
    densities = []
    for i in range(len(idx_lists)):
        points_array, density_array = _shape_slice(
            point_lists[i], density_lists[i], midplane_normals[i]
        )
        points.append(points_array)
        densities.append(density_array)

    return points, densities


def _shape_slice(points: NDArray, density, normal: NDArray):
    """Arrange lists of coordinates and density values into 2D arrays."""
    n = np.cross(np.array([0, 0, 1]), normal)
    n1 = n[0]
    n2 = n[1]
    n3 = n[2]
    theta = np.arccos(np.dot(normal, [0, 0, 1]))
    Rn = np.array(
        [
            [
                np.cos(theta) + (n1**2) * (1 - np.cos(theta)),
                n1 * n2 * (1 - np.cos(theta)) - n3 * np.sin(theta),
                n1 * n3 * (1 - np.cos(theta)) + n2 * np.sin(theta),
            ],
            [
                n1 * n2 * (1 - np.cos(theta)) + n3 * np.sin(theta),
                np.cos(theta) + (n2**2) * (1 - np.cos(theta)),
                n2 * n3 * (1 - np.cos(theta)) - n1 * np.sin(theta),
            ],
            [
                n1 * n3 * (1 - np.cos(theta)) - n2 * np.sin(theta),
                n2 * n3 * (1 - np.cos(theta)) + n1 * np.sin(theta),
                np.cos(theta) + (n3**2) * (1 - np.cos(theta)),
            ],
        ]
    )
    x_prime = Rn @ np.array([1, 0, 0])
    y_prime = Rn @ np.array([0, 1, 0])

    # project points onto y' and group indices of projected points
    point_list = []
    density_list = []

    for _, yp_group in groupby(
        sorted(zip(points, density), key=lambda y: y[0].dot(y_prime)),
        key=lambda g: g[0].dot(y_prime),
    ):
        # project points onto x' and sort indices of projected points
        p = []
        d = []
        yp_list = sorted(list(yp_group), key=lambda g: g[0].dot(x_prime))

        for _, xp_group in groupby(yp_list, key=lambda x: x[0].dot(x_prime)):
            xp_list = list(xp_group)
            p.append(list(xp_list)[0][0])
            d.append(np.mean(np.array(list(zip(*xp_list))[1])))
        point_list.append(p)
        density_list.append(np.array(d))

    m = max([len(p) for p in point_list])
    n = len(point_list)
    points_array = np.zeros((m, n, 3))
    density_array = np.zeros((m, n))

    for j in range(n):
        i = int((m - len(point_list[j])) / 2)
        points_array[i : i + len(point_list[j]), j, :] = point_list[j]
        density_array[i : i + len(point_list[j]), j] = density_list[j]

    return points_array, density_array


def generate_planes_by_axis(
    axis: NDArray,
    distances: NDArray,
    origin: NDArray,
) -> List[Tuple[NDArray, NDArray]]:
    """Define slicing planes at specified intervals along a specified axis
    relative to the origin of the grid."""
    return [(origin + axis * d, axis) for d in distances]


def density_origin(density_grid: Grid) -> NDArray:
    """Find the origin of the grid."""
    return density_grid.origin.round(decimals=10)


def density_point_boundaries(density_grid: Grid) -> List[NDArray]:
    """Find the furthest point from origin of the grid."""
    return density_grid.grid.shape * density_grid.delta.round(
        decimals=10
    ) + density_origin(density_grid)


def crop_slices(
    points: list[NDArray],
    densities: list[NDArray],
    coordinate_bounds: list[list[NDArray]],
):
    cropped_points = []
    cropped_densities = []
    """Crops point and density slice arrays by user-specified 2D coordinates."""
    if len(coordinate_bounds) < len(points):
        coordinate_bounds  *= len(points)
    for i, (point_array, density_array) in enumerate(zip(points, densities)):
        indexes = [
        np.argmin(norm(point_array - c)) for c in coordinate_bounds[i]
    ]
        cropped_points.append(point_array[indexes[:2], indexes[2:]])
        cropped_densities.append(density_array[indexes[:2], indexes[2:]])
        
    return cropped_points, cropped_densities


def visualize_slicing_plane(point: NDArray, normal: NDArray) -> None:
    c = -point.dot(normal / norm(normal))
    x = range(20)
    y = range(20)
    z = (-normal[0] * x - normal[1] * y - c) / normal[2]

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    plt.show()
    return
