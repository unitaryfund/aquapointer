# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from itertools import groupby, product
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gridData import Grid
from numpy.linalg import norm
from numpy.typing import NDArray
from aquapointer.density_canvas.DensityCanvas import DensityCanvas


def density_file_to_grid(filename: str) -> Grid:
    """Load density file as a ``Grid`` object."""
    return Grid(filename)


def density_slices_by_plane_and_offsets(
    density_grid: Grid, points: NDArray, offsets: List[float]
) -> List[DensityCanvas]:
    """Slice 3D density grid at specified intervals along a specified axis
    and flatten slices into 2D density arrays positioned at each midplane."""
    slicing_planes = [points]
    u = (points[1, :] - points[0, :]) / norm(points[1, :] - points[0, :])
    v = (points[2, :] - points[0, :]) / norm(points[2, :] - points[0, :])
    for f in offsets:
        slicing_planes.append(points + f * np.cross(u, v))

    return density_slices_by_planes(density_grid, slicing_planes)


def density_slices_by_planes(
    density_grid: Grid,
    slicing_planes: List[NDArray],
) -> List[DensityCanvas]:
    """Slice 3D density grid by planes specified by a list of point and axis
    pairs and flatten slices into 2D density arrays positioned at each
    midplane."""
    idx_lists = [[] for _ in range(len(slicing_planes) + 1)]
    point_lists = [[] for _ in range(len(slicing_planes) + 1)]
    density_lists = [[] for _ in range(len(slicing_planes) + 1)]
    normals = []
    for s in slicing_planes:
        u = (s[1, :] - s[0, :]) / norm(s[1, :] - s[0, :])
        v = (s[2, :] - s[0, :]) / norm(s[2, :] - s[0, :])
        n = np.cross(u, v)
        if np.dot(n, np.ones(3,)) >= 0:
            normals.append(n / norm(n))
        else:
            normals.append(-n / norm(n))

    origin = density_origin(density_grid)
    endpoint = density_point_boundaries(density_grid)

    midplane_points = (
        [(origin + slicing_planes[0][0, :]) / 2]
        + [
            (slicing_planes[s][0, :] + slicing_planes[s + 1][0, :]) / 2
            for s in range(len(slicing_planes) - 1)
        ]
        + [
            slicing_planes[-1][0, :]
            + normals[-1][1]
            * (endpoint - slicing_planes[-1][0, :]).dot(normals[-1][1])
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
                d = (center - slicing_planes[s][0, :]).dot(normals[s])
                if d < 0:
                    break

            elif 0 < s < len(slicing_planes):
                # slice between two planes
                d1 = (center - slicing_planes[s - 1][0, :]).dot(normals[s - 1])
                d2 = (center - slicing_planes[s][0, :]).dot(normals[s])
                if d1 >= 0 and d2 < 0:
                    break

            else:
                # slice with one plane, in direction of the normal
                d = (center - slicing_planes[s - 1][0, :]).dot(normals[s - 1])
                if d > 0:
                    break

        idx_lists[s].append(ind)
        if s == 0:
            a = origin
            b = slicing_planes[s][0, :]
        elif 0 < s < len(slicing_planes):
            a = slicing_planes[s - 1][0, :]
            b = slicing_planes[s][0, :]
        else:
            a = slicing_planes[s - 1][0, :]
            b = endpoint

        point_lists[s].append(
            center
            - (center - midplane_points[s]).dot(midplane_normals[s])
            * midplane_normals[s]
        )
        density_lists[s].append(density)

    density_canvases = []
    for i in range(len(idx_lists)):
        shape = [g for d, g in enumerate(density_grid.grid.shape) if d !=np.argmax(midplane_normals[i])]
        density_array, slice_og, slice_endpt = _shape_slice(
            point_lists[i], density_lists[i], midplane_normals[i], shape,
        )
        length_x = slice_endpt[0] - slice_og[0]
        length_y = slice_endpt[1] - slice_og[1]
        dc = DensityCanvas(origin, length_x, length_y, density_array.shape[0], density_array.shape[1])
        dc.set_density_from_slice(density_array.transpose())
        dc.set_canvas_rotation(_generate_slice_rotation_matrix(midplane_normals[i]))
        dc.set_canvas_ref_point(slice_og)
        density_canvases.append(dc)
    return density_canvases


def _generate_slice_rotation_matrix(normal: NDArray):
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
    return Rn

def _shape_slice(points: NDArray, density, normal: NDArray, shape):
    """Arrange lists of coordinates and density values into 2D arrays."""
    Rn = _generate_slice_rotation_matrix(normal)
    x_prime = Rn @ np.array([1, 0, 0])
    y_prime = Rn @ np.array([0, 1, 0])

    # project points onto x', y'
    points = [np.array([p.dot(x_prime), p.dot(y_prime)]) for p in points]
    density_list = []
    point_list = []
    
    if abs(normal.dot([0, 0, 1])) not in [0.0, 1.0]:
        x_sort = sorted(points, key=lambda x: x[0])
        y_sort = sorted(points, key=lambda y: y[1])
        dx = (x_sort[-1][0] - x_sort[0][0]) / shape[0]
        dy = (y_sort[-1][1] - y_sort[0])[1] / shape[1]
        snapped_pts = []
        for p in points:
            ix = p[0] // dx
            iy = p[1] // dy
            snapped_pts.append(np.array([ix * dx, iy * dy]) + np.array([x_sort[0][0], y_sort[0][1]]))
        points = snapped_pts

    for _, yp_group in groupby(
        sorted(zip(points, density), key=lambda y: y[0][1]),
        key=lambda g: g[0][1],
    ):
        # project points onto x' and sort indices of projected points
        p = []
        d = []
        yp_list = sorted(list(yp_group), key=lambda g: g[0][0])

        for _, xp_group in groupby(yp_list, key=lambda x: x[0][0]):
            xp_list = list(xp_group)
            p.append(list(zip(*xp_list))[0][0])
            d.append(np.mean(np.array(list(zip(*xp_list))[1])))
        point_list.append(p)
        density_list.append(np.array(d))
        m = max([len(density) for density in density_list])
        n = len(density_list)
    density_array = np.zeros((m, n))
    for j in range(n):
        i = int((m - len(density_list[j])) / 2)
        density_array[i : i + len(density_list[j]), j] = density_list[j]

    return density_array, point_list[0][0], point_list[-1][-1]


def density_origin(density_grid: Grid) -> NDArray:
    """Find the origin of the grid."""
    return density_grid.origin.round(decimals=10)


def density_point_boundaries(density_grid: Grid) -> List[NDArray]:
    """Find the furthest point from origin of the grid."""
    return density_grid.grid.shape * density_grid.delta.round(
        decimals=10
    ) + density_origin(density_grid)
