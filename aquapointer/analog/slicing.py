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
    density_3d_array = Grid(filename)
    densities = [density_3d_array[x_slice, y_slice, z_slice]]
    for s in range(len(slicing_planes) - 1):
        plane1 = construct_plane(slicing_planes[s][0], slicing_planes[s][1])
        plane2 = construct_plane(slicing_planes[s + 1][0], slicing_planes[s + 1][1])
        density = density_3d_array[x_slice, y_slice, z_slice]
        densities.append(density)
    density = density_3d_array[x_slice, y_slice, z_slice]
    densities.append(density)
    return densities


def density_slices_by_axis(
    filename: str, axis: NDArray, steps: NDArray
) -> List[NDArray]:
    slicing_planes = planes_along_axis(axis, steps)
    return density_slices_by_plane(filename, slicing_planes)


def planes_along_axis(axis: NDArray, steps: NDArray) -> List[Tuple[NDArray, NDArray]]:
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
