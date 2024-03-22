# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from pulser import Pulse, Sequence, Register
from pulser.devices import IroiseMVP, MockDevice
from pulser.channels import Rydberg
from pulser_simulation import Simulation
import matplotlib.pyplot as plt
import qutip


def rescale_register(pos, density, pad=0.2):
    """Given a register with qubits in position `pos`, rescale the positions
    so that they correspond as much as possible to the indices of the density.
    This means that the density value for the k-th qubit is found as:
    density[i,j] where i = int(rescaled_pos[k][1]), j = int(rescaled_pos[k][0])"""
    reg_xlim = (np.min(pos[:, 0]), np.max(pos[:, 0]))
    reg_ylim = (np.min(pos[:, 1]), np.max(pos[:, 1]))
    reg_len = (reg_xlim[1] - reg_xlim[0], reg_ylim[1] - reg_ylim[0])
    density_xlim = (0, density.shape[0] - 1 - pad)
    density_ylim = (0, density.shape[1] - 1 - pad)
    density_len = (density_xlim[1] - density_xlim[0], density_ylim[1] - density_ylim[0])
    shift = (density_xlim[0] - reg_xlim[0], density_ylim[0] - reg_ylim[0])
    scale = min(density_len[0] / reg_len[0], density_len[1] / reg_len[1])
    return scale * (pos + shift)


def density_of_qubit(k, density, rescaled_pos):
    """Returns the density value corresponding to the k-th qubit"""
    i = int(np.round(rescaled_pos[k][1]))
    j = int(np.round(rescaled_pos[k][0]))
    return density[i, j]


def move_blocks(positions, indices, movement, unit_vectors):
    """
    `unit_vectors` is the directional unit vectors of the lattice
    `movement` is a tuple (nx,ny) where nx is the number of
    positions to move in direction `unit_vectors[0]`
    and similarly for ny
    the qubits indexed by `indices` are the onse to be moved"""
    nx, ny = movement
    for i in range(len(positions)):
        if i not in indices:
            continue
        positions[i] += nx * unit_vectors[0] + ny * unit_vectors[1]
    return positions


def decimate_register(
    full_pos,
    rescaled_full_pos,
    density,
    threshold,
    center=None,
    movement=None,
    indices=None,
    unit_vectors=None,
):
    """Given a full array of positions, keeps only the positions corresponding to a
    high enough value of the density. If center is given, then the register and
    the array of positions is shifted so that the central position is in `center`,
    but the rescaled positions are not shifted"""
    max_density = np.max(density)
    pos = []
    rescaled_pos = []
    for k, node in enumerate(full_pos):
        if density_of_qubit(k, density, rescaled_full_pos) / max_density > threshold:
            pos.append(node)
            rescaled_pos.append(rescaled_full_pos[k])
    pos = np.array(pos)
    if movement is not None:
        pos = move_blocks(pos, indices, movement, unit_vectors)
        reg = Register.from_coordinates(pos)
    rescaled_pos = np.array(rescaled_pos)
    if center is not None:
        middle = select_middle_coordinate(pos, full_pos)
        pos = pos - middle + center
    reg = Register.from_coordinates(pos, center=False)
    return (reg, pos, rescaled_pos)


def select_middle_coordinate(coords, full_coords):
    ideal_center = np.average(coords, axis=0)
    d = 1e10
    for c in full_coords:
        if np.linalg.norm(c - ideal_center) < d:
            d = np.linalg.norm(c - ideal_center)
            real_center = c
    return real_center
