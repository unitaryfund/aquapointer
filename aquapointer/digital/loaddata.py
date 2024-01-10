# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pickle

from pathlib import Path

BASE_PATH = str(Path.cwd().parent)
DENS_DIR = "/data/MUP1/MUP1_logfilter8_slices/"
PP_DIR = "/data/MUP1/MUP1_logfilter8_points/"
REG_DIR = "/registers/"

class LoadData:

    def __init__(self) -> None:
        self.d_list = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        self.densities = self.load_density_slices(path=BASE_PATH + DENS_DIR)
        self.plane_points = self.load_plane_points(path=BASE_PATH + PP_DIR)
        self.register_positions = self.load_register_positions(path=BASE_PATH + REG_DIR)
        self.rescaled_register_positions = self.load_rescaled_register_positions(path=BASE_PATH + REG_DIR)

    def load_density_slices(self,path: str) -> list[np.ndarray]:
        # The 3D-RISM density slices are saved as pickled files in the folder MUP1.
        # They are indexed by a number (see d_list) which represents the distance in Angstrom
        # from the central slice.
        basename = "_density_slice_MUP1_logfilter8.p"
        densities = []
        for d in self.d_list:
            filename = path + f"d{d}" + basename
            with open(filename, 'rb') as file_in:
                densities.append(pickle.load(file_in))
                
        return densities

    def load_plane_points(self, path: str) -> list[np.ndarray]:
        # import slice coordinates (these are 3D coordinates in
        # angstroms, they are needed at the very end to map
        # excited qubits to positions in the protein cavity)
        basename = "_plane_points_MUP1.p"
        points = []
        for d in self.d_list:
            filename = path + f"d{d}" + basename
            with open(filename, 'rb') as file_in:
                points.append(pickle.load(file_in))
        
        return points

    def load_register_positions(self, path: str) -> list[np.ndarray]:
        # The register associated to each slide can be found in the folder nb/registers.
        # Two types of files are saved there:
        # - position_<#>.npy: the positions of the qubits in micrometers, as if they were in the QPU
        # - rescaled_position_<#>.npy: the positions of the qubits on the same scale as the density slices

        # import registers
        basename = "position_"
        positions = []
        for i in range(len(self.d_list)):
            filename = path + basename + f"{i}.npy"
            with open(filename, 'rb') as file_in:
                pos = np.load(file_in)
            positions.append(pos)
        
        return positions

    def load_rescaled_register_positions(self, path: str) -> list[np.ndarray]:
        basename = "rescaled_position_"
        rescaled_positions = []
        for i in range(len(self.d_list)):
            filename = path + basename+f"{i}.npy"
            with open(filename, 'rb') as file_in:
                res_pos = np.load(file_in)
            rescaled_positions.append(res_pos)
        
        return rescaled_positions

