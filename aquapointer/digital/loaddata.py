# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pickle
# from aquapointer.slicing import density_file_to_grid, density_slices_by_axis, find_density_origin, find_density_point_boundaries

from pathlib import Path

BASE_PATH = str(Path.cwd().parent)
DENS_DIR = "/data/MUP1/MUP1_logfilter8_slices/"
PP_DIR = "/data/MUP1/MUP1_logfilter8_points/"
REG_DIR = "/registers/"

RISM3D_DIR = "../data/3D-RISM_densities/"

class LoadData:

    def __init__(self, protein: str) -> None:
        self.d_list = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

        if protein == 'MUP1':
            self.densities = self.load_density_slices(path=BASE_PATH + DENS_DIR)
            self.plane_points = self.load_plane_points(path=BASE_PATH + PP_DIR)
            
            self.register_positions = self.load_register_positions(path=BASE_PATH + REG_DIR)
            self.rescaled_register_positions = self.load_rescaled_register_positions(path=BASE_PATH + REG_DIR)
            
        else:
            with open(REG_DIR + protein + '/rescaled_positions.pkl', 'rb') as handle:
                self.rescaled_register_positions = pickle.load(handle)
            with open(RISM3D_DIR + protein + '/slices.pkl', 'rb') as handle:
                self.densities = pickle.load(handle)
            

    def load_density_slices(self, path: str) -> list[np.ndarray]:
        r"""The 3D-RISM density slices are saved as pickled files in the folder MUP1.
        They are indexed by a number (see d_list) which represents the distance in Angstrom
        from the central slice. This function loads the files.

        Args:
            path: Path to 3D-RISM density slices files.

        Returns:
            List of numpy arrays containing the slices.
        """
        basename = "_density_slice_MUP1_logfilter8.p"
        densities = []
        for d in self.d_list:
            filename = path + f"d{d}" + basename
            with open(filename, 'rb') as file_in:
                densities.append(pickle.load(file_in))
                
        return densities


    def load_plane_points(self, path: str) -> list[np.ndarray]:
        r"""Load slice coordinates (these are 3D coordinates in
        angstroms, they are needed at the very end to map
        excited qubits to positions in the protein cavity).

        Args:
            path: Path to plane points files.
        
        Returns:
            List of numpy arrays containing the plane points.
        """
        basename = "_plane_points_MUP1.p"
        points = []
        for d in self.d_list:
            filename = path + f"d{d}" + basename
            with open(filename, 'rb') as file_in:
                points.append(pickle.load(file_in))
        
        return points

    def load_register_positions(self, path: str) -> list[np.ndarray]:
        r"""The register associated to each slice can be found in the folder nb/registers.
        - position_<#>.npy: the positions of the qubits in micrometers, as if they were in the QPU

        Args:
            path: Path to register positions files.

        Returns:
            List of numpy arrays containing the register positions.
        """

        basename = "position_"
        positions = []
        for i in range(len(self.d_list)):
            filename = path + basename + f"{i}.npy"
            with open(filename, 'rb') as file_in:
                pos = np.load(file_in)
            positions.append(pos)
        
        return positions

    def load_rescaled_register_positions(self, path: str) -> list[np.ndarray]:
        r"""The register associated to each slice can be found in the folder nb/registers.
        - rescaled_position_<#>.npy: the positions of the qubits on the same scale as the density slices

        Args:
            path: Path to register positions files.

        Returns:
            List of numpy arrays containing the rescaled register positions.
        """
        basename = "rescaled_position_"
        rescaled_positions = []
        for i in range(len(self.d_list)):
            filename = path + basename+f"{i}.npy"
            with open(filename, 'rb') as file_in:
                res_pos = np.load(file_in)
            rescaled_positions.append(res_pos)
        
        return rescaled_positions

