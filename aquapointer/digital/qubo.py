# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from aquapointer.digital.qubo_utils import gaussian, gaussian_mixture, gamma, Vij
from aquapointer.density_canvas.DensityCanvas import DensityCanvas

class Qubo:

    def __init__(self, densities, rescaled_register_positions) -> None:
        
        # self.qubo_matrices = self.get_qubo_matrices(densities=densities, rescaled_positions=rescaled_register_positions)
        self.qubo_matrices = self.get_qubo_matrices_canvas(densities=densities)
        hamiltonians = [self.get_ising_hamiltonian(qubo=qubo) for qubo in self.qubo_matrices]
        self.qubo_hamiltonian_pairs = list(zip(self.qubo_matrices, hamiltonians))

    def get_qubo_matrices_canvas(self, densities: list[np.ndarray]) -> list[np.ndarray]:
        estimated_variance = 50
        estimated_amplitude = 6

        origin = (-20, -20)
        length = 40
        npoints = densities[0].shape[0]
        canvas = DensityCanvas(
            origin=origin,
            length_x=length,
            length_y=length,
            npoints_x=npoints,
            npoints_y=npoints,
        )
        qubo_matrices = []
        for density in densities:    
            canvas.set_density_from_slice(density)
            # canvas.set_poisson_disk_lattice(spacing=(2,10))
            canvas.set_rectangular_lattice(num_x=8, num_y=8, spacing=4)
            canvas.calculate_pubo_coefficients(
                p = 2, #order of the PUBO, p=2 effectively creates a QUBO
                params = [estimated_amplitude, estimated_variance]
            )
            # canvas.decimate_lattice()

            coefficients = canvas._pubo["coeffs"]
            linear = coefficients[1]
            quadratic = coefficients[2]

            qubo = np.zeros((len(linear), len(linear)))
            
            for i, key in enumerate(linear.keys()):
                qubo[i][i] = linear[key]

            for key in quadratic.keys():
                qubo[key] = quadratic[key]
                qubo[key[::-1]] = quadratic[key]

            qubo_matrices.append(qubo)
        
        return qubo_matrices


    def get_qubo_matrices(self, densities: list[np.ndarray], rescaled_positions: list[np.ndarray]) -> list[np.ndarray]:
        r""" Given the density slices and rescaled positions of the registers,
        one can compute the corresponding QUBO matrices.

        Args:
            densities: List of numpy arrays containing the 3D-RISM slices.
            rescaled_positions: List of numpy arrays containing the rescaled positions of the register.
        
        Returns:
            List of numpy arrays containing the QUBO matrices for each slice.        
        
        """        
        variance = 50
        amplitude = 6
        qubo_matrices = []

        for k, density in enumerate(densities):
            rescaled_pos = rescaled_positions[k]
            # use function to calculate the one-body coefficients of the QUBO
            gamma_list = [gamma(density, pos, variance) for pos in rescaled_pos]
            qubo = np.zeros((len(rescaled_pos), len(rescaled_pos)))
            
            for i in range(len(gamma_list)):
                qubo[i][i] = -gamma_list[i]

            for i in range(len(rescaled_pos)):
                for j in range(i+1, len(rescaled_pos)):
                    qubo[i][j] = Vij(shape=density.shape, 
                                    mean1=tuple(rescaled_pos[i]), 
                                    mean2=tuple(rescaled_pos[j]), 
                                    var=variance, 
                                    amp=amplitude)
                    qubo[j][i] = qubo[i][j]
            qubo_matrices.append(qubo)
        
        return qubo_matrices

    def ising_energy(self, assignment: np.ndarray, qubo: np.ndarray) -> float:
        r""" Given a binary string x and a QUBO matrix Q, computes the inner product <x, Qx>.

        Args:   
            assignment: numpy array, 0,1-valued.
            qubo: 2d numpy array.

        Returns:
            Float given by computing the inner product <x, Qx>.
        
        
        """ 
        return np.transpose(assignment) @ qubo @ assignment

    def _bitfield(self, n: int, L: int) -> list[int]:
        result = np.binary_repr(n, L)
        return [int(digit) for digit in result]

    def find_optimum(self, qubo: np.ndarray) -> tuple[str, float]:
        r""" Brute-force approach to solving the QUBO problem: finding the optimal
        bitstring x that minimizes <x, Qx> where Q is the QUBO matrix.

        Args:
            qubo: 2d numpy array.
        
        Returns:
            Tuple of a bitstring and minimal energy (such that <x, Qx> is minimized).   
        
        """ 
        shape = qubo.shape
        L = shape[0]

        min_energy = np.inf

        for n in range(2**L):
            b = self._bitfield(n=n, L=L)
            energy = self.ising_energy(assignment=b, qubo=qubo)
            if energy < min_energy:
                min_energy = energy
                optimal_b = b
        
        sol = ''.join(map(str, optimal_b))        
        return sol, min_energy

    def _sparse_sigmaz_string(self, length: int, pos: list[int]) -> str:
        r""" Given a list positions and integer length, returns a string of the
        given length consisting of a Z at positions from the list of positions
        and I otherwise. This is interpreted as a sparse Pauli operator.

        Args:
            length: Integer indicating the length of the string.
            pos: List of integers for the positions of Z.

        Returns:
            String consisting of I's and Z's.  
                
        """ 
        sparse_sigmaz_str = ""
        for i in range(length):
            if i in pos:
                sparse_sigmaz_str += "Z"
            else:
                sparse_sigmaz_str += "I"
        return sparse_sigmaz_str

    def get_ising_hamiltonian(self, qubo: np.ndarray) -> SparsePauliOp:
        r""" Given a QUBO matrix, one can associate with it a sparse Pauli operator.
        This is done by mapping a binary variable x -> z := (1-x)/2.

        Args:
            qubo: 2d numpy array.
        
        Returns:
            SparsePauliOp corresponding to the QUBO matrix.       
        
        """ 
        #the constant term (coefficient in front of II...I)
        coeff_id = 0.5*np.sum([qubo[i][i] for i in range(len(qubo))])+0.5*np.sum([np.sum([qubo[i][j] for j in range(i+1,len(qubo))]) for i in range(len(qubo))])

        #the linear terms (coefficient in front of I ... I sigma^z_i I ... I)
        coeff_linear = [0.5*np.sum([qubo[i][j] for j in range(len(qubo))]) for i in range(len(qubo))]

        #the quadratic terms (coefficient in front of I ... I sigma^z_i I ... I sigma^z_j I ... I)
        coeff_quadratic = 0.25*qubo

        #creat the list of sparse pauli operators and coefficients
        sparse_list = [(self._sparse_sigmaz_string(len(qubo), []), coeff_id)]

        for i in range(len(qubo)):
            sparse_list.append((self._sparse_sigmaz_string(len(qubo), [i]), coeff_linear[i]))
            for j in range(len(qubo)):
                if i != j:
                    sparse_list.append((self._sparse_sigmaz_string(len(qubo), [i, j]), coeff_quadratic[i][j]))

        hamiltionian = SparsePauliOp.from_list(sparse_list)

        return hamiltionian