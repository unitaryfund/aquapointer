# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from aquapointer.digital.loaddata import LoadData
from aquapointer.digital.qubo_utils import gaussian, gaussian_mixture, gamma, Vij

class Qubo:

    def __init__(self, loaddata: LoadData) -> None:
        self.ld = loaddata
        self.qubo_matrices = self.get_qubo_matrices(densities=self.ld.densities, rescaled_positions=self.ld.rescaled_register_positions)
        hamiltonians = [self.get_ising_hamiltonian(qubo=qubo) for qubo in self.qubo_matrices]
        self.qubo_hamiltonian_pairs = list(zip(self.qubo_matrices, hamiltonians))

    def get_qubo_matrices(self, densities: list[np.ndarray], rescaled_positions: list[np.ndarray]) -> list[np.ndarray]:
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

    #Ising energy function (objective function to minimize)
    def ising_energy(self, assignment: np.ndarray, qubo: np.ndarray) -> float:
        return np.transpose(assignment) @ qubo @ assignment

    #for the classical brutef-force approach
    def _bitfield(self, n: int, L: int) -> list[int]:
        result = np.binary_repr(n, L)
        return [int(digit) for digit in result]

    #find for a given qubo matrix the optimal bitstring that minimizes energy by going over all possible bitstrings.
    def find_optimum(self, qubo: np.ndarray) -> tuple[str, float]:
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
        sparse_sigmaz_str = ""
        for i in range(length):
            if i in pos:
                sparse_sigmaz_str += "Z"
            else:
                sparse_sigmaz_str += "I"
        return sparse_sigmaz_str

    def get_ising_hamiltonian(self, qubo: np.ndarray) -> SparsePauliOp:
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