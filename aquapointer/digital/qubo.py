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
        """
        Initializes the Qubo instance.

        Args:
            densities (list[np.ndarray]): List of density slices.
            rescaled_register_positions (list[np.ndarray]): List of rescaled positions for the registers.
        """
        self.qubo_matrices = self._get_qubo_matrices(densities, rescaled_register_positions)
        hamiltonians = [self._get_ising_hamiltonian(qubo) for qubo in self.qubo_matrices]
        self.qubo_hamiltonian_pairs = list(zip(self.qubo_matrices, hamiltonians))

    def _get_qubo_matrices(self, densities: list[np.ndarray], rescaled_positions: list[np.ndarray]) -> list[np.ndarray]:
        """
        Computes QUBO matrices from density slices and rescaled positions.

        Args:
            densities (list[np.ndarray]): List of density slices.
            rescaled_positions (list[np.ndarray]): List of rescaled positions for the registers.
        
        Returns:
            list[np.ndarray]: List of QUBO matrices.
        """
        variance = 50
        amplitude = 6
        qubo_matrices = []

        for density, rescaled_pos in zip(densities, rescaled_positions):
            gamma_list = [gamma(density, pos, variance) for pos in rescaled_pos]
            qubo = np.zeros((len(rescaled_pos), len(rescaled_pos)))

            for i, gamma_value in enumerate(gamma_list):
                qubo[i][i] = -gamma_value

            for i in range(len(rescaled_pos)):
                for j in range(i+1, len(rescaled_pos)):
                    interaction = Vij(
                        shape=density.shape,
                        mean1=tuple(rescaled_pos[i]),
                        mean2=tuple(rescaled_pos[j]),
                        var=variance,
                        amp=amplitude
                    )
                    qubo[i][j] = qubo[j][i] = interaction

            qubo_matrices.append(qubo)
        
        return qubo_matrices

    def _get_qubo_matrices_canvas(self, densities: list[np.ndarray]) -> list[np.ndarray]:
        """
        Alternative method to compute QUBO matrices using DensityCanvas.

        Args:
            densities (list[np.ndarray]): List of density slices.
        
        Returns:
            list[np.ndarray]: List of QUBO matrices.
        """
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
            canvas.set_rectangular_lattice(num_x=8, num_y=8, spacing=4)
            canvas.calculate_pubo_coefficients(
                p=2,  # p=2 effectively creates a QUBO
                params=[estimated_amplitude, estimated_variance]
            )

            qubo = self._extract_qubo_from_canvas(canvas)
            qubo_matrices.append(qubo)
        
        return qubo_matrices

    def _extract_qubo_from_canvas(self, canvas: DensityCanvas) -> np.ndarray:
        """
        Extracts QUBO matrix from DensityCanvas.

        Args:
            canvas (DensityCanvas): The DensityCanvas object.
        
        Returns:
            np.ndarray: Extracted QUBO matrix.
        """
        coefficients = canvas._pubo["coeffs"]
        linear = coefficients[1]
        quadratic = coefficients[2]

        qubo = np.zeros((len(linear), len(linear)))
        
        for i, key in enumerate(linear.keys()):
            qubo[i][i] = linear[key]

        for key in quadratic.keys():
            qubo[key] = qubo[key[::-1]] = quadratic[key]

        return qubo

    def ising_energy(self, assignment: np.ndarray, qubo: np.ndarray) -> float:
        """
        Computes the energy for a given assignment and QUBO matrix.

        Args:
            assignment (np.ndarray): Binary string (0,1-valued) as a numpy array.
            qubo (np.ndarray): QUBO matrix.
        
        Returns:
            float: Computed energy.
        """
        return np.dot(assignment.T, np.dot(qubo, assignment))

    def _bitfield(self, n: int, L: int) -> list[int]:
        """
        Converts an integer to a binary bitfield.

        Args:
            n (int): The integer to convert.
            L (int): The length of the bitfield.
        
        Returns:
            list[int]: List of binary digits.
        """
        return [int(digit) for digit in np.binary_repr(n, L)]

    def find_optimum(self, qubo: np.ndarray) -> tuple[str, float]:
        """
        Finds the optimal solution for the QUBO problem using brute force.

        Args:
            qubo (np.ndarray): QUBO matrix.
        
        Returns:
            tuple[str, float]: Optimal bitstring and minimal energy.
        """
        L = qubo.shape[0]
        min_energy = float('inf')
        optimal_b = None

        for n in range(2**L):
            b = self._bitfield(n, L)
            energy = self.ising_energy(np.array(b), qubo)
            if energy < min_energy:
                min_energy = energy
                optimal_b = b
        
        sol = ''.join(map(str, optimal_b))
        return sol, min_energy

    def _sparse_sigmaz_string(self, length: int, positions: list[int]) -> str:
        """
        Creates a sparse Pauli-Z string.

        Args:
            length (int): Length of the string.
            positions (list[int]): Positions of 'Z' in the string.
        
        Returns:
            str: Sparse Pauli-Z string.
        """
        return ''.join('Z' if i in positions else 'I' for i in range(length))

    def _get_ising_hamiltonian(self, qubo: np.ndarray) -> SparsePauliOp:
        """
        Converts a QUBO matrix to an Ising Hamiltonian.

        Args:
            qubo (np.ndarray): QUBO matrix.
        
        Returns:
            SparsePauliOp: Corresponding Ising Hamiltonian.
        """
        coeff_id = 0.5 * np.sum(np.diag(qubo)) + 0.5 * np.sum(np.triu(qubo, 1))
        coeff_linear = [0.5 * np.sum(qubo[i]) for i in range(len(qubo))]
        coeff_quadratic = 0.25 * qubo

        sparse_list = [(self._sparse_sigmaz_string(len(qubo), []), coeff_id)]
        for i in range(len(qubo)):
            sparse_list.append((self._sparse_sigmaz_string(len(qubo), [i]), coeff_linear[i]))
            for j in range(i + 1, len(qubo)):
                sparse_list.append((self._sparse_sigmaz_string(len(qubo), [i, j]), coeff_quadratic[i][j]))

        return SparsePauliOp.from_list(sparse_list)
