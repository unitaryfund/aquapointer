# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from qiskit.primitives import Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit

from scipy.optimize import minimize
from src.qubo_utils import ising_energy

class VQE:
    def __init__(self, qubo: np.ndarray, ansatz: QuantumCircuit, ising_ham: SparsePauliOp, sampler: Sampler, params: np.ndarray | None) -> None:
        self.qubo = qubo
        self.ansatz = ansatz
        self.ising_ham = ising_ham
        self.sampler = sampler

        if params!=None:
            self.params = params
        else:
            self.params = np.array([np.random.random()]*self.ansatz.num_parameters)

    def run(self, alpha: float, method="COBYLA"):
        res = minimize(self.cvar_energy, self.params, args=(alpha, ), method=method, options={'disp': False} )
        self.params = res.x
        return res
       
    def _compute_cvar(self, probabilities: np.ndarray, values: np.ndarray, confidence_level: float) -> float:
        """
        Compute Conditional Value at Risk (CVaR) for given probabilities, values, and confidence level.

        Parameters:
        - probabilities: List or array of probabilities
        - values: List or array of corresponding values
        - confidence_level: Confidence level (e.g., 0.95 for 95% confidence)

        Returns:
        - CVaR
        """

        # Sort values in ascending order
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_probabilities = probabilities[sorted_indices]

        # Find the index where the cumulative probability exceeds the confidence level
        cumulative_prob = np.cumsum(sorted_probabilities)
        exceed_index = np.argmax(cumulative_prob >= confidence_level)

        # Calculate CVaR
        cvar_values = sorted_values[:exceed_index + 1]
        cvar_probabilities = sorted_probabilities[:exceed_index + 1]

        cvar = np.sum(cvar_values * cvar_probabilities) / np.sum(cvar_probabilities)

        return cvar
    
    def cvar_energy(self, params: np.ndarray, alpha: float) -> float:
        """ 
        Function that takes parameters to bind to the ansatz and confidence level
        alpha, to compute the cvar energy (by sampling the ansatz and computing cvar).
        
        Attributes:
        - params: list/array of probabilities
        - alpha: confidence level
        
        Returns:
        - CVaR energy
        """

        qc = self.ansatz.assign_parameters(params)
        # Add measurements to our circuit
        qc.measure_all()
        # Sample ansatz
        samp_dist = self.sampler.run(qc, shots=int(1e4)).result().quasi_dists[0]

        samp_dist_binary=samp_dist.binary_probabilities()

        prob_energy = []
        for key in samp_dist_binary.keys():
            reverse_key = key[::-1] #qiskit reverses measured bitstrings
            keynot = [(int(b)+1)%2 for b in reverse_key]
            prob_energy.append([samp_dist_binary[key], ising_energy(keynot, self.qubo)])

        prob_energy = np.array(prob_energy)
        cvar_energy = self._compute_cvar(prob_energy[:, 0], prob_energy[:, 1], alpha)

        return cvar_energy
