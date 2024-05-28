# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from qiskit.primitives import BackendSampler
from qiskit import QuantumCircuit
from scipy.optimize import minimize
from aquapointer.digital.qubo_utils import ising_energy

class VQE:
    def __init__(self, qubo: np.ndarray, ansatz: QuantumCircuit, sampler: BackendSampler, params: np.ndarray = None, track_opt_solution: bool = True) -> None:
        """
        Initializes the VQE instance.

        Args:
            qubo (np.ndarray): The QUBO matrix for the problem.
            ansatz (QuantumCircuit): The quantum circuit used as the ansatz.
            sampler (BackendSampler): The sampler backend for running quantum circuits.
            params (np.ndarray, optional): Initial parameters for the ansatz. If None, random parameters are used.
            track_opt_solution (bool, optional): If True, track the probability of the optimal solution. Default is True.
        """
        self.qubo = qubo
        self.ansatz = ansatz
        self.sampler = sampler
        self.params = params if params is not None else np.random.random(ansatz.num_parameters)
        self.r = 0.1
        self.track_opt_solution = track_opt_solution
        self.history = []

    def run(self, alpha: float, maxiter: int, method: str = "COBYLA"):
        """
        Runs the optimization.

        Args:
            alpha (float): Confidence level.
            maxiter (int): Maximum number of iterations.
            method (str, optional): Method for updating parameters. Default is "COBYLA".
        
        Returns:
            OptimizeResult: Result from running scipy.optimize.minimize.
        """
        res = minimize(self.cvar_energy, self.params, args=(alpha,), method=method, tol=1e-8, options={"maxiter": maxiter})
        self.params = res.x
        return res

    def _compute_cvar(self, probabilities: np.ndarray, values: np.ndarray, confidence_level: float) -> float:
        """
        Compute Conditional Value at Risk (CVaR) for given probabilities, values, and confidence level.

        Args:
            probabilities (np.ndarray): Array of probabilities.
            values (np.ndarray): Array of corresponding values.
            confidence_level (float): Confidence level (e.g., 0.95 for 95% confidence).

        Returns:
            float: Computed CVaR.
        """
        # Sort values in ascending order along with their probabilities
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
        r""" Function that takes parameters to bind to the ansatz and confidence level
        alpha, to compute the cvar energy (by sampling the ansatz and computing cvar).
        
        Args:
            params: numpy array of parameters for the ansatz.
            alpha: Confidence level.
        
        Returns:
            CVaR energy.
        """

        qc = self.ansatz.assign_parameters(params)
        # Add measurements to our circuit
        qc.measure_all()
        # Sample ansatz
        samp_dist = self.sampler.run(qc, shots=int(1e4)).result().quasi_dists[0]

        samp_dist_binary=samp_dist.binary_probabilities()

        correct_dist = {}
        for key in samp_dist_binary.keys():
            reverse_key = key[::-1]
            keynot = [(int(b)+1)%2 for b in reverse_key]
            correct_dist[''.join(map(str, keynot))] = samp_dist_binary[key]

        prob_energy = []
        bitstrings = []
        for key in correct_dist.keys():
            key_np = np.fromiter(map(int, key), dtype=int)
            prob_energy.append([correct_dist[key], ising_energy(key_np, self.qubo)])
            bitstrings.append(key)

        bitstrings = np.array(bitstrings)
        prob_energy = np.array(prob_energy)

        sorted_indices = np.argsort(prob_energy[:, 1])
        sorted_keys = bitstrings[sorted_indices]
        sorted_probs = prob_energy[:, 0][sorted_indices]
        sorted_values = prob_energy[:, 1][sorted_indices]

        opt_energy = sorted_values[0]
        if opt_energy < 0:
            factor = 1 - self.r
        else:
            factor = 1 + self.r

        # now obtain the energies that are 10% close to optimal
        # eps_rel_energies = []
        # for i, val in enumerate(sorted_values):
        #     if val <= factor * opt_energy: 
        #         eps_rel_energies.append(i)

        # I want the average energy for now
        eps_rel_energies = []
        for i, val in enumerate(sorted_values):
            eps_rel_energies.append(i)

        opt_b = sorted_keys[0]
        opt_prob = sorted_probs[0]
        prob_energy = np.array(prob_energy)
        cvar_energy = self._compute_cvar(prob_energy[:, 0], prob_energy[:, 1], alpha)

        top_opt_prob = np.sum(sorted_probs[eps_rel_energies])
        avg_top_energies = np.mean(sorted_probs[eps_rel_energies]*sorted_values[eps_rel_energies])

        # save intermediate optimal bitstring and energy to self.history
        if self.track_opt_solution:
            self.history.append([opt_b, np.round(opt_prob,7), opt_energy])
        else:
            self.history.append([np.round(top_opt_prob, 7), avg_top_energies])



        return cvar_energy
