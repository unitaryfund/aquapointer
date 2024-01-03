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
    
    def average_energy(self, params: np.ndarray) -> float:

        qc = self.ansatz.assign_parameters(params)
        # Add measurements to our circuit
        qc.measure_all()
        # Sample ansatz at optimal parameters
        samp_dist = self.sampler.run(qc, shots=int(1e4)).result().quasi_dists[0]

        samp_dist_binary=samp_dist.binary_probabilities()

        prob_energy = []
        for key in samp_dist_binary.keys():
            reverse_key = key[::-1] #qiskit reverses measured bitstrings
            keynot = [(int(b)+1)%2 for b in reverse_key]
            prob_energy.append([samp_dist_binary[key], ising_energy(keynot, self.qubo)])

        prob_energy = np.array(prob_energy)
        avg_energy = np.sum([x[0]*x[1] for x in prob_energy])

        return avg_energy
    
    def _compute_cvar(self, probabilities: np.ndarray, values: np.ndarray, alpha: float) -> float:
        """ 
        Auxilliary method to computes CVaR for given probabilities, values, and confidence level.
        
        Attributes:
        - probabilities: list/array of probabilities
        - values: list/array of corresponding values
        - alpha: confidence level
        
        Returns:
        - CVaR
        """
        sorted_indices = np.argsort(values)
        probs = probabilities[sorted_indices]
        vals = values[sorted_indices]
        cvar = 0
        total_prob = 0
        for i, (p, v) in enumerate(zip(probs, vals)):
            done = False
            if p >= alpha - total_prob:
                p = alpha - total_prob
                done = True
            total_prob += p
            cvar += p * v
        cvar /= total_prob
        return cvar
    
    def cvar_energy(self, params: np.ndarray, alpha: float) -> float:
        """ 
        Function that takes parameters to bind to the ansatz and confidence level
        alpha, to compute the cvar energy (by sampling the ansatz and computing cvar).
        
        Attributes:
        - params: list/array of probabilities
        - alpha: confidence level
        
        Returns:
        - cvar energy
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
