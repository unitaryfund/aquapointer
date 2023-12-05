import numpy as np
import numpy as np
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit

from scipy.optimize import minimize
from src.qubo_utils import ising_energy

class VQE:
    def __init__(self, ansatz: QuantumCircuit, ising_ham: SparsePauliOp, estimator: Estimator, sampler: Sampler) -> None:
        self.ansatz = ansatz
        self.ising_ham = ising_ham
        self.estimator = estimator
        self.sampler = sampler

        self.params = np.array([np.random.random()]*self.ansatz.num_parameters)

    # to add in the future: store sampled bitstrings while computing costs in Estimator, maybe there are good solutions in there
    def cost_func(self, params: np.ndarray, ansatz: QuantumCircuit, ising_ham: SparsePauliOp, estimator: Estimator) -> float:
        """Return estimate of energy from estimator

        Parameters:
            params (ndarray): Array of ansatz parameters
            ansatz (QuantumCircuit): Parameterized ansatz circuit
            hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
            estimator (Estimator): Estimator primitive instance

        Returns:
            float: Energy estimate
        """
        result = estimator.run(ansatz, ising_ham, parameter_values=params).result()
        cost = result.values[0]
        return cost
    
    def run(self, params: np.ndarray):
        res = minimize(self.cost_func, params, args=(self.ansatz, self.ising_ham, self.estimator), method="COBYLA")
        return res


    def iterate(self, iterations: int) -> np.ndarray:

        # initial run
        res = self.run(self.params)
        params = res.x

        for _ in range(iterations):
            res = self.run(params)
            params = res.x
        self.params = params
    
    def average_energy(self, params: np.ndarray, qubo: np.ndarray) -> float:

        qc = self.ansatz.assign_parameters(params)
        # Add measurements to our circuit
        qc.measure_all()
        # Sample ansatz at optimal parameters
        samp_dist = self.sampler.run(qc, shots=int(1e4)).result().quasi_dists[0]

        samp_dist_binary=samp_dist.binary_probabilities()

        solutions_list=[]
        prob_energy = []
        for key in samp_dist_binary.keys():
            reverse_key = key[::-1] #qiskit reverses measured bitstrings
            keynot = [(int(b)+1)%2 for b in reverse_key]
            solutions_list.append([tuple(keynot), samp_dist_binary[key], ising_energy(keynot, qubo)])
            prob_energy.append([samp_dist_binary[key], ising_energy(keynot, qubo)])

        prob_energy = np.array(prob_energy)
        avg_energy = np.sum([x[0]*x[1] for x in prob_energy])

        return avg_energy
