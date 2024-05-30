import numpy as np
from qiskit.circuit.library import QAOAAnsatz, TwoLocal
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from scipy.optimize import minimize
from aquapointer.digital.qubo_utils import get_ising_hamiltonian

class QAOA_ansatz:
    def __init__(self, qubo, warm_start=True, reps=1):
        self.qubo = qubo
        self.warm_start = warm_start
        self.reps = reps
        self.num_qubits = len(qubo)
        self.hamiltonian = get_ising_hamiltonian(qubo=qubo)
        self.qaoa_ansatz = self.create_qaoa_ansatz()

    @staticmethod
    def quadratic_function(x, sigma, mu):
        return x.T @ sigma @ x + mu.T @ x

    def create_initial_state(self):
        bounds = [(0, 1)] * self.num_qubits
        x0 = np.zeros(self.num_qubits)
        result = minimize(
            self.quadratic_function, x0, 
            args=(self.qubo - np.diag(np.diagonal(self.qubo)), np.diagonal(self.qubo)), 
            bounds=bounds
        )
        x_stars = result.x
        initial_state = QuantumCircuit(self.num_qubits)
        thetas = [2 * np.arcsin(np.sqrt(x_star)) for x_star in x_stars]

        for idx, theta in enumerate(thetas):
            initial_state.ry(theta, idx)
        
        return initial_state, thetas

    def create_mixer_hamiltonian(self, thetas):
        beta = Parameter("β")
        mixer_ham = QuantumCircuit(self.num_qubits)
        for idx, theta in enumerate(thetas):
            mixer_ham.ry(-theta, idx)
            mixer_ham.rz(-2 * beta, idx)
            mixer_ham.ry(theta, idx)
        return mixer_ham

    def create_qaoa_ansatz(self):
        if self.warm_start:
            initial_state, thetas = self.create_initial_state()
            # mixer_ham = self.create_mixer_hamiltonian(thetas)
            qaoa_ansatz = QAOAAnsatz(
                self.hamiltonian, 
                reps=self.reps, 
                initial_state=initial_state, 
                # mixer_operator=mixer_ham
            )
        else:
            qaoa_ansatz = QAOAAnsatz(self.hamiltonian, reps=self.reps)
        
        return qaoa_ansatz
