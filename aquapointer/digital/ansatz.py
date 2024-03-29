import numpy as np

from qiskit.circuit.library import QAOAAnsatz, TwoLocal
from qiskit import QuantumCircuit
from scipy.optimize import minimize

from aquapointer.digital.qubo_utils import get_ising_hamiltonian

def quadratic_function(x, sigma, mu):
    return x.T @ sigma @ x + mu.T @ x

def QAOA_ansatz(qubo, warm_start=True, reps=1):
    hamiltonian = get_ising_hamiltonian(qubo=qubo)
    num_qubits = len(qubo)

    if warm_start:
        bounds = [(0, 1)] * len(qubo)
        x0 = np.zeros(len(qubo))

        result = minimize(quadratic_function, x0, args=(qubo - np.diag(np.diagonal(qubo)), np.diagonal(qubo)), bounds=bounds)
        x_stars = result.x
        initial_state = QuantumCircuit(num_qubits)
        thetas = [2 * np.arcsin(np.sqrt(x_star)) for x_star in x_stars]


        for idx, theta in enumerate(thetas):
            initial_state.ry(theta, idx)

        # QAOA ansatz circuit
        qaoa_ansatz = QAOAAnsatz(hamiltonian, reps=reps, initial_state=initial_state)

        return qaoa_ansatz
    
    else:
        qaoa_ansatz = QAOAAnsatz(hamiltonian, reps=reps, initial_state=initial_state)

        return qaoa_ansatz
    


