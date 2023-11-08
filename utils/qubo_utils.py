import numpy as np
from qiskit.quantum_info import SparsePauliOp

def gaussian(var, m, x, y):
    """
    Returns the value at point (`x`,`y`) of a sum of isotropic normal
    distributions centered at `mean[0]`, `mean[1]`, ...
    and variance `var`
    """
    res = 0
    return np.exp(-((x-m[0])**2 +(y-m[1])**2)/(2*var))/(2*np.pi*var)

def gaussian_mixture(shape, var, means):
    res = np.zeros(shape)
    for i in range(len(res)):
        for j in range(len(res[0])):
            for mean in means:
                res[j,i] += gaussian(var, mean, i, j)
    return res

def gamma(density, m, var, amp=1):
    Nm = amp*gaussian_mixture(density.shape, var, [m])
    res = 0
    for i in range(len(Nm[0])):
        for j in range(len(Nm)):
            res += 2*Nm[i,j]*density[i,j]
            res -= Nm[i,j]*Nm[i,j]
    return res

def Vij(shape: tuple[int, int], mean1: tuple[float, float], mean2: tuple[float, float], var: float, amp: float) -> float:
    Nm1 = amp*gaussian_mixture(shape, var, [mean1])
    Nm2 = amp*gaussian_mixture(shape, var, [mean2])
    res = 0
    for i in range(len(Nm1[0])):
        for j in range(len(Nm1)):
            res += Nm1[i,j]*Nm2[i,j]
    return res

def get_qubo_matrices(densities: list[np.ndarray], rescaled_positions: list[np.ndarray]) -> list[np.ndarray]:
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
                #compute V_ij (the quadratic coefficients in the QUBO)
                #if i!=j:
                qubo[i][j] = Vij(shape=density.shape, 
                                mean1=tuple(rescaled_pos[i]), 
                                mean2=tuple(rescaled_pos[j]), 
                                var=variance, 
                                amp=amplitude)
                qubo[j][i] = qubo[i][j]
        qubo_matrices.append(qubo)
    
    return qubo_matrices

#Ising energy function (objective function to minimize)
def Ising_energy(assignment: np.ndarray, qubo: np.ndarray) -> float:
    return np.transpose(assignment) @ qubo @ assignment

#for the classical brutef-force approach
def bitfield(n: int, L: int) -> list[int]:
    result = np.binary_repr(n, L)
    return [int(digit) for digit in result]

#find for a given qubo matrix the optimal bitstring that minimizes energy by going over all possible bitstrings.
def find_optimum(qubo: np.ndarray) -> tuple[list[int], float]:
    shape = qubo.shape
    L = shape[0]

    min_energy = np.inf

    for n in range(2**L):
        b = bitfield(n=n, L=L)
        energy = Ising_energy(assignment=b, qubo=qubo)
        if energy < min_energy:
            min_energy = energy
            optimal_b = b
    
    return optimal_b, min_energy

def sparse_sigmaz_string(length: int, pos: list[int]) -> str:
    sparse_sigmaz_str=""
    for i in range(length):
        if i in pos:
            sparse_sigmaz_str+="Z"
        else:
            sparse_sigmaz_str+="I"
    return sparse_sigmaz_str

def get_Ising_hamiltonian(qubo: np.ndarray) -> SparsePauliOp:
    #the constant term (coefficient in front of II...I)
    coeff_id=0.5*np.sum([qubo[i][i] for i in range(len(qubo))])+0.5*np.sum([np.sum([qubo[i][j] for j in range(i+1,len(qubo))]) for i in range(len(qubo))])

    #the linear terms (coefficient in front of I ... I sigma^z_i I ... I)
    coeff_linear=[0.5*np.sum([qubo[i][j] for j in range(len(qubo))]) for i in range(len(qubo))]

    #the quadratic terms (coefficient in front of I ... I sigma^z_i I ... I sigma^z_j I ... I)
    coeff_quadratic=0.25*qubo

    #creat the list of sparse pauli operators and coefficients
    sparse_list=[(sparse_sigmaz_string(len(qubo), []), coeff_id)]

    for i in range(len(qubo)):
        sparse_list.append((sparse_sigmaz_string(len(qubo), [i]), coeff_linear[i]))
        for j in range(len(qubo)):
            if i!=j:
                sparse_list.append((sparse_sigmaz_string(len(qubo), [i, j]), coeff_quadratic[i][j]))

    I2=SparsePauliOp.from_list(sparse_list)

    return I2

def get_most_likely(msrmnts: dict) -> list[int]:
    b = max(msrmnts, key=msrmnts.get)
    bnot = [(int(x)+1)%2 for x in b]
    
    return bnot
        