import numpy as np
from scipy.special import binom
import itertools as it
from scipy.stats import multivariate_normal

def integer_partitions(n):
    """ Function to calculate integer partitions of n
    (taken from stackoverflow)"""
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]

def partition_eqclasses_size(r):
    """Calculates the equivalence classes and returns their size.
    The equivalence relation is r_i ~ r_j iff r_i = r_j"""
    res = []
    placed = []
    for i in range(len(r)):
        if i in placed:
            continue
        eqclass_i = [r[i]]
        placed.append(i)
        for j in range(i+1, len(r)):
            if j in placed:
                continue
            if r[j] == r[i]:
                eqclass_i.append(r[j])
                placed.append(j)
        res.append(eqclass_i)
    return [len(eq) for eq in res]

def partition_symmetries(r):
    """Calculates the symmetries of partition r=(r_0, r_1, ...)
    which will be used as the denominator of the multiplicity factor"""
    res = 1
    for i in range(len(r)):
        res *= np.math.factorial(r[i])
    eqclasses_size = partition_eqclasses_size(r)
    for i in range(len(eqclasses_size)):
        res *= np.math.factorial(eqclasses_size[i])
    return res

def integral(p, k, indices, powers, base, component, params):
    res = base()**(p-k)
    for idx, pow in zip(indices, powers):
        res *= component(idx, params)**pow
    return float(res)

def Lp_ising_energy(x, p, base, component, params, high=None, low=None):
    """Calculates the Lp ising energy of a set of binary variables x=[x_0, x_1, ..., x_n]
    high and low are optional parameters that can truncate the computation at some
    order, from above or from below"""

    if high is None:
        high = p
    if low is None:
        low = 1

    res = 0
    for k in range(1, p+1):
        fact_k = np.math.factorial(k)
        res_k = 0
        partitions = integer_partitions(k)
        for r in partitions:
            # skip orders that you want to neglect
            if len(r)<low or len(r)>high:
                continue

            # calculate multiplicity of partition
            multiplicity = fact_k/partition_symmetries(r)
            
            # cycle over all permutations
            res_r = 0
            index_permutations = it.permutations(range(len(x)), r=len(r))
            for indices in index_permutations:
                indices = list(indices)
                if np.any(x[indices] == 0):
                    continue
                res_r += integral(p, k, indices, r, base, component, params)
            res_k += multiplicity*res_r
        res += (-1)**k * binom(p, k) * res_k
            
    return res

def Lp_ising_energy_second_order_truncation(x, p, base, component, params):
    """Calculates the Lp ising energy of a set of binary variables x=[x_0, x_1, ..., x_n]
    truncated to second order terms"""
    
    if p//2 != p/2:
        raise ValueError("power p must be even")
    
    res = 0

    # first order
    for i1 in range(len(x)):
        if x[i1] == 0:
            continue
        indices = [i1]
        for k in range(1, p+1):
            partition = [k]
            res += binom(p,k) * (-1)**k * integral(p, k, indices, partition, base, component, params)
        # second order
        for i2 in range(len(x)):
            if i1 == i2:
                continue
            if x[i2] == 0:
                continue

            indices = [i1, i2]
            # k = 2j even
            for j in range(1, p//2+1):
                partition = [j, j]
                binom_even = binom(p, 2*j)
                fact_even = np.math.factorial(2*j)
                res += binom_even * fact_even/(2*np.math.factorial(j)**2) * integral(p, 2*j, indices, partition, base, component, params)
                for l in range(1, j):
                    partition = [j-l, j+l]
                    res += binom_even * fact_even/(np.math.factorial(j-l)*np.math.factorial(j+l)) * integral(p, 2*j, indices, partition, base, component, params)
            # k = 2j+1 odd
            for j in range(1, (p-1)//2+1):
                binom_odd = binom(p, 2*j+1)
                fact_odd = np.math.factorial(2*j+1)
                for l in range(j):
                    partition = [j-l, j+l+1]
                    res -= binom_odd * fact_odd/(np.math.factorial(j-l)*np.math.factorial(j+l+1)) * integral(p, 2*j+1, indices, partition, base, component, params)

    return res

def Lp_ising_energy_first_order_truncation(x, p, base, component, params):
    """Calculates the Lp ising energy of a set of binary variables x=[x_0, x_1, ..., x_n]
    truncated to first order terms"""
    
    if p//2 != p/2:
        print("Error: power p must be even")
        return
    
    res = 0

    # first order
    for i1 in range(len(x)):
        if x[i1] == 0:
            continue
        for k in range(1, p+1):
            res += binom(p,k) * (-1)**k * integral(p, k, [i1], [k], base, component, params)

    return res

def Lp_coefficients(coords, p, base, component, params, high=None, low=None, efficient_qubo=True):
    """Calculates the Lp coefficients and stores them in a dictionary
    high and low are optional parameters that can truncate the computation at some
    order, from above or from below"""

    M = len(coords)

    if high is None:
        high = p
    if low is None:
        low = 1
    
    if high > p:
        print(f"Warning: calculating up to maximum degree {p}")
        high = p
    if low < 1:
        print(f"Warning: calculating up to minimum degree {1}")
        low = 1

    # initialize every coefficient to 0
    coefficients = {}
    for i in range(low, high+1):
        coefficients[i] = {}
        idxs = it.combinations(range(M), i)
        for idx in idxs:
            coefficients[i][idx] = 0


    # calculate coefficients
    for k in range(1, p+1):
        fact_k = np.math.factorial(k)
        partitions = integer_partitions(k)
        for r in partitions:
            # skip orders that you want to neglect
            if len(r)<low or len(r)>high:
                continue

            # calculate multiplicity of partition
            multiplicity = fact_k/partition_symmetries(r)
            
            # cycle over all permutations
            index_permutations = it.permutations(range(M), r=len(r))
            for indices in index_permutations:
                prefactor = (-1)**k * binom(p, k) * multiplicity
                if p==2 and k==2 and efficient_qubo:
                    if len(r)==1:
                        d = np.linalg.norm(coords[indices[0]] - coords[indices[0]])
                    else:
                        d = np.linalg.norm(coords[indices[0]] - coords[indices[1]])
                    value = params[0]*params[0]*np.exp(-d**2/(2*2*params[1]))/(2*np.pi*2*params[1])
                else:
                    value = integral(p, k, list(indices), r, base, component, params)
                coefficients[len(r)][tuple(sorted(indices))] += prefactor*value
                print(k, r, tuple(sorted(indices)), prefactor, value)
    return coefficients

def Lp_coefficients_first_order(coords, p, base, component, params):
    """Calculates the first order Lp coefficients"""

    if p//2 != p/2:
        raise ValueError("power p must be even")

    M = len(coords)

    # calculate first order
    coefficients = []
    for i1 in range(M):
        res = 0
        for k in range(1, p+1):
            prefactor = binom(p,k) * (-1)**k
            value = integral(p, k, [i1], [k], base, component, params)
            res += prefactor * value
        coefficients.append(res)
    return coefficients









