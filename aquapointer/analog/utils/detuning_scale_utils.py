# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from pulser import Pulse, Register, Sequence
from pulser.devices import Chadoq2, MockDevice
from pulser.waveforms import BlackmanWaveform, InterpolatedWaveform
from pulser_simulation import Simulation

from aquapointer.analog.utils import benchmark_utils as bmu


def gaussian(var, m, x, y):
    """
    Returns the value at point (`x`,`y`) of a sum of isotropic normal
    distributions centered at `mean[0]`, `mean[1]`, ...
    and variance `var`
    """
    return np.exp(-((x-m[0])**2 +(y-m[1])**2)/(2*var))/(2*np.pi*var)


def gaussian_mixture(shape, var, means):
    res = np.zeros(shape)
    for i in range(len(res)):
        for j in range(len(res[0])):
            for mean in means:
                res[j,i] += gaussian(var, mean, i, j)
    return res


def create_interp_pulse(amp_params, det_params, T):
    """
    Creates a pulse of duratioin `T` by interpolating the points
    in `amp_params` and `det_params`
    """
    return Pulse(
        InterpolatedWaveform(T, [*amp_params]),
        InterpolatedWaveform(T, [*det_params]),
        0,
    )


def plot_density(
    density,
    rescaled_positions=None,
    figsize=(10, 6),
    vmin=None,
    vmax=None,
    title=None,
    name=None,
):
    fig, ax = plt.subplots(figsize=figsize)
    c = ax.pcolormesh(density, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x (arbitrary units)")
    ax.set_ylabel("y (arbitrary units)")
    fig.colorbar(c, ax=ax)
    if rescaled_positions is not None:
        ax.scatter(rescaled_positions[:, 0], rescaled_positions[:, 1], c="red")
        for i, p in enumerate(rescaled_positions):
            plt.annotate(str(i), xy=(p[0], p[1]), xytext=(p[0], p[1]))
    if name is not None:
        plt.savefig(name, bbox_inches="tight")
    # plt.show()
    return (fig, ax)


def positions_from_bitstring(bitstring, positions):
    pos = []
    for i, c in enumerate(bitstring):
        if int(c):
            pos.append(positions[i])
    return np.array(pos)


def plot_energy_landscape(alphas, betas, energy_levels, figsize=(10, 6)):
    """
    Plots the energy level of the true solution in the Rydberg Hamiltonian
    as a function of:
    alpha = scale factor between the Rydberg detuning and the linear term in the problem Hamiltonian
    beta = overall shift in the Rydberg detuning
    """
    fig, ax = plt.subplots(figsize=figsize)

    c = ax.pcolormesh(alphas, betas, energy_levels.T, cmap="viridis", shading="auto")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")

    fig.colorbar(c, ax=ax)

    plt.show()
    return


def plot_energy_landscape_contour(
    alphas, betas, energy_levels, levels=8, figsize=(10, 6)
):
    """
    Plots the energy level of the true solution in the Rydberg Hamiltonian
    as a function of:
    alpha = scale factor between the Rydberg detuning and the linear term in the problem Hamiltonian
    beta = overall shift in the Rydberg detuning
    """
    fig, ax = plt.subplots(figsize=figsize)

    c = ax.contour(alphas, betas, energy_levels.T, levels=levels, colors="black")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")

    fig.colorbar(c, ax=ax)

    plt.show()
    return


def identity(x):
    return x


def logabs(x):
    return np.log(np.abs(x))


def power_6(x):
    return np.power(x, 1 / 6)


def invert(x):
    return np.power(x, -1)


def find_possible_distances(nodes):
    distances = []
    for k1 in range(len(nodes)):
        for k2 in range(k1 + 1, len(nodes)):
            d = np.linalg.norm(nodes[k1] - nodes[k2])
            if np.any(np.abs(distances - d) < 1e-3):
                continue
            else:
                distances.append(d)
    return np.array(distances)


def gamma(density, m, var, amp=1):
    Nm = amp * gaussian_mixture(density.shape, var, [m])
    res = 0
    for i in range(len(Nm[0])):
        for j in range(len(Nm)):
            res += 2 * Nm[i, j] * density[i, j]
            res -= Nm[i, j] * Nm[i, j]
    return res


def gamma_list(density, positions, var, amp=1):
    res = []
    for m in positions:
        res.append(gamma(density, m, var, amp))
    return np.array(res)


def neighbouring_gamma_list(density, positions, center, radius, var, amp=1):
    res = []
    for i, m in enumerate(positions):
        if np.linalg.norm(m - center) < radius:
            res.append(gamma(density, m, var, amp))
    return np.array(res)


def V(density, d, var, amp=1):
    shape = density.shape
    Nm1 = amp * gaussian_mixture(shape, var, [(shape[0] / 2, shape[0] / 2)])
    Nm2 = amp * gaussian_mixture(shape, var, [(shape[0] / 2, shape[0] / 2 + d)])
    res = 0
    for i in range(len(Nm1[0])):
        for j in range(len(Nm1)):
            res += Nm1[i, j] * Nm2[i, j]
    return res


def V_list(density, distances, var, amp=1):
    res = []
    for d in distances:
        res.append(2 * V(density, d, var, amp))
    return np.array(res)


def U(d, C6=5420158.53):
    return C6 / (d**6)


def U_list(distances, C6=5420158.53):
    res = []
    for d in distances:
        res.append(2 * U(d, C6=C6))
    return np.array(res)


def ising_energies(nodes, density, var, bitstrings, brad, amp=1):
    distances = find_possible_distances(nodes)
    interaction_coefficients = []
    for d in distances:
        interaction_coefficients.append(2 * V(density, d, var, amp))

    ising_energies = {}
    n = len(nodes)
    for i, bitstring in enumerate(bitstrings):
        if not bmu.is_IS(bitstring, nodes, brad):
            continue
        mixture = gaussian_mixture(
            density.shape, var, positions_from_bitstring(bitstring, nodes)
        )
        energy = 0
        for k1 in range(n):
            if int(bitstring[k1]):
                energy -= gamma(density, nodes[k1], var)
                for k2 in range(k1 + 1, n):
                    if int(bitstring[k2]):
                        d = np.linalg.norm(nodes[k1] - nodes[k2])
                        (idx,) = np.where(np.isclose(distances, d))[0]
                        energy += interaction_coefficients[idx]
        ising_energies[bitstring] = energy
    return sorted(ising_energies.items(), key=lambda x: x[1])


def rydberg_energies(nodes, reg, dets, bitstrings, C6=5420158.53):
    seq = Sequence(reg, MockDevice)

    # local detuning for everyone
    for i, q in enumerate(nodes):
        name = "ising_l" + str(i)
        seq.declare_channel(name, "rydberg_local")
        seq.target(i, name)
        P = create_interp_pulse([1, 0], [0, dets[i]], 4)
        seq.add(P, name)
    # seq.draw()

    simul = Simulation(seq)

    H = simul.get_hamiltonian(3)

    def invert_bits(bitstring):
        return bitstring.replace("1", "2").replace("0", "1").replace("2", "0")

    rydberg_energies = {}
    for i, bs in enumerate(bitstrings):
        rydberg_energies[invert_bits(bs)] = np.real(H[i, i])
    return sorted(rydberg_energies.items(), key=lambda x: x[1])


def rydberg_samples(nodes, reg, brad, dets, bitstrings, T=4000, C6=5420158.53):
    seq = Sequence(reg, MockDevice)
    omega = MockDevice.rabi_from_blockade(brad)

    # local detuning for everyone
    for i, q in enumerate(nodes):
        name = "ising_l" + str(i)
        seq.declare_channel(name, "rydberg_local")
        seq.target(i, name)
        P = create_interp_pulse([0, omega, 0], [-20, 0, dets[i]], T)
        seq.add(P, name)
    # seq.draw()

    simul = Simulation(seq)
    res = simul.run()

    samples = res.sample_final_state()
    return samples


def compute_cubic_up(
    rescaled_positions,
    brad,
    gamma_list,
    V_list,
    U_list,
    distances,
    alpha=1,
    beta=0,
    C6=5420158.53,
):
    ising = 0
    rydberg = 0
    n = len(rescaled_positions)
    for k1 in range(n):
        p1 = rescaled_positions[k1]
        for k2 in range(k1 + 1, n):
            p2 = rescaled_positions[k2]
            d = np.linalg.norm(p1 - p2)
            if d > brad:
                gamma1 = gamma_list[k1]
                gamma2 = gamma_list[k2]
                (idx,) = np.where(np.isclose(distances, d))[0]
                V = V_list[idx]
                U = U_list[idx]
                ising += (alpha * gamma1 - beta) * (alpha * gamma2 - beta) * V
                rydberg += gamma1 * gamma2 * U
    return (ising, rydberg)


def compute_cubic_down(
    rescaled_positions,
    brad,
    gamma_list,
    V_list,
    U_list,
    distances,
    alpha=1,
    beta=0,
    C6=5420158.53,
):
    ising = 0
    rydberg = 0
    n = len(rescaled_positions)
    for k1 in range(n):
        p1 = rescaled_positions[k1]
        for k2 in range(k1 + 1, n):
            p2 = rescaled_positions[k2]
            d = np.linalg.norm(p1 - p2)
            if d > brad:
                gamma1 = gamma_list[k1]
                gamma2 = gamma_list[k2]
                (idx,) = np.where(np.isclose(distances, d))[0]
                V = V_list[idx]
                U = U_list[idx]
                ising += V / (gamma1 * gamma2)
                rydberg += U / ((alpha * gamma1 - beta) * (alpha * gamma2 - beta))
    return (ising, rydberg)


def compute_quadratic_deviation(ising, rydberg, function_ising, function_rydberg):
    return (function_ising(ising) - function_rydberg(rydberg)) ** 2
