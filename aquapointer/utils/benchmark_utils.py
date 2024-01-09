# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from pulser import Pulse, Sequence, Register
from pulser.devices import IroiseMVP, MockDevice, VirtualDevice
from pulser.channels import Rydberg
from pulser.waveforms import CustomWaveform
from pulser_simulation import Simulation, SimConfig
import matplotlib.pyplot as plt
import qutip
import collections

def generate_binary_strings(bit_count):
    """Generates all binary strings of length `bit_count`"""
    binary_strings = []
    def genbin(n, bs=''):
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + '0')
            genbin(n, bs + '1')
    genbin(bit_count)
    return binary_strings

def probability_from_0(bitstring, epsilon):
    """Calculates the probability of obtaining `bitstring` from '0000..0' 
    assuming an `epsilon` probability of measuring '1' instead of '0' """
    n1 = bitstring.count('1')
    n0 = bitstring.count('0')
    return (epsilon**n1) * ((1-epsilon)**n0)

def bitstring_projector(bitstring):
    """Generates the projector corresponding to the computational basis state
    `bitstring`"""
    inv_bit = bitstring.replace('1', '2').replace('0', '1').replace('2', '0')
    return qutip.ket(inv_bit).proj()

def flatten_samples(samples):
    """Takes a counter `samples` and returns a list where each
    entry appears as many times as its count in `samples`"""
    flat_samples = []
    for key, val in samples.items():
        flat_samples.extend([key for _ in range(val)])
    return flat_samples

def magnetization_observable(n, i):
    """Generates the magnetization operator of qubit `i`
    out of `n` qubits"""
    obs = [qutip.qeye(2) for j in range(n)]
    obs[i] = (qutip.sigmaz() + qutip.qeye(2)) / 2
    return qutip.tensor(obs)

def bitstring_probability_exact(bitstring, results, t_i, t_f):
    """Calculates the exact probability of measuring `bitstring` between
    times `t_i` and `t_f` for a Pulser emulation that gives `results`
    (an object pulser_simulation.simresults.CoherentResults) as the output
    of a noiseless emulation"""
    obs = bitstring_projector(bitstring)
    return results.expect([obs])[0][t_i:t_f]

def bitstring_probability_sampling(bitstring, samples):
    """Calculates the probability of measuring `bitstring` from a
    sampling `samples` of a state"""
    tot = np.sum(list(samples.values()))
    return samples[bitstring]/tot 

def bitstring_probability_sampling_binned(bitstring, samples, nbins):
    """Calculates the probability of measuring `bitstring` from a
    sampling `samples` of a state
    Split the samples in `nbins` bins to get error estimate"""
    flat_samples = flatten_samples(samples)
    np.random.shuffle(flat_samples)
    binned_samples = np.array_split(flat_samples, nbins)
    probs = np.zeros(nbins)
    for i in range(nbins):
        samples_i = collections.Counter(binned_samples[i])
        tot = np.sum(list(samples_i.values()))
        probs[i] = samples_i[bitstring]/tot 
    return (np.mean(probs), np.std(probs))

def average_bitstring_probability_sampling(bitstring, results, n_samples, repetitions):
    """Calculates the average probability of measuring `bitstring` out of
    `repetitions` inependent samplings, each containing `n_samples` samples, from a Pulser
    emulation that gives `results`
    (an object pulser_simulation.simresults.CoherentResults/NoisyResults) as the output
    of an emulation"""
    probs = [] 
    for i in range(repetitions):
        samples = results.sample_final_state(n_samples)
        probs.append(bitstring_probability_sampling(bitstring, samples))
    
    return (np.mean(probs), np.std(probs))

def bitstring_distribution_exact(results):
    """ Calculate the exact probability of every bitstring, in order
    from '1111..1' to '000..0' """
    state = results.get_final_state()
    N = state.shape[0]
    probs = np.zeros(N)
    for i, c in enumerate(state):
        probs[i] = np.real(c)**2 + np.imag(c)**2
    return probs

def bitstring_distribution_sampling(samples, n):
    """ Calculate the probability of every bitstring of length `n` from
    sampling `samples`, in order from '1111..1' to '000..0' """
    bitstrings = np.flip(generate_binary_strings(n))
    probs = np.zeros(2**n)
    tot = np.sum(list(samples.values()))
    for i, bitstring in enumerate(bitstrings):
        try:
            probs[i] = samples[bitstring]/tot
        except KeyError:
            probs[i] = 0
    return probs

def bitstring_distribution_sampling_binned(samples, n, nbins):
    """ Calculate the probability of every bitstring of length `n` from
    sampling `samples`, in order from '1111..1' to '000..0' 
    Split the samples in `nbins` bins to get error estimate"""
    flat_samples = flatten_samples(samples)
    np.random.shuffle(flat_samples)
    binned_samples = np.array_split(flat_samples, nbins)
    probs = np.zeros((nbins, 2**n))
    for i in range(nbins):
        samples_i = collections.Counter(binned_samples[i])
        dist = bitstring_distribution_sampling(samples_i, n)
        for j, d in enumerate(dist):
            probs[i,j] = d
    return (np.mean(probs, axis=0), np.std(probs, axis=0))

def average_bitstring_distribution_sampling(results, n_samples, repetitions=1000):
    n = results._size
    probs = np.zeros((repetitions, 2**n))
    for i in range(repetitions):
        samples = results.sample_final_state(n_samples)
        dist = bitstring_distribution_sampling(samples, n)
        for j, d in enumerate(dist):
            probs[i,j] = d
    return (np.mean(probs, axis=0), np.std(probs, axis=0))
    
def magnetization_exact(k, n, results, t_i, t_f):
    """Calculate the exact magnetization of qubit `k` out
    of `n` qubits between times `t_i` and `t_f` for a Pulser emulation that
    gives `results` (an object pulser_simulation.simresults.CoherentResults)
    as the output of a noiseless emulation"""
    return results.expect([magnetization_observable(n, k)])[0][t_i:t_f]

def magnetization_sampling(k, samples):
    """Calculate the magnetization of qubit `k` from a sampling
    `samples` of a state"""
    mag = 0
    tot = 0
    for key, val in samples.items():
        if key[k] == '1':
            mag += val
        tot += val
    return mag/tot

def magnetization_sampling_binned(k, samples, nbins):
    """Calculate the magnetization of qubit `k` from a sampling
    `samples` of a state
    Split the samples in `nbins` bins to get error estimate"""
    flat_samples = flatten_samples(samples)
    np.random.shuffle(flat_samples)
    binned_samples = np.array_split(flat_samples, nbins)
    mags = np.zeros(nbins)
    for i in range(nbins):
        samples_i = collections.Counter(binned_samples[i])
        mag = 0
        tot = 0
        for key, val in samples_i.items():
            if key[k] == '1':
                mag += val
            tot += val
        mags[i] = mag/tot
    return (np.mean(mags), np.std(mags))

def average_magnetization_sampling(k, results, n_samples, repetitions):
    """Calculates the average magnetization of qubit `k` out of
    `repetitions` inependent samplings, each contining `n_samples` samples, from a Pulser
    emulation that gives `results`
    (an object pulser_simulation.simresults.CoherentResults/NoisyResults) as the output
    of an emulation"""
    mag = []
    for i in range(repetitions):
        samples = results.sample_final_state(n_samples)
        mag.append(magnetization_sampling(k, samples))
    return (np.mean(mag), np.std(mag))


def is_IS(bitstring, pos, brad):
    """Returns `True` if `bitstring` is an independent set of the graph
    defined by nodes in position `pos` and blockade radius `brad`"""

    for i in range(len(bitstring)):
        b1 = bitstring[i]
        if b1=='0':
            continue
        for j in range(i+1, len(bitstring)):
            b2 = bitstring[j]
            if b2=='0':
                continue
            if np.linalg.norm(pos[i]-pos[j]) < brad:
                return False
    return True

def separate_IS(bitstrings, pos, brad):
    """Among all bitstrings in the list `bitstrings`, returns only those that
    are independent sets of the graph defined by nodes in position `pos` and
    blockade radius `brad`"""
    res = []
    for bitstring in bitstrings:
        if is_IS(bitstring, pos, brad):
            res.append(bitstring)
    return res

def generate_random_register(rows, cols, n, spacing, seed=346):
    """Generates a random register with `n` qubits as a subregister of a
    triangular lattice or size `rows` x `cols` and spacing `spacing`"""
    if n>rows*cols:
        raise ValueError("lattice is not large enough")
    np.random.seed(seed)
    reg = Register.triangular_lattice(rows=rows, atoms_per_row=cols, spacing=spacing)
    pos = reg._coords
    while len(pos)>n:
        k = np.random.randint(0, len(pos))
        pos = [pos[i] for i in range(len(pos)) if i!=k]
    reg = Register.from_coordinates(pos)
    return reg, pos

def observable(bitstring):
    """Generates the projector corresponding to the computational basis state
    `bitstring`"""
    inv_bit = bitstring.replace('1', '2').replace('0', '1').replace('2', '0')
    return qutip.ket(inv_bit).proj()

def pulse_landscape(register, device, independent_sets, omega, time_list, delta_list, verbose=True):
    """For each state in `independent_sets`, returns expectation value (and
    various other retlated stats) of its projector over a time range defined by
    `time_list` and over a range of detuning values defined by `delta_list`"""

    if time_list.dtype != int:
        raise ValueError("times must be int")

    IS_dictionary = dict()
    for bitstring in independent_sets:
        IS_dictionary[bitstring] = {
            "excitations" : bitstring.count('1'),
            "landscape" : np.zeros((len(time_list), len(delta_list))),
            "max_prob" : 0,
            "min_prob" : 0,
            "location_max" : (-1,-1),
            "location_min" : (-1,-1),
        }

    T = time_list[-1]

    for j, d in enumerate(delta_list):
        print(f"Simulating detuning {j+1} of {len(delta_list)}" if verbose else "", end = "\n" if verbose else "")
        seq = Sequence(register, device)
        seq.declare_channel("ch", "rydberg_global")
        pulse = Pulse.ConstantPulse(T, omega, d, 0)
        seq.add(pulse, "ch")
        sim = Simulation(seq, with_modulation=False)
        res = sim.run(progress_bar=verbose)
        for bitstring in independent_sets:
            obs = [observable(bitstring)]
            exp = res.expect(obs)
            for i, t in enumerate(time_list):
                IS_dictionary[bitstring]["landscape"][i,j] = exp[0][t]

    for bitstring in independent_sets:

        IS_dictionary[bitstring]["max_prob"] = np.max(IS_dictionary[bitstring]["landscape"])
        IS_dictionary[bitstring]["min_prob"] = np.min(IS_dictionary[bitstring]["landscape"])
        IS_dictionary[bitstring]["location_max"] = (
            np.unravel_index(
                IS_dictionary[bitstring]["landscape"].argmax(),
                IS_dictionary[bitstring]["landscape"].shape
            )
        )
        IS_dictionary[bitstring]["location_min"] = (
            np.unravel_index(
                IS_dictionary[bitstring]["landscape"].argmin(),
                IS_dictionary[bitstring]["landscape"].shape
            )
        )

    return IS_dictionary

def apply_meas_errors(samples, eps, eps_prime):
    """Copied from Pulser"""
    shots = list(samples.keys())
    n_detects_list = list(samples.values())

    # Convert shots to a 2D array
    shot_arr = np.array([list(shot) for shot in shots], dtype=int)
    # Compute flip probabilities
    flip_probs = np.where(shot_arr == 1, eps_prime, eps)
    # Repeat flip_probs based on n_detects_list
    flip_probs_repeated = np.repeat(flip_probs, n_detects_list, axis=0)
    # Generate random matrix of shape (sum(n_detects_list), len(shot))
    random_matrix = np.random.uniform(
        size=(np.sum(n_detects_list), len(shot_arr[0]))
    )
    # Compare random matrix with flip probabilities
    flips = random_matrix < flip_probs_repeated
    # Perform XOR between original array and flips
    new_shots = shot_arr.repeat(n_detects_list, axis=0) ^ flips
    # Count all the new_shots
    # We are not converting to str before because tuple indexing is faster
    detected_sample_dict: collections.Counter = collections.Counter(map(tuple, new_shots))
    return collections.Counter(
        {"".join(map(str, k)): v for k, v in detected_sample_dict.items()}
    )

def pulse_landscape_SPAM(register, device, independent_sets, omega, time_list, delta_list, epsilon, epsilon_prime, verbose=True):
    """For each state in `independent_sets`, returns expectation value (and
    various other retlated stats) of its projector over a time range defined by
    `time_list` and over a range of detuning values defined by `delta_list`.
    Inlcudes measurement errors epsilon and epsilon_prime"""

    if time_list.dtype != int:
        raise ValueError("times must be int")

    IS_dictionary = dict()
    for bitstring in independent_sets:
        IS_dictionary[bitstring] = {
            "excitations" : bitstring.count('1'),
            "landscape" : np.zeros((len(time_list), len(delta_list))),
            "max_prob" : 0,
            "min_prob" : 0,
            "location_max" : (-1,-1),
            "location_min" : (-1,-1),
        }

    T = time_list[-1]

    for j, d in enumerate(delta_list):
        print(f"Simulating detuning {j+1} of {len(delta_list)}" if verbose else "", end = "\n" if verbose else "")
        seq = Sequence(register, device)
        seq.declare_channel("ch", "rydberg_global")
        pulse = Pulse.ConstantPulse(T, omega, d, 0)
        seq.add(pulse, "ch")
        sim = Simulation(seq, with_modulation=False)
        res = sim.run(progress_bar=verbose)
        for bitstring in independent_sets:
            for i, t in enumerate(time_list):
                samples = res.sample_state(t/1000, n_samples=10000)
                err_samples = apply_meas_errors(samples, epsilon, epsilon_prime)
                IS_dictionary[bitstring]["landscape"][i,j] = err_samples[bitstring]/10000
    for bitstring in independent_sets:

        IS_dictionary[bitstring]["max_prob"] = np.max(IS_dictionary[bitstring]["landscape"])
        IS_dictionary[bitstring]["min_prob"] = np.min(IS_dictionary[bitstring]["landscape"])
        IS_dictionary[bitstring]["location_max"] = (
            np.unravel_index(
                IS_dictionary[bitstring]["landscape"].argmax(),
                IS_dictionary[bitstring]["landscape"].shape
            )
        )
        IS_dictionary[bitstring]["location_min"] = (
            np.unravel_index(
                IS_dictionary[bitstring]["landscape"].argmin(),
                IS_dictionary[bitstring]["landscape"].shape
            )
        )
    return IS_dictionary


def pulse_landscape_samples(register, device, independent_sets, omega, time_list, delta_list, verbose=True, n_samples=150):
    """For each state in `independent_sets`, returns expectation value (and
    various other retlated stats) of its projector over a time range defined by
    `time_list` and over a range of detuning values defined by `delta_list`"""

    if time_list.dtype != int:
        raise ValueError("times must be int")

    IS_dictionary = dict()
    for bitstring in independent_sets:
        IS_dictionary[bitstring] = {
            "excitations" : bitstring.count('1'),
            "landscape" : np.zeros((len(time_list), len(delta_list))),
            "samples" : np.empty((len(time_list), len(delta_list)), dtype=object),
            "max_prob" : 0,
            "min_prob" : 0,
            "location_max" : (-1,-1),
            "location_min" : (-1,-1),
        }

    T = time_list[-1]

    for j, d in enumerate(delta_list):
        print(f"Simulating detuning {j+1} of {len(delta_list)}" if verbose else "", end = "\n" if verbose else "")
        seq = Sequence(register, device)
        seq.declare_channel("ch", "rydberg_global")
        pulse = Pulse.ConstantPulse(T, omega, d, 0)
        seq.add(pulse, "ch")
        sim = Simulation(seq, with_modulation=False)
        res = sim.run(progress_bar=verbose)
        for bitstring in independent_sets:
            for i, t in enumerate(time_list):
                samples = res.sample_state(t/1e3, n_samples=n_samples)
                val = 0
                for key, count in samples.items():
                    if key==bitstring:
                        val = count/n_samples
                IS_dictionary[bitstring]["landscape"][i,j] = val
                IS_dictionary[bitstring]["samples"][i,j] = samples

    for bitstring in independent_sets:

        IS_dictionary[bitstring]["max_prob"] = np.max(IS_dictionary[bitstring]["landscape"])
        IS_dictionary[bitstring]["min_prob"] = np.min(IS_dictionary[bitstring]["landscape"])
        IS_dictionary[bitstring]["location_max"] = (
            np.unravel_index(
                IS_dictionary[bitstring]["landscape"].argmax(),
                IS_dictionary[bitstring]["landscape"].shape
            )
        )
        IS_dictionary[bitstring]["location_min"] = (
            np.unravel_index(
                IS_dictionary[bitstring]["landscape"].argmin(),
                IS_dictionary[bitstring]["landscape"].shape
            )
        )

    return IS_dictionary


def pulse_landscape_selected(register, device, independent_sets, omega, shape, params_list, indices_list, verbose=True):
    """For each state in `independent_sets`, returns expectation value (and
    various other retlated stats) of its projector over a range of parameters
    defined by `params_list`"""

    IS_dictionary = dict()
    for bitstring in independent_sets:
        IS_dictionary[bitstring] = {
            "excitations" : bitstring.count('1'),
            "landscape" : np.zeros(shape),
            "max_prob" : 0,
            "min_prob" : 0,
            "location_max" : (-1,-1),
            "location_min" : (-1,-1),
        }

    for k, (t, d) in enumerate(params_list):
        print(f"Simulating parameters {k+1} of {len(params_list)}" if verbose else "", end = "\n" if verbose else "")
        print(f"time: {t}     det: {d}" if verbose else "", end = "\n" if verbose else "")
        seq = Sequence(register, device)
        seq.declare_channel("ch", "rydberg_global")
        pulse = Pulse.ConstantPulse(t, omega, d, 0)
        seq.add(pulse, "ch")
        sim = Simulation(seq, with_modulation=False)
        res = sim.run(progress_bar=False)
        for bitstring in independent_sets:
            obs = [observable(bitstring)]
            exp = res.expect(obs)
            i = indices_list[k][0]
            j = indices_list[k][1]
            print(f"i: {i}     j: {j}" if verbose else "", end = "\n" if verbose else "")
            print(f"p: {exp[0][-1]}" if verbose else "", end = "\n" if verbose else "")
            IS_dictionary[bitstring]["landscape"][i,j] = exp[0][-1]

    for bitstring in independent_sets:

        IS_dictionary[bitstring]["max_prob"] = np.max(IS_dictionary[bitstring]["landscape"])
        IS_dictionary[bitstring]["min_prob"] = np.min(IS_dictionary[bitstring]["landscape"])
        IS_dictionary[bitstring]["location_max"] = (
            np.unravel_index(
                IS_dictionary[bitstring]["landscape"].argmax(),
                IS_dictionary[bitstring]["landscape"].shape
            )
        )
        IS_dictionary[bitstring]["location_min"] = (
            np.unravel_index(
                IS_dictionary[bitstring]["landscape"].argmin(),
                IS_dictionary[bitstring]["landscape"].shape
            )
        )

    return IS_dictionary

def pulse_landscape_modulation(register, device, independent_sets, omega, time_list, delta_list, verbose=True):
    """For each state in `independent_sets`, returns expectation value (and
    various other retlated stats) of its projector over a time range defined by
    `time_list` and over a range of detuning values defined by `delta_list`.
    Simulation is done with modulation, which means that each time value needs
    its own independent simulation."""

    if time_list.dtype != int:
        raise ValueError("times must be int")

    IS_dictionary = dict()
    for bitstring in independent_sets:
        IS_dictionary[bitstring] = {
            "excitations" : bitstring.count('1'),
            "landscape" : np.zeros((len(time_list), len(delta_list))),
            "max_prob" : 0,
            "min_prob" : 0,
            "location_max" : (-1,-1),
            "location_min" : (-1,-1),
        }

    for i, t in enumerate(time_list):
        print(f"Simulating time {i+1} of {len(time_list)}" if verbose else "", end = "\n" if verbose else "")
        for j, d in enumerate(delta_list):
            print(f"    Simulating detuning {j+1} of {len(delta_list)}" if verbose else "", end = "\n" if verbose else "")
            seq = Sequence(register, device)
            seq.declare_channel("ch", "rydberg_global")
            pulse = Pulse.ConstantPulse(t, omega, d, 0)
            seq.add(pulse, "ch")
            sim = Simulation(seq, with_modulation=True)
            res = sim.run(progress_bar=verbose)
            for bitstring in independent_sets:
                obs = [observable(bitstring)]
                exp = res.expect(obs)
                IS_dictionary[bitstring]["landscape"][i,j] = exp[0][-1]

    for bitstring in independent_sets:

        IS_dictionary[bitstring]["max_prob"] = np.max(IS_dictionary[bitstring]["landscape"])
        IS_dictionary[bitstring]["min_prob"] = np.min(IS_dictionary[bitstring]["landscape"])
        IS_dictionary[bitstring]["location_max"] = (
            np.unravel_index(
                IS_dictionary[bitstring]["landscape"].argmax(),
                IS_dictionary[bitstring]["landscape"].shape
            )
        )
        IS_dictionary[bitstring]["location_min"] = (
            np.unravel_index(
                IS_dictionary[bitstring]["landscape"].argmin(),
                IS_dictionary[bitstring]["landscape"].shape
            )
        )

    return IS_dictionary

def pulse_landscape_modulation_fast(register, device, time_buffer, independent_sets, omega, time_list, delta_list, EOM=True, verbose=True):
    """For each state in `independent_sets`, returns expectation value (and
    various other retlated stats) of its projector over a time range defined by
    `time_list` and over a range of detuning values defined by `delta_list`.
    Simulation is done with modulation, which theoretically means that each
    time value needs its own independent simulation. However, this function
    uses a shortcut that saves a lot of time."""

    if time_list.dtype != int:
        raise ValueError("times must be int")

    IS_dictionary = dict()
    for bitstring in independent_sets:
        IS_dictionary[bitstring] = {
            "excitations" : bitstring.count('1'),
            "landscape" : np.zeros((len(time_list), len(delta_list))),
            "max_prob" : 0,
            "min_prob" : 0,
            "location_max" : (-1,-1),
            "location_min" : (-1,-1),
        }


    for j, d in enumerate(delta_list):
        print(f"Simulating detuning {j+1} of {len(delta_list)}" if verbose else "", end = "\n" if verbose else "")

        print(f" Preprocessing shortcut: start" if verbose else "", end = "\n" if verbose else "")
        T = time_list[-1]
        full_seq = Sequence(register, device)
        full_seq.declare_channel("ch", "rydberg_global")
        if EOM:
            full_seq.enable_eom_mode("ch", amp_on=omega, detuning_on=d)
            full_seq.add_eom_pulse("ch", duration=T, phase=0.0)
        else:
            full_pulse = Pulse.ConstantPulse(T, omega, d, 0)
            full_seq.add(full_pulse, "ch")
        full_sim = Simulation(full_seq, with_modulation=True)
        full_amp = full_sim.samples_obj.samples_list[0].amp
        full_det = full_sim.samples_obj.samples_list[0].det
        amp_tail = full_amp[T:]
        det_tail = full_det[T:]
        wf_amp = CustomWaveform(amp_tail)
        wf_det = CustomWaveform(det_tail)
        pulse_tail = Pulse(amplitude=wf_amp, detuning=wf_det, phase=0)
        Tp = len(amp_tail)
        full_res = full_sim.run(progress_bar=verbose)
        init_states = []
        for i, t in enumerate(time_list):
            if t>=time_buffer:
                init_states.append(full_res.get_state(t/1000))
        print(f" Preprocessing shortcut: end" if verbose else "", end = "\n" if verbose else "")

        k = 0
        for i, t in enumerate(time_list):
            if t<time_buffer:
                # independent simulation for each time only if you are below the time buffer
                print(f"    Simulating time {i+1} of {len(time_list)}" if verbose else "", end = "\n" if verbose else "")
                seq = Sequence(register, device)
                seq.declare_channel("ch", "rydberg_global")
                if EOM:
                    seq.enable_eom_mode("ch", amp_on=omega, detuning_on=d)
                    seq.add_eom_pulse("ch", duration=t, phase=0.0)
                else:
                    pulse = Pulse.ConstantPulse(t, omega, d, 0)
                    seq.add(pulse, "ch")
                sim = Simulation(seq, with_modulation=True)
                res = sim.run(progress_bar=verbose)
                for bitstring in independent_sets:
                    obs = [observable(bitstring)]
                    exp = res.expect(obs)
                    IS_dictionary[bitstring]["landscape"][i,j] = exp[0][-1]
            else:
                # use shortcut for times above the time buffer
                print(f"    Simulating time {i+1} of {len(time_list)}" if verbose else "", end = "\n" if verbose else "")
                seq = Sequence(register, device)
                seq.declare_channel("ch", "rydberg_global")
                pulse = pulse_tail
                seq.add(pulse, "ch")
                # Run simulation WITHOUT modulation, because the pulse shape is already modulated
                sim = Simulation(seq, with_modulation=False)
                sim.initial_state = init_states[k]
                res = sim.run(progress_bar=False)
                for bitstring in independent_sets:
                    obs = [observable(bitstring)]
                    exp = res.expect(obs)
                    IS_dictionary[bitstring]["landscape"][i,j] = exp[0][-1]
                k += 1

    for bitstring in independent_sets:

        IS_dictionary[bitstring]["max_prob"] = np.max(IS_dictionary[bitstring]["landscape"])
        IS_dictionary[bitstring]["min_prob"] = np.min(IS_dictionary[bitstring]["landscape"])
        IS_dictionary[bitstring]["location_max"] = (
            np.unravel_index(
                IS_dictionary[bitstring]["landscape"].argmax(),
                IS_dictionary[bitstring]["landscape"].shape
            )
        )
        IS_dictionary[bitstring]["location_min"] = (
            np.unravel_index(
                IS_dictionary[bitstring]["landscape"].argmin(),
                IS_dictionary[bitstring]["landscape"].shape
            )
        )

    return IS_dictionary

def plot_probability_landscape(projector_expvals, time_list, delta_list, max_p_clip=1, figsize=(8,4)):
    """Plots the projector probability over a time range defined by
    `time_list` and over a range of detuning values defined by `delta_list`"""

    for bitstring, data in projector_expvals.items():
        fig, ax = plt.subplots(figsize=figsize)
        c = ax.pcolormesh(time_list, delta_list, data["landscape"].T, vmin=0, vmax=max_p_clip)
        ax.set_xlabel("time (ns)")
        ax.set_ylabel("detuning (rad/us)")
        ax.set_title(f"probability of bitstring: {bitstring}")
        fig.colorbar(c, ax=ax)
        plt.show()
        print(f"maximum probability: {data['max_prob']:.3f}")
        print(f"probability of measuring: {1-(1-data['max_prob'])**200:.2f} (200 shots)")
        i,j = data["location_max"]
        print(f"time of max_prob: {time_list[i]}")
        print(f"detuning of max_prob: {delta_list[j]:.2f}")
        print()
        print()

def plot_probability_contour(projector_expvals, time_list, delta_list, levels=8, max_p_clip=1, figsize=(8,4)):
    """Plots the projector probability over a time range defined by
    `time_list` and over a range of detuning values defined by `delta_list`"""

    for bitstring, data in projector_expvals.items():
        fig, ax = plt.subplots(figsize=figsize)
        c = ax.contour(time_list, delta_list, data["landscape"].T, vmin=0, vmax=max_p_clip, levels=levels)
        ax.clabel(c, inline=True)
        ax.set_xlabel("time (ns)")
        ax.set_ylabel("detuning (rad/us)")
        ax.set_title(f"probability of bitstring: {bitstring}")
        fig.colorbar(c, ax=ax)
        plt.show()
        print(f"maximum probability: {data['max_prob']:.3f}")
        print(f"probability of measuring: {1-(1-data['max_prob'])**200:.2f} (200 shots)")
        i,j = data["location_max"]
        print(f"time of max_prob: {time_list[i]}")
        print(f"detuning of max_prob: {delta_list[j]:.2f}")
        print()
        print()

def stats_by_is_size(projector_expvals, n):
    """Computes projector probability statistics"""
    for ex in range(1, n+1):
        print(f"{ex} excitations:")
        probs = []
        counter = 0
        for bitstring, data in projector_expvals.items():
            if data["excitations"] != ex:
                continue
            else:
                probs.append(data["max_prob"])
                counter += 1
        if counter > 0:
            print(f"average probability: {np.mean(probs)} ({counter} samples)")
            print(f"standard deviation: {np.std(probs)} ({counter} samples)")
            print(f"maximum probability: {np.max(probs)}")
            print(f"minimum probability: {np.min(probs)}")
        print()
