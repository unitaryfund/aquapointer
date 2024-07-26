import numpy as np
import scipy.ndimage as ndi
from pulser.devices import MockDevice
from pulser_simulation import QutipBackend

from aquapointer.slicing import density_file_to_grid, density_slices_by_planes
from aquapointer.analog import find_water_positions

from pathlib import Path
BASE_PATH = str(Path.cwd().parent)

def rism_to_locations(rism_file, settings_file):
    # load density grid from file
    grid = density_file_to_grid(rism_file)
    # load settings file containing processor settings and slcing points
    settings = open(settings_file, "r")
    settings_contents = settings.readlines()
    pulse_params = settings_contents[0].split()
    amplitude = float(settings_contents[1].split()[0])
    variance = float(settings_contents[1].split()[1])
    lattice_settings = settings_contents[2].split()

    if settings_contents[3].split()[0] == "filter":
        filter_settings = settings_contents[3].split()[1:]
        if settings_contents[4].split()[0] == "crop":
            crop_settings = settings_contents[4].split()[1:]
            slicing_points_lines = settings_contents[5:]
        else:
            crop_settings = None
            slicing_points_lines = settings_contents[4:]
    elif settings_contents[3].split()[0] == "crop":
        filter_settings = None
        crop_settings = settings_contents[3].split()[1:]
        slicing_points_lines = settings_contents[4:]

    else:
        filter_settings = None
        crop_settings = None
        slicing_points_lines = settings_contents[3:]

    slicing_points = []
    for line in slicing_points_lines:
            sp_list = [float(s) for s in line.split()]
            if sp_list != []:
                slicing_points.append(np.array([sp_list[0:3], sp_list[3:6], sp_list[6:]]))

    canvases = density_slices_by_planes(grid, slicing_points)

    if crop_settings:
        center = (float(crop_settings[0]), float(crop_settings[1]))
        size = (float(crop_settings[2]), float(crop_settings[3]))
        [c.crop_canvas(center, size) for c in canvases]

    if filter_settings:
        if filter_settings[0] == "gaussian-laplace":
            filter_fn = lambda x, sigma: -ndi.gaussian_laplace(x, sigma)
            sigma = float(filter_settings[1])

    [
        c.filter_density(filter_settings={"filter_function": filter_fn, "sigma": sigma})
        for c in canvases
    ]

    # Define a lattice
    if lattice_settings[0] == "poisson-disk":
        spacing = float(lattice_settings[3]), float(lattice_settings[4])
        [c.set_poisson_disk_lattice(spacing) for c in canvases]

    elif lattice_settings[0] == "rectangular":
        num_x = int(lattice_settings[1])
        num_y = int(lattice_settings[2])
        spacing = tuple(map(float, lattice_settings[3:5]))
        [c.set_rectangular_lattice(num_x, num_y, spacing) for c in canvases]

    elif lattice_settings[0] == "triangular":
        nrows = int(lattice_settings[1])
        ncols = int(lattice_settings[2])
        spacing = tuple(map(float, lattice_settings[3:5]))
        [c.set_triangular_lattice(nrows, ncols, spacing) for c in canvases]

    elif lattice_settings[0] == "hexagonal":
        nrows = int(lattice_settings[1])
        ncols = int(lattice_settings[2])
        spacing = tuple(map(float, lattice_settings[3:5]))
        [c.set_hexagonal_lattice(nrows, ncols, spacing) for c in canvases]

    else:
        raise ValueError(
            """
            lattice must be specified as one of the following supported lattice types: 
            poisson-disk, rectangular, triangular, or hexagonal.
        """
        )

    [c.calculate_pubo_coefficients(2, [amplitude, variance]) for c in canvases]

    if len(lattice_settings) == 6:
        size = int(lattice_settings[-1])
        [c.force_lattice_size(size) for c in canvases]

    brad = float(pulse_params[0])
    omega = float(pulse_params[1])
    max_det = float(pulse_params[2])
    pulse_duration = float(pulse_params[3])
    pulse_settings = {"brad": brad, "omega": omega, "pulse_duration": pulse_duration, "max_det": max_det}
    test_water_postions = find_water_positions(
        canvases, executor, MockDevice, pulse_settings
    )
    return test_water_postions


def executor(pulse_seq, num_samples, sim=QutipBackend):
    res = sim(pulse_seq).run()
    return res.sample_final_state(num_samples)


main_folder = "data"
dna_folder = f"{main_folder}/DNA"
rna_folder = f"{main_folder}/RNA"
wvv_folder = f"{main_folder}/4wvv"
sim_folder = f"{main_folder}/2a15_ph4"

def rism_file(path):
    return f"{path}/prot_3drism.O.1.dx"


main_output_folder = "aquapointer/analog/example_output"
dna_output_folder = f"{main_output_folder}/DNA"
rna_output_folder = f"{main_output_folder}/RNA"
wvv_folder = f"{main_output_folder}/4wvv"

sim_out_folder = f"{main_output_folder}/2a15_ph4"
locations = rism_to_locations(
    rism_file(sim_folder), "aquapointer/analog/analog_settings_example1"
)
np.savetxt(f"{sim_out_folder}/locations1.txt", locations)
