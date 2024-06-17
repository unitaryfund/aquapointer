import numpy as np
import scipy.ndimage as ndi
from pulser import Register
from pulser.devices import MockDevice
from pulser_simulation import QutipBackend
from aquapointer.slicing import density_file_to_grid, density_slices_by_planes
from aquapointer.analog_digital import processor
from aquapointer.analog import find_water_positions


def rism_to_locations(rism_file, settings_file):
    # load density grid from file
    grid = density_file_to_grid(rism_file)
    # load settings file containing processor settings and slcing points
    settings = open(settings_file, "r")
    settings_contents = settings.readlines()
    if settings_contents[3].split()[0] == "True":
        crop_settings = settings_contents[3].split()[1:]
        center = (float(crop_settings[0]), float(crop_settings[1]))
        size = (float(crop_settings[2]), float(crop_settings[3]))
        slicing_points_lines = settings_contents[4:]
    else:
        crop_settings = None
        slicing_points_lines = settings_contents[3:]

    slicing_points = []
    for line in slicing_points_lines:
        sp_list = [float(s) for s in line.split()]
        slicing_points.append(np.array([sp_list[0:3], sp_list[3:6], sp_list[6:]]))
    
    canvases = density_slices_by_planes(grid, slicing_points)

    if crop_settings:
        [c.crop_canvas(center, size) for c in canvases]
    
    def filter_fn(x, sigma):
        return -ndi.gaussian_laplace(x, sigma)

    [c.filter_density(filter_settings={"filter_function": filter_fn, "sigma": 0.5}) for c in canvases]

    # Define a lattice
    lattice_settings = settings_contents[2].split()
    if lattice_settings[0] == "poisson":
        spacing = float(lattice_settings[3]), float(lattice_settings[4])
        [c.set_poisson_disk_lattice(spacing) for c in canvases]

        
    elif lattice_settings[0] == "rectangular":
        num_x = int(lattice_settings[1])
        num_y = int(lattice_settings[2])
        spacing = tuple(map(float, lattice_settings[3:5]))
        [c.set_rectangular_lattice(num_x, num_y, spacing) for c in canvases]

    amplitude = float(settings_contents[1].split()[0])
    variance = float(settings_contents[1].split()[1])
    [c.calculate_pubo_coefficients(2, [amplitude, variance]) for c in canvases]

    if len(lattice_settings) == 6:
        size = int(lattice_settings[-1])
        [c.force_lattice_size(size) for c in canvases]

    # import registers
    positions = []
    registers = []
    d_list = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    for i in range(len(d_list)):
        with open(f'aquapointer/analog/registers/position_{i}.npy', 'rb') as file_in:
            pos = np.load(file_in)
        positions.append(pos)
        registers.append(Register.from_coordinates(pos)) # this is to create a Pulser register object

    rescaled_positions = []
    for i in range(len(d_list)):
        with open(f'aquapointer/analog/registers/rescaled_position_{i}.npy', 'rb') as file_in:
            res_pos = np.load(file_in)
        rescaled_positions.append(res_pos)
    pulse_params = settings_contents[0].split()
    brad = float(pulse_params[0])
    omega = float(pulse_params[1])
    max_det = float(pulse_params[2])
    pulse_duration = float(pulse_params[3])


    pulse_settings = processor.PulseSettings(brad, omega, pulse_duration, max_det)
    processor_configs = [processor.AnalogProcessor(device=MockDevice, pos=pos, pos_id=p, pulse_settings=pulse_settings) for p, pos in enumerate(positions)]
    test_water_postions = find_water_positions(canvases, executor, processor_configs)
    return test_water_postions

def executor(pulse_seq, num_samples, sim=QutipBackend):
    res = sim(pulse_seq).run()
    return res.sample_final_state(num_samples)


main_folder = "data"
dna_folder = f"{main_folder}/DNA"
rna_folder = f"{main_folder}/RNA"
wvv_folder = f"{main_folder}/4wvv"

def rism_file(path):
    return f"{path}/prot_3drism.O.1.dx"

main_output_folder = "aquapointer/analog/example_output"
dna_output_folder = f"{main_output_folder}/DNA"
rna_output_folder = f"{main_output_folder}/RNA"
wvv_folder = f"{main_output_folder}/4wvv"

locations = rism_to_locations(rism_file(dna_folder), "aquapointer/analog/analog_settings_example")
np.savetxt(f"{dna_output_folder}/locations.txt", locations)
