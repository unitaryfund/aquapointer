import numpy as np
from pulser import Register
from pulser.devices import MockDevice
from pulser_simulation import QutipBackend
from aquapointer.slicing import density_file_to_grid, density_slices_by_planes
from aquapointer.analog_digital import processor
from aquapointer.analog import find_water_positions


def rism_to_locations(rism_file, settings_file):

    grid = density_file_to_grid(rism_file)
    slicing_points = np.load(settings_file)
    canvases = density_slices_by_planes(grid, slicing_points)

    # import registers
    positions = []
    registers = []
    d_list = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    for i in range(len(d_list)):
        with open(f'../registers/position_{i}.npy', 'rb') as file_in:
            pos = np.load(file_in)
        positions.append(pos)
        registers.append(Register.from_coordinates(pos)) # this is to create a Pulser register object

    rescaled_positions = []
    for i in range(len(d_list)):
        with open(f'../registers/rescaled_position_{i}.npy', 'rb') as file_in:
            res_pos = np.load(file_in)
        rescaled_positions.append(res_pos)
    pulse_params = np.load(settings_file)
    brad = pulse_params[2]
    omega = pulse_params[1]
    max_det = pulse_params[2]
    pulse_duration = pulse_params[3]

    pulse_settings = processor.PulseSettings(brad, omega, pulse_duration, max_det)
    processor_configs = [processor.AnalogProcessor(device=MockDevice, pos=pos, pos_id=p, pulse_settings=pulse_settings) for p, pos in enumerate(positions)]
    test_water_postions = find_water_positions(canvases, executor, processor_configs)
    

def executor(pulse_seq, num_samples, sim=QutipBackend):
    res = sim(pulse_seq).run()
    return res.sample_final_state(num_samples)
