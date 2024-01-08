# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


import numpy as np 

from pulser import register


class PulseSettings:
    def __init__(self, brad, omega, pulse_duration, max_det) -> None:
        self.brad = brad
        self.omega = omega
        self.pulse_duration = pulse_duration
        self.max_det = max_det


class Processor:
    def __init__(self, device, register: register) -> None:
        self.register = register
        self.device = device


class AnalogProcessor(Processor):
    def __init__(
        self,
        device,
        pos,
        pos_id,
        pulse_settings: PulseSettings,
    ) -> None:
        self.device = device
        self.pos = pos,
        self.pos_id = pos_id,
        self.register = register.Register.from_coordinates(pos)
        self.pulse_settings = pulse_settings

    def scale_grid_to_register(self):
        """Placeholder for position scaling function."""
        with open(f'../registers/rescaled_position_{self.pos_id[0]}.npy', 'rb') as file_in:
            res_pos = np.load(file_in)
        return res_pos
        

class DigitalProcessor(Processor):
    def __init__(
        self,
        device,
        register: register,
    ) -> None:
        self.device = device
        self.register = register
