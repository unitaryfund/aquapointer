# Copyright (C) Unitary Fund, Pasqal, and Qubit Pharmaceuticals.
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from numpy.typing import NDArray
from pulser import Register
from pulser.backend.qpu import QPUBackend


class PulseSettings:
    def __init__(
        self, brad: float, omega: float, pulse_duration: float, max_det: float
    ) -> None:
        self.brad = brad
        self.omega = omega
        self.pulse_duration = pulse_duration
        self.max_det = max_det


class Processor:
    def __init__(self, device: QPUBackend, register: Register) -> None:
        self.register = register
        self.device = device


class AnalogProcessor(Processor):
    def __init__(
        self,
        device: QPUBackend,
        pos: NDArray,
        pos_id: int,
        pulse_settings: PulseSettings,
    ) -> None:
        self.device = device
        self.pos = (pos,)
        self.pos_id = (pos_id,)
        self.register = Register.from_coordinates(pos)
        self.pulse_settings = pulse_settings

    def scale_grid_to_register(self):
        """Placeholder for position scaling function."""
        with open(
            f"../registers/rescaled_position_{self.pos_id[0]}.npy", "rb"
        ) as file_in:
            res_pos = np.load(file_in)
        return res_pos


class DigitalProcessor(Processor):
    def __init__(
        self,
        device: QPUBackend,
        register: Register,
    ) -> None:
        self.device = device
        self.register = register
