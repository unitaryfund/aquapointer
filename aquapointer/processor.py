from typing import Dict
from pulser import register


class PulseSettings():
    def __init__(
        self,
        brad,
        omega,
        duration,
    ) -> None:

        self.brad = brad
        self.omega = omega
        self.duration = duration


class Processor():
    def __init__(
        self,
        device,
        register: register
    ) -> None:

        self.register = register
        self.device = device



class AnalogProcessor(Processor):

    def __init__(
        self,
        device,
        register: register,
        pulse_settings: PulseSettings,
    ) -> None:

        self.device = device
        self.register = register
        self.pulse_settings = pulse_settings


class DigitalProcessor(Processor):

    def __init__(
        self,
        device,
        register: register,
    ) -> None:

        self.device = device
        self.register = register