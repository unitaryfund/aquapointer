from typing import Dict
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
