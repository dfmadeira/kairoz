import numpy as np

from .sensor_config import *

class GPSSensor:
    def __init__(self, config: GPSSensorConfig) -> None:
        self.config = config

    def measure(self, state):

        x = state.position

        x += self.config.bias
        x += np.random.normal(0, self.config.noise_std)

        if self.config.quantization > 0:
            x = round(x / self.config.quantization) * self.config.quantization

        return x


class EncoderSensor:
    def __init__(self, config: EncoderSensorConfig) -> None:
        self.config = config

    def measure(self, state):

        x = state.position

        ticks = x * self.config.resolution

        ticks += self.config.bias
        ticks += np.random.normal(0.0, self.config.noise_std)

        ticks = round(ticks)

        return ticks / self.config.resolution


class Accelerometer:
    def __init__(self, config: AccelerometerConfig) -> None:
        self.config = config

    def measure(self, state):
        a = state.acceleration

        a += self.config.bias
        a += np.random.normal(0.0, self.config.noise_std)

        return a

class Gyroscope:
    def __init__(self, config: GyroscopeConfig) -> None:
        self.config = config

    def measure(self, state):

        w = state.angular_velocity

        w += self.config.bias
        w += np.random.normal(0.0, self.config.noise_std)

        return w

class Magnetometer:
    def __init__(self, config: MagnetometerConfig) -> None:
        self.config = config

    def measure(self, state):

        m = state.angular_velocity

        m += self.config.bias
        m += np.random.normal(0.0, self.config.noise_std)

        return m
