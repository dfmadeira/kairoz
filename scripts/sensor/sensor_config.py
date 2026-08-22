from dataclasses import dataclass
from typing import List


# ============================================================================
# Base Configuration
# ============================================================================

@dataclass
class SensorConfig:
    name: str
    sample_rate: float          # [Hz]
    delay: float                # [s]


# ============================================================================
# Position Sensors
# ============================================================================

@dataclass
class GPSSensorConfig(SensorConfig):
    noise_std: float            # [m]
    bias: float                 # [m]
    quantization: float         # [m]


@dataclass
class EncoderSensorConfig(SensorConfig):
    resolution: int             # ticks / revolution
    noise_std: float            # [ticks]
    bias: float                 # [ticks]


# ============================================================================
# Inertial Sensors
# ============================================================================

@dataclass
class AccelerometerConfig(SensorConfig):
    noise_std: float            # [m/s²]
    bias: float                 # [m/s²]


@dataclass
class GyroscopeConfig(SensorConfig):
    noise_std: float            # [rad/s]
    bias: float                 # [rad/s]


@dataclass
class MagnetometerConfig(SensorConfig):
    noise_std: float
    bias: float


# ============================================================================
# GPS
# ============================================================================

GPS = [

    GPSSensorConfig(
        name="GPS0",
        sample_rate=10,
        delay=0.05,
        noise_std=0.0,
        bias=0.0,
        quantization=0.0
    ),

    GPSSensorConfig(
        name="GPS1",
        sample_rate=10,
        delay=0.05,
        noise_std=0.20,
        bias=0.0,
        quantization=0.01
    ),

]

# ============================================================================
# Accelerometers
# ============================================================================

ACCELEROMETERS = [

    AccelerometerConfig(
        name="IMU0_ACC",
        sample_rate=200,
        delay=0.002,
        noise_std=0.0,
        bias=0.0
    ),

    AccelerometerConfig(
        name="IMU1_ACC",
        sample_rate=200,
        delay=0.002,
        noise_std=0.05,
        bias=0.0
    ),

]

# ============================================================================
# Gyroscopes
# ============================================================================

GYROSCOPES = [

    GyroscopeConfig(
        name="IMU1_GYRO",
        sample_rate=200,
        delay=0.002,
        noise_std=0.002,
        bias=0.0
    ),

]

# ============================================================================
# Magnetometers
# ============================================================================

MAGNETOMETERS = [

    MagnetometerConfig(
        name="MAG1",
        sample_rate=50,
        delay=0.01,
        noise_std=0.01,
        bias=0.0
    ),

]

# ============================================================================
# Encoders
# ============================================================================

ENCODERS = [

    EncoderSensorConfig(
        name="ENCODER0",
        sample_rate=500,
        delay=0.0,
        resolution=4096,
        noise_std=0.0,
        bias=0.0
    ),

    EncoderSensorConfig(
        name="ENCODER1",
        sample_rate=500,
        delay=0.0,
        resolution=4096,
        noise_std=0.0,
        bias=0.0
    ),

]
