
"""
config.py

Global configuration for the Kairoz simulator.
"""

# ============================================================
# SIMULATION
# ============================================================

DT = 0.01
SIM_TIME = 61.0

X0 = 1.0
V0 = 0.0

SEED = 42


# ============================================================
# PLANT
# ============================================================

MASS = 1.0
DAMPING = 0.4
SPRING = 2.0


# ============================================================
# EXCITATION
# ============================================================

EXCITATION_TYPE = "prbs"
EXCITATION_SWITCH_TIME = 30.0


# ============================================================
# CONTROLLER
# ============================================================

ENABLE_LQR = False
Q = [10, 50]
R = 10


# ============================================================
# RLS
# ============================================================

RLS_LAMBDA = 0.99


# ============================================================
# RESIDUAL MLP
# ============================================================

ENABLE_RESIDUAL_AI = True

MLP_LEARNING_RATE = 1e-3

MLP_BUFFER_SIZE = 5000

MLP_BATCH_SIZE = 64

MLP_UPDATE_EVERY = 10

MLP_START_TIME = 15.0


# ============================================================
# DISTURBANCES
# ============================================================

ENABLE_DISTURBANCES = True

CONSTANT_DISTURBANCE = True
CONSTANT_FORCE = 5.0

SINE_ENABLE = True
SINE_AMPLITUDE = 2.0
SINE_FREQUENCY = 1.2

IMPULSE_ENABLE = False
IMPULSE_TIME = 9.0
IMPULSE_DURATION = 0.2
IMPULSE_FORCE = 5.0

OU_ENABLE = False
OU_THETA = 2.0
OU_SIGMA = 0.8


# ============================================================
# SENSOR MODEL (v0.3)
# ============================================================

ENABLE_SENSOR_NOISE = False

POSITION_STD = 0.01
VELOCITY_STD = 0.01

POSITION_BIAS = 0.0
VELOCITY_BIAS = 0.0

SENSOR_RATE = 100

QUANTIZATION = None


# ============================================================
# LOGGING
# ============================================================

SAVE_DATA = False

SAVE_PLOTS = False

VERBOSE = True
