# constants.py
import numpy as np

# Physical constants & mission parameters
G0 = 9.81  # gravitational acceleration (m/s²)
ISP = np.array([282, 348, 350])  # specific impulse for stage 1 and stage 2 (s)
EPSILON = np.array([0.03, 0.07, 0.06])  # structural mass fractions for each stage
PAYLOAD_FRACTION = 0.03  # payload fraction (3% of total mass)
TOTAL_DELTA_V = 10500  # required net mission ΔV (m/s)

# Losses (m/s)
GRAVITY_LOSS = 1500
DRAG_LOSS = 500

# Design problem settings
NUM_STAGES = len(ISP)
BOUNDS = [(0.05, 0.85) for _ in range(NUM_STAGES)]
X0 = np.array([0.1, 0.1])
