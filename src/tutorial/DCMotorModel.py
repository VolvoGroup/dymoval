import control as ct
import numpy as np
import dymoval as dmv

# El motor SS representation
# define motor parameters
# Nominal values
L = 1e-3
R = 1.0
J = 5e-5
b = 1e-4
K = 0.1

# Working values
L = 0.9e-3
R = 1.1
J = 5e-5
b = 0.9e-4
K = 0.12

# L = 1e-3
# R = 1.8
# J = 4.5e-5
# b = 0.9e-4
# K = 0.12

# define motor state variable model
A = np.array([[-R / L, 0, -K / L], [0, 0, 1], [K / J, 0, -b / J]]).round(
    dmv.NUM_DECIMALS
)
B = np.array([[1.0 / L], [0], [0]]).round(dmv.NUM_DECIMALS)
C = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 9.5493]]).round(
    dmv.NUM_DECIMALS
)  # Convert units from rad/s to RPM
D = np.array([[0], [0], [0]]).round(dmv.NUM_DECIMALS)

DCMotor_ct = ct.ss(A, B, C, D)

# Discretization
Ts = 0.01  # s
DCMotor_dt = ct.sample_system(DCMotor_ct, Ts, method="zoh")
