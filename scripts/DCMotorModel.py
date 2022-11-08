from scipy import signal
import numpy as np

# El motor SS representation
# define motor parameters
# Nominal values
# L = 1e-3
# R = 1.0
# J = 5e-5
# b = 1e-4
# K = 0.1

# Working values
L = 0.9e-3
R = 1.1
J = 5e-5
b = 0.9e-4
K = 0.13

L = 1.4e-3
R = 1.8
J = 3.5e-5
b = 0.9e-4
K = 0.13

# define motor state variable model
A = np.array([[-R / L, 0, -K / L], [0, 0, 1], [K / J, 0, -b / J]])
B = np.array([[1.0 / L], [0], [0]])
C = np.array(
    [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 9.5493]]
)  # Convert units from rad/s to RPM
D = np.array([[0], [0], [0]])

DCMotor_ct = signal.StateSpace(A, B, C, D)

# Discretization
Ts = 0.01  # s
DCMotor_dt = DCMotor_ct.to_discrete(dt=Ts, method="bilinear")

print(f"DC motor model loaded, sampling time Ts = {Ts}.")
