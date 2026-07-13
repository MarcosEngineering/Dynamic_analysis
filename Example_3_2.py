# -*- coding: utf-8 -*-
"""
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

"""

#-------------------------------------------------
# Free Vibration of a Mass with Spring, damping
# system and harmonic Force applied
# 
#-------------------------------------------------
# Author : Marco Campolo 2026/07/12

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Parameters of System (Example 3.2)
# ---------------------------------------------------------
m = 5.0          # Mass (kNs^2/m)
k = 100.0        # Spring constant (kN/m)
zeta = 0.05      # Damping ratio (5%)
P0 = 10.0        # Force amplitude (kN)
omega_bar = 10.0 # Forcing frequency (rad/sec)

# Derived variables
omega = np.sqrt(k / m)      # Natural frequency (rad/sec)
period = 2 * np.pi / omega  # Natural period of vibration (sec)
c = 2 * zeta * m * omega    # Damping coefficient (kNs/m)

# ---------------------------------------------------------
# Numerical Integration setup
# ---------------------------------------------------------
def Diff_equ(z, t):
    x, v = z
    F = P0 * np.sin(omega_bar * t)  # Harmonic force p(t)
    dxdt = v
    dvdt = (F - c * v - k * x) / m
    return [dxdt, dvdt]

# Initial conditions: x(0)=0.02 m, v(0)=0.05 m/s
IC = [0.02, 0.05] 

# Total duration (set to 15 seconds to view transient decay and steady state)
tf = 15.0          

t = np.arange(0, tf, 0.005)
z = odeint(Diff_equ, IC, t)

# State variables
x_total = z[:, 0]  # Total Displacement
v_total = z[:, 1]  # Total Velocity

# ---------------------------------------------------------
# Steady-State and Transient Displacement Calculations
# ---------------------------------------------------------
# Calculate the analytical Steady-State amplitude (X) and phase angle (phi)
X_amp = P0 / np.sqrt((k - m * omega_bar**2)**2 + (c * omega_bar)**2)
phi = np.arctan2(c * omega_bar, k - m * omega_bar**2)

# Steady-state response: x_p(t) = X * sin(w_bar * t - phi)
x_steady = X_amp * np.sin(omega_bar * t - phi)

# Transient response: x_c(t) = Total response - Steady-state response
x_transient = x_total - x_steady

# ---------------------------------------------------------
# Acceleration and Forces Calculation
# ---------------------------------------------------------
# Applied harmonic force over time
F_applied = P0 * np.sin(omega_bar * t)

# Acceleration from equation of motion: a = (F - cv - kx) / m
a_total = (F_applied - c * v_total - k * x_total) / m

# Individual resisting forces
F_spring = k * x_total
F_damping = c * v_total

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
# Create a figure with 4 stacked subplots sharing the x-axis
fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

# 1. Displacement Plot
axs[0].plot(t, x_transient, 'r:', lw=1.5, label='Transient Response')
axs[0].plot(t, x_steady, 'k--', lw=1.5, label='Steady-State Response')
axs[0].plot(t, x_total, 'b-', lw=2.0, label='Total Response') 
axs[0].set_ylabel('Displacement (m)')
axs[0].grid(True)
axs[0].legend(loc='upper right', ncol=3, fontsize='small')
axs[0].set_title('SDOF System Response -Damping forced system ')

# 2. Velocity Plot
axs[1].plot(t, v_total, 'g-', label='Total Velocity')
axs[1].set_ylabel('Velocity (m/s)')
axs[1].grid(True)
axs[1].legend(loc='upper right')

# 3. Acceleration Plot
axs[2].plot(t, a_total, 'r-', label='Total Acceleration')
axs[2].set_ylabel('Acceleration (m/s²)')
axs[2].grid(True)
axs[2].legend(loc='upper right')

# 4. Forces Plot
axs[3].plot(t, F_applied, 'k--', label='Applied Force P(t)')
axs[3].plot(t, F_spring, 'c-', label='Spring Force (kx)')
axs[3].plot(t, F_damping, 'm-', label='Damping Force (cv)')
axs[3].set_ylabel('Force (kN)')
axs[3].set_xlabel('Time (sec)')
axs[3].set_xlim(0, tf)
axs[3].grid(True)
axs[3].legend(loc='upper right', ncol=3, fontsize='small')

# Adjust layout so subplots fit well and save
plt.tight_layout()

plt.savefig("example_3_2.pdf", format="pdf")
plt.show()