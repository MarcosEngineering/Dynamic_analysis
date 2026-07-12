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
# Free Vibration of a Mass with Spring
# and harmonic Force applyed
# 
#-------------------------------------------------
# Author : Marco Campolo 2026/07/12

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Parameters of System
# ---------------------------------------------------------
m = 33.33        # Mass (kN-s^2/m)
k = 7500.0       # Spring constant (kN/m)
c = 0.0          # Damping coefficient (Undamped, c = 0)
P0 = 100.0       # Force amplitude (kN)
omega_bar = 3.5  # Forcing frequency (rad/sec)

# Derived variables for analytical components
omega = np.sqrt(k / m)      # Natural frequency
beta = omega_bar / omega    # Frequency ratio

# ---------------------------------------------------------
# Numerical Integration setup
# ---------------------------------------------------------
def Diff_equ(z, t):
    x, v = z
    F = P0 * np.sin(omega_bar * t)  # Harmonic force p(t)
    dxdt = v
    dvdt = (F - c * v - k * x) / m
    return [dxdt, dvdt]

IC = [0.0, 0.0]  # Initial conditions: system at rest x(0)=0, v(0)=0
tf = 3.0         # Total duration to match x-axis

t = np.arange(0, tf, 0.005)
z = odeint(Diff_equ, IC, t)

# Total response from the differential equation solver
x_total = z[:, 0]  # Displacement
v_total = z[:, 1]  # Velocity

# Calculate Total Acceleration using the equation of motion
# a = (F(t) - c*v - k*x) / m
F_t = P0 * np.sin(omega_bar * t)
a_total = (F_t - c * v_total - k * x_total) / m

# ---------------------------------------------------------
# Analytical Components for Displacement
# ---------------------------------------------------------
# Steady state response
x_steady = (P0 / k) * (1 / (1 - beta**2)) * np.sin(omega_bar * t)

# Transient response
x_transient = - (P0 / k) * (beta / (1 - beta**2)) * np.sin(omega * t)

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
# Create a figure with 3 stacked subplots sharing the x-axis
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# 1. Displacement Plot
axs[0].plot(t, x_transient, 'k:', label='Transient response')
axs[0].plot(t, x_steady, 'k--', label='Steady state response')
axs[0].plot(t, x_total, 'b-', label='Total response') 
axs[0].set_ylabel('Displacement (m)')
axs[0].set_ylim(-0.02, 0.02)
axs[0].grid(True)
# Place legend outside/above the first plot
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, frameon=False)

# 2. Velocity Plot
axs[1].plot(t, v_total, 'g-', label='Velocity')
axs[1].set_ylabel('Velocity (m/s)')
axs[1].grid(True)

# 3. Acceleration Plot
axs[2].plot(t, a_total, 'r-', label='Acceleration')
axs[2].set_ylabel('Acceleration (m/s²)')
axs[2].set_xlabel('Time (sec)')
axs[2].set_xlim(0, 3.0)
axs[2].grid(True)

# Adjust layout so subplots fit well and save
plt.tight_layout()
fig.subplots_adjust(top=0.9) # Give a little extra room at the top for the legend

plt.savefig("example_3_1.pdf", format="pdf")
plt.show()