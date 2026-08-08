"""
Created on Sun Jul 26 03:42:26 2026
# -*- coding: utf-8 -*-

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

@author: mcamp
"""

#-------------------------------------------------
# Free Vibration of a Mass with Spring, damping
# system and Fourier Triangular Force applied
# 
#-------------------------------------------------
# Author : Marco Campolo 2026/07/12 (Modified)

import numpy as np
from scipy.integrate import odeint, quad
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Fourier Force Definitions (Triangular Function)
# ---------------------------------------------------------
def curve1(t, T, P):
    if t <= T / 2:
        return (1 / T) * (2 * t * P / T)
    else:
        return (1 / T) * (2 * (T - t) * P / T)

def curve1c(t, n, T, P):
    w = 2 * np.pi / T
    if t <= T / 2:
        p1 = 2 * t * P / T
    else:
        p1 = 2 * (T - t) * P / T
    return (2 / T) * p1 * np.cos(w * t * n)

def curve1s(t, n, T, P):
    w = 2 * np.pi / T
    if t <= T / 2:
        p1 = 2 * t * P / T
    else:
        p1 = 2 * (T - t) * P / T
    return (2 / T) * p1 * np.sin(w * t * n)

# ---------------------------------------------------------
# Parameters of System
# ---------------------------------------------------------
m = 5.0          # Mass (kNs^2/m)
k = 100.0        # Spring constant (kN/m)
zeta = 0.05      # Damping ratio (5%)

# Force parameters
P0 = 10.0        # Force amplitude (kN)
T_force = 2.0    # Period of the triangular force (sec)
N_terms = 10     # Number of Fourier terms to include

# Derived variables
omega = np.sqrt(k / m)      # Natural frequency (rad/sec)
period = 2 * np.pi / omega  # Natural period of vibration (sec)
c = 2 * zeta * m * omega    # Damping coefficient (kNs/m)

# ---------------------------------------------------------
# Pre-calculate Fourier Coefficients
# ---------------------------------------------------------
print("Calculating Fourier coefficients...")
a0, _ = quad(curve1, 0, T_force, args=(T_force, P0), epsabs=1e-9)
a = np.zeros(N_terms)
b = np.zeros(N_terms)

for n in range(1, N_terms + 1):
    a[n-1], _ = quad(curve1c, 0, T_force, args=(n, T_force, P0), epsabs=1e-9)
    b[n-1], _ = quad(curve1s, 0, T_force, args=(n, T_force, P0), epsabs=1e-9)

def force_fourier(t):
    """Reconstructs the force at a specific time t using Fourier series."""
    F = a0
    for n in range(1, N_terms + 1):
        omega_n = 2 * np.pi * n / T_force
        F += a[n-1] * np.cos(omega_n * t) + b[n-1] * np.sin(omega_n * t)
    return F

# ---------------------------------------------------------
# Numerical Integration setup
# ---------------------------------------------------------
def Diff_equ(z, t):
    x, v = z
    F = force_fourier(t)  # Applied Fourier force p(t)
    dxdt = v
    dvdt = (F - c * v - k * x) / m
    return [dxdt, dvdt]

# Initial conditions: x(0)=0.02 m, v(0)=0.05 m/s
IC = [0.02, 0.05] 

# Total duration
tf = 15.0         

t = np.arange(0, tf, 0.005)
print("Integrating equation of motion...")
z = odeint(Diff_equ, IC, t)

# State variables
x_total = z[:, 0]  # Total Displacement
v_total = z[:, 1]  # Total Velocity

# ---------------------------------------------------------
# Steady-State and Transient Displacement Calculations
# ---------------------------------------------------------
# Steady-state response calculated via Superposition of harmonics
x_steady = np.ones_like(t) * (a0 / k) # Static response to a0 term

for n in range(1, N_terms + 1):
    omega_n = 2 * np.pi * n / T_force
    
    # Denominator and phase for the n-th harmonic
    denom = np.sqrt((k - m * omega_n**2)**2 + (c * omega_n)**2)
    phi_n = np.arctan2(c * omega_n, k - m * omega_n**2)
    
    # Response to cosine terms
    x_steady += (a[n-1] / denom) * np.cos(omega_n * t - phi_n)
    # Response to sine terms
    x_steady += (b[n-1] / denom) * np.sin(omega_n * t - phi_n)

# Transient response: x_c(t) = Total response - Steady-state response
x_transient = x_total - x_steady

# ---------------------------------------------------------
# Acceleration and Forces Calculation
# ---------------------------------------------------------
# Reconstruct the applied harmonic force over the entire time array
F_applied = np.zeros_like(t) + a0
for n in range(1, N_terms + 1):
    omega_n = 2 * np.pi * n / T_force
    F_applied += a[n-1] * np.cos(omega_n * t) + b[n-1] * np.sin(omega_n * t)

# Acceleration from equation of motion: a = (F - cv - kx) / m
a_total = (F_applied - c * v_total - k * x_total) / m

# Individual resisting forces
F_spring = k * x_total
F_damping = c * v_total

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
print("Generating plots...")
# Create a figure with 4 stacked subplots sharing the x-axis
fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

# 1. Displacement Plot
axs[0].plot(t, x_transient, 'r:', lw=1.5, label='Transient Response')
axs[0].plot(t, x_steady, 'k--', lw=1.5, label='Steady-State Response')
axs[0].plot(t, x_total, 'b-', lw=2.0, label='Total Response') 
axs[0].set_ylabel('Displacement (m)')
axs[0].grid(True)
axs[0].legend(loc='upper right', ncol=3, fontsize='small')
axs[0].set_title('SDOF System Response - Fourier Triangular Force')

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

output_pdf = "example_4_1_Fourier.pdf"
plt.savefig(output_pdf, format="pdf")
print(f"Plot saved successfully as {output_pdf}")

plt.show()