# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 21:53:35 2026

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
# Free Vibration of a Mass with Spring and dash-pot
# UnderDumped system
# 
#-------------------------------------------------
# Author : Marco Campolo 2026/06/24

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the differential equation
def define_dif_equation(z, t):
    m = 5.0     # Define mass (m) in kNs^2/m
    k = 100.0   # Define stiffness (k) in kN/m
    zai = 0.05  # Given damping ratio (xi = 5%)
    
    omega = np.sqrt(k / m)

    # Formulate the state-space equations
    x, v = z[0], z[1]
    dxdt = v
    dvdt = -2 * zai * omega * v - (omega ** 2) * x

    return [dxdt, dvdt]

# System parameters for envelope calculation
m = 5.0
k = 100.0
zai = 0.05
omega = np.sqrt(k / m)
omega_d = omega * np.sqrt(1 - zai**2)

# Initial conditions and time span
IC = [0.02, 0.05]  # Initial conditions x(0) and v(0)
t = np.arange(0, 10.005, 0.005)  # Time duration from 0 to 10 sec with 0.005 sec interval

# Solve the ODE
z = odeint(define_dif_equation, IC, t)

# Extract kinematic arrays
displacement = z[:, 0]
velocity = z[:, 1]
# Calculate acceleration using the equation of motion: a = -2*zai*omega*v - omega^2*x
acceleration = -2 * zai * omega * velocity - (omega ** 2) * displacement

# Calculate theoretical exponential decay envelope
A = IC[0]
B = (IC[1] + zai * omega * IC[0]) / omega_d
amplitude = np.sqrt(A**2 + B**2)
envelope_upper = amplitude * np.exp(-zai * omega * t)
envelope_lower = -amplitude * np.exp(-zai * omega * t)

# Find maximum and minimum values of displacement, velocity, and acceleration
dismax, maxindex1 = np.max(displacement), np.argmax(displacement)
dismin, minindex1 = np.min(displacement), np.argmin(displacement)
dtimemax = t[maxindex1]
dtimemin = t[minindex1]

velmax, maxindex2 = np.max(velocity), np.argmax(velocity)
velmin, minindex2 = np.min(velocity), np.argmin(velocity)
vtimemax = t[maxindex2]
vtimemin = t[minindex2]

accmax, maxindex3 = np.max(acceleration), np.argmax(acceleration)
accmin, minindex3 = np.min(acceleration), np.argmin(acceleration)
atimemax = t[maxindex3]
atimemin = t[minindex3]

# Display extreme peak values and their timestamps
print(f"\ndismax = {dismax:.4f} m at time = {dtimemax:.3f} sec")
print(f"dismin = {dismin:.4f} m at time = {dtimemin:.3f} sec")
print(f"velmax = {velmax:.4f} m/s at time = {vtimemax:.3f} sec")
print(f"velmin = {velmin:.4f} m/s at time = {vtimemin:.3f} sec")
print(f"accmax = {accmax:.4f} m/s^2 at time = {atimemax:.3f} sec")
print(f"accmin = {accmin:.4f} m/s^2 at time = {atimemin:.3f} sec")

# Plotting Results
plt.figure(figsize=(10, 10)) # Increased height to accommodate 3 subplots

# 1. Displacement Plot
plt.subplot(311)
plt.plot(t, displacement, 'b-', lw=1.5, label='Displacement')
plt.plot(t, envelope_upper, 'k--', lw=1.2, label='Exponential Decay ')
plt.plot(t, envelope_lower, 'k--', lw=1.2)
plt.grid(True)
plt.xlabel('Time (sec)')
plt.ylabel('Displacement (m)')
plt.title('Displacement Time History (Underdamped Free Vibration)')
plt.legend(loc='upper right')

# 2. Velocity Plot
plt.subplot(312)
plt.plot(t, velocity, 'g-', lw=1.5)
plt.grid(True)
plt.xlabel('Time (sec)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Time History')

# 3. Acceleration Plot
plt.subplot(313)
plt.plot(t, acceleration, 'r-', lw=1.5)
plt.grid(True)
plt.xlabel('Time (sec)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Acceleration Time History')

plt.tight_layout()
plt.savefig("example_2_2.pdf", format="pdf")
plt.show()