# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:40:25 2026

Copyright (c) 2026, Campolo Marco
All rights reserved.

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

#----------------------------------------------------
# Free Vibration of a Mass on a Simply Supported Beam
# 
#----------------------------------------------------
# Author : Marco Campolo 2026/06/13


import numpy as np
import matplotlib.pyplot as plt

# 1. Define System Parameters
m = 1500.0   # Mass (kg)
L = 6.0      # Length of the beam (m)
E = 200e9    # Elastic Modulus of steel (Pa or N/m^2)
I = 4e-5     # Moment of Inertia (m^4)

# 2. Calculate Dynamic Properties
k = (48 * E * I) / (L**3)  # Equivalent stiffness (N/m)
omega_n = np.sqrt(k / m)   # Natural circular frequency (rad/s)
f_n = omega_n / (2 * np.pi) # Natural frequency (Hz)

print(f"System Stiffness (k): {k:.2f} N/m")
print(f"Natural Frequency (fn): {f_n:.2f} Hz")

# 3. Time Vector for Simulation
t = np.linspace(0, 2, 500) # Simulate for 2 seconds, 500 data points

# 4. Initial Conditions
x0 = 0.02  # Initial displacement of 2 cm (0.02 m) pulled downward
v0 = 0.0   # Released from rest (initial velocity = 0)

# 5. Equation of Motion (Analytical Solution)
# x(t) = x0*cos(w_n*t) + (v0/w_n)*sin(w_n*t)
x_t = x0 * np.cos(omega_n * t) 

# 6. Plotting the Movement
plt.figure(figsize=(10, 5))
plt.plot(t, x_t, label="Displacement $x(t)$", color='blue', linewidth=2)
plt.axhline(0, color='black', linestyle='--') # Equilibrium line
plt.title("Free Vibration of a Mass on a Simply Supported Beam")
plt.xlabel("Time (seconds)")
plt.ylabel("Displacement (meters)")
plt.grid(True, linestyle=':', alpha=0.7)
filename = 'Dynamic structure exemple 1.1.pdf'
plt.savefig(filename, bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.legend()
plt.show()