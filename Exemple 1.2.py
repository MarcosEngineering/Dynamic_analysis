# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:43:33 2026

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

#----------------------------------------------
# Free Vibration of a Mass in cantilever-spring
# Beam 
#----------------------------------------------
# Author : Marco Campolo 2026/06/14

import numpy as np
import matplotlib.pyplot as plt

# 1. Define System Parameters
m = 500.0          # Mass of the block (kg)
L = 3.0            # Length of cantilever beam (m)
E = 200e9          # Elastic Modulus of steel (Pa or N/m^2)
I = 1.5e-5         # Moment of Inertia (m^4)
k_spring = 2e6     # Stiffness of the suspended spring (N/m)

# 2. Calculate Individual Stiffnesses
# Formula for a cantilever beam with a point load at the free end
k_beam = (3 * E * I) / (L**3)  

# 3. Calculate Equivalent System Stiffness (Springs in Series)
k_eq = (k_beam * k_spring) / (k_beam + k_spring)

# 4. Calculate Dynamic Properties
omega_n = np.sqrt(k_eq / m)    # Natural circular frequency (rad/s)
f_n = omega_n / (2 * np.pi)    # Natural frequency (Hz)

print(f"Beam Stiffness: {k_beam:,.2f} N/m")
print(f"Spring Stiffness: {k_spring:,.2f} N/m")
print(f"Equivalent Stiffness (ke): {k_eq:,.2f} N/m")
print(f"Natural Frequency (fn): {f_n:.2f} Hz")

# 5. Time Vector for Simulation
t = np.linspace(0, 2, 500)     # Simulate for 2 seconds, 500 data points

# 6. Initial Conditions
x0 = -0.15                     # Initial dynamic displacement of -15 cm (-0.15 m)
v0 = 0.0                       # Released from rest

# 7. Equation of Motion (Analytical Solution)
# x(t) = x0*cos(w_n*t)
x_t = x0 * np.cos(omega_n * t) 

# 8. Plotting the Movement
plt.figure(figsize=(10, 5))
plt.plot(t, x_t, label="Total Displacement $x(t)$", color='#D32F2F', linewidth=2)
plt.axhline(0, color='black', linestyle='--') # Equilibrium line
plt.title("Free Vibration: Mass on a Spring Attached to a Cantilever Beam")
plt.xlabel("Time (seconds)")
plt.ylabel("Displacement (meters)")
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc="lower right")
filename = 'Dynamic structure exemple.pdf'
plt.savefig(filename, bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.show()