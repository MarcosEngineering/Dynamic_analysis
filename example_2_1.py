# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 10:19:03 2026


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
# Free Vibration of a Mass in Springs
# Undumped system
# 
#----------------------------------------------
# Author : Marco Campolo 2026/06/20

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the spring-mass system equation of motion function
def spring_mass(z, t, k, m):
    # z[0] is displacement (x), z[1] is velocity (v)
    # returns [velocity, acceleration]
    return [z[1], -k * z[0] / m]

# Values from Example 2.1:
# mass = 40 kNs^2/m, stiffness k = 3500 kN/m
m = 40.0
k = 3500.0

# Calculate the natural frequency and period of vibration
omega = np.sqrt(k / m)
period = 2 * np.pi / omega

# Display the calculated values
print("Mass = {:.4f} kNs^2/m".format(m))
print("Stiffness = {:.4f} kN/m".format(k))
print("Natural frequency = {:.4f} rad/sec".format(omega))
print("Period of vibration = {:.4f} sec".format(period))

# Initial conditions from Example 2.1: x(0) = 0.01 m, v(0) = 0.1 m/s
ic = np.array([0.01, 0.1])

# Time points from 0 to 5 seconds (matching textbook graphs)
t = np.linspace(0, 5, 500)

# Solve the ordinary differential equation using odeint
y = odeint(spring_mass, ic, t, args=(k, m))

displacement = y[:, 0]
velocity = y[:, 1]

# Calculate acceleration, elastic force, and inertia force
# From equation of motion: acceleration = -k * displacement / m
acceleration = -k * displacement / m
elastic_force = k * displacement
inertia_force = m * acceleration

#----------------------------------------------------
# Plotting the responses to match Figure 2.6
#----------------------------------------------------
plt.figure(figsize=(10, 12))

# (a) Displacement
plt.subplot(4, 1, 1)
plt.plot(t, displacement, 'k-')
plt.ylabel("Displacement (m)")
plt.xlim(0, 5)
plt.grid(True)

# (b) Velocity
plt.subplot(4, 1, 2)
plt.plot(t, velocity, 'k-')
plt.ylabel("Velocity (m/s)")
plt.xlim(0, 5)
plt.grid(True)

# (c) Acceleration
plt.subplot(4, 1, 3)
plt.plot(t, acceleration, 'k-')
plt.ylabel("Acceleration (m/s²)")
plt.xlim(0, 5)
plt.grid(True)

# (d) Elastic and Inertia Forces
plt.subplot(4, 1, 4)
plt.plot(t, elastic_force, 'k--', label="Elastic force")
plt.plot(t, inertia_force, 'k-', label="Inertia force")
plt.xlabel("Time (sec)")
plt.ylabel("Force, kN")
plt.xlim(0, 5)
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()

# Save the figure as a PDF file
plt.savefig("Example_2_1.pdf", format="pdf")
plt.show()