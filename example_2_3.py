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
#  SCritical Dumped system
# 
#-------------------------------------------------
# Author : Marco Campolo 2026/06/23


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the differential equation for a critically damped system
def define_dif_equation(z, t):
    m = 1.0     # Define mass (m) in kg
    k = 100.0   # Define stiffness (k) in N/m
    zai = 1.0   # Given damping ratio (xi = 1.0 for Critical Damping)
    
    omega = np.sqrt(k / m)

    # Formulate the state-space equations
    x, v = z[0], z[1]
    dxdt = v
    dvdt = -2 * zai * omega * v - (omega ** 2) * x

    return [dxdt, dvdt]

# Initial conditions matching Figure 2.3 (cases i, ii, iii)
IC_cases = {
    "(i) x(0)=0.0, v(0)=0.1": [0.0, 0.1],
    "(ii) x(0)=0.01, v(0)=0.1": [0.01, 0.1],
    "(iii) x(0)=-0.01, v(0)=-0.1": [-0.01, -0.1]
}
t = np.arange(0, 2.005, 0.005)  # Time duration from 0 to 2 sec with 0.005 sec interval

# Plotting Results Setup
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

# Loop through each case to solve and plot
for label, IC in IC_cases.items():
    # Solve the ODE
    z = odeint(define_dif_equation, IC, t)

    # Find maximum and minimum values of displacement and velocity
    dismax, maxindex1 = np.max(z[:, 0]), np.argmax(z[:, 0])
    dismin, minindex1 = np.min(z[:, 0]), np.argmin(z[:, 0])
    dtimemax = t[maxindex1]
    dtimemin = t[minindex1]

    velmax, maxindex2 = np.max(z[:, 1]), np.argmax(z[:, 1])
    velmin, minindex2 = np.min(z[:, 1]), np.argmin(z[:, 1])
    vtimemax = t[maxindex2]
    vtimemin = t[minindex2]

    # Display extreme peak values and their timestamps
    print(f"\n--- Case {label} ---")
    print(f"dismax = {dismax:.4f} m at time = {dtimemax:.3f} sec")
    print(f"dismin = {dismin:.4f} m at time = {dtimemin:.3f} sec")
    print(f"velmax = {velmax:.4f} m/s at time = {vtimemax:.3f} sec")
    print(f"velmin = {velmin:.4f} m/s at time = {vtimemin:.3f} sec")

    # Plot onto subplots
    ax1.plot(t, z[:, 0], lw=1.5, label=label)
    ax2.plot(t, z[:, 1], lw=1.5, label=label)

# Finalize Displacement Plot
ax1.grid(True)
ax1.set_xlabel('Time (sec)')
ax1.set_ylabel('Displacement (m)')
ax1.set_title('Displacement Time History (Critically Damped System)')
ax1.legend()

# Finalize Velocity Plot
ax2.grid(True)
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_title('Velocity Time History')
ax2.legend()

plt.tight_layout()
plt.savefig("critically_damped_rev_2.pdf", format="pdf")
plt.show()