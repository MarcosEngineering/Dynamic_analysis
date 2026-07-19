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
#   Dynamic magnification factor plot
#   Phase angle plot
#   beta = frequency ratio
# 
#-------------------------------------------------
# Author : Marco Campolo 2026/07/13



import numpy as np
import matplotlib.pyplot as plt

# Values of zai (damping ratio) and beta (frequency ratio)
# Reshaping zai to (5, 1) allows it to broadcast against beta (4.01,)
zai = np.array([0.01, 0.10, 0.20, 0.70, 1.0]).reshape(-1, 1)
beta = np.arange(0, 4.01, 0.01)

# 1. Calculate dynamic magnification factor (d) using broadcasting
d = 1 / np.sqrt((1 - beta**2)**2 + (2 * zai * beta)**2)

# 2. Calculate phase angles (theta) using arctan2
# np.arctan2(y, x) handles x=0 automatically and places the angle in the correct quadrant
theta_rad = np.arctan2(2 * zai * beta, 1 - beta**2)
theta_deg = np.degrees(theta_rad)

# --- Plotting ---

# Plot dynamic magnification factor
plt.figure(figsize=(8, 5))
for i, z in enumerate(zai.flatten()):
    plt.plot(beta, d[i], label=f'ξ = {z:.4g}')
plt.grid(True)
plt.xlabel('Frequency Ratio ($\\beta$)')
plt.ylabel('Magnification Factor D')
plt.axis([0, 4, 0, 10])
plt.legend()
plt.title("Dynamic Magnification Factor vs. Frequency Ratio")
plt.tight_layout()
plt.savefig("Dynamic Magnification_Factor vs Frequency Ratio.pdf", format="pdf")
plt.show()

# Plot phase angles
plt.figure(figsize=(8, 5))
for i, z in enumerate(zai.flatten()):
    plt.plot(beta, theta_deg[i], label=f'ξ = {z:.4g}')
plt.grid(True)
plt.xlabel('Frequency Ratio ($\\beta$)')
plt.ylabel('Phase Angle ϑ (degrees)')
plt.yticks(np.arange(0, 181, 45)) # Helpful to show 0, 45, 90, 135, 180
plt.legend()
plt.title("Phase Angle vs. Frequency Ratio")
plt.tight_layout()
plt.savefig("Phase Angle vs. Frequency Ratio.pdf", format="pdf")
plt.show()





