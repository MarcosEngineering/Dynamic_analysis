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

# ==========================================
# 1. Setup Parameters
# ==========================================
# Frequency ratios (beta) for the steady-state plot
beta = np.linspace(0.0, 2.5, 1000)

# Time array for the time-domain plot (0 to 60 seconds)
t = np.linspace(0, 60, 2000)

# Natural frequency (set to 1 rad/s for normalized simulation)
omega = 1.0 

# Damping ratios for comparison
xi_small = 0.05  # 5% damping
xi_large = 0.15  # 15% damping
xi_med = 0.70    # 70% damping
xi_crit = 1.00   # 100% damping (Critical damping)

# ==========================================
# 2. Define the Equations from the Text
# ==========================================

# Eq (3.17): Damped Steady-State Dynamic Magnification Factor
def D_steady_state(beta, xi):
    return 1.0 / np.sqrt((1 - beta**2)**2 + (2 * beta * xi)**2)

# Eq (3.16e): Undamped Steady-State (Avoid division by zero at beta=1)
# We use a masked array to safely plot the asymptote
D_undamped_ss = np.divide(1.0, np.abs(1 - beta**2), out=np.zeros_like(beta), where=(beta!=1))
D_undamped_ss[np.isclose(beta, 1.0, atol=1e-2)] = np.nan 

# Eq (3.23): Undamped Time-Domain Resonant Response
D_t_undamped = 0.5 * (np.sin(omega * t) - omega * t * np.cos(omega * t))

# Eq (3.22): Damped Time-Domain Resonant Response
D_t_damped_small = (1 / (2 * xi_small)) * (np.exp(-xi_small * omega * t) - 1) * np.cos(omega * t)
D_t_damped_large = (1 / (2 * xi_large)) * (np.exp(-xi_large * omega * t) - 1) * np.cos(omega * t)
D_t_damped_med   = (1 / (2 * xi_med))   * (np.exp(-xi_med * omega * t)   - 1) * np.cos(omega * t)
D_t_damped_crit  = (1 / (2 * xi_crit))  * (np.exp(-xi_crit * omega * t)  - 1) * np.cos(omega * t)

# ==========================================
# 3. Create the Visualization
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Panel A: Steady-State Response (Frequency Domain) ---
ax1.plot(beta, D_steady_state(beta, xi_small), 'b-', label=f'Damped system, $\\xi$={xi_small}')
ax1.plot(beta, D_steady_state(beta, xi_large), 'g-', label=f'Damped system, $\\xi$={xi_large}')
ax1.plot(beta, D_steady_state(beta, xi_med), color='orange', label=f'Damped system, $\\xi$={xi_med}')
ax1.plot(beta, D_steady_state(beta, xi_crit), color='purple', label=f'Damped system, $\\xi$={xi_crit}')
ax1.plot(beta, D_undamped_ss, 'r--', label='Undamped system, $\\xi$=0')

# Highlight Resonance
ax1.axvline(x=1.0, color='gray', linestyle=':', label='Resonance ($\\beta=1$)')
ax1.set_ylim(0, 12)
ax1.set_xlim(0, 2.5)
ax1.set_title("Steady-State Magnification Factor vs. Frequency Ratio")
ax1.set_xlabel("Frequency Ratio $\\beta = \\omega / \\bar{\\omega}$")
ax1.set_ylabel("Magnification Factor $D$")
ax1.grid(True, alpha=0.3)
ax1.legend()

# --- Panel B: Time-Domain Response at Perfect Resonance (beta = 1) ---
ax2.plot(t, D_t_undamped, 'r-', label='Undamped system')
ax2.plot(t, D_t_damped_small, 'b-', label=f'Damped system $\\xi$={xi_small}')
ax2.plot(t, D_t_damped_large, 'g-', label=f'Damped system $\\xi$={xi_large}')
ax2.plot(t, D_t_damped_med, color='orange', label=f'Damped system $\\xi$={xi_med}')
ax2.plot(t, D_t_damped_crit, color='purple', label=f'Damped system $\\xi$={xi_crit}')

# Plot the linear envelope for the undamped system to show the math clearly
envelope = 0.5 * omega * t
ax2.plot(t, envelope, 'r:', alpha=0.7, label="Linear Envelope ($0.5 \\omega t$)")
ax2.plot(t, -envelope, 'r:', alpha=0.7)

# Limits
limit_small = 1 / (2 * xi_small)
ax2.axhline(y=limit_small, color='b', linestyle=':', alpha=0.7, label=f"Damped Limit $\\xi$={xi_small} ($D_{{max}}={limit_small:.1f}$)")
ax2.axhline(y=-limit_small, color='b', linestyle=':', alpha=0.7)

limit_large = 1 / (2 * xi_large)
ax2.axhline(y=limit_large, color='g', linestyle=':', alpha=0.7, label=f"Damped Limit $\\xi$={xi_large} ($D_{{max}}={limit_large:.1f}$)")
ax2.axhline(y=-limit_large, color='g', linestyle=':', alpha=0.7)

limit_med = 1 / (2 * xi_med)
ax2.axhline(y=limit_med, color='orange', linestyle=':', alpha=0.7, label=f"Damped Limit $\\xi$={xi_med} ($D_{{max}}={limit_med:.2f}$)")
ax2.axhline(y=-limit_med, color='orange', linestyle=':', alpha=0.7)

limit_crit = 1 / (2 * xi_crit)
ax2.axhline(y=limit_crit, color='purple', linestyle=':', alpha=0.7, label=f"Damped Limit $\\xi$={xi_crit} ($D_{{max}}={limit_crit:.1f}$)")
ax2.axhline(y=-limit_crit, color='purple', linestyle=':', alpha=0.7)

ax2.set_xlim(0, 60)
ax2.set_title("Time-Domain Response at Resonance ($\\beta = 1$)")
ax2.set_xlabel("Time $t$ (seconds)")
ax2.set_ylabel("Time-Dependent Magnification $D(t)$")
ax2.grid(True, alpha=0.3)

# Place the legend slightly outside to handle the extra entries cleanly
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

plt.tight_layout()
plt.savefig("example_3_4.pdf", format="pdf")
plt.show()