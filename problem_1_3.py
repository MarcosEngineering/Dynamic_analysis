# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 18:48:38 2026


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
# Free Vibration of a Mass in Springs System
# 
#----------------------------------------------
# Author : Marco Campolo 2026/06/18

import math
import numpy as np
import matplotlib.pyplot as plt

def calculate_displacement(m, k1, k2, k3, initial_pull, time_duration):
    """
    Calculates static deflection and plots dynamic vertical displacement.
    
    Parameters:
    m (float): Mass (kg)
    k1, k2, k3 (float): Spring stiffnesses (N/m)
    initial_pull (float): Initial downward displacement to start vibration (meters)
    time_duration (float): How many seconds to simulate (seconds)
    """
    g = 9.81 # Acceleration due to gravity (m/s^2)
    
    # 1. System Properties (from previous analysis)
    k_parallel = k2 + k3
    k_eq = (k1 * k_parallel) / (k1 + k_parallel)
    omega_n = math.sqrt(k_eq / m)
    
    # 2. Static Deflection Calculation
    # How far the mass sags just by hanging there
    static_deflection = (m * g) / k_eq
    
    # 3. Dynamic Displacement Calculation (over time)
    # Generate an array of time points from 0 to time_duration
    t = np.linspace(0, time_duration, 1000)
    
    # Calculate vertical position at each time point x(t) = X0 * cos(w_n * t)
    # Note: This represents displacement from the static equilibrium position
    x_t = initial_pull * np.cos(omega_n * t)
    
    # --- Console Output ---
    print("--- Displacement Analysis ---")
    print(f"Equivalent Stiffness (k_eq): {k_eq:.2f} N/m")
    print(f"Static Deflection: {static_deflection:.4f} m ({static_deflection * 1000:.2f} mm)")
    print(f"Natural Circular Frequency: {omega_n:.2f} rad/s")
    
    # --- Plotting the Dynamic Displacement ---
    plt.figure(figsize=(10, 5))
    plt.plot(t, x_t, label="Dynamic Displacement $x(t)$", color='blue')
    
    # Formatting the plot
    plt.axhline(0, color='black', linewidth=1, linestyle='--') # Equilibrium line
    plt.title("Dynamic Vertical Displacement over Time (Free Vibration)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Vertical Displacement (meters)")
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # --- CHANGED LINE HERE ---
    plt.legend(loc='lower left')
    
    # Show the graph
    filename = 'Dynamic_structure_problem_1_3.pdf'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.show()
    
# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # System parameters
    mass_value = 10.0      # kg
    k1_value = 500.0       # N/m
    k2_value = 200.0       # N/m
    k3_value = 300.0       # N/m
    
    # Vibration parameters
    initial_stretch = -1.5 # Pulled Up 1.5 m (1.5m) before releasing
    simulation_time = 5.0  # Simulate for 5 seconds
    
    # Run the function
    calculate_displacement(
        m=mass_value, 
        k1=k1_value, 
        k2=k2_value, 
        k3=k3_value, 
        initial_pull=initial_stretch, 
        time_duration=simulation_time
    )