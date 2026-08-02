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
#   Triangular Periodic loading
#   Fourier series
# 
#-------------------------------------------------
# Author : Marco Campolo 2026/08/02

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ------------------
# CODE CURVE1
# ------------------
def curve1(t, T, P):
    """Fourier representation of a periodic force: Triangular function (a0 term)"""
    if t <= T / 2:
        p1 = 2 * t * P / T
    else:
        p1 = 2 * (T - t) * P / T
    return (1 / T) * p1

# ------------------
# CODE CURVE1C
# ------------------
def curve1c(t, n, T, P):
    """Computes constants for cosine terms"""
    w = 2 * np.pi / T
    if t <= T / 2:
        p1 = 2 * t * P / T
    else:
        p1 = 2 * (T - t) * P / T
    return (2 / T) * p1 * np.cos(w * t * n)

# ------------------
# CODE CURVE1S
# ------------------
def curve1s(t, n, T, P):
    """Computes constants for sine terms"""
    w = 2 * np.pi / T
    if t <= T / 2:
        p1 = 2 * t * P / T
    else:
        p1 = 2 * (T - t) * P / T
    return (2 / T) * p1 * np.sin(w * t * n)

# ------------------
# MAIN CODE 
# ------------------
def main():
    P = 10
    T = 2
    
    print(f"Amplitude of periodic force = {P:.3g} N")
    print(f"Period of force = {T:.3g} sec")
    
    a0, _ = quad(curve1, 0, T, args=(T, P), epsabs=1e-9)
    
    # Setup interactive mode for the pause effect
    plt.ion()
    fig = plt.figure(figsize=(10, 12))
    
    # Pre-allocate arrays for coefficients
    a = np.zeros(10)
    b = np.zeros(10)
    
    # Define time vector: 0 to T*2 with 0.05 step
    t_vals = np.arange(0, (T * 2) + 0.05, 0.05)
    
    for N in range(1, 11): # Loops from 1 to 10 inclusive
        
        # Calculate Fourier coefficients
        for n in range(1, N + 1):
            a[n-1], _ = quad(curve1c, 0, T, args=(n, T, P), epsabs=1e-9)
            b[n-1], _ = quad(curve1s, 0, T, args=(n, T, P), epsabs=1e-9)
            
        Outp = np.zeros(len(t_vals))
        
        # Reconstruct the waveform
        for x, t in enumerate(t_vals):
            c = 0
            s = 0
            for n in range(1, N + 1):
                c += a[n-1] * np.cos(t * 2 * np.pi * n / T)
                s += b[n-1] * np.sin(t * 2 * np.pi * n / T)
            Outp[x] = s + c + a0
            
        # Plotting
        ax = fig.add_subplot(5, 2, N)
        ax.plot(t_vals, Outp, '.-')
        ax.grid(True)
        ax.set_title(f"N = {N}")
        
        # --- ADDED UNITS OF MEASUREMENT --------
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N)")
        # ---------------------------------------
        
        plt.pause(1) # Pauses execution for 1 second to watch it build
        
    plt.ioff() # Turn off interactive mode
    
    # Apply tight_layout ONCE after all subplots are created
    plt.tight_layout()
    
    # --- FIXED PROCEDURE TO SAVE AS PDF ---
    # Using fig.savefig() guarantees the specific figure object is saved
    pdf_filename = "Fourier_Triangular_Function.pdf"
    fig.savefig(pdf_filename, format="pdf", bbox_inches="tight")
    print(f"Plot successfully saved to: {pdf_filename}")
    # --------------------------------------

    plt.show() # Keep the final plot open

if __name__ == "__main__":
    main()