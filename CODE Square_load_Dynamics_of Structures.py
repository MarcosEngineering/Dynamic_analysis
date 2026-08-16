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
#   Square Pulse Periodic loading
#   Fourier series (Odd Harmonics Only)
# 
#-------------------------------------------------
#   Author : Marco Campolo 2026/08/16


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ------------------
# CODE CURVE1
# ------------------
def curve1(t, T, P):
    """Fourier representation of a periodic force: Square pulse (a0 term)"""
    if t < T / 2:
        p1 = P
    else:
        p1 = -P
    return (1 / T) * p1

# ------------------
# CODE CURVE1C
# ------------------
def curve1c(t, n, T, P):
    """Computes constants for cosine terms"""
    w = 2 * np.pi / T
    if t < T / 2:
        p1 = P
    else:
        p1 = -P
    return (2 / T) * p1 * np.cos(w * t * n)

# ------------------
# CODE CURVE1S
# ------------------
def curve1s(t, n, T, P):
    """Computes constants for sine terms"""
    w = 2 * np.pi / T
    if t < T / 2:
        p1 = P
    else:
        p1 = -P
    return (2 / T) * p1 * np.sin(w * t * n)

# ------------------
# MAIN CODE 
# ------------------
def main():
    P = 1
    T = 1
    
    print(f"Amplitude of square pulse = {P:.3g} N")
    print(f"Period of force = {T:.3g} sec")
    
    a0, _ = quad(curve1, 0, T, args=(T, P), epsabs=1e-9, limit=100)
    
    plt.ion()
    
    # Adding constrained_layout=True as an extra safeguard for spacing
    fig = plt.figure(figsize=(10, 12), constrained_layout=True)
    
    a = np.zeros(20)
    b = np.zeros(20)
    
    t_vals = np.arange(0, T + 0.01, 0.01)
    
    for plot_idx, N in enumerate(range(1, 21, 2), start=1): 
        
        for n in range(1, N + 1):
            a[n-1], _ = quad(curve1c, 0, T, args=(n, T, P), epsabs=1e-9, points=[T/2])
            b[n-1], _ = quad(curve1s, 0, T, args=(n, T, P), epsabs=1e-9, points=[T/2])
            
        Outp = np.zeros(len(t_vals))
        
        for x, t in enumerate(t_vals):
            c = 0
            s = 0
            for n in range(1, N + 1):
                c += a[n-1] * np.cos(t * 2 * np.pi * n / T)
                s += b[n-1] * np.sin(t * 2 * np.pi * n / T)
            Outp[x] = s + c + a0
            
        ax = fig.add_subplot(5, 2, plot_idx)
        ax.plot(t_vals, Outp, 'k-')
        ax.grid(True)
        
        ax.set_title(f"{plot_idx} Harmonic(s) (up to n={N})")
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Force coeff.")
        ax.set_ylim([-1.5, 1.5]) 
        
        # --- FIXED SPACING ---
        # Forces Matplotlib to fix overlapping text on every iteration
        fig.tight_layout() 
        # ---------------------
        
        plt.pause(1) 
        
    plt.ioff() 
    
    pdf_filename = "Fourier_Square_Pulse.pdf"
    fig.savefig(pdf_filename, format="pdf", bbox_inches="tight")
    print(f"Plot successfully saved to: {pdf_filename}")

    plt.show() 

if __name__ == "__main__":
    main()