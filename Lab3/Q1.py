import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

Cs = 15.925e-12         # sampling capacitor
fLO = 2.4e9             # LO frequency (Hz)
N = 8                   # Number of cycle
T = 1/fLO

b_FIR = np.ones(N)
b_FIR = b_FIR * (0.5*T/Cs)


w, h_FIR = signal.freqz(b_FIR, fs = fLO)

h_IIR = h_FIR * 1/(1 - np.exp(-1j * 2 * np.pi * w * 2 * np.pi / fLO))  # IIR response with Z^(-1) at the output

################## Plot the result ##################
fig, axes = plt.subplots(1, 1, figsize=(9, 8))

axes.plot(w/1e6, 20 * np.log10(abs(h_FIR)), 'blue', label = 'FIR (Discharged Cap)')
axes.plot(w/1e6, 20 * np.log10(abs(h_IIR)), 'black', label = 'IIR (Non-discharged Cap)')
axes.set_title(f"Gm-C filter response with N = {N:.0f}, fLO = {fLO/1e9:.1f}GHz", fontweight="bold", loc="center")
axes.set_xlim([0, fLO/(2 * 1e6)])
axes.set_ylabel("Voltage gain (dB)")
axes.set_xlabel("Frequency (MHz)")
axes.grid()
axes.legend()

plt.show()
