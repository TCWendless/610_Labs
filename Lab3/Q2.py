import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

CR = 0.5e-12         # Rotating capacitor = 0.5 pF
CH = 15.425e-12      # History capacitor = 15.425 pF
a1 = CH/(CH + CR)
a2 = 1 - a1
fLO = 2.4e9             # LO frequency (Hz)
N = 8                   # Number of cycle
T = 1/fLO

b = np.ones(N)
b = b * (0.5*T/(CR + CH))


w, h_FIR = signal.freqz(b, fs = fLO)

z = np.exp(1j * 2 * np.pi * (2 * np.pi * w) / fLO)        # z = e^(j * 2pi * omega / Fs)
h_out = h_FIR * 1/(1 - a1 * z**(-1) - a2 * z**(-2))       # transfer function for IIR generateing by CR



################## Plot the result ##################
fig, axes = plt.subplots(1, 1, figsize=(9, 8))

#axes.plot(w/1e6, 20 * np.log10(abs(h_FIR)), 'blue', label = 'FIR (Discharged Cap)')
axes.plot(w/1e6, 20 * np.log10(abs(h_out)), 'blue', label = 'IIR (Non-discharged Cap)')
axes.set_title(f"Gm-C filter response with N = {N:.0f}, fLO = {fLO/1e9:.1f}GHz, CR = {CR*1e12:.1f} pF, CH = {CH*1e12:.3f} pF", fontweight="bold", loc="center")
axes.set_xlim([0, fLO/(2 * 1e6)])
axes.set_ylabel("Voltage gain (dB)")
axes.set_xlabel("Frequency (MHz)")
axes.grid()
axes.legend()

plt.show()