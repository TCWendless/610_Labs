import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

# ------------------------ Q1.a. ------------------------
# FIR filter
fir_num = [1, 1, 1, 1, 1]
fir_denom = [1]

w_fir, h_fir = signal.freqz(fir_num, fir_denom, worN=1024)

# IIR filter
iir_num = [1, 1]
iir_denom = [1, -1]

w_iir, h_iir = signal.freqz(iir_num, iir_denom, worN=1024)

# ------------------------ Q1.b. ------------------------
zero_fir, pole_fir, gain_fir = signal.tf2zpk(fir_num, fir_denom)
zero_iir, pole_iir, gain_iir = signal.tf2zpk(iir_num, iir_denom)


# ------------------------ Plot ------------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# --- FIR Transfer Function ---

axes[0,0].plot(w_fir / np.pi, 20 * np.log10(abs(h_fir)), label="FIR response", color="blue")
axes[0,0].set_title("FIR Filter Frequency Response", fontweight="bold", loc="center")
axes[0,0].set_xlabel("Normalized Frequency (π rad/sample)")
axes[0,0].set_ylabel("Magnitude (dB)")
axes[0,0].grid()
axes[0,0].legend()

# --- Plot FIR Poles and Zeros ---

# --- Plot unit circle ---
theta = np.linspace(0 , 2*np.pi, 100)
ejtheta = np.exp(1j*theta)
axes[1,0].plot(np.real(ejtheta), np.imag(ejtheta))

for zr in zero_fir:
   axes[1,0].plot(np.real(zr), np.imag(zr), 'ro')

for pl in pole_fir:
    axes[1,0].plot(np.real(pl), np.imag(pl), 'rx')


axes[1,0].set_title("Poles and Zeros", fontweight="bold", loc="center")
axes[1,0].set_xlabel("Real Axis")
axes[1,0].set_ylabel("Complex Axis")
axes[1,0].grid()

# IIR transfer function
axes[0,1].plot(w_iir / np.pi, 20 * np.log10(abs(h_iir)), label="IIR response", color="blue")
axes[0,1].set_title("IIR Filter Frequency Response", fontweight="bold", loc="center")
axes[0,1].set_xlabel("Normalized Frequency (π rad/sample)")
axes[0,1].set_ylabel("Magnitude (dB)")
axes[0,1].grid()
axes[0,1].legend()

# --- Plot IIR Poles and Zeros

# --- Plot unit circle ---

axes[1,1].plot(np.real(ejtheta), np.imag(ejtheta))

for zr in zero_iir:
   axes[1,1].plot(np.real(zr), np.imag(zr), 'ro')

for pl in pole_iir:
    axes[1,1].plot(np.real(pl), np.imag(pl), 'rx')


axes[1,1].set_title("Poles and Zeros", fontweight="bold", loc="center")
axes[1,1].set_xlabel("Real Axis")
axes[1,1].set_ylabel("Complex Axis")
axes[1,1].grid()

plt.tight_layout()
plt.show()
