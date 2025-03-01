from scipy.fft import fft, fftshift
import numpy as np
import matplotlib.pyplot as plt

F1 = 300e6
Fs = 500e6
T = 10/F1
Ts = 1/Fs

# Continuous time signal
t = np.linspace(0, T, 1000)
x_t = np.cos(2 * np.pi * F1 * t)

# Sampled signal
t_sampled = np.arange(0, T-Ts, Ts)  # discrete time 0 ~ T-Ts sample every Ts
x_sampled = np.cos(2 * np.pi * F1 * t_sampled)

# Reconstruct continuous signal from sampled signal
def sinc_interp(xn, tn, t_interp): #(sample point, sample point's time, reconstruct time)
    """ sinc to get original signal """
    Ts = tn[1] - tn[0]  # sample time= xn[i] - xn[i-1]= Ts= 1/Fs
    return np.sum(xn * np.sinc((t_interp[:, None] - tn) / Ts), axis=1)

x_reconstructed = sinc_interp(x_sampled, t_sampled, t)

# Reconstruct signal with shifted smapling time
t_sampled_shifted = np.arange(Ts/2, T-Ts/2, Ts)                 # Sample points shifted by Ts/2
x_sampled_shifted = np.cos(2 * np.pi * F1 * t_sampled_shifted)
x_halfTs = sinc_interp(x_sampled_shifted, t_sampled_shifted, t)

# Calculate MSE
mse_Ts = np.mean((x_reconstructed - x_t) ** 2)
mse_halfTs = np.mean((x_halfTs - x_t) ** 2)

# -------------------------------------------- Plotting --------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(9, 9))

# original signal + sample point
axes[0].plot(t*1e9, x_t, label="Continuous time Signal")
axes[0].stem(t_sampled*1e9, x_sampled, 'black', markerfmt='black', label="Sampled with Ts")
axes[0].set_title(f"Original Signal and Sampling at Ts (Sampling Frequency = {Fs/1e6 : .0e}MHz)", fontweight="bold", loc="center")
axes[0].set_xlabel("Time (ns)")
axes[0].set_ylabel("Amplitude")
axes[0].legend()


# Reconstructed signal
axes[1].plot(t * 1e9, x_t, 'b', label="Original Continuous Signal")
axes[1].plot(t * 1e9, x_reconstructed, 'black', label="Reconstructed Signal (Ts)")
axes[1].set_title(f"Reconstructed Signal from Samples at Ts (MSE: {mse_Ts : .2e})", fontweight="bold", loc="center")
axes[1].set_xlabel("Time (ns)")
axes[1].set_ylabel("Amplitude")
axes[1].legend()

# Reconstructed signal with shifted sampling
axes[2].plot(t * 1e9, x_t, 'b', label="Original Continuous Signal")
axes[2].plot(t * 1e9, x_halfTs, 'black', label="Reconstructed Signal (Ts)")
axes[2].set_title(f"Reconstructed Signal from shited Samples at Ts (MSE: {mse_halfTs : .2e})", fontweight="bold", loc="center")
axes[2].set_xlabel("Time (ns)")
axes[2].set_ylabel("Amplitude")
axes[2].legend()


plt.tight_layout()
plt.show()