import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

# Signal freq = 6 MHz, Amplitude = 1 V, Sampling frequency = 5 MHz
# SNR = 50 dB
Fsig = 2e6
Amp = 1
Fs = 5e6
SNR = 50
sample_points = 64

Tsig = 1/Fsig       # Tsig = 5e-7
Ts = 1/Fs           # Ts = 2e-7

t = np.linspace(0, sample_points*Ts, 1000)
y_t = Amp * np.sin(2 * np.pi * Fsig * t)

t_sample = np.arange(0, sample_points*Ts, Ts)
y_sample = Amp * np.sin(2 * np.pi * Fsig * t_sample)


# Calculate power of the sine wave
P_signal = Amp**2 / 2

# Calculate the noise variance for Gaussian noise to achieve the target SNR
P_noise_gaussian = P_signal / (10 ** (SNR / 10))

# Gaussian noise
mean = 0
var = P_noise_gaussian
sigma = np.sqrt(var)
G_noise = np.random.normal(mean, sigma, t_sample.shape)

# Noisy signal with Hanning window
window = np.hamming(sample_points)
noisy_signal = (y_sample + G_noise)*window

# Calculate the PSD using DFT
N = len(t_sample)
frequencies = fftfreq(N, 1/Fs)
frequencies = fftshift(frequencies)  # Shift zero frequency to the center


# Perform DFT on the noisy signal
df = Fs / sample_points
noisy_signal_dft = fft(noisy_signal, N)
psd_noisy = np.abs(fftshift(noisy_signal_dft))**2 / (N * Fs) * df
psd_dB = 10 * np.log10(psd_noisy)


################## Plot the result ##################
fig, axes = plt.subplots(3, 1, figsize=(9, 8))

axes[0].stem(t_sample, y_sample, 'r', label = 'sample points')
axes[0].plot(t, y_t, label = 'input signal')
axes[0].set_title("Input signal and sample points", fontweight="bold", loc="center")
axes[0].set_ylabel("Magnitude")
axes[0].set_xlabel("Time (us)")
axes[0].grid()
axes[0].legend()


axes[1].plot(t_sample, noisy_signal, 'black', label = 'Noisy signal')
axes[1].set_title("Noisy sampled signal with Hamming window", fontweight="bold", loc="center")
axes[1].set_ylabel("Magnitude")
axes[1].set_xlabel("Time (us)")
axes[1].grid()
axes[1].legend()


axes[2].plot(frequencies, psd_dB, 'green', label = 'PSD of noisy signal')
axes[2].set_title("PSD of Noisy Signal with Hamming window", fontweight="bold", loc="center")
axes[2].set_ylabel("Power (dB)")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].grid()
axes[2].legend()

plt.tight_layout()
plt.show()


psd_half = psd_dB[:int(sample_points/2)]
signal_idx1 = np.argmax(psd_half)
signal_power_estimate = psd_half[signal_idx1]

# Estimate noise power as the average PSD in the non-signal frequencies
noise_power_estimate = np.mean(psd_half[psd_half < signal_power_estimate])

# Estimate the SNR from the PSD
SNR_from_psd = signal_power_estimate - noise_power_estimate

# Print results
print(f"Theoretical SNR (from the input noise power): {SNR} dB")
print(f"Estimated SNR from the PSD plot: {SNR_from_psd:.2f} dB")