import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

res = 12         # Resolution
levels = 2**res
f = 300e6       # Input frequency
fs = 4e9      # Sampling frequency
Amp = 1
full_scale = 2 * Amp
SNR_ideal = 38

T = 1 / f
Ts = 1 / fs
delta = full_scale / (levels - 1)

# Total number of samples for 30 period
num_periods = 30

# Input signal after sampling
t = np.linspace(0, num_periods * T, 1000 * int(T/Ts), endpoint=False)
input_signal = np.sin(2 * np.pi * f * t)

# Quantize the signal
quantized_signal = np.round((input_signal + Amp) / delta) * delta - Amp


### Question starts here ###

# Perform FFT to compute the PSD
# Number of points in the FFT (typically a power of 2 for efficiency)
N = int(num_periods * fs / f)
t_sample = np.arange(N) / fs


input_signal_sample = Amp * np.sin(2 * np.pi * f * t_sample)

quantized_signal_sample = np.round((input_signal_sample + Amp) / delta) * delta - Amp
quantized_noise = input_signal_sample - quantized_signal_sample

mean = 0
var = abs(1.585e-4 - np.mean(quantized_noise**2))
sigma = np.sqrt(var)
G_noise = np.random.normal(mean, sigma, quantized_noise.shape)

quantized_signal_sample = quantized_signal_sample + G_noise

quantized_noise = quantized_noise + G_noise


input_power = np.mean(input_signal_sample**2)
noise_power = np.mean(quantized_noise**2)

SNR = 10 * np.log10(input_power / noise_power)


df = fs / N
fft_result = fft(quantized_signal_sample, N)
fft_result_shift = fftshift(fft_result)

# Calculate the magnitude of the FFT and then the power spectral density (PSD)
psd = df * (np.abs(fft_result_shift) ** 2 / (N * fs))
quantized_psd_dB = 10 * np.log10(psd)

# Frequency axis for the PSD plot
frequencies = fftfreq(N, Ts)
frequencies_shift = fftshift(frequencies)     # Shift zero frequency to the center



################## Plot the result ##################
fig, axes = plt.subplots(3, 1, figsize=(9, 8))

axes[0].plot(t, input_signal, label = 'Input signal')
axes[0].set_title(f"Continuous input signal (Number of periods = {num_periods:.1f})", fontweight="bold", loc="center")
axes[0].set_ylabel("Amplitude")
axes[0].set_xlabel("Time")
axes[0].grid()
axes[0].legend()

axes[1].plot(t, quantized_signal, 'black', label = 'Quantized signal')
axes[1].set_title(f"Continuous quantized signal (Number of periods = {num_periods:.1f})", fontweight="bold", loc="center")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlabel("Time")
axes[1].grid()
axes[1].legend()


axes[2].plot(frequencies_shift, quantized_psd_dB, 'green', label = 'FFT of quantized signal')
axes[2].set_title(f"PSD (Number of periods = {num_periods:.1f})", fontweight="bold", loc="center")
axes[2].set_ylabel("Amplitdue (dB)")
axes[2].set_xlabel("frequency (GHz)")
axes[2].grid()
axes[2].legend()


plt.tight_layout()
plt.show()

print(f"Theoretical SNR (from the input noise power): {SNR_ideal} dB")
print(f"Estimated SNR from the PSD plot: {SNR:.2f} dB")