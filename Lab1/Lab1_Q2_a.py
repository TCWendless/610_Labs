from scipy.fft import fft, fftshift
import numpy as np
import matplotlib.pyplot as plt
# ----------------------    Question 2.    ----------------------
F1 = 300e6
F2 = 800e6
Fs = 500e6
N = 100

t_step = 1/Fs                                   # sampling time interval
t_sampled = np.arange(0, N*t_step, t_step)              # time steps
x1_t = np.cos(2*np.pi*F1*t_sampled)                       # time domain signal after sampling
x2_t = np.cos(2*np.pi*F2*t_sampled)


# ---------------------------------- Plot ----------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 9))

# Plot sampled x1
axes[0].plot(t_sampled*1e9, x1_t, 'b', label="X1 Sampled Signal")
axes[0].set_title("X1 Sampling at Ts", fontweight="bold", loc="center")
axes[0].set_xlabel("Time (ns)")
axes[0].set_ylabel("Amplitude")
axes[0].legend()
axes[0].grid()

# Plot sampled x2
axes[1].plot(t_sampled*1e9, x2_t, 'black', label="X2 Sampled Signal")
axes[1].set_title("X2 Sampling at Ts", fontweight="bold", loc="center")
axes[1].set_xlabel("Time (ns)")
axes[1].set_ylabel("Amplitude")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()