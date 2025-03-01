from scipy.fft import fft, fftshift
import numpy as np
import matplotlib.pyplot as plt


# ---------------------- Question 3. a. ----------------------

F1 = 2e6                                      # input signal frequency
Fs = 5e6                                        # sampling frequency
N = 64                                          # number of sampling points
T = 1/F1
Ts = 1/Fs

t = np.linspace(0, 5*T, 1000)
y_t = np.cos(2 * np.pi * F1 * t)
y_window = np.blackman(1000) * y_t

t_sample = np.arange(0, N*Ts, Ts)
y_sample = np.cos(2 * np.pi * F1 * t_sample)

Y = fftshift(fft(y_sample))
Y_mag = abs(Y) / N

f_step = int(Fs / N)
f = np.arange(-0.5*N*f_step, 0.5*N*f_step, f_step)


fig, axes = plt.subplots(2, 1, figsize=(9, 8))

axes[0].plot(t*1e6, y_window)
axes[0].set_title("Continuous time function", fontweight="bold", loc="center")
axes[0].set_ylabel("Magnitude")
axes[0].set_xlabel("Time (ns)")
axes[0].legend()

axes[1].plot(f, Y_mag)
axes[1].set_title("Magnitude of Fourier transform", fontweight="bold", loc="center")
axes[1].set_ylabel("Magnitude")
axes[1].set_xlabel("Frequecy")
axes[1].legend()

plt.tight_layout()
plt.show()


"""
# ---------------------- Question 3. b. ----------------------

F1 = 200e6
F2 = 400e6

Fs_3b = 1e9
N = 64

t_step_3b = 1/Fs_3b
t_3b = np.arange(0, N*t_step_3b, t_step_3b)
y_t_3b = np.cos(2*np.pi*F1*t_3b) + np.cos(2*np.pi*F2*t_3b)      # time domain signal after sampling

Y_3b = fft(y_t_3b)
Y_3b_mag = abs(Y_3b) / N

f_step_3b = Fs_3b / N
f_3b = np.arange(0, N*f_step_3b, f_step_3b)

fplot_3b = f_3b[0:int(N/2+1)]
Y_mag_plot_3b = 2 * Y_3b_mag[0:int(N/2+1)]
Y_mag_plot_3b[0] = Y_mag_plot_3b[0]/2

plt.subplot(2, 1, 1)
plt.plot(t_3b, y_t_3b)
plt.title("3.b. Time domain of input signal")
plt.ylabel("Magnitude")
plt.xlabel("Discrete Time")

plt.subplot(2, 1, 2)
plt.plot(fplot_3b, Y_mag_plot_3b)
plt.title("3.b. Time domain of input signal")
plt.ylabel("Magnitude")
plt.xlabel("Discrete Time")

plt.tight_layout()
plt.show()
"""

"""
# ---------------------- Question 3.d.(a.) ----------------------

F = 2e6                                         # input signal frequency
Fs = 5e6                                        # sampling frequency
N = 64                                          # number of sampling points

t_step_3d = 1/Fs                                   # sampling time interval
t_3d = np.arange(0, N*t_step_3d, t_step_3d)            # time steps

window = [0]*N
window = np.blackman(N)
y_t_3d = np.cos(2*np.pi*F*t_3d) * window              # time domain signal enveloped by Blackman window

f_step_3d = Fs / N
f_3d = np.arange(0, N*f_step_3d, f_step_3d)              # frequency steps

Y_3d = fft(y_t_3d)
Y_mag_3d = np.abs(Y_3d) / N

fplot = f_3d[0:int(N/2+1)]                         # build frequency axis
X_mag_plot_3d = 2 * Y_mag_3d[0:int(N/2+1)]
X_mag_plot_3d[0] = X_mag_plot_3d[0]/2                 # DC term does not need to multiply by 2

plt.subplot(2, 1, 1)
plt.plot(t_3d, y_t_3d)
plt.title("3.a. Time domain sample of cosine function")
plt.ylabel("Magnitude")
plt.xlabel("Discrete Time")

plt.subplot(2, 1, 2)
plt.plot(fplot, X_mag_plot_3d)
plt.title("3.a. frequency domain of cosine function")
plt.ylabel("Magnitude")
plt.xlabel("Discrete frequency")

plt.tight_layout()
plt.show()


"""
"""
# ---------------------- Question 3.d.(b.) ----------------------

F1 = 200e6
F2 = 400e6

Fs_3d = 1e9
N = 64

t_step_3d = 1/Fs_3d
t_3d = np.arange(0, N*t_step_3d, t_step_3d)
y_t_3d = np.cos(2*np.pi*F1*t_3d) + np.cos(2*np.pi*F2*t_3d)      # time domain signal after sampling

Y_3d = fft(y_t_3d)
Y_3d_mag = abs(Y_3d) / N

f_step_3d = Fs_3d / N
f_3d = np.arange(0, N*f_step_3d, f_step_3d)

fplot_3d = f_3d[0:int(N/2+1)]
Y_mag_plot_3d = 2 * Y_3d_mag[0:int(N/2+1)]
Y_mag_plot_3d[0] = Y_mag_plot_3d[0]/2

plt.subplot(2, 1, 1)
plt.plot(t_3d, y_t_3d)
plt.title("3.b. Time domain of input signal")
plt.ylabel("Magnitude")
plt.xlabel("Discrete Time")

plt.subplot(2, 1, 2)
plt.plot(fplot_3d, Y_mag_plot_3d)
plt.title("3.b. Time domain of input signal")
plt.ylabel("Magnitude")
plt.xlabel("Discrete Time")

plt.tight_layout()
plt.show()
"""