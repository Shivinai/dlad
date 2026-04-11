from ChannelSimulation import Channel
from Modem import Modem
import matplotlib.pyplot as plt
import numpy as np

og_bits = np.random.randint(0, 2, 16)
snr = np.random.randint(0, 50)
anomaly = 'dropout'

modem = Modem(fs=44100, f0=2000, f1=8000, symbol_duration=0.02)
channel = Channel(snr_db=snr)

mod_signal = modem.modulate(og_bits)
proc_signal = channel.process(mod_signal, anomaly_type=anomaly)
rec_bits = modem.demodulate(proc_signal)

print(og_bits)
print(snr)
print(rec_bits)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

time_axis = np.arange(len(proc_signal))
ax1.plot(time_axis, proc_signal, color='#1f77b4', linewidth=1)

ax1.set_title(f"Processed Signal (Anomaly: {anomaly}, SNR: {snr}dB)")
ax1.set_ylabel("Amplitude")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

bit_indices = np.arange(len(og_bits))

ax2.stem(bit_indices, og_bits, linefmt='b-', markerfmt='bo', basefmt='k-', label='Original Bits')
ax2.stem(bit_indices + 0.15, rec_bits, linefmt='r-', markerfmt='rx', basefmt='k-', label='Demodulated Bits')

ax2.set_title("Bit Comparison")
ax2.set_xlabel("Bit Index")
ax2.set_ylabel("Value")
ax2.set_xticks(bit_indices)
ax2.set_yticks([0, 1])
ax2.legend(loc="center right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()