import numpy as np

class Channel:
    @staticmethod
    def apply_noise(signal, snr_db=25):
        sig_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10.0)
        noise_power = sig_power / snr_linear
        
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
        return signal + noise

