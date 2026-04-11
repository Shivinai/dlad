import numpy as np

class Channel:

    def __init__(self, snr_db=10.0):
        self.snr_db = snr_db

        self._anomaly_handlers = {
            'dropout': self._dropout,
            'echo': self._echo,
            'clipping': self._clipping
        }
    
    def process(self, signal, anomaly_type=None, **anomaly_kwargs):
        s = signal.copy()
        
        if anomaly_type in self._anomaly_handlers:
            s = self._anomaly_handlers[anomaly_type](s, **anomaly_kwargs)
        elif anomaly_type is not None:
            raise ValueError(f"Unknown anomaly: '{anomaly_type}'. Falling back to none")

        signal_power = np.mean(np.abs(s)**2)
        
        if signal_power > 0:
            noise_power = signal_power / (10 ** (self.snr_db / 10.0))
            noise = np.random.normal(0, np.sqrt(noise_power), len(s))

            s += noise

            return s

    def _dropout(self, signal):
        corrupted = signal.copy()
        duration = int(len(signal) * np.round(np.random.uniform(0.1, 0.7), 1))
        if len(signal) > duration:
            start_idx = np.random.randint(0, len(signal) - duration)
            corrupted[start_idx : start_idx + duration] = 0.0
        return corrupted

    def _echo(self, signal):
        echo = np.zeros_like(signal)
        ds = np.random.randint(50, 2000)
        print(f"Using {ds} delay samples")
        if len(signal) > ds:
            echo[ds:] = signal[:-ds] * np.round(np.random.uniform(0.3, 0.8), 1)
        return signal + echo

    def _clipping(self, signal):
        threshold = np.round(np.random.uniform(0.1, 0.6), 1)
        return np.clip(signal, -threshold, threshold)
        