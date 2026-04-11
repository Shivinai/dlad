import numpy as np

class Channel:

    def __init__(self, snr):
        self.snr = snr

        self._anomaly_handlers = {
            'dropout': self._dropout,
            'echo': self._echo,
            'clipping': self._clipping
        }

    def _get_random_range(self, signal_length):
        duration = int(signal_length * np.round(np.random.uniform(0.1, 0.2), 1))
        start_idx = np.random.randint(0, signal_length - duration)

        return start_idx, start_idx + duration
    
    def process(self, signal, anomaly_type=None, **anomaly_kwargs):
        s = signal.copy()
        
        signal_power = np.mean(np.abs(s)**2)
        if signal_power > 0:
            noise_power = signal_power / (10 ** (self.snr / 10.0))
            noise = np.random.normal(0, np.sqrt(noise_power), len(s))
            s += noise
        
        if anomaly_type in self._anomaly_handlers:
            s = self._anomaly_handlers[anomaly_type](s, **anomaly_kwargs)
        
        return s

    def _dropout(self, signal):
        corrupted = signal.copy()
        start, end = self._get_random_range(len(signal))
        corrupted[start:end] *= np.random.uniform(0.01, 0.1)
        return corrupted

    def _echo(self, signal):
        corrupted = signal.copy()
        start, end = self._get_random_range(len(signal))
        
        ds = np.random.randint(50, 1000)
        gain = np.round(np.random.uniform(0.4, 0.7), 1)
        
        segment = signal[start:end].copy()
        echo_segment = np.zeros_like(segment)
        
        if len(segment) > ds:
            echo_segment[ds:] = segment[:-ds] * gain
            corrupted[start:end] = segment + echo_segment
            
        return corrupted

    def _clipping(self, signal):
        corrupted = signal.copy()
        start, end = self._get_random_range(len(signal))
        
        threshold = np.round(np.random.uniform(0.1, 0.4), 1)
        
        corrupted[start:end] = np.clip(corrupted[start:end], -threshold, threshold)
        return corrupted
        