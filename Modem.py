import numpy as np

class Modem:

    def __init__(self, fs, f0, f1, symbol_duration, amplitude = 1):
        self.fs = fs
        self.f0 = f0
        self.f1 = f1
        self.symbol_duration = symbol_duration
        self.samples_per_symbol = int(fs * symbol_duration)
        self.amplitude = amplitude
        
        t = np.linspace(0, self.symbol_duration, self.samples_per_symbol, endpoint=False)
        
        k = (self.f1 - self.f0) / self.symbol_duration

        phase_0 = 2 * np.pi * (self.f0 * t + 0.5 * k * (t**2))
        phase_1 = 2 * np.pi * (self.f1 * t + 0.5 * k * (t**2))
        
        self.ref_down = self.amplitude * np.cos(phase_0) 
        self.ref_up = self.amplitude * np.cos(phase_1)  
    
    def modulate(self, bits):
        
        bits = np.array(bits).astype(int)
        signal = np.zeros(len(bits) * self.samples_per_symbol)
        
        for i, bit in enumerate(bits):
            start_idx = i * self.samples_per_symbol
            symbol_wave = self.ref_up if bit == 1 else self.ref_down
            signal[start_idx : start_idx + self.samples_per_symbol] = symbol_wave
        
        return signal
    
    def demodulate(self, signal):
        
        n_symbols = len(signal) // self.samples_per_symbol
        bits = []
        
        for i in range(n_symbols):
            start_idx = i * self.samples_per_symbol
            symbol = signal[start_idx : start_idx + self.samples_per_symbol]
            
            energy_up = np.abs(np.dot(symbol, self.ref_up))
            energy_down = np.abs(np.dot(symbol, self.ref_down))
            
            bits.append(1 if energy_up > energy_down else 0)
        
        return bits