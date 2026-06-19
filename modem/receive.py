import numpy as np

class Receiver:
    def __init__(self, carriers=8, pfx_len=2):
        self.carriers = carriers
        self.pfx_len = pfx_len
        self.inv_qam_map = {-3: '00', -1: '01', 1: '11', 3: '10'}

    def _constellation_demapper(self, complex_val):
        allowed_points = [-3, -1, 1, 3]
        best_real = min(allowed_points, key=lambda x: abs(complex_val.real - x))
        best_imag = min(allowed_points, key=lambda x: abs(complex_val.imag - x))
        
        return self.inv_qam_map[best_real] + self.inv_qam_map[best_imag]

    def demodulate_symbol(self, frame_10):
        time_data = frame_10[self.pfx_len:]
        
        frequencies = np.fft.fft(time_data)
        
        bits_32 = ""
        for i in range(self.carriers):
            bits_32 += self._constellation_demapper(frequencies[i])
            
        return bits_32

