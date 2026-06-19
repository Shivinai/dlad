import numpy as np

class Transmitter:
    def __init__(self, carriers=8, pfx_len=2):
        self.carriers = carriers
        self.pfx_len = pfx_len
        self.qam_map = {'00': -3, '01': -1, '11': 1, '10': 3}
    
    def _constellation_mapper(self, bits_4):
        real_bits = bits_4[0:2]
        imag_bits = bits_4[2:4]
        return complex(self.qam_map[real_bits], self.qam_map[imag_bits])

    def modulate_symbol(self, bits_32):
        assert len(bits_32) == 32, "Expecting 32 bits"
        
        frequencies = []
        for i in range(self.carriers):
            chunk = bits_32[i*4 : (i+1)*4]
            frequencies.append(self._constellation_mapper(chunk))
            
        time_samples = np.fft.ifft(frequencies)
        
        cp = time_samples[(self.carriers - self.pfx_len):]
        
        return np.concatenate([cp, time_samples])
