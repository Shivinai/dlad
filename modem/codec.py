import numpy as np

DEFAULT_CHAR_MAP = {
    ' ': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,
    'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16,
    'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24,
    'y': 25, 'z': 26, '0': 27, '1': 28, '2': 29, '3': 30, '4': 31, '5': 32,
    '6': 33, '7': 34, '8': 35, '9': 36, '.': 37, ',': 38, '?': 39, '!': 40
}

class Codec:
    def __init__(self):
        self.bits_per_char = 6
        self.char_to_int = DEFAULT_CHAR_MAP
        self.int_to_char = {v: k for k, v in self.char_to_int.items()}

    def encode(self, text):
        bits = ''
        for char in text.lower():
            val = self.char_to_int.get(char, 0)
            bits += format(val, f'0{self.bits_per_char}b')
        return bits

    def decode(self, bit_string):
        n = self.bits_per_char
        text = ''
        for i in range(0, len(bit_string), n):
            chunk = bit_string[i:i + n]
            if len(chunk) == n:
                text += self.int_to_char.get(int(chunk, 2), '?')

        return text
