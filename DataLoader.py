import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

def build_spectrogram(signal, sr, nps):
    f,t,S = spectrogram(signal, fs=sr, nperseg=nps)
    S_log = 10 * np.log10(S + 1e-10)

    S_norm = (S_log - S_log.min()) / (S_log.max() - S_log.min())

    return S_norm,f,t

class ModemDataset(Dataset):
    def __init__(self, raw_signal, sr, nps, labels=None):
        self.raw_signal = raw_signal
        self.labels = labels
        self.sr = sr
        self.nps = nps
    
    def __len__(self):
        return len(self.raw_signal)

    def __getitem__(self, idx):
        sig = self.raw_signal[idx]

        spec_norm, _, _ = build_spectrogram(sig, self.sr, self.nps)

        spec_tensor = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0)

        # if self.labels is not None:
        #     label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        #     return spec_tensor, label_tensor

        return spec_tensor, spec_tensor