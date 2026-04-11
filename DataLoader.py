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

class Dataset2D(Dataset):
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

        return spec_tensor, spec_tensor

class Dataset1D(Dataset):
    def __init__(self, raw_signal, frame_size=256):
        if isinstance(raw_signal, list):
            self.raw_signal = np.concatenate(raw_signal) if len(raw_signal) > 0 else np.array([])
        else:
            self.raw_signal = raw_signal
        
        self.frame_size = frame_size
        
        num_frames = len(self.raw_signal) // frame_size
        self.frames = self.raw_signal[:num_frames * frame_size].reshape(num_frames, frame_size)
    
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        return torch.tensor(frame, dtype=torch.float32).unsqueeze(0), torch.tensor(frame, dtype=torch.float32).unsqueeze(0)