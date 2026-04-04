import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from SignalGenerator import ModemSignalGenerator
from NoiseGenerator import ModemNoiseGenerator
from AnomalyInjector import ModemAnomalyInjector
from DataLoader import ModemDataset
from Autoencoder import SpectrogramAE
from Autoencoder import train_model
from PerfTracker import PerformanceTracker

if __name__ == "__main__":
    NUM_TRAINING_SAMPLES = 800
    BITS_PER_FRAME = 16
    SNR_DB = 20

    tracker = PerformanceTracker()

    print("Building training dataset")
    generator = ModemSignalGenerator(fs=44100, bit_dur=0.02)
    training_signals = []

    for _ in range(NUM_TRAINING_SAMPLES):
        random_bits = np.random.randint(0, 2, BITS_PER_FRAME)

        clean_frame = generator.GenerateFrame(random_bits)

        noise_gen = ModemNoiseGenerator(clean_frame, snr_db=SNR_DB)

        noised_frame = noise_gen.GenerateNoise()

        training_signals.append(noised_frame)

    print("Generating data for PyTorch")

    dataset = ModemDataset(raw_signal=training_signals, sr=44100, nps=256)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda")
    #device = torch.device("cpu")

    if device == "cuda":
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using CPU, expect slow training times")

    model = SpectrogramAE().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training process")

    tracker.trackingStart()

    train_model(model, train_loader, optimizer, criterion, device, epochs=20)

    print("Training complete")

    torch.save(model.state_dict(), "data.pth")
    print("Data written to disk")

    tracker.trackingStop(model, "data.pth")