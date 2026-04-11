import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from DataLoader import ModemDataset
from Autoencoder import SpectrogramAE
from Autoencoder import train_model
from PerfTracker import PerformanceTracker
from Macros import generate_training_data

if __name__ == "__main__":
    NUM_TRAINING_SAMPLES = 20000
    BITS_PER_FRAME = 16
    SNR = 30
    EPOCHS = 100

    tracker = PerformanceTracker()

    print("Building dataset")
    training_signals = generate_training_data(nts=NUM_TRAINING_SAMPLES, bpf=BITS_PER_FRAME, snr=SNR)

    dataset = ModemDataset(raw_signal=training_signals, sr=44100, nps=256)

    train_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda")
    #device = torch.device("cpu")

    if device.type == "cuda":
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using CPU, expect slow training times")

    model = SpectrogramAE().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training process")

    tracker.tracking_start()

    train_model(model, train_loader, optimizer, criterion, device, epochs=EPOCHS)

    print("Training complete")

    torch.save(model.state_dict(), "detector.pth")
    print("Data written to disk")

    tracker.tracking_stop(model, "detector.pth")
