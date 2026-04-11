import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from DataLoader import Dataset2D
from DataLoader import Dataset1D
from Autoencoder import SpectrogramAE2D
from Autoencoder import SignalAE1D
from Autoencoder import train_model
from PerfTracker import PerformanceTracker
from Macros import generate_training_data
import argparse

def train_2d_autoencoder(device):
    NUM_TRAINING_SAMPLES = 10000
    BITS_PER_FRAME = 16
    SNR = 30
    EPOCHS = 100

    print("Generating raw training data...")
    training_signals = generate_training_data(nts=NUM_TRAINING_SAMPLES, bpf=BITS_PER_FRAME, snr=SNR)

    print("Building dataset...")
    dataset_2d = Dataset2D(raw_signal=training_signals, sr=44100, nps=256)
    train_loader_2d = DataLoader(dataset_2d, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    model = SpectrogramAE2D().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, train_loader_2d, optimizer, criterion, device, epochs=EPOCHS)

    torch.save(model.state_dict(), "detector_2d.pth")
    print("2D Model weights saved to 'detector_2d.pth'")
    
    return model


def train_1d_autoencoder(device):
    NUM_TRAINING_SAMPLES = 10000
    BITS_PER_FRAME = 16
    SNR = 30
    EPOCHS = 100
    
    print("Generating raw training data...")
    training_signals = generate_training_data(nts=NUM_TRAINING_SAMPLES, bpf=BITS_PER_FRAME, snr=SNR)

    print("Building dataset...")
    dataset_1d = Dataset1D(raw_signal=training_signals, frame_size=256)
    train_loader_1d = DataLoader(dataset_1d, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    model = SignalAE1D().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, train_loader_1d, optimizer, criterion, device, epochs=EPOCHS)

    torch.save(model.state_dict(), "detector_1d.pth")
    print("Weights saved to 'detector_1d.pth'")
    
    return model

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU, expect slow training times")

    parser = argparse.ArgumentParser(description="Train tandem models")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["2d", "1d", "both"], 
        default="both",
        help="What're we training?: '2d', '1d', or 'both'"
    )
    args = parser.parse_args()

    tracker = PerformanceTracker()
    tracker.tracking_start()

    model_ref = None 

    if args.model in ["2d", "both"]:
        model_ref = train_2d_autoencoder(device=device)
    
    if args.model in ["1d", "both"]:
        model_ref = train_1d_autoencoder(device=device)

    print("\nTraining complete. Data written to disk.")

    if model_ref:
        tracker.tracking_stop(model_ref, "detector.pth")
    else:
        tracker.tracking_stop(None, "detector.pth")
        