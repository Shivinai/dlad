import torch
import numpy as np
import matplotlib.pyplot as plt
from Autoencoder import SpectrogramAE2D
from DataLoader import build_spectrogram
from scipy.signal import medfilt
from Macros import generate_frame

def get_threshold(model, sr, nps, device, snr, num_calibration_frames=1000):
    model.eval()
    all_mse_values = []

    with torch.no_grad():
        for _ in range(num_calibration_frames):
            frame = generate_frame(bpf=16, snr=snr)
            spec, _, _ = build_spectrogram(frame, sr, nps)
            spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            recon = model(spec_tensor).squeeze().cpu().numpy()
            
            mse_time = np.mean((spec - recon)**2, axis=0)
            all_mse_values.extend(mse_time)
    
    all_mse_values = np.array(all_mse_values)
    threshold = np.mean(all_mse_values) + 4 * np.std(all_mse_values)
    
    print(f"Detected threshold: {threshold:.6f} (mean: {np.mean(all_mse_values):.6f}, std: {np.std(all_mse_values):.6f})")
    return threshold

def detect_anomaly(model, anomalous_signal, sr, nps, device, threshold):
    model.eval()

    anom_spec, f, t = build_spectrogram(anomalous_signal, sr, nps)
    anom_tensor = torch.tensor(anom_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        anom_recon = model(anom_tensor).squeeze().cpu().numpy()

    anom_mse_time = np.mean((anom_spec - anom_recon)**2, axis=0)

    anom_mse_smoothed = medfilt(anom_mse_time, kernel_size=3)

    anomaly_indices = np.where(anom_mse_smoothed > threshold)[0]

    time_axis_signal = np.linspace(0, len(anomalous_signal) / sr, len(anomalous_signal))

    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.4)

    axs[0].plot(time_axis_signal, anomalous_signal, color='blue', alpha=0.7)
    axs[0].set_title('Detected anomalies')
    axs[0].set_ylabel('Amp')
    axs[0].set_xlim([time_axis_signal[0], time_axis_signal[-1]])

    if len(anomaly_indices) > 0:
        for idx in anomaly_indices:
            t_anom = t[idx]
            window_dur = (nps / sr) 
            axs[0].axvspan(t_anom - window_dur/2, t_anom + window_dur/2, color='red', alpha=0.3)

    im1 = axs[1].pcolormesh(t, f, anom_spec, shading='gouraud')
    axs[1].set_title('Generated signal spectrogram')
    axs[1].set_ylabel('Freq')

    im2 = axs[2].pcolormesh(t, f, anom_recon, shading='gouraud')
    axs[2].set_title('Autoencoder reconstructed spectrogram')
    axs[2].set_ylabel('Freq')

    axs[3].plot(t, anom_mse_time, color='purple')
    axs[3].set_title('Reconstruction error')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('MSE')
    axs[3].set_xlim([t[0], t[-1]])
    
    axs[3].fill_between(t, anom_mse_time, threshold, where=(anom_mse_time > threshold), color='red', alpha=0.3)
    axs[3].legend()

    plt.show()

if __name__ == "__main__":
    SR = 44100
    NPS = 256
    BITS_PER_FRAME = 16
    SNR = 30
    DEBUG_THRESHOLD = 0.0045

    device = torch.device("cuda")
    #device = torch.device("cpu")

    if device.type == "cuda":
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using CPU, expect worse performance")
    
    model = SpectrogramAE2D().to(device)

    try:
        model.load_state_dict(torch.load("detector.pth", map_location=device))
        print("Weights loaded successfully")
    except FileNotFoundError:
        print("Weights file is missing")
        exit()
    
    #threshold = DEBUG_THRESHOLD
    threshold = get_threshold(model=model, sr=SR, nps=NPS, device=device, snr=SNR, num_calibration_frames=50)
    anomalous_frame = generate_frame(bpf=BITS_PER_FRAME, snr=SNR, anomaly='clipping')

    detect_anomaly(model, anomalous_frame, SR, NPS, device, threshold)