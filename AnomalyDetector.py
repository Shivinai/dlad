import torch
import numpy as np
import matplotlib.pyplot as plt
from Autoencoder import SpectrogramAE2D
from Autoencoder import SignalAE1D
from DataLoader import build_spectrogram
from scipy.signal import medfilt
from Macros import generate_frame

def get_threshold_2d(model, sr, nps, device, snr, num_calibration_frames=1000):
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

def get_threshold_1d(model, sr, nps, device, snr, num_calibration_frames=1000):
    model.eval()
    all_mse = []
    
    with torch.no_grad():
        for _ in range(num_calibration_frames):
            frame = generate_frame(bpf=16, snr=snr) 
            
            num_frames = len(frame) // nps
            frames_1d = frame[:num_frames * nps].reshape(num_frames, 1, nps)
            
            tensor_1d = torch.tensor(frames_1d, dtype=torch.float32).to(device)
            recon = model(tensor_1d).squeeze().cpu().numpy()
            
            mse = np.mean((frames_1d.squeeze() - recon)**2, axis=1)
            all_mse.extend(mse)
            
    all_mse = np.array(all_mse)
    return np.mean(all_mse) + 4 * np.std(all_mse)

def detect_anomaly(model_2d, model_1d, anomalous_signal, sr, nps, device, threshold_2d, threshold_1d):
    model_2d.eval()
    model_1d.eval()

    anom_spec, f, t = build_spectrogram(anomalous_signal, sr, nps)
    anom_tensor_2d = torch.tensor(anom_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        anom_recon_2d = model_2d(anom_tensor_2d).squeeze().cpu().numpy()
    mse_2d = np.mean((anom_spec - anom_recon_2d)**2, axis=0)
    mse_2d_smoothed = medfilt(mse_2d, kernel_size=5)

    num_frames = len(anomalous_signal) // nps
    trimmed_signal = anomalous_signal[:num_frames * nps]
    frames_1d = trimmed_signal.reshape(num_frames, nps)
    
    tensor_1d = torch.tensor(frames_1d, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        recon_1d = model_1d(tensor_1d).squeeze().cpu().numpy()
    
    mse_1d = np.mean((frames_1d - recon_1d)**2, axis=1)
    mse_1d_smoothed = medfilt(mse_1d, kernel_size=5)

    mask_2d = mse_2d_smoothed > threshold_2d
    mask_1d = mse_1d_smoothed > threshold_1d

    anom_intervals = []
    window_dur = nps / sr

    for idx in np.where(mask_2d)[0]:
        t_center = t[idx] if idx < len(t) else t[-1]
        anom_intervals.append((max(0, t_center - window_dur/2), t_center + window_dur/2))

    for idx in np.where(mask_1d)[0]:
        start_t = idx * window_dur
        end_t = (idx + 1) * window_dur
        anom_intervals.append((start_t, end_t))

    merged_intervals = []
    if anom_intervals:
        anom_intervals.sort(key=lambda x: x[0])
        merged_intervals = [anom_intervals[0]]
        for current in anom_intervals[1:]:
            last = merged_intervals[-1]
            if current[0] <= last[1]: 
                merged_intervals[-1] = (last[0], max(last[1], current[1]))
            else:
                merged_intervals.append(current)

    min_len = min(len(mse_2d_smoothed), len(mse_1d_smoothed))
    time_axis_signal = np.linspace(0, len(anomalous_signal) / sr, len(anomalous_signal))
    
    fig, axs = plt.subplots(5, 1, figsize=(14, 16))
    plt.subplots_adjust(hspace=0.5)

    axs[0].plot(time_axis_signal, anomalous_signal, color='blue', alpha=0.7)
    axs[0].set_title('Detected anomalies')
    axs[0].set_ylabel('Amp')
    axs[0].set_xlim([time_axis_signal[0], time_axis_signal[-1]])

    if merged_intervals:
        for start_t, end_t in merged_intervals:
            axs[0].axvspan(start_t, end_t, color='red', alpha=0.3)

    axs[1].pcolormesh(t, f, anom_spec, shading='gouraud')
    axs[1].set_title('Original Spectrogram')
    axs[2].pcolormesh(t, f, anom_recon_2d, shading='gouraud')
    axs[2].set_title('2D Reconstructed Spectrogram')

    axs[3].plot(t[:min_len], mse_2d_smoothed[:min_len], color='purple')
    axs[3].set_title('2D Reconstruction Error (Spectrogram)')
    axs[3].axhline(threshold_2d, color='red', linestyle='--')
    axs[3].set_xlim([t[0], t[min_len-1] if min_len-1 < len(t) else t[-1]])

    t_1d = np.arange(num_frames) * window_dur + (window_dur / 2)
    
    axs[4].plot(t_1d[:min_len], mse_1d_smoothed[:min_len], color='green')
    axs[4].set_title('1D Reconstruction Error (Raw Signal)')
    axs[4].axhline(threshold_1d, color='red', linestyle='--')
    axs[4].set_xlabel('Time')
    axs[4].set_xlim([t_1d[0], t_1d[min_len-1] if min_len-1 < len(t_1d) else t_1d[-1]])

    plt.show()

if __name__ == "__main__":
    SR = 44100
    NPS = 256
    BITS_PER_FRAME = 16
    SNR = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_2d = SpectrogramAE2D().to(device)
    model_1d = SignalAE1D().to(device)

    try:
        model_2d.load_state_dict(torch.load("detector_2d.pth", map_location=device))
        model_1d.load_state_dict(torch.load("detector_1d.pth", map_location=device))
        print("Both models loaded successfully")
    except FileNotFoundError as e:
        print(f"Weights file missing: {e}")
        exit()
    
    print("Calibrating 2D model...")
    threshold_2d = get_threshold_2d(model_2d, SR, NPS, device, SNR, num_calibration_frames=50)
    
    print("Calibrating 1D model...")
    threshold_1d = get_threshold_1d(model_1d, SR, NPS, device, SNR, num_calibration_frames=50)
    
    anomalous_frame = generate_frame(bpf=BITS_PER_FRAME, snr=SNR, anomaly='dropout')

    detect_anomaly(
        model_2d=model_2d, 
        model_1d=model_1d, 
        anomalous_signal=anomalous_frame, 
        sr=SR, 
        nps=NPS, 
        device=device, 
        threshold_2d=threshold_2d, 
        threshold_1d=threshold_1d
    )