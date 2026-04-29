import torch
import numpy as np
import matplotlib.pyplot as plt
from Autoencoder import SpectrogramAE2D
from Autoencoder import SignalAE1D
from DataLoader import build_spectrogram
from scipy.signal import medfilt
from Macros import generate_frame
import matplotlib.gridspec as gridspec
from torchinfo import summary

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

import matplotlib.gridspec as gridspec

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
    trimmed_signal = anomalous_signal[tenso:num_frames * nps]
    frames_1d = trimmed_signal.reshape(num_frames, nps)
    
    tensor_1d = torch.tensor(frames_1d, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        recon_1d = model_1d(tensor_1d).squeeze().cpu().numpy()
    
    mse_1d = np.mean((frames_1d - recon_1d)**2, axis=1)
    mse_1d_smoothed = medfilt(mse_1d, kernel_size=5)
    
    recon_1d_flat = recon_1d.flatten()

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
    time_axis_1d_flat = np.linspace(0, len(trimmed_signal) / sr, len(trimmed_signal))
    t_1d_err = np.arange(num_frames) * window_dur + (window_dur / 2)
    
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(4, 2, hspace=0.4, wspace=0.15)

    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(time_axis_signal, anomalous_signal, color='blue', alpha=0.7)
    ax_main.set_title('Обнаруженные аномалии')
    ax_main.set_ylabel('Амплитуда')
    ax_main.set_xlim([time_axis_signal[0], time_axis_signal[-1]])

    if merged_intervals:
        for start_t, end_t in merged_intervals:
            ax_main.axvspan(start_t, end_t, color='red', alpha=0.3)

    ax_spec_orig = fig.add_subplot(gs[1, 0])
    ax_spec_orig.pcolormesh(t, f, anom_spec, shading='gouraud')
    ax_spec_orig.set_title('Исходная спектрограмма')
    ax_spec_orig.set_ylabel('Частота, Гц')

    ax_spec_recon = fig.add_subplot(gs[1, 1])
    ax_spec_recon.pcolormesh(t, f, anom_recon_2d, shading='gouraud')
    ax_spec_recon.set_title('Восстановленная спектрограмма')


    ax_1d_orig = fig.add_subplot(gs[2, 0])
    ax_1d_orig.plot(time_axis_1d_flat, trimmed_signal, color='royalblue', alpha=0.8)
    ax_1d_orig.set_title('Исходный сигнал')
    ax_1d_orig.set_ylabel('Амплитуда')
    ax_1d_orig.set_xlim([time_axis_1d_flat[0], time_axis_1d_flat[-1]])

    ax_1d_recon = fig.add_subplot(gs[2, 1])
    ax_1d_recon.plot(time_axis_1d_flat, recon_1d_flat, color='darkorange', alpha=0.8)
    ax_1d_recon.set_title('Восстановленный сигнал')
    ax_1d_recon.set_xlim([time_axis_1d_flat[0], time_axis_1d_flat[-1]])

    ax_err_2d = fig.add_subplot(gs[3, 0])
    ax_err_2d.plot(t[:min_len], mse_2d_smoothed[:min_len], color='purple')
    ax_err_2d.set_title('Ошибка реконструкции двухмерного автоэнкодера')
    ax_err_2d.axhline(threshold_2d, color='red', linestyle='--')
    ax_err_2d.set_xlabel('Время')
    ax_err_2d.set_ylabel('СКО')
    ax_err_2d.set_xlim([t[0], t[min_len-1] if min_len-1 < len(t) else t[-1]])

    ax_err_1d = fig.add_subplot(gs[3, 1])
    ax_err_1d.plot(t_1d_err[:min_len], mse_1d_smoothed[:min_len], color='green')
    ax_err_1d.set_title('Ошибка реконструкции одномерного автоэнкодера')
    ax_err_1d.axhline(threshold_1d, color='red', linestyle='--')
    ax_err_1d.set_xlabel('Время')
    ax_err_1d.set_xlim([t_1d_err[0], t_1d_err[min_len-1] if min_len-1 < len(t_1d_err) else t_1d_err[-1]])

    plt.show()

if __name__ == "__main__":
    SR = 44100
    NPS = 256
    BITS_PER_FRAME = 16
    SNR = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU, expect perfromance issues")
    
    model_2d = SpectrogramAE2D().to(device)
    summary(model_2d, input_size=(1, 1, 129, 1))

    model_1d = SignalAE1D().to(device)
    summary(model_1d, input_size=(1, 1, 256))

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
    
    anomalous_frame = generate_frame(bpf=BITS_PER_FRAME, snr=SNR, anomaly='phase_flip')

    detect_anomaly(
        model_2d=model_2d, 
        model_1d=model_1d, 
        anomalous_signal=anomalous_frame, 
        sr=SR, 
        nps=NPS, 
        device=device, 
        threshold_2d=threshold_2d, 
        threshold_1d=0.09
    )