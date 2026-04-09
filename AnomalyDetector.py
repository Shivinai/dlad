import torch
import numpy as np
import matplotlib.pyplot as plt
from Autoencoder import SpectrogramAE
from SignalGenerator import ModemSignalGenerator
from NoiseGenerator import ModemNoiseGenerator
from AnomalyInjector import ModemAnomalyInjector
from DataLoader import build_spectrogram

def get_threshold(model, generator, sr, nps, device, snr_db=20, num_calibration_frames=50):

    model.eval()
    all_mse_values = []
        
    with torch.no_grad():
        for _ in range(num_calibration_frames):
            random_bits = np.random.randint(0, 2, 16)
            clean_frame = generator.GenerateFrame(random_bits)
            noisy_normal_frame = ModemNoiseGenerator(clean_frame, snr_db=snr_db).GenerateNoise()
            
            spec, _, _ = build_spectrogram(noisy_normal_frame, sr, nps)
            spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            recon = model(spec_tensor).squeeze().cpu().numpy()
            
            mse_time = np.mean((spec - recon)**2, axis=0)
            all_mse_values.extend(mse_time)
            
    global_mean = np.mean(all_mse_values)
    global_std = np.std(all_mse_values)
    
    robust_threshold = global_mean + 4 * global_std
    
    return robust_threshold

def detect_anomaly(model, anomalous_signal, sr, nps, device, THRESHOLD):
    model.eval()

    anom_spec, f, t = build_spectrogram(anomalous_signal, sr, nps)
    anom_tensor = torch.tensor(anom_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        anom_recon = model(anom_tensor).squeeze().cpu().numpy()

    anom_mse_time = np.mean((anom_spec - anom_recon)**2, axis=0)

    anomaly_indices = np.where(anom_mse_time > THRESHOLD)[0]
    
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
    
    axs[3].fill_between(t, anom_mse_time, THRESHOLD, where=(anom_mse_time > THRESHOLD), color='red', alpha=0.3)
    axs[3].legend()

    plt.show()

if __name__ == "__main__":
    SR = 44100
    NPS = 256
    BITS_PER_FRAME = 16
    SNR_DB = 20

    #device = torch.device("cuda")
    device = torch.device("cpu")

    if device.type == "cuda":
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using CPU, expect worse performance")
    
    model = SpectrogramAE().to(device)

    try:
        model.load_state_dict(torch.load("detector.pth", map_location=device))
        print("Weights loaded successfully")
    except FileNotFoundError:
        print("Weights file is missing")
        exit()

    generator = ModemSignalGenerator(fs=SR, bit_dur=0.02)
    
    robust_threshold = get_threshold(model=model, generator=generator, sr=SR, nps=NPS, device=device, snr_db=SNR_DB, num_calibration_frames=50)

    random_bits = np.random.randint(0, 2, BITS_PER_FRAME)
    clean_frame = generator.GenerateFrame(random_bits)

    noise_gen = ModemNoiseGenerator(clean_frame, snr_db=SNR_DB)
    reference_noisy_frame = noise_gen.GenerateNoise()

    injector = ModemAnomalyInjector(reference_noisy_frame)
    anomalous_frame = injector.apply_dropout()

    detect_anomaly(model, anomalous_frame, SR, NPS, device, robust_threshold)